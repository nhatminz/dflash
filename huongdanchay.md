# Hướng dẫn chạy DFlash + ReflexOPD

Project `dflash/` nằm ngang hàng với `SpecNaacl/` và không import code của
SpecNaacl khi huấn luyện. Hai chế độ dùng chung backend NVIDIA DFlash:

* `METHOD=dflash_reflex`: DFlash, GRPO và ReflexOPD với A được huấn luyện.
* `METHOD=dflash`: cùng DFlash/GRPO/co-training, bỏ đường correction và OPD.

Máy đang dùng để phát triển là RTX 3090, Torch 2.4/CUDA 12. Các kiểm thử CPU,
Triton và DFlash nhỏ đã chạy ở đây. **Chưa chạy NeMo GRPO hoặc pretraining
Megatron đầy đủ trên B200**, vì thiếu runtime CUDA 13 và đường dẫn model/data
được yêu cầu. Các lệnh dưới đây dành cho máy có dữ liệu và môi trường phù hợp;
không hiểu kết quả kiểm thử nhỏ là xác nhận chạy dài đã thành công.


## Cập nhật correctness/fairness cuối

Mặc định mới `FAIRNESS_PROFILE=specnaacl_matched` sẽ **dừng preflight** với backend
đã pin: LoRA QKV/gate-up gộp không tương đương bảy adapter PEFT độc lập; chưa có
cadence target bốn cohort độc lập với draft/A; và cách lọc nhóm reward/mask khác.
Không coi profile này là cấu hình chạy thành công. Để thực hiện các lệnh native
trong tài liệu, đặt rõ:

```bash
export FAIRNESS_PROFILE=native_dflash
export METRICS_MODE=production
```

Native vẫn train target đầy đủ, nên chỉ so baseline/Reflex trong cùng profile,
không kết luận tương đương hoàn toàn SpecNaacl. `BETA=.04`, `EPSILON=.1` đã đi vào
loss thật; sequence mean và gradient k3 đã đối chiếu trực tiếp với loss Spec.
Đọc [báo cáo cuối](validation/final_reflex_optimization_report.md) và
[ma trận fairness](validation/fairness_config_matrix.json) trước khi chạy B200.

Code SpecNaacl hiện tại cập nhật target sau mỗi rollout có nhóm reward biến thiên,
rồi xóa batch. `ACCUMULATION_STEPS=4` không tạo bốn-rollout readiness; bảng nominal
256 response trong prompt khác hành vi nguồn hiện tại. Draft/A vẫn cập nhật mỗi
rollout thành công; A dùng gradient OPD, B shared trong cohort và reset giữa cohort.

Đo tốc độ: dùng production cho cả hai method. Đo AAL: dùng instrumented cho cả hai.
Baseline production không thu AAL; log null, không giả số 0. Profile/diagnostics
là lần chạy riêng. Để tái hiện objective NeMo cũ hoặc resume checkpoint cũ, thêm
`GRPO_OBJECTIVE=original_nemo BETA=0 EPSILON=.2` và giữ nguyên các overrides cũ.
Nếu có khác biệt bất tương thích, preflight báo lỗi, không xóa checkpoint.

## 1. Chuẩn bị môi trường riêng

```bash
cd /đường/dẫn/SpecDecode/dflash
nvidia-smi
nvcc --version
command -v uv
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"
export NVTE_CUDA_ARCHS=100
export TORCH_CUDA_ARCH_LIST=10.0
bash scripts/setup_environment.sh
export PYTHONPATH="$PWD:$PWD/third_party/nemo-rl${PYTHONPATH:+:$PYTHONPATH}"
.venv/bin/python scripts/validate_environment.py --require-runtime \
  --output outputs/environment.json
```

Thay đường dẫn `cd` bằng thư mục thực tế. Cần Python >=3.13.14,<3.14,
Torch 2.11.0+cu130, Triton 3.6.0, vLLM 0.25.1 và CUDA development toolkit
13.x. Các dependency khác được lấy nguyên từ `uv.lock` của NeMo đã pin.
Driver CUDA 13 phải phù hợp; xem [ENVIRONMENT.md](ENVIRONMENT.md).
Không cài vào environment của SpecNaacl. Setup chỉ chạy một lần và có thể
biên dịch TE/DeepEP; launcher không tự cài hoặc nâng cấp package.

Có ba environment: `.venv` cho driver/reward, `.venv-mcore` cho pretrain và
Megatron actor, `.venv-vllm` cho inference actor. Không gộp hai extra MCore và
vLLM vì NeMo khai báo chúng xung đột. Nếu dùng environment đã chuẩn bị sẵn:

```bash
export PYTHON_BIN=/absolute/base/bin/python
export DFLASH_MCORE_PYTHON=/absolute/mcore/bin/python
export DFLASH_VLLM_PYTHON=/absolute/vllm/bin/python
"$DFLASH_VLLM_PYTHON" scripts/patch_upstream.py --vllm-installed
```

## 2. Kiểm tra model và dữ liệu

Mặc định dùng Qwen3-4B ở `/workspace/storage-shared/models/Qwen3-4B`.
Qwen3-8B và Qwen3-1.7B có launcher riêng, nhưng phải có checkpoint thật
ở đường dẫn tương ứng. Không tự tải hoặc thay bằng model khác.

```bash
.venv/bin/python scripts/validate_environment.py \
  --model /workspace/storage-shared/models/Qwen3-4B \
  --data /workspace/storage-shared/nlp/minhpn19/data/sharegpt/ShareGPT_V4.3_unfiltered_cleaned_split.json
.venv/bin/python scripts/prepare_data.py --dataset simplelr \
  --model /workspace/storage-shared/models/Qwen3-4B --max-samples 16
.venv/bin/python scripts/prepare_data.py --dataset gsm8k --max-samples 16
.venv/bin/python scripts/prepare_data.py --dataset dapo --max-samples 16
```

Ba đường dẫn GRPO mặc định:

| DATASET | File dưới DATA_ROOT |
|---|---|
| `simplelr` | `simplelr_abel_level3to5/train.parquet` |
| `gsm8k` | `gsm8k/main/train-00000-of-00001.parquet` |
| `dapo` | `DAPO-Math-17k-Processed/en/train-00000-of-00001.parquet` |

`DATA_ROOT=/workspace/storage-shared/nlp/minhpn19/data`.
`DATASET_PATH` ghi đè đúng file được chọn, không có fallback. Parser giữ
cách lấy question/prompt, ground truth, phần sau `####`, bọc `\\boxed{}`,
chọn subset theo seed và lọc prompt theo chat template như SpecNaacl.
Reward dùng math-verify và hệ số format 0.2 của SpecNaacl, không so chuỗi
đáp án đơn giản.

## 3. Pretrain DFlash trên ShareGPT

```bash
bash pretrain_qwen3_4b.sh
bash pretrain_qwen3_8b.sh
# Model nhỏ hơn, nếu đã có model local:
bash pretrain_qwen3_1p7b.sh
```

Pretrain khởi tạo drafter từ đầu, dùng target đóng băng để lấy auxiliary
hidden states và soft targets. Mỗi block có anchor + MASK, attention nhìn
trunk trước anchor và toàn bộ block của chính nó. Loss và exponential
slot weighting lấy từ NVIDIA PR #3712. Mặc định block 16 gồm **1 anchor và
15 proposal**. MASK row là **151669**, một hàng embedding chưa dùng trong
vocab Qwen3; đây không phải ID `<|fim_pad|>`.

```bash
CUDA_VISIBLE_DEVICES=0 PRETRAIN_EPOCHS=5 PRETRAIN_LR=5e-5 \
PRETRAIN_BATCH_SIZE=1 PRETRAIN_ACCUMULATION_STEPS=8 \
PRETRAIN_MAX_LENGTH=2048 PRETRAIN_SAVE_INTERVAL=500 \
bash pretrain_qwen3_4b.sh
```

Thông số pretrain chính:

| Biến | Mặc định / ý nghĩa |
|---|---|
| `PRETRAIN_DATASET` | `sharegpt` |
| `PRETRAIN_DATASET_PATH` | file ShareGPT đã nêu |
| `PRETRAIN_EPOCHS`, `PRETRAIN_LR` | 5, 5e-5 |
| `PRETRAIN_BATCH_SIZE` | 1 trên mỗi GPU |
| `PRETRAIN_ACCUMULATION_STEPS` | 8 cho 4B/1.7B, 16 cho 8B |
| `PRETRAIN_MAX_LENGTH` | 2048 |
| `PRETRAIN_MAX_SAMPLES`, `PRETRAIN_MAX_STEPS` | 0 = không giới hạn |
| `PRETRAIN_SAVE_INTERVAL` | 500 optimizer steps |
| `PRETRAIN_DATALOADER_WORKERS` | 4 |
| `PRETRAIN_WARM_START` | rỗng = khởi tạo từ đầu; có thể đặt drafter tương thích |
| `DFLASH_LAYERS`, `DFLASH_ANCHORS_PER_SEQ` | 3 draft layers, 16 anchors |
| `DFLASH_LOSS_CHUNK_SIZE` | 32 slot logits mỗi chunk loss |

BF16 dùng cho forward; master weights, gradient accumulation và Adam state
dùng FP32. Tqdm hiển thị epoch, optimizer step, loss, LR, tokens, thời gian
step và elapsed. Chỉ rank chính hiển thị progress.

Checkpoint export mới nhất nằm tại:

```text
outputs/pretrain/qwen3_4b/latest_checkpoint
outputs/pretrain/qwen3_4b/<run>/checkpoints/draft/step<N>/
outputs/pretrain/qwen3_4b/<run>/checkpoints/resume/latest.pt
```

Export chứa `model.safetensors`, `config.json`, `dflash_manifest.json` và
`opd_projector.pt`. Checkpoint resume chứa master weights/optimizer/scheduler,
RNG các rank, epoch và batch kế tiếp. File safetensors dùng tên weight tương
thích với loader DFlash của NeMo/vLLM.

## 4. GRPO với Reflex

```bash
DATASET=simplelr bash train_qwen3_4b.sh
DATASET=gsm8k bash train_qwen3_4b.sh
DATASET=dapo bash train_qwen3_4b.sh
```

Nếu không truyền `DRAFT_CHECKPOINT`, launcher lấy `latest_checkpoint` của
pretrain tương ứng. Nếu chưa pretrain, launcher dừng với lỗi đường dẫn.

```bash
DRAFT_CHECKPOINT=/absolute/path/to/compatible/dflash \
DATASET=simplelr bash train_qwen3_4b.sh
DATASET=simplelr bash train_qwen3_8b.sh
```

A có group AdamW riêng tên `draft_opd_projector`, LR 1e-4, weight decay 0.
Gradient A đến từ phản hồi rollout, được cộng theo tổng gradient/tổng trọng
số rồi cập nhật tại boundary optimizer thành công của Megatron. Optimizer và
scheduler của A được lưu riêng cùng checkpoint chung. Draft parameters thông
thường vẫn dùng distillation gradient và optimizer gốc của NVIDIA.
A mới được gửi sang inference ở boundary cohort tiếp theo, cùng nhịp refit
draft/target; kiểm tra version ngăn sử dụng feedback A cũ.

Một B FP32 được chia sẻ bởi toàn bộ `BATCH_SIZE * RESPONSES_PER_PROMPT`
responses trong cohort. B cập nhật sau verification, không reset khi một
response kết thúc, và không được lưu thành model weight. Mỗi cohort mới
reset B và tái sử dụng buffers.

## 5. Baseline DFlash

```bash
DATASET=simplelr bash train_qwen3_4b_dflash.sh
DATASET=gsm8k bash train_qwen3_4b_dflash.sh
DATASET=dapo bash train_qwen3_4b_dflash.sh
DATASET=simplelr bash train_qwen3_8b_dflash.sh
```

Baseline không gọi OPD correction/feedback kernels. Hook đếm acceptance vẫn
hoạt động để log AAL bằng cùng quy ước với Reflex. Các thông số sampling,
loss, pretrained drafter và online co-training giống nhau giữa hai chế độ.

## 6. Thông số GRPO có thể ghi đè

| Nhóm | Biến và mặc định |
|---|---|
| Model/device | `MODEL_KEY`, `MODEL`, `CUDA_VISIBLE_DEVICES=0`, `NPROC_PER_NODE=1` |
| Batch | `BATCH_SIZE=8`, `RESPONSES_PER_PROMPT=8`, `ACCUMULATION_STEPS=4`, `TRAIN_MICRO_BATCH_SIZE=1` |
| LR | `TARGET_LR=1e-6`, `DRAFT_LR=1e-4`, `DFLASH_DRAFT_LR=$DRAFT_LR` |
| Generation | `GEN_MAX_LENGTH=2048`, `MAX_PROMPT_LENGTH=2048`, `TEMPERATURE=1`, `TOP_P=0.95` |
| DFlash | `DFLASH_BLOCK_SIZE=16`, `DFLASH_LOSS_WEIGHT=0.1`, `DFLASH_MASK_TOKEN_ID=151669` |
| Reflex | `OPD_RANK=8`, `OPD_TOPK=16`, `OPD_FAST_LR=0.01`, `OPD_UPDATE_INTERVAL=1` |
| A | `OPD_TRAIN_PROJECTOR=1`, `OPD_PROJECTOR_LR=1e-4`, `OPD_PROJECTOR_WEIGHT_DECAY=0` |
| Feedback | `OPD_ACCEPTED_WEIGHT=1`, `OPD_REJECTED_WEIGHT=1` |
| Kernels | `OPD_UPDATE_STREAM=1`, `OPD_PROPOSAL_MODE=auto`, `OPD_DENSE_IMPLEMENTATION=auto` |
| Profiling | `OPD_PROFILE=0`, `OPD_DIAGNOSTICS=0`, `OPD_DISPATCH_CACHE=` |
| Data/seed | `DATASET=simplelr`, `TRAIN_SUBSET_SEED=42`, `MAX_TRAIN_SAMPLES=0` |
| Duration | `TRAIN_EPOCHS=1`, `MAX_GRPO_STEPS=1000000`, `SAVE_INTERVAL=10` |
| Logging | `LOG_INTERVAL=1`, `ROLLOUT_LOG_FLUSH_INTERVAL=1` |
| Memory/reward | `VLLM_MEMORY_UTILIZATION=0.65`, `REWARD_WORKERS=8` |
| Output | `OUTPUT_ROOT=$PWD/outputs`, `RUN_NAME=...`, `RUN_DIR=...`, `RESUME=` |

`ACCUMULATION_STEPS` quy định số microbatch tối thiểu; physical microbatch
còn bị chặn bởi `TRAIN_MICRO_BATCH_SIZE` để tiết kiệm VRAM. Với batch 8x8,
physical microbatch 1 thì Megatron dùng 64 microbatch cho một global update.
`DRAFT_ACCUMULATION_STEPS` phải bằng 1 vì upstream cập nhật draft cùng
optimizer boundary với target. Giá trị khác bị từ chối, không bị bỏ qua.

GRPO hiện hỗ trợ **một GPU, một inference engine, TP=PP=CP=1**. Không chạy
multi-engine vì như vậy B sẽ bị chia theo engine. `NPROC_PER_NODE != 1` cho
GRPO bị từ chối trước khi khởi tạo GPU. Pretrain có thể dùng nhiều rank DP;
phần đó chưa được kiểm chứng trên nhiều GPU thực.

`LOG_INTERVAL` điều khiển refresh postfix tqdm; metric vẫn có một dòng ở
mỗi DataLoader iteration. `ROLLOUT_LOG_FLUSH_INTERVAL` điều khiển tần suất
fsync bền vững của log. `OPD_DIAGNOSTICS=1` thêm B norm/active vocab rows
ở boundary cohort; không lấy scalar CPU trong từng verification round.

## 7. Resume và dry run

```bash
RESUME=auto bash pretrain_qwen3_4b.sh
RESUME=auto DATASET=simplelr bash train_qwen3_4b.sh
RESUME=/absolute/path/to/old/run DATASET=simplelr bash train_qwen3_4b.sh
DRY_RUN=true bash pretrain_qwen3_4b.sh
DRY_RUN=true DATASET=dapo bash train_qwen3_4b.sh
```

`active_run` chọn run để resume; `latest_run` cập nhật khi launcher hoàn tất.
Giữ nguyên method, dataset, model, batch, optimizer và sampling của run cũ.
Có thể tăng `MAX_GRPO_STEPS` hoặc `PRETRAIN_MAX_STEPS` để tiếp tục sau smoke
test; lịch pretrain vẫn tính theo tổng epochs ban đầu. Cấu hình không tương
thích sẽ bị từ chối. B bắt đầu lại ở cohort mới; A, pending gradients, version,
optimizer, scheduler, RNG, dataloader và cumulative counters được khôi phục.
Log sau checkpoint cuối bị cắt để tránh đếm trùng khi phục hồi sau crash.

Dry run chỉ in command/config, không import Torch, nạp model hoặc dùng GPU.
Có thể chạy dry run trước khi cài environment hay có dữ liệu local.

## 8. Smoke test ít bước trên máy đã chuẩn bị

```bash
PRETRAIN_MAX_SAMPLES=8 PRETRAIN_MAX_LENGTH=512 \
PRETRAIN_ACCUMULATION_STEPS=2 PRETRAIN_MAX_STEPS=2 \
PRETRAIN_SAVE_INTERVAL=1 PRETRAIN_DATALOADER_WORKERS=0 \
RUN_NAME=pretrain_smoke bash pretrain_qwen3_4b.sh

DATASET=simplelr BATCH_SIZE=2 RESPONSES_PER_PROMPT=2 \
ACCUMULATION_STEPS=2 TRAIN_MICRO_BATCH_SIZE=1 \
MAX_TRAIN_SAMPLES=8 MAX_PROMPT_LENGTH=1024 GEN_MAX_LENGTH=32 \
MAX_GRPO_STEPS=1 SAVE_INTERVAL=1 REWARD_WORKERS=1 \
RUN_NAME=reflex_smoke OPD_DIAGNOSTICS=1 bash train_qwen3_4b.sh
```

Để kiểm tra resume, chạy lại lệnh GRPO với cùng overrides, thêm `RESUME=auto`
và đổi `MAX_GRPO_STEPS=2`. Chạy baseline smoke bằng wrapper `_dflash.sh` với
`RUN_NAME=baseline_smoke` và cùng draft checkpoint. Các smoke test này thực
hiện optimizer step thật; chúng **chưa được chạy ở workspace hiện tại**.

Kiểm thử nhỏ không cần model local:

```bash
PYTHONPATH="$PWD:$PWD/third_party/nemo-rl" .venv-mcore/bin/python -m pytest tests -q
# Workspace phát triển hiện tại có thể dùng:
.venv-test/bin/python -m pytest tests -q
```

## 9. Đọc output và so sánh công bằng

```text
outputs/train/qwen3_4b/<run>/
  logs/console.log metrics.jsonl rollout_timing.csv timing.csv train_steps.csv
  logs/upstream/
  statistics/
  checkpoints/resume/step_*/policy/weights/
  checkpoints/resume/step_*/dflash_driver.pt
  ipc/feedback/<iteration-id>/*.pt
  summary.json run_config.json launch_config.json run_metadata.json resolved.yaml
```

Target và DFlash được lưu chung trong checkpoint NeMo, không nhân bản toàn
bộ weights vào hai thư mục `target/` và `draft/`. Sidecar A nằm trong
`policy/weights/reflex_rank0.pt`; số optimizer steps nằm trong
`dflash_stats_rank0.pt`. `dflash_driver.pt` giữ cumulative counters, RNG
driver/inference và danh sách rollout đã commit. RNG trainer gồm cả CUDA
tracker của Megatron. Packet feedback chỉ chứa gradient A, RNG và số liệu nhỏ,
không chứa target logits hay B.

`iter_aal = iter_accepted_tokens / iter_verification_rounds`. Accepted path
gồm prefix được chấp nhận và một token mới của target (recovery hoặc bonus),
sau khi cắt EOS/length. `iter_accepted_draft_tokens` chỉ đếm proposal được
chấp nhận. Bonus không được coi là accepted draft hoặc OPD feedback.
`cumulative_aal` lấy tổng accepted path chia tổng rounds, không trung bình
các AAL iteration. `step_aal` là alias của `iter_aal` trong luồng synchronous.

Acceptance rate = accepted draft / proposed draft, với đề xuất cắt theo
length/EOS và không đếm request bị discard. `iter_generation_time_s` gồm
rollout và boundary drain; `iter_wall_time_s` gồm rollout, reward, train,
refit, checkpoint và logging upstream trong iteration. Cumulative wall time
tính từ lúc setup driver hoàn tất, cộng thời gian đã commit trước resume;
không tính thời gian máy tắt hoặc one-off model/environment loading.

NeMo tính target và draft trong cùng forward/backward. Vì vậy thời gian
riêng cho hai phần là **null/ô trống**, còn `joint_train_time_s` là thời gian
đo thật. Diagnostics không bật cũng là null. Không coi các ô này là 0 ms.

```bash
.venv/bin/python scripts/summarize_run.py outputs/train/qwen3_4b/active_run
.venv/bin/python scripts/plot_aal.py outputs/train/qwen3_4b/active_run \
  --cumulative --output outputs/aal.png
.venv/bin/python scripts/plot_training_time.py outputs/train/qwen3_4b/active_run \
  --output outputs/training_time.png
```

Để so sánh baseline/Reflex, chốt một checkpoint pretrain có đường dẫn thật:

```bash
export DRAFT_CHECKPOINT="$(readlink -f outputs/pretrain/qwen3_4b/latest_checkpoint)"
export DATASET=simplelr TRAIN_SUBSET_SEED=42 BATCH_SIZE=8 RESPONSES_PER_PROMPT=8
export TEMPERATURE=1 TOP_P=0.95 GEN_MAX_LENGTH=2048 MAX_GRPO_STEPS=20
RUN_NAME=baseline_compare bash train_qwen3_4b_dflash.sh
RUN_NAME=reflex_compare bash train_qwen3_4b.sh
.venv/bin/python scripts/summarize_run.py \
  outputs/train/qwen3_4b/baseline_compare outputs/train/qwen3_4b/reflex_compare
.venv/bin/python scripts/plot_aal.py \
  outputs/train/qwen3_4b/baseline_compare outputs/train/qwen3_4b/reflex_compare \
  --cumulative --output outputs/compare_aal.png
```

Đây là lệnh để người dùng tự chạy; agent không tự chạy thí nghiệm 20 bước.
Chỉ kết luận nhanh hơn khi đo tổng wall time/throughput cùng điều kiện.
AAL cao hơn chưa đủ chứng minh end-to-end nhanh hơn.

## 10. Profile và chọn kernel

```bash
OPD_PROFILE=1 OPD_DIAGNOSTICS=1 MAX_GRPO_STEPS=1 \
DATASET=simplelr bash train_qwen3_4b.sh

PYTHONPATH="$PWD" .venv-vllm/bin/python scripts/benchmark_rollout.py \
  --mode micro --rows 64 --hidden 2560 --vocab 151936 --rank 8 --topk 16 \
  --repeats 10 --output outputs/benchmarks/b200_qwen3_4b.json \
  --cache outputs/benchmarks/b200_dispatch.json
OPD_DISPATCH_CACHE="$PWD/outputs/benchmarks/b200_dispatch.json" bash train_qwen3_4b.sh
```

Microbenchmark có ba mức active vocabulary, so sparse/fused dense/GEMM và
feedback; không chạy transformer hoặc GRPO. Cache chỉ dùng cho GPU và shape
trùng hoàn toàn. `--rows` là số MASK slots thực tế trong một proposal batch,
không phải số prompt; có thể thay bằng shape đang quan tâm. Shape không có
cache dùng sparse fused và proposal fused bảo thủ; không tự sweep. Kết quả RTX 3090 nhỏ trong
`validation/cuda_microbenchmark.json` không được dùng làm số liệu B200.

Profile CUDA event nằm trong `metrics.jsonl.profile_ms`: target forward
(gồm prefill), DFlash forward, draft head, correction, proposal scan/selection,
feedback extraction, top-k union, A gradient và B update. Correction được fuse
vào scan thì một phần thời gian nằm trong `proposal_sampling_ms`; không cộng
các trường trùng phạm vi để suy ra speedup. Refit được đo ở timing upstream.
Tắt profile khi so thời gian training bình thường.

Để benchmark rollout thật ngoài GRPO, tạo JSON list các prompt đã định dạng
và chạy trong environment vLLM đã patch:

```bash
PYTHONPATH="$PWD" .venv-vllm/bin/python scripts/benchmark_rollout.py \
  --mode rollout --model /workspace/storage-shared/models/Qwen3-4B \
  --draft "$DRAFT_CHECKPOINT" --prompts /absolute/path/prompts.json \
  --method dflash_reflex --max-tokens 64 \
  --output outputs/benchmarks/real_rollout.json
```

## 11. Warm-start reference Qwen3-8B

Chỉ dùng `z-lab/Qwen3-8B-DFlash-b16` với target Qwen3-8B tương thích.
Revision đã kiểm tra nằm trong `third_party/reference_hf_revisions.json`.
Tải checkpoint vào shared storage bằng công cụ của bạn, rồi:

```bash
MODEL=/workspace/storage-shared/models/Qwen3-8B \
PRETRAIN_WARM_START=/absolute/path/Qwen3-8B-DFlash-b16 \
bash pretrain_qwen3_8b.sh
```

Validation kiểm tra tokenizer vocabulary theo hash semantic của tokenizer
Qwen3-8B đã pin, hidden/head dimensions, 36 target layers, 5 draft layers,
auxiliary layer IDs, MASK row, block 16 và toàn bộ shapes weights. Không ghép
checkpoint 8B này với Qwen2.5 hay target 4B. Qwen2.5 có QKV bias nên không
được quảng cáo hỗ trợ trong bản tích hợp này.

## Benchmark cuối và cấu hình B200 được đề nghị

Giữ A trainable, B shared, rank 8, top-k 16, fast LR .01, feedback mỗi round.
Chạy benchmark ngắn trên B200 một lần rồi dùng cache đúng GPU:

```bash
mkdir -p outputs/benchmarks
PYTHONPATH="$PWD" .venv-vllm/bin/python scripts/benchmark_reflex_hotpath.py \
  --model-config /workspace/storage-shared/models/Qwen3-4B/config.json \
  --requests 1 8 32 64 --repeats 15 \
  --cache outputs/benchmarks/b200_dispatch.json \
  --output outputs/benchmarks/b200_hotpath.json
export OPD_DISPATCH_CACHE="$PWD/outputs/benchmarks/b200_dispatch.json"
FAIRNESS_PROFILE=native_dflash DATASET=simplelr METHOD=dflash_reflex bash train_qwen3_4b.sh
FAIRNESS_PROFILE=native_dflash DATASET=simplelr METHOD=dflash bash train_qwen3_4b_dflash.sh
# Dataset khác và resume:
FAIRNESS_PROFILE=native_dflash DATASET=gsm8k bash train_qwen3_4b.sh
FAIRNESS_PROFILE=native_dflash DATASET=dapo bash train_qwen3_4b.sh
FAIRNESS_PROFILE=native_dflash DATASET=simplelr RESUME=auto bash train_qwen3_4b.sh
```

`OPD_PIPELINE=auto` dùng đo đạc theo GPU/shape; feedback density chỉ đọc ở cuối
cohort. Có thể cố định fused/selective/selective_masked để tái lập kernel.
Các candidate đều giữ q của đúng A/B/head lúc proposal, giữ logits gốc đến verify
và chặn stale version. Không nhân bản B hoặc tính lại LM head. Thêm lifetime
logits tối đa khoảng 278 MiB cho 64 request BF16 (556 MiB nếu logits FP32);
cần tính vào memory budget.

Cache phân biệt cả dtype hidden và logits. Nếu model trả logits FP32 với hidden
BF16, benchmark phải thêm `--logits-dtype float32`; không dùng lại bucket BF16.
GEMM bật thủ công giữ đúng buffer logits đã correction cho feedback; auto không
chọn GEMM vì thứ tự reduction có thể thay quyết định khi logits gần bằng nhau.

Lệnh matched chính xác, **hiện dự kiến thất bại preflight**:

```bash
python3 scripts/compare_specnaacl_config.py --model qwen3_4b --dataset simplelr --profile specnaacl_matched
DATASET=simplelr FAIRNESS_PROFILE=specnaacl_matched METHOD=dflash_reflex bash train_qwen3_4b.sh
DATASET=simplelr FAIRNESS_PROFILE=specnaacl_matched METHOD=dflash bash train_qwen3_4b_dflash.sh
```

Chưa có model/dataset/runtime phù hợp tại máy phát triển, nên chưa chạy GRPO thật
hoặc pretrain đầy đủ. Chỉ tự chạy smoke 1–2 step ở B200 khi đã cài đúng environment;
không dùng kết quả microbenchmark để khẳng định throughput sinh hoặc training.
