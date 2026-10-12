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

Launcher mặc định `RUNTIME_CHECK=auto`: kiểm tra sâu runtime một lần, lưu kết quả
vào `.cache/runtime_checks/validation.json`, rồi dùng fingerprint nhẹ ở các lần
chạy sau. Thay Python, package, lock, commit upstream hoặc patch sẽ làm cache hết
hạn. Các job khởi động cùng lúc dùng khóa chung để tránh kiểm tra sâu trùng lặp.
Setup thành công cũng ghi cache này. Model/data/checkpoint/resume vẫn được kiểm
tra mỗi lần; `DRY_RUN=true` không import runtime. Không phải cài lại thư viện.

```bash
# Mặc định: cache kiểm tra môi trường, auto ưu tiên selective_masked nếu chưa calibration.
CUDA_VISIBLE_DEVICES=0 METRICS_MODE=instrumented bash train_qwen25_3b.sh
# B200 đã được kiểm tra: bỏ ngay phần kiểm tra thư viện nặng.
RUNTIME_CHECK=skip CUDA_VISIBLE_DEVICES=0 METRICS_MODE=instrumented bash train_qwen25_3b.sh
# Ép kiểm tra đầy đủ khi cần điều tra môi trường.
RUNTIME_CHECK=full CUDA_VISIBLE_DEVICES=0 bash train_qwen25_3b.sh
```

`OPD_PIPELINE=auto` dùng calibration phù hợp GPU/dtype/model và mật độ feedback
gần đây; thiếu calibration sẽ dùng `selective_masked`. Có thể ép bằng
`OPD_PIPELINE=selective_masked`. Số vòng của từng pipeline xuất hiện trong
`logs/metrics.jsonl` (`opd_pipeline_rounds`), `rollout_timing.csv` và `summary.json`.
Dùng `METRICS_MODE=instrumented` cho **cả hai** phương pháp để xem AAL của baseline.
Giữ eager/V1, batch 8, accum 4 và checkpoint mỗi 50 step như trước.

Xem [báo cáo tối ưu](validation/reflex_performance_fix.md) để có lệnh calibration
B200 và 8 lệnh train cho 4 model. Số đo RTX 3090 không chứng minh tốc độ B200.

## Cập nhật lỗi CUDA khi rollout trên B200

Log ngày 12/10 dừng bởi `CUDA illegal memory access` trong generation; các dòng
Transformers báo thiếu docstring và thông báo Ray ở cuối không xác định kernel
gây lỗi. Crash xảy ra khi batch còn 32 request × 16 token = 512, đúng ngưỡng
CUDA graph. Đây là dấu hiệu cần điều tra, chưa phải bằng chứng kết luận kernel.

Mặc định mới `VLLM_ENFORCE_EAGER=1` áp dụng cho cả target/draft ở cả baseline và
Reflex, tránh compile/graph replay của vLLM trong lần chạy này. Cách này có thể
giảm tốc độ; phải dùng cùng giá trị khi so hai phương pháp. Reflex đã sửa buffer
token trả cho vLLM để không dùng chung vùng nhớ top-k sẽ bị tái sử dụng. Nếu
generation lỗi, không gọi RPC đọc gradient/RNG trên engine đã lỗi; traceback
được giữ và ghi thêm ở `<run>/generation_errors/` nếu worker còn trả ngoại lệ.
Nếu native process abort trước đó, xem `<run>/logs/console.log` và log worker Ray.

Chỉ cập nhật source project trên B200, không chạy lại setup/cài thư viện hoặc
pretrain. Khởi động job mới (job đã chạy không tự nhận bản sửa):

```bash
cd /workspace/storage-shared/nlp/minhpn19/dflash
source offline_packages/activate.sh
CUDA_VISIBLE_DEVICES=2 VLLM_ENFORCE_EAGER=1 bash train_qwen3_1p7b.sh
```

Lệnh chọn pretrain checkpoint theo wrapper như trước. Nếu dùng checkpoint tùy
chọn, giữ `DRAFT_CHECKPOINT=/đường/dẫn/checkpoint` của bạn. Batch mặc định 8,
accumulation 4, checkpoint mỗi 50 GRPO iteration. Khi khởi động phải thấy
`vLLM model runner: V1; target/draft eager mode: True`. `VLLM_ENFORCE_EAGER=0` dành cho lần thử
graph riêng; chưa được xác nhận an toàn trên B200. Bản sửa đã qua test CPU/CUDA
trên 3090; chưa có kết quả full GRPO trên B200.


## Sửa AAL=0 và đường chạy Reflex

`aal=N/A` ở baseline production nghĩa là không thu thống kê acceptance, đúng
thiết kế. `aal=0` kèm `iter_verification_rounds=0`, `A_version=0`,
`B_update_rounds=0`, `iter_opd_feedback_rows=0` sau nhiều iteration cho thấy
Reflex chưa chạy; không được coi là acceptance thực bằng 0.

vLLM đã pin có thể tự chọn Model Runner V2 cho các Qwen dense. Hook của project
chỉ có trên V1; `enforce_eager` không buộc chọn V1. Mặc định mới là
`VLLM_USE_V2_MODEL_RUNNER=0` cho cả hai method, kiểm tra class runner thật trong
worker, kiểm tra đủ ba vị trí hook và dừng nếu rollout Reflex nhiều token mà
không có proposal/verification hook. Không có round hoặc proposal thì tỷ số
được ghi `null` và hiển thị `N/A`. Log JSON có thêm `rollout_hook_health`.

Sau khi cập nhật source, dùng môi trường đã cài để đảm bảo đủ Python hook:

```bash
source offline_packages/activate.sh
"$DFLASH_VLLM_PYTHON" scripts/patch_upstream.py --vllm-installed
```

Lệnh patch chỉ sửa Python đã cài, không tải/build lại thư viện. Dừng riêng job
cũ và chạy job mới từ cùng pretrain checkpoint; không resume run đã thiếu
Reflex. Để so AAL, mở hai terminal và dùng instrumented cho cả hai:

```bash
CUDA_VISIBLE_DEVICES=0 METRICS_MODE=instrumented bash train_qwen25_3b.sh
CUDA_VISIBLE_DEVICES=1 METRICS_MODE=instrumented bash train_qwen25_3b_dflash.sh
```

Chọn GPU phù hợp với máy. Không cần ghi biến runner vì launcher đã đặt mặc định.
Nếu shell có export `VLLM_USE_V2_MODEL_RUNNER=1`, unset nó trước; giá trị V2 bị
từ chối thay vì âm thầm bỏ qua Reflex. Log phải in runner thật
`vllm.v1.worker.gpu_model_runner`, và sau rollout nhiều token phải có số lần
verification/feedback dương. Khi đo tốc độ, chạy cả hai với production; baseline
sẽ tiếp tục hiển thị AAL `N/A` theo thiết kế.

## Cập nhật correctness/fairness cuối

Mặc định là `FAIRNESS_PROFILE=native_dflash`, không cần thêm biến này trước mỗi
lệnh chạy. Nếu shell còn giá trị cũ, chạy `unset FAIRNESS_PROFILE` để dùng mặc định.
Đặt rõ `FAIRNESS_PROFILE=specnaacl_matched` vẫn **dừng preflight** với backend
đã pin: LoRA QKV/gate-up gộp không tương đương bảy adapter PEFT độc lập; chưa có
cadence target bốn cohort độc lập với draft/A; và cách lọc nhóm reward/mask khác.
Không coi profile matched là cấu hình chạy thành công. Để dùng các mặc định native
và đo tốc độ trong tài liệu:

```bash
unset FAIRNESS_PROFILE
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

Bộ experiment gồm **6 model**, bỏ Llama: Qwen2.5-1.5B/3B/7B/14B-Instruct
và Qwen3-1.7B/4B. Qwen2.5 dùng target nguyên bản có QKV bias; drafter DFlash
được khởi tạo độc lập theo kiến trúc Qwen3 không bias, giữ đúng kích thước
hidden/head/GQA/vocab và dùng embedding/LM head của target. Không đổi weights
hoặc tokenizer target để giả làm Qwen3. Kiểm tra MASK 151669 trên tokenizer
local chạy trước khi cấp phát weights lên GPU.

| MODEL_KEY | Target dưới `/workspace/storage-shared/models/` | Batch GRPO × accumulation | Pretrain batch × accumulation |
|---|---|---|---|
| `qwen25_1p5b` | `Qwen2.5-1.5B-Instruct` | 16 × 2 | 8 × 4 |
| `qwen25_3b` | `Qwen2.5-3B-Instruct` | 8 × 4 | 8 × 4 |
| `qwen25_7b` | `Qwen2.5-7B-Instruct` | 8 × 4 | 8 × 4 |
| `qwen25_14b` | `Qwen2.5-14B-Instruct` | 4 × 8 | 8 × 4 |
| `qwen3_1p7b` | `Qwen3-1.7B` | 16 × 2 | 8 × 4 |
| `qwen3_4b` | `Qwen3-4B` | 8 × 4 | 8 × 4 |

Batch GRPO và accumulation lấy đúng cấu hình từng target của SpecNaacl.
Theo yêu cầu mới, pretrain cả sáu model dùng batch thực 8/GPU và accumulation
4: batch hiệu dụng 32 mẫu cho một GPU. Cấu hình này khác batch hiệu dụng cũ
16/8/4/2 của Qwen2.5 và 8 của Qwen3-1.7B/4B, đồng thời khác mặc định
pretrain SpecNaacl. So sánh công bằng cần đặt cùng batch hiệu dụng ở cả hai
dự án; mặc định mới ưu tiên cấu hình 8 × 4 do người dùng chọn.
Không lấy các biến `DRAFT_PRETRAIN_*` mà launcher SpecNaacl không truyền vào CLI.
Đây vẫn là loss
DFlash, khác Medusa, và native target GRPO vẫn full fine-tuning. Các mặc định
GRPO không thay đổi trong lần chỉnh batch pretrain này. Batch mới yêu cầu run
mới; không resume trực tiếp checkpoint có batch/accumulation cũ.

### Chạy đồng thời 6 model

Cấu hình GRPO hiện tạo cụm Ray riêng cho từng run: mặc định
`DFLASH_RAY_NUM_CPUS=8`, `DFLASH_RAY_OBJECT_STORE_GB=1` (GiB), không bật dashboard
UI. `num_cpus` là ngân sách scheduler, không phải giới hạn CPU của hệ điều hành.
Các pool RPC mặc định 2 thread, không prestart worker cho driver đầu tiên;
OMP/MKL/OpenBLAS/NumExpr mặc định 1 thread nếu chưa được người dùng đặt.
Giữ các giá trị giống nhau cho DFlash và Reflex.

Nếu log dừng ở `Started a local Ray instance`, dòng này chưa xác nhận driver
đã kết nối xong. Bản mới in `[startup pid=... GPU=...]` cho Ray, tokenizer,
dữ liệu và khởi tạo model. Sau 120 giây trong một giai đoạn, chương trình tự
ghi stack trace vào `<run>/logs/console.log` rồi tiếp tục chờ; chữ `Timeout`
trong bản dump này là chẩn đoán, không phải ngoại lệ làm dừng train.
Địa chỉ cụm, thư mục Ray riêng trong `/tmp` và trạng thái khởi tạo nằm ở
`<run>/logs/ray_startup.json`. Cập nhật source rồi chạy lại riêng các job bị
kẹt. Không cần cài lại thư viện hoặc pretrain lại; giữ nguyên job đang train.

Cài môi trường một lần trước; cần 6 GPU có đủ VRAM và đủ RAM/CPU cho 6 job.
Các lệnh sau chạy mới, mỗi job một GPU. `MODEL` đã export từ một lần chạy
model cụ thể phải được xóa, nếu không nó ghi đè đường dẫn của mọi wrapper.

```bash
cd /đường/dẫn/SpecDecode/dflash
unset MODEL MODEL_KEY MODEL_ENV RUN_NAME RUN_DIR RESUME DRAFT_CHECKPOINT
unset PRETRAIN_WARM_START FAIRNESS_PROFILE
unset PRETRAIN_BATCH_SIZE PRETRAIN_ACCUMULATION_STEPS
export NPROC_PER_NODE=1
mkdir -p outputs/launch_logs
CUDA_VISIBLE_DEVICES=0 bash pretrain_qwen25_1p5b.sh > outputs/launch_logs/qwen25_1p5b.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 bash pretrain_qwen25_3b.sh > outputs/launch_logs/qwen25_3b.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 bash pretrain_qwen25_7b.sh > outputs/launch_logs/qwen25_7b.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 bash pretrain_qwen25_14b.sh > outputs/launch_logs/qwen25_14b.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 bash pretrain_qwen3_1p7b.sh > outputs/launch_logs/qwen3_1p7b.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 bash pretrain_qwen3_4b.sh > outputs/launch_logs/qwen3_4b.log 2>&1 &
wait
```

Mỗi model ghi vào `outputs/pretrain/<MODEL_KEY>/`; không dùng chung checkpoint.
Xem log bằng `tail -f outputs/launch_logs/qwen25_14b.log`. Nếu chạy lại các
lệnh trên, log launcher bị ghi mới; console log trong từng run vẫn riêng.
Để tiếp tục một run bị gián đoạn, chạy đúng wrapper với GPU tương ứng và
`RESUME=auto`. Cả 6 model đều có wrapper GRPO Reflex và baseline, ví dụ
`DATASET=simplelr bash train_qwen25_14b.sh` và
`DATASET=simplelr bash train_qwen25_14b_dflash.sh`.

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
PRETRAIN_BATCH_SIZE=8 PRETRAIN_ACCUMULATION_STEPS=4 \
PRETRAIN_MAX_LENGTH=2048 PRETRAIN_SAVE_INTERVAL=500 \
bash pretrain_qwen3_4b.sh
```

Thông số pretrain chính:

| Biến | Mặc định / ý nghĩa |
|---|---|
| `PRETRAIN_DATASET` | `sharegpt` |
| `PRETRAIN_DATASET_PATH` | file ShareGPT đã nêu |
| `PRETRAIN_EPOCHS`, `PRETRAIN_LR` | 5, 5e-5 |
| `PRETRAIN_BATCH_SIZE` | 8/GPU cho cả sáu model; Qwen3-8B tùy chọn giữ 1 |
| `PRETRAIN_ACCUMULATION_STEPS` | 4 cho cả sáu model; Qwen3-8B tùy chọn giữ 16 |
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
Thanh progress đếm batch DataLoader, khác optimizer step. Ví dụ 67.596 mẫu
trên một GPU: batch 1 có 67.596 batch/epoch; batch 8 có 8.450 batch/epoch.
Với accumulation 4, 8.450 batch tương ứng 2.113 optimizer step/epoch (lượt
cuối dùng hai batch còn lại). Log đầu run in rõ các con số này. Tăng
batch làm tăng VRAM và công việc mỗi lượt; không đảm bảo tốc độ tăng 8 lần.

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
| Model/device | Wrapper chọn `MODEL_KEY`; tùy chỉnh `MODEL`, `CUDA_VISIBLE_DEVICES=0`, `NPROC_PER_NODE=1` |
| Batch | `BATCH_SIZE=8`, `RESPONSES_PER_PROMPT=8`, `ACCUMULATION_STEPS=4`, `TRAIN_MICRO_BATCH_SIZE=1` |
| LR | `TARGET_LR=1e-6`, `DRAFT_LR=1e-4`, `DFLASH_DRAFT_LR=$DRAFT_LR` |
| Generation | `GEN_MAX_LENGTH=2048`, `MAX_PROMPT_LENGTH=2048`, `TEMPERATURE=1`, `TOP_P=0.95` |
| DFlash | `DFLASH_BLOCK_SIZE=16`, `DFLASH_LOSS_WEIGHT=0.1`, `DFLASH_MASK_TOKEN_ID=151669` |
| Reflex | `OPD_RANK=8`, `OPD_TOPK=16`, `OPD_FAST_LR=0.01`, `OPD_UPDATE_INTERVAL=1` |
| A | `OPD_TRAIN_PROJECTOR=1`, `OPD_PROJECTOR_LR=1e-4`, `OPD_PROJECTOR_WEIGHT_DECAY=0` |
| Feedback | `OPD_ACCEPTED_WEIGHT=1`, `OPD_REJECTED_WEIGHT=1` |
| Kernels | `OPD_UPDATE_STREAM=0`, `OPD_PROPOSAL_MODE=auto`, `OPD_DENSE_IMPLEMENTATION=auto` |
| Profiling | `OPD_PROFILE=0`, `OPD_DIAGNOSTICS=0`, `OPD_DISPATCH_CACHE=` |
| Data/seed | `DATASET=simplelr`, `TRAIN_SUBSET_SEED=42`, `MAX_TRAIN_SAMPLES=0` |
| Duration | `TRAIN_EPOCHS=1`, `MAX_GRPO_STEPS=1000000`, `SAVE_INTERVAL=50` |
| Logging | `LOG_INTERVAL=1`, `ROLLOUT_LOG_FLUSH_INTERVAL=1` |
| Memory/reward | `VLLM_MEMORY_UTILIZATION=0.65`, `REWARD_WORKERS=8` |
| vLLM execution | `VLLM_ENFORCE_EAGER=1` cho cả target/draft và baseline/Reflex |
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
  --cache outputs/benchmarks/b200_thresholds.json
OPD_DISPATCH_CACHE="$PWD/outputs/benchmarks/b200_thresholds.json" bash train_qwen3_4b.sh
```

Microbenchmark có ba mức active vocabulary, so sparse/fused dense/GEMM và
feedback; không chạy transformer hoặc GRPO. Cache chỉ dùng cho GPU và shape
trùng hoàn toàn. `--rows` là số MASK slots thực tế trong một proposal batch,
không phải số prompt; có thể thay bằng shape đang quan tâm. Shape không có
cache dùng sparse correction và proposal selective_masked; không tự sweep. Kết quả RTX 3090 nhỏ trong
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
checkpoint 8B này với Qwen2.5 hay target 4B. Qwen2.5 cần drafter pretrain
riêng bằng wrapper tương ứng ở mục 3; không dùng checkpoint z-lab của Qwen3.

## Benchmark cuối và cấu hình B200 được đề nghị

Giữ A trainable, B shared, rank 8, top-k 16, fast LR .01, feedback mỗi round.
Chạy benchmark ngắn trên B200 một lần rồi dùng cache đúng GPU:

```bash
mkdir -p outputs/benchmarks
PYTHONPATH="$PWD" .venv-vllm/bin/python scripts/benchmark_reflex_performance.py \
  --model-config /workspace/storage-shared/models/Qwen3-4B/config.json \
  --requests 1 8 32 64 --prefixes 0 1 2 4 8 15 --streams 0 1 --warmup 3 --repeats 15 \
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

`OPD_PIPELINE=auto` dùng calibration schema 3 theo GPU/shape và mật độ feedback gần đây.
Thống kê trong cohort được copy async mỗi 16 vòng và chỉ đọc khi event đã xong.
Cuối cohort cập nhật tỷ lệ valid rows / tổng teacher rows cho cohort sau. Có thể
cố định fused/selective/selective_masked để tái lập kernel. `OPD_UPDATE_STREAM=0`
là mặc định mới dựa trên số đo local; so lại 0/1 trên B200 bằng rollout thật.
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
