# Báo cáo triển khai DFlash + ReflexOPD

Code nằm trong `dflash/`, ngang hàng với `SpecNaacl/`. Các đường chạy pretrain
và GRPO đã được triển khai bằng model/loss NVIDIA và proposer/verifier vLLM
được pin. Kiểm thử CPU và CUDA nhỏ đã chạy trên RTX 3090. **Chưa xác nhận
chạy end-to-end NeMo/Megatron/vLLM trên B200**: máy hiện tại thiếu runtime
CUDA 13 và các model/dataset ở shared storage. Không có kết luận speedup GRPO.


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

## 1. Cây thư mục mới

```text
dflash/
├── README.md ENVIRONMENT.md IMPLEMENTATION_REPORT.md huongdanchay.md
├── versions.lock SOURCE_MANIFEST.json requirements.txt pyproject.toml
├── pretrain_qwen3_{1p7b,4b,8b}.sh
├── pretrain_qwen25_{1p5b,3b,7b,14b}.sh
├── train_qwen3_{1p7b,4b,8b}.sh
├── train_qwen3_{1p7b,4b,8b}_dflash.sh
├── train_qwen25_{1p5b,3b,7b,14b}{,_dflash}.sh
├── configs/
│   ├── _shared/b200_common.env
│   └── qwen3_{1p7b,4b,8b}/b200.env
├── dflash_reflex/
│   ├── __init__.py config.py adapter.py kernels.py feedback.py
│   ├── rollout_state.py vllm_runtime.py projector_training.py
│   ├── nemo_bridge.py checkpointing.py metrics.py
│   ├── model_contract.py block_training.py
│   └── data.py rewards.py nemo_data.py pretrain.py train.py
├── scripts/
│   ├── launch/{common,pretrain_model,train_model}.sh
│   ├── fetch_sources.py patch_upstream.py setup_environment.sh
│   ├── prepare_run.py validate_environment.py prepare_data.py
│   └── benchmark_rollout.py summarize_run.py plot_aal.py plot_training_time.py
├── patches/{nemo-rl,vllm}.patch
├── third_party/
│   ├── nemo-rl/ vllm/ original-dflash/
│   └── reference_qwen3_8b_b16_config.json reference_hf_revisions.json
├── tests/test_{dflash,reflex,cuda,runtime,sampling,training,
│              distributed,resume,logging,launchers,tools,patches}.py
├── validation/
│   ├── report.json pytest.xml source_audit.json file_manifest.json
│   └── cuda_microbenchmark.json qwen3_4b_geometry_rtx3090.json
│       dispatch_cache_rtx3090.json
└── outputs/{pretrain,train,benchmarks}/
```

Checkouts thực có sẵn trong workspace. Chúng được ignore trong Git ngoài để
tránh gitlink không hợp lệ; `scripts/fetch_sources.py` khôi phục chúng từ SHA
khi chuyển project sang máy khác. Không import module từ `../SpecNaacl/`.

## 2. Nguồn và SHA thực tế

| Nguồn | Commit đã đọc và checkout |
|---|---|
| MarkXCloud/RL, nhánh DFlash của NVIDIA PR #3712 | `c18e462adf34739951b3d21773f436eaee324b31` |
| vLLM v0.25.1, khớp dependency NeMo | `752a3a504485790a2e8491cacbb35c137339ad34` |
| z-lab/dflash | `07ebd93db9f472af339b644bb70221ad8428328a` |

`versions.lock` và
`validation/source_audit.json` là dữ liệu kiểm chứng bằng `git rev-parse HEAD`.
Megatron Bridge gitlink: `b11414c71b15e54d333eb49346ed199f20fa9021`.
Setup cập nhật submodule đệ quy theo gitlink, không theo branch mới nhất.

Đã đối chiếu issue #3698, PR #3712, hạ tầng EAGLE của PR #3699 và các PR
#4059/#4060; không thay implementation DFlash bằng main hoặc DSpark.
Các link nguồn chính nằm trong [README.md](README.md).

Runtime lấy từ `uv.lock` thực của NeMo: Python >=3.13.14,<3.14,
Torch 2.11.0+cu130, Triton 3.6.0, Transformers 5.12.1, Ray 2.56.1,
vLLM 0.25.1, FlashAttention 2.8.1+cu13torch2.10cxx11abitrue và
Transformer Engine 2.15.0+42b8400. Hai extra `mcore`/`vllm` xung đột nên
được cài ở hai environment riêng, cộng environment driver riêng.

## 3. File thêm và thay đổi

Mọi file project mới nằm trong `dflash/`; danh sách đầy đủ và SHA256 nằm trong
`validation/file_manifest.json`. `SOURCE_MANIFEST.json` chụp checksum code
SpecNaacl trước khi triển khai; audit cuối xác nhận các file đó không đổi.

| File / nhóm file | Nội dung |
|---|---|
| `adapter.py`, `config.py`, `feedback.py` | A/B, QR init, public knobs, CPU oracle của objective SpecNaacl |
| `kernels.py` | Correction, full-vocab normalization/top-k, union/tail, A gradient và sparse B update bằng Triton |
| `rollout_state.py`, `vllm_runtime.py` | Cohort/request cache, proposal/feedback matching, CUDA stream, diagnostics, reusable buffers |
| `projector_training.py`, `nemo_bridge.py` | Optimizer A, tổng gradient/weight, NeMo boundaries, refit A, checkpoint sidecars |
| `checkpointing.py`, `metrics.py` | Atomic state, RNG, pending gradients, per-iteration logs, cumulative counters/resume |
| `pretrain.py`, `block_training.py` | Pretrain NVIDIA thật; helper nhỏ chỉ là numerical oracle để kiểm thử |
| `train.py`, `nemo_data.py`, `data.py`, `rewards.py` | GRPO NVIDIA, ShareGPT/math parsing, tokenizer, reward tương đương SpecNaacl |
| `model_contract.py` | Kiểm tra architecture/tokenizer/MASK/block và safetensors headers trước nạp GPU |
| `scripts/`, `configs/`, chín wrappers | Setup riêng, launch, overrides, dry run, validation, reporting, benchmark |
| `tests/`, `validation/`, bốn tài liệu Markdown | Kiểm thử thực đã chạy, artifact kết quả và hướng dẫn vận hành |

Patch NeMo chạm đúng bốn file:

* `nemo_rl/models/generation/vllm/vllm_worker.py`: bọc một `LLM.generate`
  bằng begin/drain cohort.
* `nemo_rl/models/policy/workers/megatron_policy_worker.py`: tạo trainer A,
  aggregate trước optimizer, cập nhật sau optimizer thành công, save/restore
  sidecar và RNG sau constructor.
* `nemo_rl/algorithms/grpo.py`: theo dõi từng DataLoader iteration và chỉ
  finalize checkpoint sau khi ghi counters/timing cuối iteration.
* `nemo_rl/environments/math_environment.py`: chọn reward math-verify và
  format coefficient 0.2 tương đương SpecNaacl.

Patch vLLM chạm đúng ba file:

* `vllm/v1/spec_decode/llm_base_proposer.py`: lấy final MASK hidden states
  trong `_greedy_sample`, sửa logits và chọn proposal; profile forward sẵn có.
* `vllm/v1/sample/rejection_sampler.py`: lấy accepted prefix trực tiếp từ
  decision kernels; chuyển teacher đã xử lý sampling sang Reflex.
* `vllm/v1/worker/gpu_model_runner.py`: cung cấp context request/length/EOS
  hiện tại và profile target forward sẵn có.

`patches/*.patch` lưu diff để review. `patch_upstream.py` là cách áp dụng
chính: kiểm tra exact source anchors, fail khi API drift, chạy lại an toàn.
Test dựng lại các file từ Git object nguyên bản, patch hai lần, so byte với
checkout hiện tại và parse AST.

## 4. Proposal và verification

DFlash vẫn forward block song song một lần: anchor + MASK, noncausal attention
trong block, context target qua K/V. Không thêm forward cho Reflex.
`H` là hidden state sau final norm, đúng đầu vào shared target LM head.
Correction là `U=H@A`, `z'=z+U@B.T`; proposal là argmax của corrected logits.

Verifier giữ `draft_probs=None`: phân phối proposal thực là one-hot. Softmax
corrected logits chỉ phục vụ OPD, không được giả làm proposal probability.
Teacher stochastic là tensor target probabilities đã được verifier áp dụng
temperature/top-p. Teacher greedy là delta tại argmax đã bị constraints.

Decision kernels ghi accepted-prefix count thực, không suy từ equality của
output. Feedback chỉ gồm accepted prefix + first rejection; bonus và phần
sau rejection bị loại. EOS, max remaining tokens, padding và discarded
requests đều mask weight. Cache theo stable request ID chịu được reorder;
proposal ordinal và pending state ngăn verify hai lần. Async scheduler bị tắt.

Objective giữ union top16 draft + top16 target khác draft, với full-vocabulary
draft normalization và một tail bucket. Điểm quan trọng từ kernel SpecNaacl:
KL/TV báo cáo có tail, nhưng gradient vận hành là union-only `w*(q-p)`,
không tự thêm đạo hàm tail. Denominator là tổng weight thật, kể cả khi <1.
A gradient đọc `B_t` trước atomic update B. Round đầu B=0 cho gradient A=0
nhưng weight vẫn thuộc denominator; các round sau có gradient A khác 0.

## 5. Huấn luyện và đồng bộ A

A FP32, rank mặc định 8, khởi tạo QR từ các hàng LM head cách đều giống
SpecNaacl. A là `torch.nn.Parameter` trainable. `ProjectorTrainer` có group
AdamW tên `draft_opd_projector`, LR 1e-4, weight decay 0, scheduler hằng số.

Gradient A đến từ `H.T @ (g @ B_t)`, cộng trên GPU trong cohort. Boundary
rollout gửi một packet CPU nhỏ chứa tổng gradient và weight; không gửi logits
hay B. Boundary optimizer cộng global numerator và denominator rồi chia
đúng một lần. Packet trùng hoặc sai A version gây lỗi. Gradient pending được
giữ nếu optimizer target thất bại; không bị reset ở microbatch.

Đây là **optimizer riêng được phối hợp tại boundary optimizer Megatron**,
không chèn tensor ngoài model vào distributed optimizer nội bộ của MCore.
Ordinary draft/target vẫn dùng joint optimizer và distillation của NVIDIA.
Sau successful joint update, A được cập nhật/publish; trước cohort kế tiếp,
vLLM nạp đúng A/version cùng nhịp refit draft/target. Test RPC boundary kiểm
tra A mới thực sự được copy vào reusable adapter.

Checkpoint lưu A, Adam moments, scheduler, version, consumed IDs, gradient
pending/weight; trainer RNG gồm Python/NumPy/Torch/CUDA và MCore CUDA tracker.
Driver và inference RNG cũng lưu riêng. NeMo mặc định không nạp RNG trong
đường setup này nên hook restore tường minh là cần thiết. Full GPU resume
vẫn cần qualification trên runtime production.

## 6. Shared B và topology

Một complete synchronous `LLM.generate` call có một UUID cohort, một inference
owner và **một B FP32 cho mọi prompt/response trong call**. B sống qua nhiều
round, dùng chung khi các requests được scheduler batch/reorder. Kết thúc
một response chỉ giải phóng request cache. Cohort sau reset B, active rows,
grad buffers và cache; tái sử dụng vùng GPU đã cấp phát. B không nằm trong
model/checkpoint.

GRPO hiện cho phép **NPROC_PER_NODE=1, một colocated engine, TP=PP=CP=1,
không SP/packing, không async scheduling**. Không chia một cohort ra nhiều
engine độc lập vì sẽ đổi shared-B thành per-shard-B. Topology không hỗ trợ
bị chặn trong `scripts/prepare_run.py::build_grpo_config`, trước GPU init.
Phép aggregate A đã có test hai process Gloo; điều này không đồng nghĩa đã
hỗ trợ hoặc kiểm chứng multi-GPU GRPO. Pretrain có DP nhiều rank, chưa chạy
nhiều GPU thực tại đây.

## 7. Giảm và đo overhead GPU

* Tính U một lần; correction, normalization và top-k selection fuse theo
  vocabulary tiles. Không materialize corrected full logits trong fused mode.
* Sparse correction bỏ qua inactive vocabulary rows; dense fused/GEMM có
  dispatch. GEMM dùng buffer tái sử dụng khi được chọn.
* Reuse top-k và normalization của proposal cho feedback; chỉ dot shared head
  ở teacher IDs ngoài draft shortlist và weight hợp lệ.
* Union, metric sums, A-gradient và atomic sparse B update chạy GPU theo batch;
  không loop Python theo token/slot, không backward qua transformer trong verify.
* B/workspaces/cache tái sử dụng qua cohorts. Stream riêng dùng wait_stream,
  CUDA event và record_stream để giữ thứ tự/lifetime. Không synchronize toàn
  device hay lấy scalar CPU ở mỗi verification round.
* Baseline bypass correction/OPD allocations/kernels; chỉ giữ acceptance
  instrumentation cần cho AAL và state checkpoint tại boundary.
* `OPD_PROFILE=1` dùng CUDA events, resolve ở cohort end. Có DFlash/head,
  target forward, verification sampler, correction/selection, extraction,
  union, A gradient, B update; refit lấy từ timing upstream. Các phạm vi fuse
  có thể overlap, không cộng mù các field để suy end-to-end time.

Microbenchmark cuối trên **RTX 3090, không phải B200**, BF16,
N=30 MASK slots, H=2560, V=151936, rank=8, top-k=16, 5 repeats:

| Active vocab | Sparse fused proposal | Dense fused | Dense GEMM |
|---|---:|---:|---:|
| 2% | 0.472 ms | 0.530 ms | 0.529 ms |
| 20% | 0.488 ms | 0.530 ms | 0.530 ms |
| 100% | 0.530 ms | 0.530 ms | 0.529 ms |

Feedback: khoảng **0.322 ms**. Đây là dữ liệu tổng hợp cùng geometry, không
có transformer/GRPO; head GEMM raw nằm ngoài proposal interval. Cache winner
theo exact GPU/shape/dtype; shape khác dùng heuristic, không tự sweep.
Không dùng cache RTX 3090 làm calibration B200. Artifact số đo đầy đủ nằm
trong `validation/qwen3_4b_geometry_rtx3090.json`.

## 8. Kết quả kiểm thử và phần chưa chạy

Lượt cuối: **46 passed, 0 failed, 0 skipped**,
12,40 giây. Ruff lint/format, AST của 40 file Python, syntax của 13 file shell
và dry run của các launcher đều qua. Kết quả máy đọc được nằm trong
`validation/report.json` và `pytest.xml`.
Các nhóm kiểm thử:

| Nhóm | Phạm vi thực đã kiểm tra |
|---|---|
| DFlash | Model z-lab thật với tiny config, CPU/CUDA forward/backward/AdamW/save/load; geometry head_dim không suy từ hidden_size |
| Training contract | Anchor/MASK, trunk/block attention, teacher slot alignment, valid-prefix mask, NVIDIA exponential slot weights, loss sanity |
| Reflex | QR, B=0, union/tail/full norm, weight<1, update order, nonzero A gradient, shared/cohort isolation/reset |
| Triton CUDA | Sparse/dense/GEMM, FP32/BF16, full-vocabulary union, sparse update so với numerical oracle |
| vLLM sampling | AST-extracted decision kernels từ đúng source pin; output khớp pristine kernels; accepted counts và 40.000 sample giữ phân phối target |
| Runtime | Request reorder/EOS/discard, stream bật/tắt, reusable B reset, RPC A version/refit, baseline bypass OPD |
| Optimizer/resume | Hai-rank Gloo global sum/weight; optimizer/scheduler/pending gradient/RNG equivalence; deferred driver checkpoint; truncate future/partial logs |
| Data/reward | Ba parquet schemas, deterministic subsets, ShareGPT assistant mask, math-verify và format reward |
| Tooling | Wrappers/dry run/overrides, config inheritance thật của NeMo, incompatible resume, patches idempotent, plot và summarize từ temporary logs |

Lệnh kiểm thử local:

```bash
cd dflash
.venv-test/bin/python -m pytest tests -q --junitxml=validation/pytest.xml
.venv-test/bin/ruff check dflash_reflex scripts tests
python3 scripts/validate_environment.py --output validation/source_audit.json
```

Môi trường kiểm thử là Python 3.12.2, Torch 2.4.1/CUDA 12.1, Triton 3.0.0,
Transformers 4.57.1 trên RTX 3090. Ba warnings đến từ protobuf deprecation
và SciPy hệ thống yêu cầu NumPy <2.3 trong environment test dùng NumPy 2.5.4;
không có test failure. Không thay package trong môi trường SpecNaacl.

**Chưa chạy:** cài CUDA-13 production dependency set; import/forward NVIDIA
Megatron DFlash thực; ShareGPT pretrain end-to-end; target/draft GRPO optimizer
step end-to-end; live vLLM engine/dynamic scheduler; distributed GPU training;
full NeMo checkpoint continuation; B200 memory/throughput hoặc speedup.
Tiny z-lab test và actual sampler-kernel test không thay thế các kiểm chứng đó.

## 9. Lệnh pretrain ShareGPT

Trên máy có shared storage và runtime phù hợp:

```bash
cd dflash
bash scripts/setup_environment.sh
bash pretrain_qwen3_4b.sh
# Nếu đã có model Qwen3-8B local:
bash pretrain_qwen3_8b.sh
```

Smoke giới hạn hai optimizer steps, không chạy thí nghiệm dài tự động:

```bash
PRETRAIN_MAX_SAMPLES=8 PRETRAIN_MAX_LENGTH=512 \
PRETRAIN_ACCUMULATION_STEPS=2 PRETRAIN_MAX_STEPS=2 \
PRETRAIN_SAVE_INTERVAL=1 PRETRAIN_DATALOADER_WORKERS=0 \
RUN_NAME=pretrain_smoke bash pretrain_qwen3_4b.sh
```

Target frozen causal forward lấy taps và teacher, NVIDIA `DFlashDraftModel`
forward anchor+MASK, `BlockDraftLossFn` thật; BF16 forward, FP32 master weights,
FP32 accumulation/Adam, global count normalization, clip và cosine schedule.
Khởi tạo được từ đầu; warm-start chỉ là tùy chọn. Export HF headless
safetensors dùng exporter gốc của NVIDIA, kèm config/manifest/A.

## 10. Lệnh GRPO và resume

```bash
DATASET=simplelr bash train_qwen3_4b.sh
DATASET=gsm8k bash train_qwen3_4b.sh
DATASET=dapo bash train_qwen3_4b.sh
DATASET=simplelr bash train_qwen3_4b_dflash.sh
RESUME=auto DATASET=simplelr bash train_qwen3_4b.sh
DRY_RUN=true DATASET=simplelr bash train_qwen3_4b.sh
```

Launcher dùng latest compatible pretraining checkpoint nếu không override
`DRAFT_CHECKPOINT`; thiếu model/data/checkpoint thì fail, không tải thay thế.
Hai method dùng cùng NVIDIA co-training và sampling. Hướng dẫn smoke GRPO,
overrides, Qwen3-8B, plotting và fair comparison nằm trong
[huongdanchay.md](huongdanchay.md). Bộ 6 model gồm Qwen2.5-1.5B/3B/7B/14B
và Qwen3-1.7B/4B, bỏ Llama. Qwen2.5 giữ target có QKV bias, dùng drafter
Qwen3-style không bias được khởi tạo độc lập với đúng geometry và shared
embedding/LM head. Có kiểm thử tiny forward/backward/export nhưng chưa có
full B200 training; xem [báo cáo hỗ trợ](validation/qwen25_support_report.md). External 8B-b16
checkpoint cần đúng target/tokenizer/aux layers/MASK/weights; MASK là unused
row 151669, không phải `<|fim_pad|>`.

## 11. Logs, thời gian và checkpoint

```text
outputs/pretrain/<model_key>/<run>/logs/metrics.jsonl
outputs/pretrain/<model_key>/<run>/checkpoints/resume/latest.pt
outputs/pretrain/<model_key>/latest_checkpoint

outputs/train/<model_key>/<run>/logs/console.log
outputs/train/<model_key>/<run>/logs/rollout_timing.csv
outputs/train/<model_key>/<run>/logs/metrics.jsonl
outputs/train/<model_key>/<run>/logs/{timing,train_steps}.csv
outputs/train/<model_key>/<run>/statistics/iteration_*.json
outputs/train/<model_key>/<run>/summary.json
outputs/train/<model_key>/<run>/checkpoints/resume/step_*/policy/{weights,optimizer}/
outputs/train/<model_key>/<run>/checkpoints/resume/step_*/dflash_driver.pt
outputs/train/<model_key>/{active_run,latest_run}
```

Mỗi DataLoader iteration có đúng một row AAL/acceptance/timing, kể cả iteration
không update. AAL = accepted path tokens / actual verification rounds;
accepted path có recovery/bonus target token, accepted draft đếm riêng.
Bonus không tăng accepted draft hoặc OPD feedback. Cumulative ratios dùng
tổng numerator/tổng denominator, không lấy trung bình các ratios.

`summary.json` có total wall time tích lũy qua resume, generation/end-to-end
throughput, số optimizer steps thành công thực và Reflex totals. Tách thời
gian startup khỏi training. NeMo chạy draft/target chung forward/backward nên
thời gian riêng là null; joint time được đo thật. Diagnostics không bật cũng
là null. Tqdm GRPO có loss/reward/AAL/acc/tok/s/elapsed; pretrain có epoch,
step/loss/LR/tokens/step time/elapsed. Không đồng bộ GPU từng round để in log.

Weights target và draft được checkpoint chung bởi NeMo; `target/`, `draft/`
trong GRPO run là layout dự phòng, không chứa bản copy giả của joint weights.
A sidecar là `policy/weights/reflex_rank0.pt`, successful steps và trainer RNG
ở `dflash_stats_rank0.pt`. Driver sidecar được ghi trước atomic finalization,
giữ counters và engine RNG. Resume khôi phục pending A gradient và loại
packet/log/statistics sau checkpoint đã commit. Launch contract chặn đổi
method, OPD hyperparameters, model/tokenizer hoặc dataset khi resume.

## 12. Giới hạn cần giải quyết trên máy chạy thực

| Giới hạn cụ thể | Vị trí code / kiểm chứng cần thiết |
|---|---|
| Máy hiện tại driver 555/CUDA 12, thiếu CUDA 13/TE/FA/MCore/vLLM locked | `scripts/setup_environment.sh`, `scripts/validate_environment.py --require-runtime`: cần hoàn tất install/import trên host phù hợp |
| Không có target và bốn dataset ở shared storage đã yêu cầu | `scripts/prepare_run.py::main`, `model_contract.py`: fail trước nạp model; cần cung cấp đúng các file local |
| Chưa chạy actual Megatron pretrain | `pretrain.py::make_model/main`: cần smoke hai bước trên host production để xác nhận FA/TE ABI, memory và exporter |
| Chưa chạy actual GRPO/co-training/refit/resume | `train.py::main`, `nemo_bridge.py`, các patch Megatron/vLLM: cần smoke optimizer thật và resume với runtime đã pin |
| Live scheduler chỉ được kiểm tra bằng runtime trace nhỏ; actual rejection kernels đã chạy | `vllm_runtime.py::Runtime`: cần live engine smoke để xác nhận scheduling/KV/cache contract trên vLLM 0.25.1 |
| Multi-GPU GRPO bị chặn có chủ đích; DP pretrain chưa kiểm thử GPU phân tán | `prepare_run.py::build_grpo_config`, `pretrain.py`: không mở topology khác mà chưa giải quyết shared-B ownership |
| Chưa có B200 calibration hoặc end-to-end time comparison | `scripts/benchmark_rollout.py`, `metrics.py`: số đo RTX 3090 không chứng minh B200 speedup |

Đây là các giới hạn xác nhận được, không được thay bằng placeholder hoặc
metric giả. Các đường production đã có code thực và lệnh chạy; vẫn cần
qualification end-to-end trên máy có runtime/model/data đúng trước run dài.

## Báo cáo tối ưu cuối

Artifacts mới: `validation/final_reflex_optimization_report.md/.json`,
`fairness_config_matrix.json`, `benchmark_before_after.json`,
`dispatch_cache_rtx3090_v2.json`, `final_pytest.xml`.
Báo cáo cuối thay thế số đo/heuristic trong các mục triển khai trước đó.
Bản kernel cũ đã giữ trong `tests/reference/`; SpecNaacl được checksum trước/sau.
Không thay pins hoặc cài thay production dependencies. Các kiểm thử mới chạy
kernel thực, loss NeMo thực qua AST và sampler thực qua AST; chưa chạy engine thật.
