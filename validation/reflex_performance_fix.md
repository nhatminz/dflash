# DFlash + ReflexOPD: performance fix

Đã sửa trực tiếp source `dflash/`. Không đóng ZIP, không cài dependency và không
sửa source `SpecNaacl/`. Máy đo là RTX 3090, Python 3.12, Torch 2.4.1+cu121,
Triton 3.0. **Chưa đo B200 hoặc rollout GRPO thật.** Các số dưới đây không phải
speedup sinh token trên B200. Số AAL tăng 4,7%, số vòng giảm 4,2%, throughput giảm
7,4% và accepted prefix khoảng 0,683 là thông tin trong prompt người dùng.

Dữ liệu từng phép đo, phase timing, memory, cấu hình, kết quả test và hash source
nằm trong [reflex_performance_benchmark.json](reflex_performance_benchmark.json).

## Thay đổi cụ thể

| File / hàm | Thay đổi |
| --- | --- |
| `dflash_reflex/kernels.py::feedback`, `_scan`, `_merge`, `_compact_teacher_rows` | Bỏ teacher `p.topk` trên toàn bộ hàng. `selective_masked` kiểm tra weight trước khi đọc vocabulary; selective/fused compact valid rows trên GPU, không lấy count về CPU. Teacher scan tile 2048 và merge nhỏ hơn; bỏ normalization teacher vì p đã chuẩn hóa. |
| `kernels.py::_argmax_scan`, `_argmax_merge`, `_target_q` | Greedy teacher cũng bỏ qua hàng weight 0; chặn đọc teacher IDs chưa khởi tạo của hàng không hợp lệ. Tie chọn token ID nhỏ nhất. |
| `kernels.py::project_hidden`, `ProposalWorkspace`, `greedy_propose` | Tái sử dụng buffer FP32 cho hidden thay vì tạo tensor mỗi proposal. Giữ tile/reduction tree của q như cũ; argmax selective dùng tile 1024 để giảm CUDA block, token IDs vẫn khớp. Tile 2048 đã thử nhưng chậm hơn 1024 ở prefix ngắn nên không giữ. |
| `dflash_reflex/runtime_metadata.py::MetadataWorkspace`, `_limits_and_counts`, `_extract` | Một upload metadata từ pinned host buffer, hai kernel lấy EOS/limit/counters và h/u/weights/raw row map. Bỏ tensor/cat/repeat_interleave/gather tạm ở đường Reflex. Không lặp Python theo draft slot. Buffer được tái sử dụng qua cohort; event query bảo vệ pinned storage đang được GPU đọc. Hỗ trợ stride output/accepted/draft/discard. |
| `dflash_reflex/vllm_runtime.py::propose`, `verified` | Cache liên tục theo proposal order, reuse request map khi thứ tự không đổi; vẫn kiểm tra lifecycle/version khi reorder hoặc preemption. Chỉ copy h/u cho selective, tránh copy q/top-k chưa dùng. |
| `vllm_runtime.py::sample_density`, `finish` | Telemetry copy async mỗi 16 batch verification khi có calibration phù hợp; chỉ đọc pinned scalar sau event.query thành công. Đây chỉ là telemetry: B vẫn cập nhật mỗi vòng. Tỷ lệ cohort dùng valid rows / tổng teacher rows, kể cả EOS/discard. |
| `dflash_reflex/dispatch.py::PipelinePolicy`, `load_cache`; `adapter.py::__init__` | Cache schema 3 khớp implementation revision, GPU name/capability, hidden/logits dtype, vocab, hidden, rank, top-k, block và stream. Chọn batch gần nhất theo khoảng cách log, nội suy latency theo feedback density. Tính trước curves/buckets để giảm Python overhead. Thiếu cache phù hợp dùng selective_masked; measured selective/fused vẫn có thể thắng. Cache schema 2 vẫn dùng được cho sparse/dense thresholds; pipeline winners cũ không gồm runtime overhead nên không dùng. |
| `dflash_reflex/config.py`; `configs/_shared/b200_common.env` | `OPD_UPDATE_STREAM=0` là mặc định mới theo số đo local: ít overhead hơn ở batch nhỏ, gần tương đương ở batch lớn. `1` vẫn được hỗ trợ và đã test. Chưa xác nhận lựa chọn tối ưu trên B200. |
| `dflash_reflex/metrics.py::record`; `vllm_runtime.py::finish` | Ghi `opd_pipeline_rounds`, pipeline cuối, ba cột `iter_pipeline_*_rounds` và cumulative counts vào summary, giữ AAL và các metric cũ. **Pipeline counts đếm batch gọi verified có draft rows; iter_verification_rounds đếm từng request. Hai đơn vị khác nhau.** |
| `scripts/runtime_checks.py` | Auto/full/skip, fingerprint stdlib, file lock chống 8 job deep-check trùng nhau, chỉ cache sau khi validation thành công và fingerprint không đổi. |
| `scripts/launch/common.sh::startup_checks`; shared env; `scripts/launch/train_model.sh` | Mặc định RUNTIME_CHECK=auto; bỏ các import runtime/helper preflight lặp mỗi launch. Model/data/checkpoint/config/resume vẫn luôn được kiểm tra. |
| `scripts/validate_environment.py::main` | Setup/full qualification ghi cache thành công. Auto bỏ source audit SpecNaacl; explicit full và setup vẫn giữ audit. Không fetch/install/apply patch ở train launch. |
| `scripts/prepare_run.py` | Chế độ kiểm tra startup không thuộc pretrain resume contract; vẫn kiểm tra toàn bộ thiết lập học/model/data/draft. |
| `scripts/benchmark_reflex_performance.py` | Benchmark Runtime thật với input synthetic: metadata, correction, extraction, teacher top-k, KL/gradient, B update, dependency stream. Xuất calibration schema 3. Profile riêng để event profiling không làm lệch phép đo chính. |
| `scripts/benchmark_raw_cache.py` | So retained row indexing với copy valid raw rows có buffer tái sử dụng; kiểm tra IDs/q giống hệt. |
| `tests/test_performance_parity.py` | Reference numerical tests đã được tạo và chạy trước sửa kernel, rồi chạy lại sau từng thay đổi. |
| `tests/test_runtime_metadata.py`, `test_pipeline_policy.py`, `test_runtime_check_cache.py`, `test_startup_checks.py`, `test_logging.py` | EOS, discard, draft rỗng/invalid, reorder/stride, dynamic batch, stream, async density, A/B không zero sau nhiều vòng, cache invalidate/concurrency/failure, log và resume. |
| `tests/test_qwen25_support.py` | Sửa assertion cũ kỳ vọng prompt batch theo SpecNaacl: DFlash theo yêu cầu người dùng là batch 8. Vẫn giữ số SpecNaacl và nhãn DIFFERENT_BY_DESIGN khi khác. Không đổi cấu hình học để làm test pass. |
| `README.md`, `ENVIRONMENT.md`, `huongdanchay.md`, báo cáo này | Cập nhật chính sách startup, calibration, fallback và lệnh chạy. |

## Bảo toàn thuật toán và fairness

A vẫn trainable, gradient rollout vẫn được tích lũy, optimizer/sync A tới inference
workers giữ nguyên. B dùng chung toàn cohort, reset ở ranh giới cohort, update sau
mỗi verification. Gradient A đọc B_t trước update B. Không đổi top-16 union + tail,
accepted-prefix/first-rejection weights, greedy proposal hoặc rejection sampler.
Không thêm target/draft forward, backward rollout hoặc bản sao B đầy đủ.

Giữ raw `compute_logits` gốc tới verification để giữ head bias, logit processing
và rounding; không tái tính head rồi giả định đó là cùng distribution. FP32 atomic
accumulation có sai số làm tròn như trước, không cam kết bitwise gradient giữa các
lần chạy. Token IDs phải bằng nhau trong tests; tolerance được ghi rõ bên dưới.

Production baseline tiếp tục bypass OPD. Instrumented baseline giữ counters cũ,
không bị ép chạy correction/teacher top-k. Hai phương pháp dùng cùng draft/target
checkpoint, GRPO settings, eager/V1. Batch 8, accum 4, SimpleLR và checkpoint 50
step giữ nguyên. GSM8K, DAPO, ShareGPT pretrain và resume được regression-test.

## Kiểm chứng

- Trước sửa kernel: **18/18** reference tests qua với prefix 0,1,2,4,8,15 và ba pipeline.
- Sau teacher tile 2048: **31/31** proposal/feedback/reference tests qua.
- Sau tối ưu argmax: **38/38** numerical/metadata tests qua.
- Runtime + actual extracted sampling kernels + numerical tests: **36/36** qua.
- Full project regression cuối: **333/333** qua. Có warning dependency/deprecation local;
  không có test fail. Hai assertion batch cũ được sửa theo yêu cầu batch 8 đã có.
- Sau đổi default stream: **83/83** runtime/launch/fairness tests qua.
- Sau bổ sung density denominator/EOS: **39/39** metadata/log/resume/policy tests qua.
- Startup cache cuối: **15/15** qua; tám process cùng cache chỉ deep-validate một lần.
- Ruff các Python file đã sửa/thêm và `bash -n` các shell file qua.
- 144 cell synthetic Runtime benchmark đều có A gradient và B norm không zero,
  16 update/cell (3 warmup + 10 timed + 3 profile). A gradient vòng đầu bằng 0 vì
  B_t=0; từ các vòng sau có gradient. Đây **không phải** xác nhận rollout B200 thật.

Tolerance reference FP32: q rtol=5e-5/atol=1e-7; B rtol=3e-4/atol=3e-7;
A gradient rtol=6e-4/atol=5e-7; weighted KL rtol=2e-4/atol=5e-6. Tests khác còn có
oracle FP64 cho union/tail và B_t. Benchmark before/after có B norm bằng nhau;
sai khác tương đối A-gradient norm lớn nhất khoảng 7.69e-08, không thay thế test
so từng phần tử. Test sampler giữ target sampling và accepted-prefix behavior.

## Đo overhead trên RTX 3090

Qwen3-4B geometry thật: hidden 2560, vocab 151936, BF16 hidden, FP32 logits,
block 16, rank 8, top-k 16. Requests 1/8/32/64, prefix 0/1/2/4/8/15, stream 0/1,
ba pipeline = 144 cell mỗi phiên before/after; 3 warmup, 10 timed rounds/cell.
Baseline snapshot lấy trước mọi sửa kernel/runtime trong task này.

Đơn vị ms/verification batch, wall time gồm CPU submission và GPU dependency;
không gồm model forward, scheduler/rejection sampling, reward hoặc GRPO. Profile
phases đo riêng ba vòng; phase lồng nhau không cộng hai lần vào total.

| Requests=64, stream=0, prefix | selective_masked trước | sau | Giảm overhead |
| --- | ---: | ---: | ---: |
| 0 | 4.337 | 2.191 | 49.5% |
| 1 | 5.863 | 3.210 | 45.2% |
| 2 | 7.505 | 4.291 | 42.8% |
| 4 | 10.940 | 6.383 | 41.7% |
| 8 | 17.245 | 10.822 | 37.2% |
| 15 | 27.165 | 17.556 | 35.4% |

Batch 64 prefix 0: extraction khoảng 0.801 → 0.027 ms; masked teacher top-k
0.858 → 0.248 ms; proposal argmax khoảng 1.529 → 0.945 ms.
Prefix dài fused vẫn thắng selective: prefix 15 fused khoảng 15,9 ms,
selective_masked khoảng 17,6 ms. Vì vậy auto không cố định selective.
Batch 1/8 cũng giảm overhead; đầy đủ số đo GPU median/p95, wall, phase và memory
nằm trong JSON. Không cell nào có wall overhead xấu hơn before trong phiên này;
không suy diễn kết quả đó thành cải thiện throughput B200.

Stream 1 tạo khoảng 0,06–0,10 ms overhead thêm ở các batch nhỏ; batch 32/64 gần
như ngang stream 0 trong số đo local. Default chọn 0 để giảm CPU/event overhead,
giữ 1 để so trên B200. Cả hai đều có update mỗi vòng và next proposal dùng B mới.

Raw FP32 cache cho 64×15×151936 là 583.434.240 bytes (~556 MiB), BF16 ~278 MiB.
So retained indexing với valid-row copy + statistics (ms, RTX 3090):

| Prefix | Retain | Copy valid rows trước statistics | Bộ nhớ copy thêm |
| --- | ---: | ---: | ---: |
| 0 | 0.873 | 0.980 | 38.895.616 bytes |
| 2 | 2.555 | 2.672 | 116.686.848 bytes |
| 15 | 11.507 | 13.216 | 583.434.240 bytes |

Đã cấp trước copy buffer và cung cấp row IDs oracle cho phương án copy; chưa tính
chi phí tìm valid rows, nên đây là so sánh ưu ái copy. Copy không giúp giải phóng
raw trước khi biết kết quả verification, lại thêm đọc/ghi. Giữ raw cache. Không
benchmark head recomputation vì chưa có cách chứng minh giữ nguyên bias/processing/
dtype của compute_logits mà không chạy thêm head; chưa đủ cơ sở triển khai.

## Runtime checks

`RUNTIME_CHECK=auto` mặc định. Cache `.cache/runtime_checks/validation.json` gồm
fingerprint của Python executables/pyvenv, metadata+RECORD/native file stat,
versions.lock/uv.lock, upstream HEAD+recursive submodule commits+modified files,
installed vLLM hook contents,
patch/validator sources và host/driver. Cache không phải audit byte-by-byte mọi
binary. Auto hit chỉ đọc/stat nhẹ và chạy stdlib Python probes, không import GPU
stacks. Đo local khoảng **0,44 giây**; chưa đo filesystem B200. Setup qualification
thành công ghi cache; nếu chưa có, first auto launch chạy deep validation một lần.

`full` ép kiểm tra sâu và reference audit; `skip` bỏ runtime/import qualification.
Cả ba vẫn giữ kiểm tra model/data/draft/checkpoint/config/resume của prepare_run.
Không network/download/install/patch lúc launch. DRY_RUN vẫn không cần GPU stack.
MCore helper được worker build/reuse như trước; bỏ prebuild lặp ở launcher.

## Lệnh chạy trên B200

Cập nhật source project rồi chạy job mới; không cần cài lại thư viện hoặc pretrain.
Nếu offline installer đã tạo các môi trường .venv-local thì dùng:

```bash
cd /workspace/storage-shared/nlp/minhpn19/dflash
source offline_packages/activate.sh
unset METHOD
export RUNTIME_CHECK=auto
export METRICS_MODE=instrumented
export OPD_PIPELINE=auto
export OPD_UPDATE_STREAM=0
```

`instrumented` áp dụng cho cả hai để baseline có AAL. `production` baseline bypass
hooks và AAL sẽ N/A như thiết kế. Không export METHOD chung khi dùng wrapper pair.
Nếu môi trường B200 đã được xác nhận và muốn vào ngay: `export RUNTIME_CHECK=skip`.
Vẫn phải tải model và khởi tạo Ray/vLLM/Megatron; đây không phải phần được cache.

Mỗi dòng dưới chạy ở một terminal riêng, cùng thư mục project, sau đoạn env trên:

```bash
CUDA_VISIBLE_DEVICES=0 bash train_qwen25_1p5b_dflash.sh
CUDA_VISIBLE_DEVICES=1 bash train_qwen25_1p5b.sh
CUDA_VISIBLE_DEVICES=2 bash train_qwen25_3b_dflash.sh
CUDA_VISIBLE_DEVICES=3 bash train_qwen25_3b.sh
CUDA_VISIBLE_DEVICES=4 bash train_qwen3_1p7b_dflash.sh
CUDA_VISIBLE_DEVICES=5 bash train_qwen3_1p7b.sh
CUDA_VISIBLE_DEVICES=6 bash train_qwen3_4b_dflash.sh
CUDA_VISIBLE_DEVICES=7 bash train_qwen3_4b.sh
```

Cặp 7B/14B nếu cần thay vào GPU trống:

```bash
CUDA_VISIBLE_DEVICES=0 bash train_qwen25_7b_dflash.sh
CUDA_VISIBLE_DEVICES=1 bash train_qwen25_7b.sh
CUDA_VISIBLE_DEVICES=2 bash train_qwen25_14b_dflash.sh
CUDA_VISIBLE_DEVICES=3 bash train_qwen25_14b.sh
```

Batch size=8, accumulation=4 đã mặc định. Wrapper tự chọn latest pretrain checkpoint
đúng model. Có thể chỉ định cùng `DRAFT_CHECKPOINT=/absolute/path` cho cả hai.
Không ghép target khác model với draft đó. Để resume job cũ dùng
`RESUME=/absolute/path/to/run`; giữ config gốc. Nếu run cũ stream=1 phải thêm
`OPD_UPDATE_STREAM=1` khi resume; đổi default stream không cho phép đổi learning
configuration ngầm trong checkpoint. Các lệnh trên tạo run mới nếu RESUME không set.

Calibration B200 (chỉ benchmark, không GRPO), chạy trên GPU đang trống:

```bash
mkdir -p outputs/benchmarks
CUDA_VISIBLE_DEVICES=0 "$DFLASH_VLLM_PYTHON" scripts/benchmark_reflex_performance.py \
  --model-config /workspace/storage-shared/models/Qwen3-4B/config.json \
  --requests 1 8 32 64 --prefixes 0 1 2 4 8 15 --streams 0 1 \
  --logits-dtype float32 --warmup 3 --repeats 15 \
  --output outputs/benchmarks/b200_qwen3_4b_runtime.json \
  --cache outputs/benchmarks/b200_qwen3_4b_dispatch.json
OPD_DISPATCH_CACHE="$PWD/outputs/benchmarks/b200_qwen3_4b_dispatch.json" \
  CUDA_VISIBLE_DEVICES=0 METRICS_MODE=instrumented bash train_qwen3_4b.sh
```

Chọn đúng dtype logits runtime; cache BF16 không áp dụng FP32 và ngược lại.
Calibration Qwen3-4B không áp dụng target có hidden/vocab khác. Đổi config model
và tên cache để calibrate các model khác. RTX cache không bao giờ được dùng trên
B200. Cache schema 2 cũ cần regenerate bằng utility mới cho pipeline selection.
Không có calibration thì auto vẫn dùng selective_masked.

Có thể ép đối chiếu một run ngắn (người dùng tự chạy, agent chưa chạy B200):

```bash
OPD_PIPELINE=selective_masked OPD_DIAGNOSTICS=1 MAX_GRPO_STEPS=2 \
  CUDA_VISIBLE_DEVICES=0 METRICS_MODE=instrumented bash train_qwen25_3b.sh
```

Kiểm tra logs: `opd_pipeline_rounds.selective_masked > 0`, `B_update_rounds > 0`,
`B_valid_feedback_rows > 0`; A gradient vòng đầu có thể bằng 0, phải kiểm tra vòng
sau và `A_optimizer_steps/A_version` sau optimizer. So AAL, generation tokens/s,
wall time cả rollout/training với baseline cùng seed/checkpoint/eager/config.
Không kết luận throughput nhanh hơn chỉ từ bảng OPD synthetic ở trên.
