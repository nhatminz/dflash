# Bàn giao phân tích verification feedback DFlash

Ngày kiểm tra: 13/10/2026 (Asia/Saigon).

## Đã triển khai

Launcher `run_dflash_motivation.sh` có `--mode check|collect|analyze|all`,
`--checkpoints`, `--smoke`, `--resume`. Đây là pipeline inference/analysis riêng,
không có nhánh train, không gọi GRPO setup/train, không tạo optimizer/projector.
Không sửa script train, pretrain, NeMo/vLLM vendored, thư viện hoặc checkpoint gốc.

Khôi phục native checkpoint bằng API NeMo/Megatron đã pin, `load_optimizer=False`.
Đọc tracker Bridge, metadata DCP, common state, storage ranges, training step,
tokenizer/MASK/block/vocab và provenance. Trước restore, kiểm tra mọi tensor/factory
của model được checkpoint bao phủ. Export target và draft vào cache riêng, ghi
SHA256 từng parameter và file. Không dùng pretrained draft weights thay cho draft
đã train; chỉ đọc config/provenance draft gốc để dựng kiến trúc.

Hook thụ động bọc proposer và sampler V1 theo process, đối chiếu AST với live tree.
Chỉ clone token IDs trên GPU, chuyển một packed buffer ở cuối rollout. Không thay
logits, kernel rejection, RNG hay weights; không có thêm forward trong rollout.
Tái dựng output đúng request ID, bao gồm token target-only có nhãn censor;
ordinary events chỉ gồm accepted prefix và reject đầu tiên trước EOS/length cap.
Không suy rejection từ equality hoặc dùng target argmax thay token đã sample.

Thống kê E2/E3/E6 được port sang đặc trưng block DFlash: lịch sử theo global
verification boundary, khoảng cách theo per-response round, donor/recipient rời
nhau, raw/matched RR, exact-token/exposure common support, bootstrap toàn rollout.
Không có tree depth/k/pruning giả. Mỗi checkpoint được phân tích riêng.

Rollout được commit bằng Parquet/JSON/checksum/complete marker. Resume xác minh và
bỏ qua rollout hoàn tất; đủ một stage thì bỏ qua cả model init. Smoke dùng folder
riêng nhưng chia sẻ cache export thật với production. Analyze không import Torch,
Transformers, vLLM hoặc NeMo, và có cache thống kê theo đúng input/source identity.

## Xác nhận đã chạy và phần chưa thể xác nhận

| Mục | Kết quả trên máy phát triển |
|---|---|
| Toàn bộ test project | **375 passed**, 29.38 giây ở lượt kiểm tra cuối |
| Test mới của analysis | **42 tests**, gồm real Torch DCP CPU fixtures và kiểm thử orchestration với fake NeMo API |
| Offline analyze không import model stack | Pass; test xuất đủ 4 PDF và PNG từ fixture |
| Ruff, shell syntax, Git whitespace | Pass |
| Source checkpoint thật Step 250 | Không có trên máy này; chưa restore |
| Source checkpoint thật Step 600 | Không có trên máy này; chưa restore |
| Genuine vLLM frozen rollout/sampling parity B200 | Chưa chạy; được bắt buộc trong `collect --smoke` |
| Số prompt/response/round/miss thực nghiệm | Chưa đo |
| RR/CI/availability Figure 1 thật | Chưa đo |
| Source LaTeX SPARK | Chưa tìm thấy trong workspace; đã đọc prompt và code tham chiếu E2/E3/E6 |

Phần trăm/CI trong CPU fixture chỉ dùng kiểm thử, không phải kết quả Qwen2.5-3B.
Không có full training hoặc genuine checkpoint giả được tạo trong output production.

Máy hiện tại: RTX 3090, Python 3.12, Torch 2.4.1/CUDA 12.1. Runtime server đã pin
là Python 3.13, Torch 2.11/CUDA 13, vLLM 0.25.1. Vì thiếu runtime/checkpoints/dataset
B200, không khẳng định loader Megatron/vLLM đã pass trên hai checkpoint thật.

Lệnh `--mode check` đã được gọi trên development host: dừng chính xác với
`FileNotFoundError: No recorded policy configuration under ...20261012T180453Z__1108564`.
Báo cáo trạng thái được ghi tại `outputs/analysis/qwen25_3b_simplelr/analysis_report.md`;
đó là báo cáo blocker, không chứa kết quả thực nghiệm.

## Provenance bắt buộc khi chạy trên server

```text
.../qwen25_3b__train__dflash__simplelr__seed42__20261012T180453Z__1108564/
  checkpoints/resume/step_250/policy/weights/<iteration đã kiểm tra>/
    target và draft đều từ đây
  checkpoints/resume/step_600/policy/weights/<iteration đã kiểm tra>/
    target và draft đều từ đây
```

Không lấy weights từ các folder target/draft trống. Không lấy target Step 600
ghép với draft Step 250. MCore iteration bên trong có thể khác nhãn GRPO step;
`training_info.json.total_steps` là bước của run và phải khớp 250/600.
Khi export thành công, `checkpoint_manifest.json` ghi đúng nguồn, complete
coverage, cache path và parameter hashes. Khi collect thành công, từng rollout
ghi frozen hash parity; rollout đầu mỗi checkpoint/session còn có replay tắt
logging so sánh sequence/termination/native acceptance counters.

## File thêm mới

```text
analysis/__init__.py
analysis/verification_feedback/__init__.py
analysis/verification_feedback/config.py
analysis/verification_feedback/checkpoints.py
analysis/verification_feedback/data.py
analysis/verification_feedback/events.py
analysis/verification_feedback/hook.py
analysis/verification_feedback/collector.py
analysis/verification_feedback/statistics.py
analysis/verification_feedback/analysis.py
analysis/verification_feedback/plotting.py
analysis/verification_feedback/io.py
analysis/verification_feedback/README_VI.md
scripts/run_dflash_motivation.py
run_dflash_motivation.sh
tests/test_motivation_checkpoints.py
tests/test_motivation_collection.py
tests/test_motivation_events.py
tests/test_motivation_statistics.py
tests/test_motivation_data.py
tests/test_motivation_cli.py
validation/DFLASH_MOTIVATION_IMPLEMENTATION.md
validation/dflash_motivation_validation.json
validation/dflash_motivation_tests.xml
```

Tất cả là file mới trong repo; không thay đổi file code train hiện có.
Chưa stage/commit/push. Cần đưa các file mới vào commit khi chuyển qua GitHub.
Không đóng ZIP, không thay đổi thư viện trong lần triển khai này.

## Chạy và đọc kết quả

Hướng dẫn đầy đủ bằng tiếng Việt:
`analysis/verification_feedback/README_VI.md`.

```bash
cd /workspace/storage-shared/nlp/minhpn19/dflash
source offline_packages/activate.sh
export CUDA_VISIBLE_DEVICES=0 RUNTIME_CHECK=skip
bash run_dflash_motivation.sh --mode check
bash run_dflash_motivation.sh --mode collect --smoke --resume
bash run_dflash_motivation.sh --mode collect --resume
CUDA_VISIBLE_DEVICES="" bash run_dflash_motivation.sh --mode analyze
```

Chọn GPU trống. Bộ cài/environment cũ không cần cài lại cho hook phân tích.
Native PEFT, nhiều rank hoặc stop strings không được âm thầm chuyển sang recipe
khác; các trường hợp ngoài hợp đồng bị từ chối rõ ràng.

Kết quả thật sau chạy: `outputs/analysis/qwen25_3b_simplelr/` gồm các partition
`collection/step_<step>/rollout_XXXX/{events,rounds}.parquet`,
`statistics_step_{250,600}.json`, `summary.csv`, `summary.json`,
`fig1{a,b,c}_dflash.pdf`, `fig1_dflash_combined.{pdf,png}` và `analysis_report.md`.
Khoảng tin cậy thiếu support được báo null với số event/cluster, không giả định
RR > 1. Các kết quả là liên hệ quan sát, không phải chứng minh nhân quả.
