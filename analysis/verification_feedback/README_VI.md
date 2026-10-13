# Phân tích verification feedback DFlash cho Figure 1 SPARK

Pipeline này chỉ đọc cặp target/draft đã train ở **Step 250** và **Step 600**.
Không chạy pretrain, không resume GRPO, không tạo optimizer hoặc projector,
không cập nhật Reflex A/B. Các script train hiện có giữ nguyên.

## Chạy trên B200

Chuyển các file code mới trong repo `dflash` sang server bằng Git như bình thường.
Không cần build lại vLLM để cài hook phân tích: hook chỉ được gắn trong process
inference của launcher phân tích. Không cài thêm thư viện từ Internet.
Môi trường NeMo offline hiện có cung cấp PyTorch, Transformers, Pandas,
PyArrow, Matplotlib, tqdm và Safetensors.

Trong một terminal, chọn **một GPU đang trống**:

```bash
cd /workspace/storage-shared/nlp/minhpn19/dflash
source offline_packages/activate.sh
export CUDA_VISIBLE_DEVICES=0

# Môi trường đã cài/kiểm tra trước đó: bỏ qualification nặng.
# Metadata, checksum và compatibility checkpoint vẫn luôn được kiểm tra.
export RUNTIME_CHECK=skip

# 1. Kiểm tra checkpoint, cấu hình, tokenizer và train/test disjointness.
# Không tạo model, optimizer, Ray cluster hoặc chạy GRPO.
bash run_dflash_motivation.sh --mode check

# 2. Một rollout nhỏ: 2 prompt x 2 response mỗi checkpoint.
# Giữ nguyên maximum generation length và dừng EOS tự nhiên.
# Bắt buộc so sánh logging bật/tắt, native acceptance counters và hash weights.
bash run_dflash_motivation.sh --mode collect --smoke --resume

# 3. Đầy đủ: 20 rollout x 8 prompt x 8 response mỗi checkpoint.
# --resume kiểm tra rồi bỏ qua rollout đã commit hợp lệ.
bash run_dflash_motivation.sh --mode collect --resume

# 4. Thống kê và figures, không cần GPU hoặc truy cập model/checkpoint gốc.
CUDA_VISIBLE_DEVICES="" bash run_dflash_motivation.sh --mode analyze
```

Để thực hiện liên tiếp check, collect và analyze:

```bash
CUDA_VISIBLE_DEVICES=0 RUNTIME_CHECK=skip \
  bash run_dflash_motivation.sh --mode all --resume
```

Không chạy các lệnh `train_qwen*.sh` cho thí nghiệm này.
`--resume` ở launcher này **chỉ tiếp tục thu thập analysis**, không resume optimization.

Mặc định `RUNTIME_CHECK=auto`: dùng cache qualification của project nếu vẫn hợp lệ.
`RUNTIME_CHECK=skip` chỉ bỏ qualification môi trường; không bỏ kiểm tra checkpoint.
Launcher tự ép `METHOD=dflash`, `OPD_ENABLED=0`, `METRICS_MODE=production`,
`OPD_PROFILE=0` và V1 runner. Logger riêng vẫn thu đầy đủ event phân tích.

## Input mặc định và nguồn weights

```text
SOURCE_RUN=/workspace/storage-shared/nlp/minhpn19/dflash/outputs/train/qwen25_3b/qwen25_3b__train__dflash__simplelr__seed42__20261012T180453Z__1108564
TEST_DATASET_PATH=/workspace/storage-shared/nlp/minhpn19/data/simplelr_abel_level3to5/test.parquet
CHECKPOINTS=250,600
SEED=2026
EVAL_ROLLOUTS=20
PROMPTS_PER_ROLLOUT=8
RESPONSES_PER_PROMPT=8
BOOTSTRAP_SAMPLES=2000
```

Checkpoint được lấy trực tiếp từ:

```text
$SOURCE_RUN/checkpoints/resume/step_250/policy/weights/
$SOURCE_RUN/checkpoints/resume/step_600/policy/weights/
```

Không dùng hai thư mục `checkpoints/target` / `checkpoints/draft` đang trống.
Kiểm tra `training_info.json.total_steps` phải đúng step yêu cầu. Iteration bên
trong Megatron được đọc riêng, có thể là `iter_0000000`: nó không tự động đồng
nghĩa với số GRPO iteration ở thư mục ngoài. Bridge tracker có ưu tiên như loader
gốc; export trỏ thẳng vào iteration đã xác nhận, không chọn lại “latest”.

`check` đọc DCP metadata, storage ranges, native backend/common state,
run/checkpoint configurations, MASK, vocab, tokenizer fingerprint và checksum.
Nếu checkpoint có `integrity.json`, sử dụng API MCore để kiểm tra manifest đó.

`collect` khôi phục bằng NeMo `setup_model_and_optimizer(load_optimizer=False)`.
Hook kiểm tra coverage của các ShardedTensor và ShardedTensorFactory trước load.
Không đọc pretrained draft weights; chỉ dùng **config và provenance manifest**
của draft gốc được ghi trong recipe để dựng đúng kiến trúc, rồi restore toàn bộ
weights target/draft từ cùng checkpoint resume. Nếu các file config đó đã bị
xóa, lệnh dừng rõ ràng; không dựng draft khác để thay thế.

Weights đã restore được export Safetensors vào:

```text
outputs/analysis/qwen25_3b_simplelr/.cache/inference_step_250/
outputs/analysis/qwen25_3b_simplelr/.cache/inference_step_600/
```

Cache export được dùng chung giữa smoke và production, có khóa độc quyền,
source identity và checksum. Kết quả smoke nằm trong thư mục `smoke/` riêng;
không ghi đè production. Checkpoint gốc chỉ được đọc.

Settings lấy từ `run_config.json` đã expand, hoặc `config.yaml` đã lưu trong
checkpoint. Các file launch/resolved/metadata hiện có được liệt kê, hash và đối
chiếu. Không tự lấy batch/temperature/generation length từ shell train hiện tại.
Train split lấy từ recipe gốc. `--test-data` cho phép chỉ định rõ split test khác,
nhưng vẫn phải đủ prompt và không trùng question hash với train gốc.

## Thu thập và tính đúng của event

Hook kiểm tra AST các hàm sampler/proposer/V1 runner **đang được import** so với
source vendored live. Khác semantics thì dừng trước collection. Commit/package
pins nằm trong `versions.lock`; không dùng ZIP code cũ làm nguồn xác thực.

Target token là ID thật trong output của rejection sampler. Với greedy DFlash,
`draft_probs=None` nghĩa là proposal singleton có xác suất 1. Recovery loại ID
proposal. Accepted prefix được suy ra từ hợp đồng output đã kiểm tra của hai
kernel: **prefix accepted + recovery hoặc bonus**, sau đó là các sentinel `-1`.
Không dùng token equality để xác định accepted prefix, không dùng target argmax.
Equality chỉ kiểm tra lại candidate-set miss và hợp đồng singleton.

Chỉ ghi các token thực sự phát ra: accepted proposal, proposal bị reject đầu
tiên, và các token target-only có nhãn censor riêng để tái dựng response đầy đủ.
Loại post-rejection/post-EOS/padding slots khỏi ordinary denominators.
EOS, length cap, bonus và prefill/target-only không phải ordinary miss.
`accepted_prefix_length` giữ số accepted của sampler trước censor;
`visited_proposal_count` trong rounds ghi số vị trí thực sự phát ra trước bonus.

Request ID thật nối GPU batch với response ID chuẩn, không dựa vào thứ tự batch
cố định. Các copy trong vòng nóng chỉ chứa token IDs trên GPU. Một packed D2H ở
cuối rollout; không copy vocabulary logits, không thêm target forward, không
synchronize từng token/round.

Logging-disabled replay bắt buộc cho rollout đầu tiên của mỗi checkpoint/session.
Giữ seed của từng request; reset prefix cache giữa các rollout độc lập và trước
replay, giữ nguyên caching setting bên trong rollout. Tokens, termination và
native speculative acceptance counters phải trùng. Hash toàn bộ parameter của
target/draft trước evaluation và sau mỗi rollout phải giữ nguyên. Chi phí hash,
export và logging không được báo là inference latency; timing replay là wall
time ở ranh giới rollout, không phải kernel-only GPU time.

## Thống kê

Phương pháp tham chiếu: `analysis_spec/e2e6/analysis.py`, E2/E3/E6.
Không import runtime FastGRPO/SpecNaacl vào inference DFlash.

- 1(a): raw/matched risk ratio trong response, distance 1, 2–4, 5–16, 17–64, >64;
  control donor/recipient độc lập trong từng checkpoint, chia hai nửa ID đã sort,
  ghép một-một không dùng lại donor; chỉ donor completion ordinal nhỏ hơn recipient.
- 1(b): nguồn loại trừ nhau theo ưu tiên same response → sibling → different prompt
  → none. So sánh chính sibling/none; còn báo so sánh sibling bất kỳ/no sibling.
- 1(c): số ordinary miss có feedback trước đó trong response và trong prompt
  **bao gồm current response**, chia cho toàn bộ ordinary miss.

Lịch sử được cập nhật sau toàn bộ global verification boundary. Không có
same-batch/future leakage. Khoảng cách riêng trong response dùng per-response rounds.
Matching dùng token ID, block slot/length, valid proposal count, active-size bin,
response-position bin, verification-progress bin và opportunity/target-token exposure.
Không gán tree depth, draft-k hay pruning label giả cho DFlash.

CI percentile 95% với 2.000 bootstrap draw của toàn rollout; control resample các
cặp độc lập theo recipient rollout. Giữ toàn bộ sibling responses cùng cluster.
Matched risks chỉ trên common support; bootstrap không điền missing risk mà
chuẩn hóa trọng số thực nghiệm trên support có cả hai điều kiện trong draw đó.
JSON báo unsupported strata/events, số cluster và số draw hợp lệ. Không đủ
support/cluster thì giá trị hoặc CI là `null`, không được thay bằng 0 hay ép RR > 1.
Đây là phân tích quan sát, không chứng minh feedback sẽ sửa được miss.

## Output

Mặc định: `outputs/analysis/qwen25_3b_simplelr/`.

```text
checkpoint_manifest.json
evaluation_manifest.json
collection/step_250/rollout_0000/events.parquet
collection/step_250/rollout_0000/rounds.parquet
collection/step_600/rollout_0000/events.parquet
collection/step_600/rollout_0000/rounds.parquet
statistics_step_250.json
statistics_step_600.json
summary.csv
summary.json
fig1a_dflash.pdf
fig1b_dflash.pdf
fig1c_dflash.pdf
fig1_dflash_combined.pdf
fig1_dflash_combined.png
analysis_report.md
```

Mỗi `rollout_XXXX/` chứa `events.parquet`, `rounds.parquet`, `responses.json`,
`requests.json`, `audit.json`, `complete.json`. Đây là dạng partitioned của
`events_step_<step>.parquet`/`rounds_step_<step>.parquet`, tránh gom toàn bộ event
vào một file lớn. Complete marker chỉ được ghi sau file fsync và checksum.
Lệnh resume bỏ qua marker đã xác minh; file hoàn tất bị hỏng gây lỗi, không
âm thầm recollect. Rollout chưa có complete marker mới được chạy lại.

Một checkpoint có đủ rollout hợp lệ sẽ được bỏ qua cả model initialization
khi `--resume`. `analyze` dùng cache thống kê theo input hashes/source/seed và
bootstrap size. Có thể thêm checkpoint thật bằng `--checkpoints 250,600,<step>`;
không cần thay logic thống kê hoặc tạo nhãn “middle stage” giả.

## Kiểm thử và giới hạn xác nhận

```bash
# Development environment; không phải lệnh cài lại thư viện trên B200.
.venv-test/bin/python -m pytest -q tests/test_motivation_*.py
```

CPU tests dùng fixture nhỏ tách khỏi output production để kiểm tra DCP loading,
alignment, censoring, histories, donor controls, denominator, cluster bootstrap,
atomic resume, exporter orchestration và xuất PDF/PNG không import model stack.
Fixture không phải checkpoint Qwen2.5-3B Step 250/600 thật.

Máy phát triển hiện là RTX 3090, Torch 2.4.1/CUDA 12.1; thiếu hai checkpoint B200,
test split B200 và runtime vLLM/MCore tương ứng Torch 2.11/CUDA 13.0. Vì vậy chưa
xác nhận genuine GPU restore, frozen rollout hoặc sampling parity trên B200.
Chưa có số miss/RR/CI thực nghiệm hoặc figure kết quả thật để đưa vào paper.
Source LaTeX SPARK chưa tìm thấy trong workspace; đã đối chiếu prompt và code E2/E3/E6.
Kết quả thật chỉ được báo sau khi các lệnh trên hoàn tất và validate thành công.
