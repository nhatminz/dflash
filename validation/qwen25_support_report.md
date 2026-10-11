# Hỗ trợ bộ 6 model của SpecNaacl, bỏ Llama

Bổ sung Qwen2.5-1.5B/3B/7B/14B-Instruct bên cạnh Qwen3-1.7B/4B. Mỗi model
có launcher pretrain, GRPO Reflex và baseline DFlash. Các wrapper Qwen3-8B
đã có vẫn được giữ như lựa chọn ngoài bộ experiment này.

Target Qwen2.5 giữ nguyên architecture và QKV bias. Drafter DFlash headless
được khởi tạo độc lập bằng architecture Qwen3 bias-free, với hidden size,
head dimension, GQA, vocabulary và RoPE base tương ứng target. Drafter dùng
full block attention, không kế thừa trường sliding-window của Qwen2.5. Target
embedding, LM head và auxiliary hidden states được chia sẻ như đường Qwen3
đã có. Không chuyển weights target sang architecture Qwen3.

MASK 151669 phải nằm trong embedding table và không thuộc vocabulary thực,
added tokens hoặc BOS/EOS/PAD. Preflight, direct pretraining và checkpoint
validation kiểm tra điều này từ file local trước khi tải weights. Checkpoint
Qwen2.5 cần manifest khớp fingerprint target/tokenizer; không chấp nhận draft
Qwen3-8B của z-lab như warm start cho Qwen2.5. Cấu hình drafter Qwen3 đã có
được giữ nguyên để không thay đổi contract resume cũ.

Batch/accumulation GRPO lấy từ cấu hình từng model của SpecNaacl: 16/2, 8/4,
8/4, 4/8 cho Qwen2.5 theo thứ tự tăng size. Pretrain Qwen2.5 dùng microbatch
1 với accumulation 16/8/4/2, đối chiếu batch hiệu dụng với CLI launcher thực
của SpecNaacl; các biến DRAFT_PRETRAIN_* không được launcher đó sử dụng.
Mặc định pretrain Qwen3 hiện có được giữ nguyên. Native full target training
và các khác biệt sample/mask/cadence với SpecNaacl vẫn được công khai trong
từng ma trận `fairness_<MODEL_KEY>.json`; đây không phải strict equivalence.

Kiểm chứng: **98 tests passed, 0 failed, 0 skipped**. Có forward/backward và
optimizer step với target HF Qwen2 thật ở kích thước nhỏ, QKV bias hiện hữu,
drafter DFlash gốc không bias, shared embedding/head, checkpoint export/header
validation và reload. Test còn kiểm tra cấu hình chính thức của cả 4 Qwen2.5,
batch từ CLI SpecNaacl thực, và 6 pretrain launcher dry-run đồng thời có run
directory riêng. Đây không phải 6 job GPU training thật.

21 wrapper qua dry-run và Bash syntax; Ruff/compileall qua. Các SHA upstream
giữ nguyên và audit SpecNaacl không thấy file nguồn bị đổi. Không sửa vendor
patch để thêm target family này. [JUnit](qwen25_support_tests.xml),
[source validation](qwen25_source_validation.json),
[JSON report](qwen25_support_report.json). Các config Qwen2.5 tham chiếu và
revision HF chính thức nằm trong `third_party/reference_qwen25_revisions.json`.

Chưa chạy full pretraining/NeMo GRPO trên B200 với các model/dataset local.
Lệnh chạy 6 job trên 6 GPU nằm trong
[hướng dẫn](../huongdanchay.md#chạy-đồng-thời-6-model).
