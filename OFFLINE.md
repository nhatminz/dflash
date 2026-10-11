# B200 truy cập pip index nhưng không truy cập GitHub

Lỗi `No route to host` ở bước fetch NeMo là thiếu source, chưa phải lỗi CUDA.
Một `.git` rỗng sau fetch thất bại không chứa commit để chạy. Backend được pin
cần Python >=3.13.14,<3.14, khác Python 3.12 đang dùng để gọi setup trong log.

## Bộ cài được chuẩn bị trên máy có internet

`scripts/prepare_local_packages.py` đọc chính `uv.lock` và dùng `uv export
--frozen` độc lập cho base, mcore, vllm. Nó chọn Linux x86_64 / CPython 3.13,
không dùng Python/CUDA của máy tải để chọn wheel. Các artifact ngoài PyPI
được đặt trong `offline_packages/wheels/`:

* Torch 2.11.0+cu130, TorchVision 0.26.0+cu130, Triton 3.6.0;
* FlashAttention 2.8.1+cu13torch2.10cxx11abitrue cp313;
* FlashInfer JIT cache 0.6.8.post1+cu130 cho mcore và 0.6.13+cu130 cho vLLM;
* vLLM 0.25.1 Linux x86_64 ABI3.

Mỗi wheel được kiểm SHA256 theo lockfile. CDN PyTorch chính thức được dùng
thay CDN r2 khi URL r2 bị từ chối, vẫn phải đúng hash. Python standalone
3.13.x (>=3.13.14) được tải kèm tar.gz và ghi URL/hash vào manifest để B200
không phải tải interpreter từ GitHub.

13 dependency được pin bằng Git được checkout vào `offline_packages/sources/`
với đúng commit và recursive submodules. Chúng gồm Lens, TransferQueue,
DeepEP/DeepGEMM, TransformerEngine, Mamba, causal-conv1d, FlashMLA và các
dependency liên quan. Đây là source, **không phải wheel đã build**; một wheel
PyPI cùng tên/version không thay thế được commit tùy biến. Chúng được build
trên B200 bằng CUDA development toolkit 13. Không build chúng bằng Torch
2.4/CUDA 12 của máy 3090 rồi coi là wheel production tương thích.

Ba repo đã pin của project và các submodule NeMo/Megatron cũng được chuẩn bị
trong `third_party/`, bao gồm metadata `.git`. Không đổi `versions.lock` hoặc
`uv.lock`, không cài cả MCore/vLLM vào cùng environment.

Lệnh tải/tiếp tục tải trên máy có internet (Python >=3.11 và `packaging`, `uv`):

```bash
cd /đường/dẫn/dflash
.venv-test/bin/python scripts/prepare_local_packages.py
```

Nếu máy tải không có `.venv-test`, dùng Python phù hợp đã cài `packaging`.
`--plan-only` chỉ lập danh sách; `--wheels-only` tải wheel/Python nhưng manifest
vẫn `ready=false` vì thiếu source. Đừng chuyển một bộ mới tải dở như bộ hoàn tất.
Các file tải thành công được tái sử dụng nếu hash đúng. Các file `.partial`
không được dùng để cài.

## Chuyển sang B200

Chuyển code project đã sửa, **toàn bộ** `offline_packages/` và `third_party/`.
Không dùng quy tắc `.gitignore` khi copy dependency; không bỏ các thư mục
`.git` nằm trong `third_party/` và `offline_packages/sources/`. `git clone`
project chính không mang các artifact bị ignore này sang.

Có sẵn ba file ZIP trên máy 3090 (đường dẫn lưu cụ thể được ghi trong
`validation/b200_transfer_report.json`):

* `dflash_b200_sources.zip`: source dependency, metadata `.git`, submodule,
  Python standalone và manifest/requirements;
* `dflash_b200_wheels.zip`: 6 wheel ngoài PyPI;
* `dflash_b200_torch.zip`: wheel Torch, lưu riêng trên SSD vì ổ dự án thiếu chỗ.

Chuyển cả ba ZIP và `SHA256SUMS` sang cùng một thư mục trên B200. Code project lấy từ GitHub như
bình thường, rồi giải nén **từ thư mục gốc `dflash`**:

```bash
cd /đường/dẫn/chứa/ba/zip
sha256sum -c SHA256SUMS
cd /workspace/storage-shared/nlp/minhpn19/dflash
unzip -oq /đường/dẫn/chứa/ba/zip/dflash_b200_sources.zip
unzip -oq /đường/dẫn/chứa/ba/zip/dflash_b200_wheels.zip
unzip -oq /đường/dẫn/chứa/ba/zip/dflash_b200_torch.zip
python3 scripts/install_local_packages.py --check-only
bash scripts/setup_environment.sh --local-packages
source offline_packages/activate.sh
```

ZIP giữ symlink và quyền Unix; dùng `unzip` trên Linux. Không cần chuyển
virtualenv của máy 3090, cache test hoặc `outputs/`. Bộ ZIP không chứa model
weights và dataset; giữ các đường dẫn dữ liệu/model local đã có trên B200.

Ví dụ đóng gói dependency để chuyển qua shared storage/file transfer:

```bash
tar -czf /đường/dẫn/còn-đủ-dung-lượng/dflash_b200_deps.tar.gz \
  offline_packages third_party
```

Giải nén vào project B200 sau khi đồng bộ code mới:

```bash
cd /workspace/storage-shared/nlp/minhpn19/dflash
tar --no-same-owner -xzf /đường/dẫn/dflash_b200_deps.tar.gz
python3 scripts/install_local_packages.py --check-only
```

Không cần xóa `.git` rỗng do lần fetch lỗi: bộ source đầy đủ thay thế metadata
thiếu khi giải nén. Installer từ chối source sai SHA hoặc artifact sai hash.

## Cài trên B200

Chế độ `--local-packages` chỉ dùng file local cho wheel ngoài PyPI và Git
source. Các package PyPI còn lại được lấy từ index đang truy cập được. Installer
đọc `PIP_INDEX_URL`/`PIP_EXTRA_INDEX_URL` nếu bạn chưa cấu hình index riêng cho
uv. Nó không tự đổi sang một phiên bản thư viện khác khi thiếu dependency.

Nếu log dừng ở `nvidia-resiliency-ext` với `required files libcupti.so and
cupti.h not found`, lỗi xảy ra trong setup, trước khi có `activate.sh`. Bản
installer mới lấy CUPTI từ `nvidia-cuda-cupti==13.0.85` đã cài trong môi trường
MCore: wheel có `nvidia/cu13/include/cupti.h` và `lib/libcupti.so.13`, nhưng
thiếu tên linker `libcupti.so` mà upstream tìm. Installer tạo symlink tên này,
trỏ `CUDA_PATH` riêng cho build Resiliency, bổ sung headers từ development
toolkit qua `CPPFLAGS` và RUNPATH cho thư viện. Không tắt CUPTI, không đổi pin
và không đổi `CUDA_HOME` của các extension khác.

Cập nhật `scripts/install_local_packages.py` (hoặc giải nén gói nhỏ
`dflash_b200_cupti_fix.zip` từ máy 3090) rồi chạy lại setup. Các environment
và package đã cài được tái sử dụng; không cần xóa chúng hay tải lại ba ZIP lớn.
Chỉ source activation sau khi setup thành công:

```bash
bash scripts/setup_environment.sh --local-packages && source offline_packages/activate.sh
```

Nếu build báo `No such file or directory: 'clang++'`, compiler được chọn không
có trên máy. Installer mới đặt `CC`/`CXX` rõ ràng cho mọi backend và build
isolation: ưu tiên `gcc`/`g++`, rồi `clang`/`clang++`, rồi `cc`/`c++` có trong
`PATH`. Nó giữ cấu hình `CC`/`CXX` do bạn đặt nếu executable tồn tại; cấu hình
sai sẽ báo lỗi trước khi cài package. Bản sửa có trong
`dflash_b200_compiler_fix.zip` (đã bao gồm bản sửa CUPTI).

Nếu máy đã có GCC, cũng có thể chạy lại ngay với installer cũ:

```bash
export CC="$(command -v gcc)"
export CXX="$(command -v g++)"
if [[ -n "$CC" && -n "$CXX" ]]; then
  bash scripts/setup_environment.sh --local-packages && source offline_packages/activate.sh
else
  echo "Cần C/C++ compiler trong PATH; chưa tìm thấy gcc/g++."
fi
```

```bash
cd /workspace/storage-shared/nlp/minhpn19/dflash
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"
export NVTE_CUDA_ARCHS=100
export TORCH_CUDA_ARCH_LIST=10.0
export CUDA_VISIBLE_DEVICES=0

bash scripts/setup_environment.sh --local-packages
source offline_packages/activate.sh

"$PYTHON_BIN" scripts/validate_environment.py --require-runtime \
  --model /workspace/storage-shared/models/Qwen3-1.7B \
  --data /workspace/storage-shared/nlp/minhpn19/data/sharegpt/ShareGPT_V4.3_unfiltered_cleaned_split.json

unset MODEL MODEL_KEY MODEL_ENV RUN_NAME RUN_DIR RESUME DRAFT_CHECKPOINT
bash pretrain_qwen3_1p7b.sh
```

Environment mới là `.venv-local`, `.venv-local-mcore`, `.venv-local-vllm`;
không ghi đè `.venv` đã có. **Mỗi terminal mới cần source
`offline_packages/activate.sh` trước khi chạy launcher** để chọn chúng.
Build các CUDA extension vẫn cần compiler, cuDNN headers và system libraries
đúng theo upstream; DeepEP cần các thư viện phát triển RDMA. Bộ artifact không
thay thế các package hệ điều hành đó. Full installation/build/runtime check
trên B200 chưa được chạy ở máy 3090.

Có một mâu thuẫn metadata ngay trong upstream đã pin: NeMo yêu cầu Python
3.13, nhưng gitlink Megatron-Bridge vẫn khai báo `<3.13`. Installer giữ nguyên
commit, chỉ dùng `pip --ignore-requires-python --no-deps --no-build-isolation`
cho Bridge và in thông báo rõ ràng. Đây không phải bằng chứng Bridge chạy
đúng trên B200: bước kiểm tra import MCore/Bridge/TE cuối cùng vẫn bắt buộc.
Các source khác dùng chính sách build isolation của upstream; phiên bản của
ModelOpt/Resiliency lấy từ lockfile thay vì suy đoán từ Git history shallow.

## Máy chặn cả pip index

Đó là trường hợp khác: cần cache/wheelhouse đầy đủ của mọi package và Python.
`bash scripts/setup_environment.sh --offline` không fetch Git/submodule, đặt
`UV_OFFLINE=true`, tắt tải Python tự động và chỉ dùng cache sẵn có. Bộ local
artifacts ở trên chỉ thay các nguồn ngoài PyPI, không tuyên bố chứa toàn bộ
PyPI. Nếu thiếu cache, setup dừng và báo lỗi; không tự bật mạng.
