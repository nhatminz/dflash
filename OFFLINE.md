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

### Source build CUDA 13 và log lỗi Ninja

Nếu DeepGEMM đã cài/import thành công nhưng xác nhận cuối báo
`mcore: flash_attn differs from the NeMo lock`, đây là lỗi đối chiếu tên wheel
với metadata: wheel đã pin tên `2.8.1+cu13torch2.10cxx11abiTRUE`, nhưng
`METADATA` bên trong ghi `2.8.1`. Validator mới kiểm tra SHA256 wheel từ
`uv.lock` và 84 file đã cài theo RECORD, gồm extension CUDA; không chấp nhận
một gói PyPI/CUDA 12 khác chỉ vì có cùng số `2.8.1`. Với cài trực tiếp bằng
`uv sync`, yêu cầu provenance URL và SHA256 đúng lock.

Khi các gói đã cài xong, chỉ cần ZIP nhỏ chứa code, không chuyển lại wheel:

```bash
cd /workspace/storage-shared/nlp/minhpn19/dflash
unzip -oq library/dflash_b200_runtime_metadata_fix.zip
python3 scripts/install_local_packages.py --finalize-only \
  && source offline_packages/activate.sh
```

`--finalize-only` chỉ áp dụng patch vLLM idempotent, ghi activation và kiểm
tra ba môi trường đã có. Không gọi pip/uv install, NVCC hoặc source build.
Validator vẫn yêu cầu đúng pin runtime, CUDA 13/BF16, import MCore/TE/DFlash,
DeepGEMM và vLLM/Reflex. Nếu một backend lỗi, nó kiểm tra tiếp các backend
còn lại và ghi đầy đủ lỗi cùng phiên bản thực vào
`offline_packages/runtime_validation.json`; tiến trình vẫn trả lỗi.
Không chạy lại full setup hoặc resume DeepGEMM sau lệnh này.

Kiểm chứng wheel thật: `validation/flash_attn_metadata_real_probe.json`.
Probe trên máy chuyển file chỉ kiểm tra danh tính binary; import và kernel
GPU B200 chưa được chạy trên máy đó.

Lỗi DeepGEMM `/usr/bin/ld: cannot find -lnvrtc` là lỗi link sau khi C++ đã
compile xong. Setup.py ở commit đã pin link `cudart` và `nvrtc` nhưng chỉ thêm
`${CUDA_HOME}/lib64`; wheel CUDA 13 của backend đặt thư viện ở
`nvidia/cu13/lib` và chỉ mang SONAME `.so.13`. Installer tạo symlink tương đối
`libnvrtc.so`/`libcudart.so`, thêm library search path và RUNPATH riêng cho
DeepGEMM. Không đổi CUDA_HOME, pin dependency hay source DeepGEMM; không tải
thêm NVRTC nếu các wheel trong registry đã cài đầy đủ. Probe kiểm tra link
`-lnvrtc -lcudart`, nạp bằng process mới và xác nhận NVRTC 13.0 trước build.

Nếu lỗi đang ở backend vLLM/DeepGEMM, chép ZIP nhỏ mới nhất vào `library/`:

```bash
cd /workspace/storage-shared/nlp/minhpn19/dflash
unzip -oq library/dflash_b200_deep_gemm_nvrtc_fix.zip
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"
python3 scripts/install_local_packages.py --backend vllm \
  --resume-from-source deep-gemm \
  && source offline_packages/activate.sh
```

Lệnh này bỏ qua base, MCore/TE và những source vLLM trước DeepGEMM; hoàn tất
setup/patch/runtime check. Không chạy lại full setup sau đó. ZIP chứa cả hai
wheel header của bản vá trước nhưng không cần chuyển lại bundle lớn. Kiểm
chứng link/load với wheel NVIDIA thật và compile PTX SM100:
`validation/deep_gemm_nvrtc_real_probe.json`; probe PTX preload builtins như
Torch 2.11, không chạy kernel GPU. Toàn bộ wheel DeepGEMM và runtime B200 vẫn
cần được kiểm tra trên máy đích.

Lỗi Transformer Engine tại `util/cuda_nvml.h: fatal error: nvml.h` cần
`nvidia-nvml-dev==13.0.87`. Gói NVIDIA này chỉ chứa header C/C++, không thay
driver hay `libnvidia-ml.so.1`; `nvidia-ml-py`/`pynvml` không cung cấp header đó.
Bản vá `b200_transfer/dflash_b200_te_nvml_resume_fix.zip` chứa cả wheel NVML
(125 KB) và profiler (21 KB), cùng SHA256 trong `requirements-build-cuda13.txt`.
Installer cài chúng trước probe/source CUDA, kiểm tra `nvml.h` ngay từ đầu và
compile thử chính `cuda_nvml.cpp` của TE đã pin trước build lớn.

Đã cài đến Transformer Engine trong MCore thì dùng **resume**, không chạy lại
full setup sau lệnh này:

```bash
cd /workspace/storage-shared/nlp/minhpn19/dflash
unzip -oq library/dflash_b200_te_nvml_resume_fix.zip
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"
python3 scripts/install_local_packages.py --backend mcore \
  --resume-from-source transformer-engine \
  && source offline_packages/activate.sh
```

Resume bỏ qua backend `base`, registry MCore và các source trước TE; cài tiếp
TE/Megatron-LM/Bridge, hoàn tất NeMo MCore, rồi setup backend vLLM và chạy kiểm
tra runtime như full setup. Các backend trước/đang resume phải có Python đã
tạo. Đây là ranh giới do người chạy chỉ định dựa vào log đã hoàn tất, không
suy đoán thành công chỉ từ version metadata. Các dependency vLLM chưa từng
cài vẫn cần được cài lần đầu qua bundle/index đang có.

Không xóa `.venv-local*`, source TE hay `transformer-engine/build/cmake`.
Bản vá không đổi architecture/compiler/CMake flags; headers được bổ sung qua
đường dẫn môi trường. Ninja tái sử dụng object đã hoàn thành khi cấu hình,
source và đường dẫn build giữ nguyên; số bước hiển thị có thể bắt đầu lại.
Kiểm chứng: `validation/te_nvml_real_probe.json` compile chính translation unit
bị lỗi; `validation/te_nvml_resume_tests.xml` kiểm tra luồng resume và giữ
object qua một lần configure/build CMake/Ninja thật.

Lỗi DeepEP `allocator.cuh: fatal error: cuda_profiler_api.h: No such file or
directory` cần thêm **development headers**, không phải đổi commit DeepEP.
Installer bổ sung `nvidia-cuda-profiler-api==13.0.85` theo SHA256 trong
`requirements-build-cuda13.txt` trước probe/source CUDA, áp dụng cả full setup,
resume và `--retry-source deep-ep`. Đây là gói chỉ chứa headers; các pin runtime và
upstream lock giữ nguyên. ZIP sửa lỗi mang sẵn wheel 21 KB trong
`offline_packages/toolchain_wheels/`, được cài bằng `--no-index --require-hashes`.
ZIP mới nhất chứa đủ cả hai wheel development. Nếu bundle thiếu một wheel,
installer dùng thêm pip index đã cấu hình.

HybridEP ở commit đã pin còn link `-lnvtx3interop`. Wheel NVTX đã có trong lock
chỉ chứa `libnvtx3interop.so.1`: installer tạo symlink tương đối `.so`, bổ sung
library search path và RUNPATH riêng cho build DeepEP. Không sửa CUDA toolkit
hay tắt profiler/HybridEP. Trước build lớn, probe NVCC 13.0 compile và link một
shared object SM100 có tham chiếu profiler/NVTX, rồi thử nạp bằng process mới;
không chạy kernel hay gọi profiler trên GPU.

Chuyển `b200_transfer/dflash_b200_deep_ep_fix.zip` sang folder `library/` trên
B200 rồi chạy:

```bash
cd /workspace/storage-shared/nlp/minhpn19/dflash
unzip -oq library/dflash_b200_deep_ep_fix.zip
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"
python3 scripts/install_local_packages.py --backend mcore --retry-source deep-ep \
  && bash scripts/setup_environment.sh --local-packages \
  && source offline_packages/activate.sh
```

Kiểm chứng trên máy chuẩn bị: `validation/deep_ep_real_probe.json` và
`validation/deep_ep_toolchain_tests.xml`. Probe compile/link thành công không
thay thế việc build toàn bộ wheel DeepEP và kiểm tra runtime B200.

`ninja: build stopped: subcommand failed` là lỗi tổng kết. Các dòng
`ptxas info` ngay trước nó có thể thuộc một compilation khác đã thành công.
Không thể kết luận nguyên nhân từ đoạn đuôi này; cần lỗi `FAILED:`,
`fatal error:`, `error:` hoặc `Killed` ở đầu output.

Installer hiện chạy các source qua `python -m pip install --verbose --no-deps`,
giữ chính sách build isolation của upstream và ghi toàn bộ stdout/stderr vào
`offline_packages/build_logs/<timestamp>/<backend>/<package>.log`. Nếu thất
bại, nó in lại diagnostic đầu tiên cùng lệnh compiler, thay vì chỉ đoạn đuôi
dài của `ptxas`. Package registry vẫn được cài bằng uv từ các requirement đã
export frozen. Index pip được giữ nguyên hoặc lấy từ cấu hình index uv.

Cấu hình build đã bổ sung đường dẫn `CUDA_HOME/include/cccl` vào
`CPLUS_INCLUDE_PATH`, theo Dockerfile NeMo đã pin. Nếu toolkit thiếu CCCL,
installer thử headers từ wheel `nvidia-cuda-cccl` đã được lock và cài trong
chính backend; nó không tải một phiên bản thay thế. Trước khi build source
MCore/vLLM, installer kiểm tra NVCC 13.0, compile thử các Torch/CUDA headers
bằng C++20 và compile một CUDA object SM100. Đây là kiểm tra toolchain;
không thay thế bước kiểm tra runtime B200 cuối setup.

Nếu lỗi là `CUDAContextLight.h: fatal error: cusparse.h: No such file or
directory`, headers của các math library cũng cần nằm trong đường dẫn compiler.
Các wheel cuSPARSE/cuBLAS/cuSOLVER đã pin cho CUDA 13 chứa headers ở
`<backend-site-packages>/nvidia/cu13/include`; cuDNN dùng
`nvidia/cudnn/include`. Installer bổ sung các thư mục headers tồn tại của
chính backend vào `CPATH`/`CPLUS_INCLUDE_PATH`, đồng thời truyền `-I` qua
`NVCC_PREPEND_FLAGS` cho frontend NVCC. Headers development toolkit đứng
trước wheel headers để CRT từ pip không ghi đè CRT phù hợp với NVCC 13.0.
Các flags NVCC do người dùng đặt được giữ nguyên. Không copy headers vào
`/usr/local/cuda` hay dùng headers của môi trường khác.

Bản sửa này có trong `dflash_b200_cuda_headers_fix.zip`, đã gồm các bản sửa
trước. Nó kiểm tra lại cuSPARSE/cuBLAS/cuSOLVER trong cả probe C++ và CUDA
SM100. Sau khi giải nén, dùng lệnh retry Hadamard bên dưới rồi full setup;
không cần xóa môi trường hoặc tải lại các ZIP dependency lớn.

Mặc định giới hạn build `MAX_JOBS=2`, `NVCC_THREADS=1`,
`NVTE_BUILD_MAX_JOBS=2`, `NVTE_BUILD_THREADS_PER_JOB=1`; giá trị bạn đặt được
giữ nguyên. Những biến này không thay đổi batch size hay thuật toán training.
Một số setup.py tự hardcode số thread/architecture riêng; chẳng hạn bản
Hadamard đã pin vẫn build SM80/90/100 dù `TORCH_CUDA_ARCH_LIST=10.0`.

Gói `dflash_b200_source_build_fix.zip` chứa installer, helper và các bản sửa
CUPTI/compiler trước đó. Giải nén vào project rồi chạy full setup như trên.
Để chỉ thử lại package đang lỗi trong môi trường MCore đã có sẵn:

```bash
python3 scripts/install_local_packages.py \
  --backend mcore --retry-source fast-hadamard-transform
```

Lệnh retry không cài lại các backend và không tạo activation/đánh dấu setup
hoàn tất. Sau khi thành công, vẫn chạy full setup và kiểm tra runtime. Bản
source Hadamard ở commit `f134af63...` tự khai version `1.0.4.post1`, dù
upstream lock ghi `1.1.0`; installer tiếp tục giữ đúng commit này.

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
