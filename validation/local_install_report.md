# B200 local artifact preparation

Bộ artifact đã tải hoàn tất; `manifest.json` có `ready=true`.
`install_local_packages.py --check-only` đã pass: SHA256 của 7 wheel và
Python archive, commit của 13 Git dependency và recursive submodules đều hợp lệ.

| Artifact ngoài PyPI | Dung lượng |
|---|---|
| `torch-2.11.0+cu130-cp313-cp313-manylinux_2_28_x86_64.whl` | 506.5 MiB |
| `torchvision-0.26.0+cu130-cp313-cp313-manylinux_2_28_x86_64.whl` | 7.2 MiB |
| `triton-3.6.0-cp313-cp313-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl` | 179.6 MiB |
| `flash_attn-2.8.1+cu13torch2.10cxx11abiTRUE-cp313-cp313-linux_x86_64.whl` | 118.8 MiB |
| `flashinfer_jit_cache-0.6.8.post1+cu130-cp39-abi3-manylinux_2_28_x86_64.whl` | 1994.0 MiB |
| `flashinfer_jit_cache-0.6.13+cu130-cp39-abi3-manylinux_2_28_x86_64.whl` | 1462.3 MiB |
| `vllm-0.25.1-cp38-abi3-manylinux_2_28_x86_64.whl` | 238.5 MiB |

Python standalone: 3.13.16, Linux x86_64 GNU. Toàn bộ `offline_packages/`
khoảng 5.4 GiB; `third_party/` khoảng 619 MiB. Chuyển cả hai cùng code mới,
giữ metadata `.git`. Không cần chuyển `.venv-test` hoặc `outputs`.

Full test suite: 104 passed, 0 skipped; Ruff check/format,
Bash syntax và compile các script mới đều pass. SpecNaacl không thay đổi.
`versions.lock` và NeMo `uv.lock` không thay đổi.

Đã build wheel Megatron-Bridge đúng gitlink bằng Python 3.13.16 + setuptools
83.0.0. Ngoại lệ metadata `<3.13` của Bridge được báo rõ trong installer;
không mở rộng ngoại lệ cho các dependency khác.

Chưa chạy full install hoặc build extension CUDA 13 trên B200. Các wheel
được chọn từ lockfile upstream; kiểm tra ABI/import GPU cuối cùng vẫn bắt buộc.
Các gói PyPI còn lại dùng index truy cập được; đây không phải wheelhouse chứa
mọi dependency PyPI. Hướng dẫn: [OFFLINE.md](../OFFLINE.md).
