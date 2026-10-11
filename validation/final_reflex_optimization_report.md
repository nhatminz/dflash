# Báo cáo cuối: DFlash + ReflexOPD correctness, fairness và hot path

Bản sửa được áp dụng trực tiếp vào project hiện có. **83 kiểm thử qua, không skip; chưa xác minh production B200/NeMo GRPO.** Chỉ thay đổi trong `dflash/` và patch dependency do project quản lý. SHA của 439 file nguồn SpecNaacl trước/sau trùng nhau; ba upstream commit giữ nguyên. Không chạy pretrain đầy đủ, GRPO dài hoặc sweep siêu tham số.

`FAIRNESS_PROFILE=specnaacl_matched` mặc định **thất bại preflight có chủ đích**. Backend đã pin chưa hỗ trợ đầy đủ LoRA độc lập/cadence/lọc mẫu yêu cầu. Cấu hình chạy được ở mức resolver là `native_dflash`, vẫn full fine-tune target và không được coi tương đương hoàn toàn SpecNaacl.

## A. Những lỗi và bất biến đã xử lý

### C01: Feedback q was reconstructed using a different head reduction and omitted actual logit processing.

- Nguồn: `dflash_reflex/kernels.py::_target_q`.
- Nguyên nhân: Recomputed H @ head may differ from model.compute_logits (BF16 rounding, bias, scaling, processors).
- Đường hiện tại: `kernels.py::_union_terms/statistics; vllm_runtime.py::Runtime.propose/verified`.
- Sửa/giữ đúng: Retain the exact raw proposal logits until verification; map original rows across request reordering; fuse teacher-coordinate q into union using A_t/U_t/B_t. Never copy B or run a second head forward. Reject stale B/A/head versions.
- Kiểm chứng: `test_feedback_parity.py::test_selected_feedback_uses_exact_raw_and_old_b`, `test_runtime_alignment.py::test_variable_lengths_reordered_selective_and_version_guard`.

### C02: Greedy proposals computed full top-k/normalization for every slot.

- Nguồn: `kernels.py::propose/_scan/_merge`.
- Nguyên nhân: Proposal selection and learning statistics shared the same full-vocabulary path unconditionally.
- Đường hiện tại: `kernels.py::greedy_propose/_argmax_scan/_argmax_merge/statistics; vllm_runtime.py::Runtime.propose`.
- Sửa/giữ đúng: Keep Candidate A and add Candidate B corrected argmax followed by masked valid-row statistics. Preserve serial FP32 correction order, disabled FP fusion and minimum-token-ID ties. Select from complete hot-path calibration rather than isolated argmax.
- Kiểm chứng: `test_proposal_parity.py::test_exact_argmax_and_statistics`, `test_feedback_parity.py::test_fp64_union_tail_a_gradient_and_b_update_oracle`.

### C03: Teacher top-k ties could change the selected operational gradient.

- Nguồn: `kernels.py::feedback (torch.topk)`.
- Nguyên nhân: torch.topk does not guarantee SpecNaacl minimum-ID boundary ties; selected coordinates matter even when probabilities tie.
- Đường hiện tại: `kernels.py::_teacher_flags/feedback`.
- Sửa/giữ đúng: Canonicalize K+1 teacher candidates. Only positive K/K+1 boundary ties scan ascending vocabulary IDs. Masked teacher scan uses the same tie rule; zero-probability teacher-only coordinates are excluded as in SpecNaacl.
- Kiểm chứng: `test_feedback_parity.py::test_teacher_topk_ties_are_identical_for_all_candidates`.

### C04: Unused feedback slots still consumed vocabulary work; greedy teachers needlessly required a dense one-hot tensor.

- Nguồn: `kernels.py::feedback; vllm_runtime.py::Runtime.verified`.
- Nguyên nhân: All padded/post-rejection rows passed through teacher top-k and draft statistics.
- Đường hiện tại: `kernels.py::_scan/_merge/_union_terms/feedback; vllm_runtime.py::Runtime.verified`.
- Sửa/giữ đúng: GPU masks return before expensive vocabulary loads on invalid rows. Benchmark masked versus full teacher top-k; greedy p is an argmax delta without a new dense one-hot allocation. Use actual sampler decisions, cap/EOS/variable-length/request masks.
- Kiểm chứng: `test_feedback_parity.py::test_greedy_teacher_delta_without_vocabulary_probability_allocation`, `test_sampling.py`, `test_runtime_alignment.py`, `test_reflex.py`.

### C05: Growing host density bound could force dense correction on repeatedly visited token IDs.

- Nguồn: `kernels.py::propose/feedback (host_active_bound)`.
- Nguyên nhân: n * 2 * K per round counts duplicate IDs repeatedly and is not active-row density.
- Đường hiện tại: `kernels.py::_update/_scan/_argmax_scan; adapter.py; dispatch.py`.
- Sửa/giữ đúng: Unique GPU active bitmap/count and compact IDs; GPU threshold branch reads actual active_count. GPU counters record chosen backends and materialize only at cohort boundaries. Unknown hardware/shape stays sparse. Legacy measured fused cache migrates; no arbitrary vocab/8 threshold or automatic GEMM.
- Kiểm chứng: `test_sparse_dense_dispatch.py::test_gpu_density_dispatch_and_repeated_active_ids`, `test_sparse_dense_dispatch.py::test_legacy_measured_cache_migrates_without_host_density_bound`.

### C06: A learning, update order, skipped steps and resume needed an explicit invariant audit.

- Nguồn: `kernels.py::_union_terms/_update; projector_training.py; nemo_bridge.py`.
- Nguyên nhân: Selective reconstruction must precede mutation; target/draft/A cadence must be read from actual source rather than the accumulation label.
- Đường hiện tại: `kernels.py::feedback; vllm_runtime.py::Runtime.finish; metrics.py; existing ProjectorTrainer/optimizer boundary hooks retained`.
- Sửa/giữ đúng: Compute A numerator from B_t before sparse B updates; accumulate FP32/global valid weight; A stays persistent and trainable, B shared within one cohort and reset only between cohorts. Preserve pending feedback on unsuccessful joint steps and checkpoint optimizer/scheduler/RNG/version. One A update per successful native joint boundary, not one update per four cohorts.
- Kiểm chứng: `test_projector_training.py`, `test_optimizer_cadence.py`, `test_cohort_lifecycle.py`, `test_resume.py`, `test_distributed.py`.

### C07: KL=0, clip=.2 and NeMo score-factor gradients did not match the intended Spec target loss.

- Nguồn: `scripts/prepare_run.py::build_grpo_config; nemo_rl/algorithms/loss/loss_functions.py::ClippedPGLossFn`.
- Nguyên nhân: Equal coefficients alone do not match sequence reduction, on-policy ratio, KL clamping or the score-factor gradient.
- Đường hiện tại: `prepare_run.py; dflash_reflex/grpo_loss.py; patches/nemo-rl.patch`.
- Sửa/giữ đúng: Default BETA=.04/EPSILON=.1 reach the actual loss. specnaacl_sequence uses sequence mean, detached-current old logp for the one on-policy pass, unclamped k3 and direct Spec KL gradient. original_nemo retains the old four-key loss contract for reproducibility/resume.
- Kiểm chứng: `test_optimizer_cadence.py (extracted actual Spec and patched actual NeMo loss value/gradient)`, `test_config_fairness.py`.

### C08: Full-target training and native physical microbatching could be mislabeled SpecNaacl matched.

- Nguồn: `prepare_run.py/configs/_shared/b200_common.env`.
- Nguyên nhân: Pinned Bridge LoRA shares input factors for fused QKV/gate-up; native joint optimizer does not express the requested independent four-cohort target accumulation. Filtering/masks also differ.
- Đường hiện tại: `fairness.py; prepare_run.py; scripts/compare_specnaacl_config.py`.
- Sửa/giữ đúng: Strict specnaacl_matched preflight fails before GPU/model access, including dry run. native_dflash is explicit and valid as a native profile. Reject unsupported LoRA overrides, legacy token budgets and multi-epoch GRPO knobs rather than silently accepting them. Produce a stdlib-only matrix from actual shell/source/resolved config.
- Kiểm chứng: `test_config_fairness.py`, `test_launchers.py`, `test_tools.py`.

### C09: Production baseline paid for Reflex instrumentation and could expose fabricated AAL zeros.

- Nguồn: `patches/vllm.patch; vllm_runtime.py`.
- Nguyên nhân: Unconditional Python wrappers, acceptance-history and GPU counter allocation for METHOD=dflash.
- Đường hiện tại: `scripts/patch_upstream.py; patches/vllm.patch; vllm_runtime.py; metrics.py`.
- Sửa/giữ đúng: Import process flags once. Production baseline directly follows original forward/proposer/sampler path without optional wrapper, GPU acceptance buffer or OPD state. instrumented mode gives both methods identical core sampler metrics. Missing production metrics stay null, including cumulative metrics after resume.
- Kiểm chứng: `test_sampler_correctness.py`, `test_sampling.py`, `test_logging.py::test_production_missing_metrics_remain_null_on_resume`, `test_patches.py`.

### C10: Whitespace handling and the vLLM seed could diverge from the source data/reward contract.

- Nguồn: `data.py::normalize_record; pinned vllm_worker.py engine seed`.
- Nguyên nhân: Unconditional answer strip and node/bundle seed do not match SpecNaacl extraction/experiment seed.
- Đường hiện tại: `data.py; patches/nemo-rl.patch::vllm_worker.py; tests/test_reward_parity.py`.
- Sửa/giữ đúng: Preserve original whitespace unless extracting after ####; verify question/gold/subset/template and actual math-verify reward on shared synthetic records/completions for all three datasets. Engine seed uses TRAIN_SUBSET_SEED. Retain and disclose native missing-gold/overlength/group-normalization differences.
- Kiểm chứng: `test_reward_parity.py`, `test_training.py`, `test_patches.py`.

### C11: Diagnostics could produce Inf on underflow and incomplete wall/metric totals on resume.

- Nguồn: `feedback.py::topk_union_tail; kernels.py::_union_terms; metrics.py; scripts/summarize_run.py`.
- Nguyên nhân: log(0) diagnostic and last-iteration wall time omit final checkpoint drain; absent counters do not mean zero.
- Đường hiện tại: `feedback.py; kernels.py::_union_terms; metrics.py; nemo_bridge.py; scripts/summarize_run.py/plot_aal.py`.
- Sửa/giữ đúng: Floor only q in reported KL logs at FP32 tiny, leaving operational q-p untouched. Keep final complete wall time, actual cohort/response/successful-step/A/B counters, reward/losses and checkpoint continuity. Fused or unmeasured profile components are null rather than invented timings.
- Kiểm chứng: `test_feedback_parity.py::test_zero_q_zero_tail_finite_diagnostic_and_sparse_gradient`, `test_logging.py`, `test_resume.py`.

### C12: Qwen3-4B tied heads and inference-mode tensors needed runtime-safe handling.

- Nguồn: `prepare_run.py::load_head_projector; vllm_runtime.py::Runtime.propose; adapter.py/workspaces`.
- Nguyên nhân: Tied model checkpoint may not contain lm_head.weight, and inference tensors lack version counters.
- Đường hiện tại: `prepare_run.py (model.embed_tokens.weight fallback); adapter.py/kernels.py (normal persistent tensors); vllm_runtime.py::tensor_version`.
- Sửa/giữ đúng: Initialize A from the actual tied head when configured; allocate persistent versioned A/scratch outside inference mode; retain raw logits so head recomputation is unnecessary. Enforce refit at existing cohort boundaries.
- Kiểm chứng: `test_runtime_alignment.py::test_actual_inference_mode_proposal_has_versioned_projector`, `test_dflash.py`, `test_resume.py`.

Sai khác có chủ đích: KL chẩn đoán floor q tại FP32 tiny để tránh Inf khi underflow. Đây là số đo KL đã chặn ở mức máy, không phải KL toán học vô hạn; selected-coordinate gradient `w*(q-p)` và update không bị floor. GEMM thay thứ tự reduction nên không tự chọn cho proposal cần exact near-tie parity.

### C13: Explicit dense GEMM feedback could reconstruct a different correction than the proposal used.

- Nguồn: `kernels.py::propose/feedback/_target_q`.
- Nguyên nhân: GEMM and serial Triton sums need not round identically; computing teacher q with a second reduction violates the exact proposal-distribution contract.
- Đường hiện tại: `kernels.py::ProposalWorkspace/propose/feedback/_union_terms`.
- Sửa: Store a view of the already allocated GEMM-corrected proposal buffer. Feedback gathers those exact logits with CORRECTED_RAW=True and never applies B a second time. A terms still read B_t. Automatic GEMM selection stays disabled.
- Kiểm chứng: `test_feedback_parity.py::test_explicit_gemm_feedback_uses_the_exact_corrected_proposal_logits`.

Cache có thêm suffix dtype khi logits khác hidden; bucket BF16 không được tái dùng cho logits FP32. Lệnh benchmark có `--logits-dtype float32` cho trường hợp đó.

### C14: RESUME=auto could select a baseline or another dataset after running a comparison pair.

- Nguồn: `scripts/launch/common.sh::resolve_run/prepare_run`.
- Nguyên nhân: A single per-model active_run symlink was shared across METHOD and DATASET.
- Đường hiện tại: `scripts/launch/common.sh::resolve_run/prepare_run`.
- Sửa: Maintain method/dataset-specific active links, prefer the matching link for auto resume and retain the previous legacy active_run fallback. Existing strict configuration validation still rejects an incompatible legacy checkpoint; nothing is silently reset.
- Kiểm chứng: `test_config_fairness.py::test_auto_resume_selects_the_requested_method_and_dataset`, `test_launchers.py::test_resume_rejects_changed_method_or_reflex_parameters`.

### C15: Training wall time excluded Ray/model setup, and null production counters could prevent carrying completed-session drain time into resume.

- Nguồn: `train.py::main; metrics.py::Metrics.__init__/finalize`.
- Nguyên nhân: Metrics clock began after setup; checkpoint/summary identity comparison required rounds even when production intentionally records rounds=null.
- Đường hiện tại: `train.py::main; nemo_bridge.py::configure_driver; metrics.py::Metrics.__init__`.
- Sửa: Start the session clock at train.main, pass it across setup, include final checkpoint/worker shutdown, exclude downtime between resumes. Compare only available identity counters for final-summary drain recovery. Preserve historical checkpoint wall totals without inventing startup time missing from old runs.
- Kiểm chứng: `test_logging.py::test_production_resume_wall_includes_startup_and_completed_session_drain`, `test_logging.py::test_final_wall_time_includes_checkpoint_drain_and_survives_resume`.

## B. Đo hiệu năng thực tế

GPU đo: **NVIDIA GeForce RTX 3090, SM 8.6; Torch 2.4.1, CUDA 12.1, Triton 3.0**. Không phải B200. Kích thước lấy từ [config Qwen3-4B chính thức](https://huggingface.co/Qwen/Qwen3-4B/raw/main/config.json), lưu tại [snapshot local](../third_party/reference_qwen3_4b_config.json): H=2560, V=151936, rank=8, K=16, block=16 (15 draft slots). Head/hidden/raw BF16, A/B FP32, teacher p FP32 tổng hợp. Các candidate nhận cùng A/B/hidden/raw/p/weights trước mỗi lần đo. Warmup 3, đo 15 lần; CUDA events loại compile/reset khỏi interval. p95 là percentile thực nghiệm của 15 mẫu, không phải khoảng tin cậy.

- Original: nguồn kernel trước sửa đã đóng băng tại `tests/reference/`.
- A/fused: proposal tính sẵn top-k + normalization; feedback dùng raw thật và union hợp nhất.
- B/selective: argmax trước, thống kê q trên hàng feedback hợp lệ; teacher torch.topk + canonical tie.
- B/masked: như B, thêm teacher scan chỉ hàng có weight dương.
- Bỏ head recomputation ở teacher coordinates; dùng lại U và B đã load trong union/A term. Không nhân bản toàn B, không backward qua backbone trong rollout.

Bảng dưới là **median tổng proposal + feedback + A-gradient + B-update**, đơn vị ms. Prefix=0 là reject đầu; prefix=15 là chấp nhận toàn bộ draft block. Đây không phải thời gian transformer, verification hay GRPO.

| Request | Prefix | Original | A fused | B selective | B masked | Best đo/original |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0 | 0.461 | 0.470 | 0.470 | 0.432 | 1.07× |
| 1 | 2 | 0.456 | 0.464 | 0.486 | 0.440 | 1.03× |
| 1 | 7 | 0.456 | 0.475 | 0.473 | 0.461 | 0.99× |
| 1 | 15 | 0.447 | 0.469 | 0.529 | 0.541 | 0.95× |
| 8 | 0 | 2.220 | 2.216 | 0.996 | 0.531 | 4.18× |
| 8 | 2 | 2.220 | 2.203 | 1.189 | 0.905 | 2.45× |
| 8 | 7 | 2.211 | 2.202 | 1.700 | 1.896 | 1.30× |
| 8 | 15 | 2.217 | 2.203 | 2.391 | 3.235 | 1.01× |
| 32 | 0 | 8.207 | 8.272 | 3.348 | 1.799 | 4.56× |
| 32 | 2 | 8.220 | 8.238 | 4.158 | 3.317 | 2.48× |
| 32 | 7 | 8.281 | 8.237 | 6.200 | 7.303 | 1.34× |
| 32 | 15 | 8.374 | 8.295 | 9.101 | 12.826 | 1.01× |
| 64 | 0 | 16.433 | 16.468 | 6.558 | 3.402 | 4.83× |
| 64 | 2 | 16.459 | 16.473 | 8.240 | 6.571 | 2.50× |
| 64 | 7 | 16.510 | 16.475 | 12.326 | 14.444 | 1.34× |
| 64 | 15 | 16.693 | 16.490 | 18.097 | 25.690 | 1.01× |

Với 64 request/reject đầu, original 16.433 ms → B masked 3.402 ms, khoảng 4.83× riêng hot path. Khi full acceptance, A fused 16.490 ms tốt hơn B selective 18.097 ms và B masked 25.690 ms. Với một request, A vẫn có thể chậm hơn original vài phần trăm do canonicalization/guard/bookkeeping cần cho correctness; không tuyên bố thắng mọi shape. Không dùng original feedback sai raw để lấy lại số đo đẹp.

Auto chỉ dùng lựa chọn đã đo theo GPU/shape và feedback density của **cohort hoàn tất trước đó**, cộng ngưỡng lợi ích 5% để tránh lựa chọn do nhiễu. Không biết trước acceptance của round hiện tại. Khi chưa có density, dùng aggregate winner trên bốn outcome đã đo; không có cache thì A fused. B200 không dùng cache 3090. Nếu phân bố acceptance đổi giữa cohort, policy có thể chưa chọn candidate nhanh nhất ngay; không gắn tốc độ best-per-outcome trong bảng với tốc độ auto thực tế.

### Sparse/dense và atomics

Dispatch đọc unique active_count trên GPU, không đọc scalar về CPU mỗi round. Các threshold đo trên 3090 đều là V+1: chưa có thắng lợi dense fused ≥5% đơn điệu ở các density 2%, 20%, 100%. Dense/GEMM vẫn có thể dùng explicit để tái lập benchmark; GEMM không tự được chọn vì near-tie reduction khác. Chỉ một atomic counter cho loại dispatch mỗi proposal, materialize ở cuối cohort.

| 64 request, density | Sparse fused | Dense fused | Dense GEMM | GPU auto |
|---:|---:|---:|---:|---:|
| 2% | 12.346 | 14.241 | 14.070 | 12.483 |
| 20% | 12.710 | 14.241 | 14.174 | 12.671 |
| 100% | 14.358 | 14.239 | 14.259 | 13.976 |

Repeated-ID stress (64 request, 64 unique IDs): atomic update 0.0377 ms so với sort/unique/index_add 0.3079 ms. Giữ sparse atomics; không thêm sorting vào production chỉ vì có repeated IDs.

Peak allocated toàn harness 1952.0 MiB, bao gồm head/teacher/shared inputs và workspace của mọi candidate; không dùng số này để khẳng định candidate nào tiết kiệm RAM. Retain raw logits tối đa 278.2 MiB cho 64 request BF16, hoặc gấp đôi nếu raw FP32; đây là tăng lifetime để giữ q chính xác, không phải copy B. Idle CUDA synchronize đo 21.7 µs; không có scalar sync bổ sung mỗi verification round.

JSON gốc chứa median/p95, proposal riêng, feedback riêng, overhead so với argmax, rows/s, memory và dispatch cho tất cả shape: [benchmark_before_after.json](benchmark_before_after.json). Không suy ra AAL, tokens/s rollout hoặc wall time GRPO từ các số này.

## C. Fairness so với SpecNaacl

Ma trận [fairness_config_matrix.json](fairness_config_matrix.json) đọc shell config thực, AST LoRA trong SpecNaacl và config resolver DFlash; chạy được bằng `python3 -S`, không import Torch/load weights/khởi tạo CUDA. Status thuộc MATCH, EQUIVALENT, DIFFERENT_BY_DESIGN, UNSUPPORTED, UNVERIFIED. SpecNaacl không có model env qwen3_8b; tool ghi các giá trị model/batch tương ứng UNVERIFIED, không tự điền số liệu hoặc traceback.

| Thuộc tính | SpecNaacl thực tế | DFlash native hiện tại | Kết luận |
|---|---|---|---|
| Model/path, seed, sampling | Qwen3-4B, 42, T=1, p=.95, 2048 response | Cùng giá trị cấu hình; seed đi vào engine | MATCH cấu hình, file weights/tokenizer local chưa xác minh |
| Prompt / response | 8 / 8 | 8 / 8 | MATCH |
| Target effective batch | 8 × số nhóm reward không hằng còn lại mỗi rollout | 64 response/cohort; incomplete/valid count ghi runtime | DIFFERENT_BY_DESIGN; không phải 256 |
| Target optimizer cadence | Sau mỗi rollout còn batch; source lines 1535/1699/2041 | Một joint target+draft step mỗi cohort thành công | Gần nhau về cadence thường; không cùng filtering/denominator |
| Draft / A cadence | Mỗi rollout trainable | Mỗi joint optimizer boundary thành công; pending A giữ khi skip | EQUIVALENT thường, khác khi skip/overflow; không giảm A bốn lần |
| KL / clipping | beta .04, epsilon .1; sequence mean k3 | Scalar đã map loss thật; direct k3 gradient đối chiếu source | MATCH trên tensor/mask/ref giống nhau, chưa match toàn pipeline |
| LoRA target | 7 adapter PEFT độc lập, r64 alpha32 dropout0 | Full training; Bridge fused factors không bị giả là tương đương | UNSUPPORTED matched |
| Reward / prompt parser | math_verify accuracy + .2 format; prompt_dict/math | Đối chiếu hàm thực trên completion/record tổng hợp 3 dataset | MATCH hàm; dataset/tokenizer thực UNVERIFIED |
| Reward normalization / mask | numpy population std không epsilon; bỏ nhóm target có std=0; retokenize assistant | Population std +1e-6; giữ nhóm hằng; mask token sinh | DIFFERENT_BY_DESIGN |
| A/B Reflex | trainable QR A; cohort-shared temporary B | Giữ nguyên contract và union-only update | MATCH invariants kiểm thử |
| Optimizer weight decay | torch AdamW mặc định .01 | native target/draft/A .0; A có override thực | DIFFERENT_BY_DESIGN được công khai |
| Draft loss / token budgets | Hidden/logit regression; MAX_TRAINING_TOKEN/padding/chunk | NVIDIA block loss + anchors/chunks | DIFFERENT_BY_DESIGN; reject legacy knobs không tương đương |
| Total generated tokens / checkpoint init | Đọc log/checkpoint thật | Counters thật và strict checkpoint contract | Chưa có full run để xác minh |

Prompt mô tả target accumulation bốn cohort (nominal 256 response), nhưng source local không thực hiện readiness đó: `all_ranks_ready = len(batch_data['messages']) > 0`, target `.step()`, rồi clear batch. Không sửa SpecNaacl để đổi điều này. Tăng NeMo global_batch lên 256 không giải quyết yêu cầu draft/A độc lập và scope B. Do đó matched profile bị chặn, không che bằng dry run thành công.

LoRA blocker cụ thể: `nemo_rl/models/megatron/setup.py` dùng Bridge LoRA trên `linear_qkv`/`linear_fc1`; các input factors được share cho Q/K/V và gate/up, khác bảy PEFT adapters độc lập. Chưa tích hợp/kiểm chứng canonical injection, optimizer groups, merged vLLM refit và checkpoint round-trip. Nguồn [Bridge LoRA tại gitlink đã pin](https://raw.githubusercontent.com/NVIDIA-NeMo/Megatron-Bridge/b11414c71b15e54d333eb49346ed199f20fa9021/src/megatron/bridge/peft/lora.py). Full fine-tune không được thay tên thành LoRA.

## D. Cấu hình đề nghị duy nhất cho B200 hiện tại

Chọn **Qwen3-4B + SimpleLR + native_dflash + dflash_reflex**, loss sequence đã sửa, A trainable, B shared, r8/K16/fastLR .01, feedback mỗi verification, pipeline auto có calibration tại đúng B200. Đây là cấu hình native để đo tác dụng Reflex trong cùng framework; chưa đủ điều kiện so sánh strict với SpecNaacl. Giữ A weight decay .0 của native hiện tại để tránh đổi contract checkpoint; khác Spec .01 đã ghi ma trận, không coi là match.

Chuẩn bị môi trường theo [ENVIRONMENT.md](../ENVIRONMENT.md), patch lại wheel vLLM bằng script nếu environment đã cài trước. Pretrain vẫn dùng ShareGPT, không chạy tự động trong task này. Các lệnh dưới đây yêu cầu local models/data/runtime production thật.

```bash
cd dflash
bash scripts/setup_environment.sh
.venv/bin/python scripts/validate_environment.py --require-runtime
bash pretrain_qwen3_4b.sh

# Calibration ngắn, không load model weights hay chạy GRPO.
mkdir -p outputs/benchmarks
PYTHONPATH="$PWD" .venv-vllm/bin/python scripts/benchmark_reflex_hotpath.py \
  --model-config /workspace/storage-shared/models/Qwen3-4B/config.json \
  --requests 1 8 32 64 --warmup 3 --repeats 15 \
  --cache outputs/benchmarks/b200_dispatch.json \
  --output outputs/benchmarks/b200_hotpath.json

export FAIRNESS_PROFILE=native_dflash METRICS_MODE=production
export GRPO_OBJECTIVE=specnaacl_sequence BETA=.04 EPSILON=.1
export BATCH_SIZE=8 RESPONSES_PER_PROMPT=8 ACCUMULATION_STEPS=4
export OPD_TRAIN_PROJECTOR=1 OPD_RANK=8 OPD_TOPK=16 OPD_FAST_LR=.01
export OPD_PROJECTOR_LR=1e-4 OPD_PROJECTOR_WEIGHT_DECAY=0.0
export OPD_UPDATE_INTERVAL=1 OPD_UPDATE_STREAM=1 OPD_PIPELINE=auto
export OPD_PROPOSAL_MODE=auto OPD_DENSE_IMPLEMENTATION=auto
export OPD_PROFILE=0 OPD_DIAGNOSTICS=0
export OPD_DISPATCH_CACHE="$PWD/outputs/benchmarks/b200_dispatch.json"

# Native pair, cùng checkpoint pretrain/target và config.
DATASET=simplelr METHOD=dflash_reflex bash train_qwen3_4b.sh
DATASET=simplelr METHOD=dflash bash train_qwen3_4b_dflash.sh

# Dataset và resume vẫn được giữ.
DATASET=gsm8k METHOD=dflash_reflex bash train_qwen3_4b.sh
DATASET=dapo METHOD=dflash_reflex bash train_qwen3_4b.sh
DATASET=simplelr METHOD=dflash_reflex RESUME=auto bash train_qwen3_4b.sh
```

Để kiểm tra thật trên B200 trước khi dành ngân sách training: dùng cùng cấu hình trên với `MAX_GRPO_STEPS=2 MAX_TRAIN_SAMPLES=16 SAVE_INTERVAL=1`, cho cả baseline và Reflex. Đây là smoke cần thực hiện trên host có runtime/model, **chưa được chạy tại đây**. Tạo run riêng cho kiểm tra AAL với `METRICS_MODE=instrumented` ở cả hai method; lần profiling riêng bật `OPD_PROFILE=1`. Không trộn latency profile/instrumented với production.

Native recipe cũ/resume checkpoint cũ: `GRPO_OBJECTIVE=original_nemo BETA=0 EPSILON=.2`, giữ các batch/draft/init overrides và projector optimizer settings đúng checkpoint. Resume kiểm tra full config, sidecars, optimizer/scheduler/RNG và model contract; mismatch báo lỗi, không xóa/reset checkpoint. Auto resume ưu tiên `active_run_<METHOD>_<DATASET>` để không chọn nhầm run sau baseline/dataset khác; vẫn đọc link legacy khi chưa có link riêng. Mode/pipeline/profiling quan sát được có thể đổi khi resume; nếu một phần production không có core counters thì cumulative AAL vẫn null.

## E. Lệnh matched chính xác và kết quả mong đợi

Các lệnh sau hiện **phải thất bại preflight**, không phải lệnh train đã được hỗ trợ. Default matched không có fallback âm thầm sang native.

```bash
python3 -S scripts/compare_specnaacl_config.py \
  --model qwen3_4b --dataset simplelr --profile specnaacl_matched
# Exit 2, ghi ma trận với blockers.
DATASET=simplelr FAIRNESS_PROFILE=specnaacl_matched \
  METHOD=dflash_reflex bash train_qwen3_4b.sh
DATASET=simplelr FAIRNESS_PROFILE=specnaacl_matched \
  METHOD=dflash bash train_qwen3_4b_dflash.sh
```

## F. Kiểm chứng và giới hạn còn lại

83 passed, 0 failed, 0 skipped. Ba warning thuộc protobuf/SciPy trong development environment; không giả production dependencies đã được xác minh. [JUnit](final_tests.xml), [static checks](final_static_checks.json), [published patches](final_published_patch_check.json).

- `compileall`: qua; `ruff check`/`ruff format --check`: qua; `bash -n`: 13 script qua (gồm 9 wrapper gốc).
- Pinned source SHAs qua; published patches apply đúng pristine source, patcher chạy hai lần không đổi và migration hook vLLM cũ được kiểm thử.
- Exact proposal ID parity large vocabulary BF16/FP32/B0/sparse/dense/near ties. q normalization rtol 3e-5, atol 1e-7. FP64 oracle cho B rtol 2e-5/atol 3e-9, A rtol 2e-5/atol 5e-9; không có mismatch phải nới tolerance tùy tiện.
- Chạy actual tiny original DFlash forward/backward/save, source-extracted NVIDIA BlockDraftLossFn, actual patched NeMo loss class, actual pinned rejection sampler kernels (greedy exact và stochastic distribution với 40k mẫu), runtime reorder/stream/cohort/projector update/resume primitives. Đây là tests có phạm vi nhỏ, không phải full-engine/full-model production integration.
- A/B diagnostics, target/draft/A successful steps, actual prompts/responses/cohorts, reward/loss và wall time được ghi ở các boundary. Wall time mới bắt đầu tại `train.main`, gồm Ray/model setup và shutdown; không gồm pretrain riêng, launcher preflight hoặc downtime giữa session. Không tự dựng lại startup time đã thiếu trong log checkpoint cũ. B norm chỉ đo khi diagnostics bật. Profiling có dflash_forward/head/proposal/extraction/topk/union/A/B/target phases; normalization nằm trong fused statistics nên thời gian riêng null. `weight_sync_ms` bao gồm prepare/sleep/wake/refit; refit-only null. Không cộng các nested phases để phóng đại tổng OPD time.
- Không có B200, CUDA13 actor environments, Ray hoặc local target/draft/dataset yêu cầu. [Preflight production thất bại thật](production_preflight_failure.txt). Chưa xác minh pretrain đầy đủ, live vLLM weights/refit, full GRPO optimizer step, checkpoint distributed thật, AAL improvement hay rollout/training wall speedup.
- Cần chạy smoke ngắn ở host B200 trước experiment; không gọi bản này đã production-qualified. Fallback tied-head đã kiểm tra code, chưa load checkpoint target thật. Không có critical placeholder được báo thành công: các feature matched chưa có bị reject rõ.

Báo cáo này thay thế các số đo/validation lịch sử trong `validation/report.json`, `pytest.xml`, `final_pytest.xml` và các đoạn báo cáo cũ. Nguồn số liệu hiện tại: [JSON đầy đủ](final_reflex_optimization_report.json), [matrix](fairness_config_matrix.json), [benchmark](benchmark_before_after.json).
