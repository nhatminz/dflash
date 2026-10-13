"""Opt-in passive wrappers for the pinned V1 proposer/sampler. No weight updates.

Only compact token IDs are cloned on device in the hot path. A single packed
GPU-to-CPU transfer is made at the END of a rollout. The rejection sampler and
its RNG/kernel launches are never replaced or modified.
"""

import inspect

_hook = None
_installed = False


def verify_contract():
    """Compare actual installed core functions to the authoritative live tree."""
    import ast
    import textwrap

    from vllm.v1.sample import rejection_sampler as sampler
    from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    from .config import ROOT
    from .io import digest, file_hash

    checks = []
    sampler_path = ROOT / "third_party/vllm/vllm/v1/sample/rejection_sampler.py"
    for name in (
        "rejection_sample",
        "rejection_greedy_sample_kernel",
        "rejection_random_sample_kernel",
        "sample_recovered_tokens",
        "sample_recovered_tokens_kernel",
    ):
        fn = getattr(sampler, name)
        checks.append((sampler_path, name, getattr(fn, "fn", fn), None))
    checks += [
        (
            ROOT / "third_party/vllm/vllm/v1/worker/gpu_model_runner.py",
            "_sample",
            GPUModelRunner._sample,
            "GPUModelRunner",
        ),
        (
            ROOT / "third_party/vllm/vllm/v1/spec_decode/llm_base_proposer.py",
            "_greedy_sample",
            SpecDecodeBaseProposer._greedy_sample,
            "SpecDecodeBaseProposer",
        ),
    ]
    signatures = {}
    for path, name, actual, class_name in checks:
        tree = ast.parse(path.read_text())
        nodes = tree.body
        if class_name:
            nodes = next(
                n.body for n in nodes if isinstance(n, ast.ClassDef) and n.name == class_name
            )
        expected = next(
            n
            for n in nodes
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name
        )
        observed = ast.parse(textwrap.dedent(inspect.getsource(actual))).body[0]
        expected.decorator_list = []
        observed.decorator_list = []
        if ast.dump(expected, include_attributes=False) != ast.dump(
            observed, include_attributes=False
        ):
            raise RuntimeError(
                f"Installed vLLM {name} differs from live pinned source {path}; refusing unaudited event semantics"
            )
        signatures[name] = digest(ast.dump(observed, include_attributes=False))
    return dict(
        core_function_sha256=signatures,
        installed_sampler_file_sha256=file_hash(inspect.getfile(sampler)),
    )


class PassiveHook:
    def __init__(self):
        self.enabled = False
        self.packets = []
        self.context = None
        self.sample_logged = False
        self.proposal_calls = 0

    def begin(self):
        self.packets.clear()
        self.context = None
        self.proposal_calls = 0
        self.enabled = True

    def capture(self, output, lengths, draft_ids):
        if self.context is None:
            raise RuntimeError("Verification sampler ran without the V1 request context")
        self.packets.append(
            dict(
                **self.context,
                lengths=list(lengths),
                outputs_gpu=output.detach().clone(),
                draft_gpu=draft_ids.detach().clone() if draft_ids is not None else None,
            )
        )
        self.sample_logged = True

    def drain(self):
        import torch

        self.enabled = False
        tensors = []
        descriptors = []
        for packet in self.packets:
            for key in ("outputs_gpu", "draft_gpu"):
                value = packet[key]
                if value is not None:
                    tensors.append(value.reshape(-1).to(torch.int64))
                    descriptors.append((packet, key, value.shape, value.numel()))
        if tensors:
            # Synchronization is only at the rollout boundary; no .item(),
            # .cpu(), events.synchronize() or logits transfer inside a round.
            packed = torch.cat(tensors).cpu()
            offset = 0
            for packet, key, shape, size in descriptors:
                packet["outputs" if key == "outputs_gpu" else "draft_ids"] = (
                    packed[offset : offset + size].reshape(shape).tolist()
                )
                offset += size
        for packet in self.packets:
            packet.setdefault("draft_ids", [])
            packet.pop("outputs_gpu")
            packet.pop("draft_gpu")
        packets, self.packets = self.packets, []
        return packets


def install():
    global _hook, _installed
    if _installed:
        return _hook
    verify_contract()
    import torch
    from vllm.v1.sample import rejection_sampler as sampler
    from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    original_sample = GPUModelRunner._sample
    original_rejection = sampler.rejection_sample
    original_propose = SpecDecodeBaseProposer._greedy_sample
    signature = inspect.signature(original_rejection)
    required = {"draft_token_ids", "num_draft_tokens", "draft_probs", "synthetic_mode"}
    if not required <= signature.parameters.keys():
        raise RuntimeError("Pinned rejection_sample signature changed; refusing an unaudited hook")
    _hook = PassiveHook()

    def sample(runner, *args, **kwargs):
        if not _hook.enabled:
            return original_sample(runner, *args, **kwargs)
        if runner.use_async_scheduling:
            raise RuntimeError("Passive analysis requires the source run's synchronous scheduler")
        ids = list(runner.input_batch.req_ids)
        _hook.context = dict(
            request_ids=ids,
            discard=runner.discard_request_mask.np[: len(ids)].astype(bool).tolist(),
        )
        _hook.sample_logged = False
        result = original_sample(runner, *args, **kwargs)
        if not _hook.sample_logged:
            _hook.capture(result.sampled_token_ids, [0] * len(ids), None)
        _hook.context = None
        return result

    def rejection(*args, **kwargs):
        if not _hook.enabled:
            return original_rejection(*args, **kwargs)
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        if bound.arguments["synthetic_mode"] or bound.arguments["draft_probs"] is not None:
            raise RuntimeError("Expected actual singleton greedy DFlash rejection sampling")
        result = original_rejection(*args, **kwargs)
        _hook.capture(
            result, bound.arguments["num_draft_tokens"], bound.arguments["draft_token_ids"]
        )
        return result

    def propose(proposer, hidden):
        result = original_propose(proposer, hidden)
        if _hook.enabled:
            if proposer.method != "dflash" or result.dtype not in (torch.int32, torch.int64):
                raise RuntimeError("Unexpected proposer in pure DFlash analysis")
            _hook.proposal_calls += 1
        return result

    GPUModelRunner._sample = sample
    sampler.rejection_sample = rejection
    SpecDecodeBaseProposer._greedy_sample = propose
    _installed = True
    return _hook


def worker_begin(worker):
    from dflash_reflex.vllm_runtime import HOOKS_ENABLED, current

    if HOOKS_ENABLED or current() is not None:
        raise RuntimeError("Frozen pure-DFlash analysis unexpectedly enabled a Reflex runtime")
    return install().begin()


def worker_drain(worker):
    return install().drain()


def worker_synchronize(worker):
    import torch

    torch.cuda.synchronize()


def parameter_audit(worker):
    import hashlib

    import torch

    runner = worker.model_runner
    modules = {"target": runner.model, "draft": runner.drafter.model}
    result = {}
    for label, module in modules.items():
        module.eval()
        module.requires_grad_(False)
        hashes = {}
        for name, value in module.named_parameters():
            h = hashlib.sha256()
            flat = value.detach().reshape(-1)
            for start in range(0, flat.numel(), 4 * 1024 * 1024):
                h.update(
                    flat[start : start + 4 * 1024 * 1024]
                    .contiguous()
                    .view(torch.uint8)
                    .cpu()
                    .numpy()
                    .tobytes()
                )
            hashes[name] = dict(
                shape=list(value.shape), dtype=str(value.dtype), sha256=h.hexdigest()
            )
        result[label] = dict(
            parameters=hashes,
            training=module.training,
            trainable_parameters=sum(p.requires_grad for p in module.parameters()),
        )
    return result
