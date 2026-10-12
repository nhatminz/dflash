"""Compare exact retained-row indexing with an extra valid-row raw-logit copy."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import torch

    from dflash_reflex import kernels
    from dflash_reflex.adapter import Adapter, initialize_projector
    from dflash_reflex.config import ReflexConfig

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(884)
    config = json.loads((ROOT / "third_party/reference_qwen3_4b_config.json").read_text())
    d, v, n = config["hidden_size"], config["vocab_size"], 64 * 15
    head = torch.randn(v, d, device="cuda", dtype=torch.bfloat16) * 0.02
    adapter = Adapter(head, initialize_projector(head), ReflexConfig(update_stream=False))
    raw = torch.randn(n, v, device="cuda")
    u = torch.randn(n, 8, device="cuda")
    results = []
    for prefix in (0, 2, 15):
        rows = torch.tensor(
            [i * 15 + j for i in range(64) for j in range(min(prefix + 1, 15))], device="cuda"
        )
        selected_u = u.index_select(0, rows)
        weights = torch.ones(len(rows), device="cuda")
        workspace = kernels.ProposalWorkspace(len(rows), v, 8, 16, "cuda")
        # Reusable buffer gives the copy alternative the best possible case.
        copied = torch.empty(len(rows), v, device="cuda")

        def retained():
            kernels.statistics(adapter, selected_u, raw, workspace, weights, rows)

        def compact_copy():
            torch.index_select(raw, 0, rows, out=copied)
            kernels.statistics(adapter, selected_u, copied, workspace, weights)

        retained()
        reference_ids, reference_q = workspace.ids.clone(), workspace.q.clone()
        compact_copy()
        torch.testing.assert_close(workspace.ids, reference_ids, rtol=0, atol=0)
        torch.testing.assert_close(workspace.q, reference_q, rtol=0, atol=0)
        measurements = {}
        for name, function in (
            ("retained_row_map", retained),
            ("valid_copy_then_statistics", compact_copy),
        ):
            for _ in range(3):
                function()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(20):
                function()
            end.record()
            end.synchronize()
            measurements[name + "_ms"] = start.elapsed_time(end) / 20
        results.append(
            dict(
                prefix=prefix,
                valid_rows=len(rows),
                exact_parity=True,
                extra_copy_bytes=copied.numel() * copied.element_size(),
                **measurements,
            )
        )
    report = dict(
        gpu=torch.cuda.get_device_name(),
        requests=64,
        hidden=d,
        vocab=v,
        raw_bytes=raw.numel() * raw.element_size(),
        results=results,
        scope="Synthetic exact valid-row copy vs retained indexing. Oracle row IDs supplied; compaction/discovery excluded. No head recomputation or real rollout.",
        decision="retain original logits; valid-row copy adds work and cannot release originals before verification; head recomputation is not qualified for exact bias/processing/dtype parity",
    )
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
