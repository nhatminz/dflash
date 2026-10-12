"""Iteration ratios are derived from sums, including across checkpoint resume."""

import csv
import json
import math
import os
import time
from pathlib import Path

FIELDS = (
    "iteration",
    "epoch",
    "global_step",
    "dataset",
    "method",
    "iter_rollout_tokens",
    "iter_generated_tokens",
    "iter_verification_rounds",
    "iter_accepted_tokens",
    "iter_accepted_draft_tokens",
    "iter_proposed_draft_tokens",
    "iter_aal",
    "cumulative_aal",
    "iter_acceptance_rate",
    "cumulative_acceptance_rate",
    "iter_generation_time_s",
    "iter_generation_tokens_per_s",
    "iter_wall_time_s",
    "iter_end_to_end_tokens_per_s",
    "iter_draft_train_time_s",
    "iter_target_train_time_s",
    "iter_reward_time_s",
    "cumulative_generation_time_s",
    "cumulative_wall_time_s",
    "iter_opd_feedback_rows",
    "iter_opd_updates",
    "iter_opd_kl",
    "iter_opd_tv",
    "iter_opd_active_rows",
    "iter_opd_B_norm",
    "iter_opd_A_grad_norm",
)
FIELDS += (
    "A_optimizer_steps",
    "A_version",
    "A_gradient_rows",
    "A_gradient_weight",
    "A_gradient_norm",
    "A_parameter_norm",
    "B_update_rounds",
    "B_valid_feedback_rows",
    "B_active_rows",
    "B_parameter_norm",
    "target_optimizer_steps",
    "draft_optimizer_steps",
    "opd_feedback_rows",
    "opd_update_rounds",
    "opd_active_rows",
    "opd_A_grad_norm",
    "opd_B_norm",
    "total_training_wall_time_s",
    "iter_target_loss",
    "iter_draft_loss",
    "iter_reward",
    "FAIRNESS_PROFILE",
    "METRICS_MODE",
    "iter_rollout_cohorts",
    "cumulative_rollout_cohorts",
    "iter_rollout_responses",
    "iter_rollout_prompts",
    "cumulative_rollout_responses",
    "iter_target_response_count",
    "step_aal",
    "step_acceptance_rate",
    "opd_A_norm",
    "opd_sparse_dispatch_count",
    "opd_dense_dispatch_count",
    "opd_update_time_s",
    "opd_correction_time_s",
)


def ratio(n, d):
    return n / d if d else 0.0


def json_value(value):
    if isinstance(value, dict):
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    if hasattr(value, "tolist"):
        return json_value(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def truncate_jsonl(path, key, completed):
    """Discard uncommitted rows, tolerating a crash during the final write."""
    path = Path(path)
    if not path.is_file():
        return
    lines = path.read_text().splitlines()
    retained = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if index != len(lines) - 1:
                raise
            break
        if row[key] <= completed:
            retained.append(json.dumps(row) + "\n")
    path.write_text("".join(retained))


class Metrics:
    def __init__(self, run_dir, method, dataset, state=None, *, wall_start=None):
        self.run_dir = Path(run_dir)
        self.method, self.dataset = method, dataset
        self.state = dict(state or {})
        summary_path = self.run_dir / "summary.json"
        if state is not None and summary_path.is_file():
            summary = json.loads(summary_path.read_text())
            # A normally ended session may have measured final checkpoint
            # draining AFTER the checkpoint sidecar was committed.
            keys = ("iteration", "generated", "optimizer_steps")
            if self.state.get("core_metrics_available", True):
                keys += ("rounds",)
            if all(summary.get(k) == self.state.get(k) for k in keys):
                self.state["wall"] = max(self.state.get("wall", 0), summary.get("wall", 0))
        self.start = time.perf_counter() if wall_start is None else wall_start
        self.previous_wall = self.state.get("wall", 0.0)
        metadata = self.run_dir / "run_metadata.json"
        self.pretrain_time = (
            json.loads(metadata.read_text()).get("pretrain_time_s") if metadata.is_file() else None
        )
        self.flush_interval = int(os.environ.get("ROLLOUT_LOG_FLUSH_INTERVAL", "1"))
        if self.flush_interval < 1:
            raise ValueError("ROLLOUT_LOG_FLUSH_INTERVAL must be positive")
        logs = self.run_dir / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "statistics").mkdir(parents=True, exist_ok=True)
        self.path = logs / "rollout_timing.csv"
        # Crash may have written rows AFTER the last committed checkpoint.
        if state is not None:
            self.truncate_after(self.state.get("iteration", 0))

    def truncate_after(self, iteration):
        if self.path.exists():
            with self.path.open() as f:
                rows = [r for r in csv.DictReader(f) if int(r["iteration"]) <= iteration]
            with self.path.open("w", newline="") as f:
                writer = csv.DictWriter(f, FIELDS)
                writer.writeheader()
                writer.writerows(rows)
        for name in ("metrics.jsonl", "timing.csv", "train_steps.csv"):
            path = self.path.parent / name
            if path.exists():
                if name.endswith("jsonl"):
                    truncate_jsonl(path, "iteration", iteration)
                else:
                    with path.open() as f:
                        reader = csv.DictReader(f)
                        fields, rows = reader.fieldnames, list(reader)
                    with path.open("w", newline="") as f:
                        writer = csv.DictWriter(f, fields)
                        writer.writeheader()
                        writer.writerows(r for r in rows if int(r["iteration"]) <= iteration)
        for path in (self.run_dir / "statistics").glob("iteration_*.json"):
            if int(path.stem.split("_")[-1]) > iteration:
                path.unlink()

    def record(self, epoch, step, packets, timing, *, wall, train_metrics=None):
        sums = [0.0] * 8
        generated = updates = 0
        profile = {}
        for packet in packets:
            for i, value in enumerate(packet["counters"].tolist()):
                sums[i] += value
            generated += packet["generated_tokens"]
            updates += packet["updates"]
            for name, value in packet["profile_ms"].items():
                if value is None:
                    profile.setdefault(name, None)
                else:
                    profile[name] = (profile.get(name) or 0.0) + value
        accepted, rounds, accepted_draft, proposed, weight, kl, tv, feedback_rows = sums
        if profile and "prepare_for_generation/total" in timing:
            profile["weight_sync_ms"] = timing["prepare_for_generation/total"] * 1000
        if profile:
            profile["feedback_compaction_ms"] = profile.get("feedback_extraction_ms")
            # Normalization shares the draft statistics scan. Weight sync also
            # includes sleep/wake and cannot be called a refit-only measurement.
            profile.setdefault("feedback_normalization_ms", None)
            profile.setdefault("weight_refit_ms", None)
        generation = sum(p["generation_time_s"] for p in packets)
        core_available = all(p.get("core_metrics_available", True) for p in packets)
        self.state["core_metrics_available"] = (
            self.state.get("core_metrics_available", True) and core_available
        )
        self.state["rollout_cohorts"] = self.state.get("rollout_cohorts", 0) + len(packets)
        for key, value in dict(
            accepted=accepted,
            rounds=rounds,
            accepted_draft=accepted_draft,
            proposed=proposed,
            generated=generated,
            generation=generation,
        ).items():
            self.state[key] = self.state.get(key, 0) + value
        self.state["iteration"] = self.state.get("iteration", 0) + 1
        self.state["wall"] = self.previous_wall + time.perf_counter() - self.start
        self.state["updates"] = self.state.get("updates", 0) + updates
        self.state["feedback_rows"] = self.state.get("feedback_rows", 0) + feedback_rows
        for key in ("policy_training", "reward_calculation", "checkpointing"):
            self.state[key] = self.state.get(key, 0) + timing.get(key, 0)
        self.state["optimizer_steps"] = (train_metrics or {}).get(
            "total_optimizer_steps", self.state.get("optimizer_steps", 0)
        )
        for key in ("sparse_dispatch_count", "dense_dispatch_count"):
            self.state[key] = self.state.get(key, 0) + sum(p.get(key, 0) for p in packets)
        row = dict(
            iteration=self.state["iteration"],
            epoch=epoch,
            global_step=step,
            dataset=self.dataset,
            method=self.method,
            iter_rollout_tokens=generated,
            iter_generated_tokens=generated,
            iter_verification_rounds=int(rounds),
            iter_accepted_tokens=int(accepted),
            iter_accepted_draft_tokens=int(accepted_draft),
            iter_proposed_draft_tokens=int(proposed),
            iter_aal=ratio(accepted, rounds) if rounds else None,
            cumulative_aal=ratio(self.state["accepted"], self.state["rounds"])
            if self.state["rounds"]
            else None,
            iter_acceptance_rate=ratio(accepted_draft, proposed) if proposed else None,
            cumulative_acceptance_rate=ratio(self.state["accepted_draft"], self.state["proposed"])
            if self.state["proposed"]
            else None,
            iter_generation_time_s=generation,
            iter_generation_tokens_per_s=ratio(generated, generation),
            iter_wall_time_s=wall,
            iter_end_to_end_tokens_per_s=ratio(generated, wall),
            # NVIDIA computes draft and target in ONE forward/backward.
            # There is no honest separately measured phase duration.
            iter_draft_train_time_s=None,
            iter_target_train_time_s=None,
            iter_reward_time_s=timing.get("reward_calculation"),
            cumulative_generation_time_s=self.state["generation"],
            cumulative_wall_time_s=self.state["wall"],
            iter_opd_feedback_rows=int(feedback_rows),
            iter_opd_updates=updates,
            iter_opd_kl=ratio(kl, weight) if weight else None,
            iter_opd_tv=ratio(tv, weight) if weight else None,
            iter_opd_active_rows=max((p.get("active_rows", 0) for p in packets), default=0)
            if any("active_rows" in p for p in packets)
            else None,
            iter_opd_B_norm=max((p.get("B_norm", 0) for p in packets), default=0)
            if any("B_norm" in p for p in packets)
            else None,
            iter_opd_A_grad_norm=(train_metrics or {}).get("opd_A_grad_norm"),
        )
        row.update(
            step_aal=row["iter_aal"],
            step_acceptance_rate=row["iter_acceptance_rate"],
            opd_A_norm=(train_metrics or {}).get("opd_A_norm"),
            opd_sparse_dispatch_count=sum(p.get("sparse_dispatch_count", 0) for p in packets),
            opd_dense_dispatch_count=sum(p.get("dense_dispatch_count", 0) for p in packets),
            opd_update_time_s=(
                profile.get("topk_union_ms", 0)
                + profile.get("A_gradient_ms", 0)
                + profile.get("B_update_ms", 0)
            )
            / 1000
            if profile
            else None,
            # Correction and argmax share a fused scan. Report that whole
            # proposal phase, including U, instead of a fabricated zero for B.
            opd_correction_time_s=profile["corrected_argmax_ms"] / 1000
            if "corrected_argmax_ms" in profile
            else profile["logit_correction_ms"] / 1000
            if "logit_correction_ms" in profile
            else None,
        )
        tm = train_metrics or {}
        response_count = (
            sum(p["rollout_responses"] for p in packets)
            if all("rollout_responses" in p for p in packets)
            else None
        )
        if response_count is not None:
            self.state["rollout_responses"] = (
                self.state.get("rollout_responses", 0) + response_count
            )
        row.update(
            iter_rollout_responses=response_count,
            iter_rollout_prompts=sum(p["rollout_prompts"] for p in packets)
            if all("rollout_prompts" in p for p in packets)
            else None,
            cumulative_rollout_responses=self.state.get("rollout_responses"),
            iter_target_response_count=tm.get("global_valid_seqs"),
            A_optimizer_steps=tm.get("opd_A_version"),
            A_version=tm.get("opd_A_version"),
            A_gradient_rows=int(feedback_rows),
            A_gradient_weight=weight,
            A_gradient_norm=tm.get("opd_A_grad_norm"),
            A_parameter_norm=tm.get("opd_A_norm"),
            B_update_rounds=updates,
            B_valid_feedback_rows=int(feedback_rows),
            B_active_rows=row["iter_opd_active_rows"],
            B_parameter_norm=row["iter_opd_B_norm"],
            target_optimizer_steps=self.state["optimizer_steps"],
            draft_optimizer_steps=self.state["optimizer_steps"],
            opd_feedback_rows=int(feedback_rows),
            opd_update_rounds=updates,
            opd_active_rows=row["iter_opd_active_rows"],
            opd_A_grad_norm=tm.get("opd_A_grad_norm"),
            opd_B_norm=row["iter_opd_B_norm"],
            total_training_wall_time_s=self.state["wall"],
            iter_target_loss=tm.get("loss"),
            iter_draft_loss=tm.get("draft_loss"),
            iter_reward=tm.get("reward"),
            FAIRNESS_PROFILE=os.environ.get("FAIRNESS_PROFILE", "native_dflash"),
            METRICS_MODE=os.environ.get("METRICS_MODE", "production"),
            iter_rollout_cohorts=len(packets),
            cumulative_rollout_cohorts=self.state["rollout_cohorts"],
        )
        if not core_available:
            for name in (
                "iter_verification_rounds",
                "iter_accepted_tokens",
                "iter_accepted_draft_tokens",
                "iter_proposed_draft_tokens",
                "iter_aal",
                "cumulative_aal",
                "iter_acceptance_rate",
                "cumulative_acceptance_rate",
                "step_aal",
                "step_acceptance_rate",
            ):
                row[name] = None
        if not self.state["core_metrics_available"]:
            row["cumulative_aal"] = row["cumulative_acceptance_rate"] = None
        row = json_value(row)
        new = not self.path.exists() or self.path.stat().st_size == 0
        with self.path.open("a", newline="") as f:
            writer = csv.DictWriter(f, FIELDS)
            if new:
                writer.writeheader()
            writer.writerow(row)
        extra = dict(
            row,
            profile_ms=profile,
            upstream_timing=timing,
            upstream_metrics=train_metrics or {},
            rollout_hook_health=[
                dict(
                    group_id=p.get("group_id"),
                    model_runner_module=p.get("model_runner_module"),
                    proposal_calls=p.get("proposal_calls"),
                    verification_calls=p.get("verification_calls"),
                    updates=p.get("updates"),
                )
                for p in packets
            ],
        )
        (self.run_dir / "statistics" / f"iteration_{row['iteration']:08d}.json").write_text(
            json.dumps(json_value(dict(row, profile_ms=profile)), allow_nan=False) + "\n"
        )
        with (self.path.parent / "metrics.jsonl").open("a") as f:
            f.write(json.dumps(json_value(extra), allow_nan=False) + "\n")
        for name in ("timing.csv", "train_steps.csv"):
            path = self.path.parent / name
            new = not path.exists()
            with path.open("a", newline="") as f:
                writer = csv.DictWriter(
                    f, ["iteration", "global_step", "joint_train_time_s", "weight_sync_time_s"]
                )
                if new:
                    writer.writeheader()
                writer.writerow(
                    dict(
                        iteration=row["iteration"],
                        global_step=step,
                        joint_train_time_s=timing.get("policy_training"),
                        weight_sync_time_s=timing.get("prepare_for_generation/total"),
                    )
                )
        summary = dict(
            self.state,
            method=self.method,
            dataset=self.dataset,
            aal=row["cumulative_aal"],
            acceptance_rate=row["cumulative_acceptance_rate"],
            total_training_time_s=self.state["wall"],
            end_to_end_tokens_per_s=ratio(self.state["generated"], self.state["wall"]),
        )
        summary.update(
            total_training_wall_time_s=self.state["wall"],
            cumulative_elapsed_time_s=self.state["wall"],
            total_rollout_time_s=self.state["generation"],
            total_joint_train_time_s=self.state["policy_training"],
            total_target_train_time_s=None,
            total_draft_train_time_s=None,
            total_pretrain_time_s=self.pretrain_time,
            total_generated_tokens=self.state["generated"],
            total_verification_rounds=self.state["rounds"],
            total_accepted_draft_tokens=self.state["accepted_draft"],
            total_proposed_draft_tokens=self.state["proposed"],
            total_target_optimizer_steps=self.state["optimizer_steps"],
            total_draft_optimizer_steps=self.state["optimizer_steps"],
            generation_tokens_per_s=ratio(self.state["generated"], self.state["generation"]),
            average_iteration_time_s=ratio(self.state["wall"], self.state["iteration"]),
            average_grpo_step_time_s=ratio(self.state["wall"], self.state["optimizer_steps"]),
            opd_total_feedback_rows=self.state["feedback_rows"],
            opd_total_updates=self.state["updates"],
            opd_sparse_dispatch_count=self.state["sparse_dispatch_count"],
            opd_dense_dispatch_count=self.state["dense_dispatch_count"],
            opd_A_norm=row["opd_A_norm"],
            opd_A_grad_norm=row["iter_opd_A_grad_norm"],
            opd_B_norm=row["iter_opd_B_norm"],
            opd_active_vocab_rows=row["iter_opd_active_rows"],
        )
        summary.update(
            A_optimizer_steps=row["A_optimizer_steps"],
            A_version=row["A_version"],
            FAIRNESS_PROFILE=row["FAIRNESS_PROFILE"],
            METRICS_MODE=row["METRICS_MODE"],
        )
        if not self.state["core_metrics_available"]:
            for name in (
                "aal",
                "acceptance_rate",
                "total_verification_rounds",
                "total_accepted_draft_tokens",
                "total_proposed_draft_tokens",
                "accepted",
                "rounds",
                "accepted_draft",
                "proposed",
            ):
                summary[name] = None
        (self.run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        # Rows are visible every iteration. The public flush knob controls
        # durable fsync frequency; resume removes uncommitted later rows.
        if row["iteration"] % self.flush_interval == 0:
            for name in ("metrics.jsonl", "rollout_timing.csv", "timing.csv", "train_steps.csv"):
                with (self.path.parent / name).open("rb") as stream:
                    os.fsync(stream.fileno())
        return row

    def finalize(self):
        """Include final checkpoint drain/worker shutdown in actual run time."""
        path = self.run_dir / "summary.json"
        if not path.is_file():
            return
        self.state["wall"] = self.previous_wall + time.perf_counter() - self.start
        summary = json.loads(path.read_text())
        summary.update(self.state)
        summary.update(
            total_training_time_s=self.state["wall"],
            total_training_wall_time_s=self.state["wall"],
            cumulative_elapsed_time_s=self.state["wall"],
            end_to_end_tokens_per_s=ratio(self.state["generated"], self.state["wall"]),
            average_iteration_time_s=ratio(self.state["wall"], self.state["iteration"]),
            average_grpo_step_time_s=ratio(self.state["wall"], self.state["optimizer_steps"]),
        )
        if not self.state.get("core_metrics_available", True):
            for name in ("accepted", "rounds", "accepted_draft", "proposed"):
                summary[name] = None
        path.write_text(json.dumps(json_value(summary), indent=2, allow_nan=False) + "\n")
