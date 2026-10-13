"""E2/E3/E6 methodology adapted to DFlash's visited block slots.

Histories become observable only after a global verification batch completes.
Per-response rounds are used ONLY for within-response recurrence distance.
No tree depths, tree budgets or pruned-child labels are invented.
"""

import math
from collections import defaultdict

import numpy as np
import pandas as pd

DISTANCES = ("1", "2–4", "5–16", "17–64", ">64")


def distance_bin(value):
    if value is None:
        return "none"
    if value < 1:
        raise ValueError("Prior feedback must precede the current verification round")
    for maximum, label in ((1, "1"), (4, "2–4"), (16, "5–16"), (64, "17–64")):
        if value <= maximum:
            return label
    return ">64"


def exposure_bin(value):
    return 0 if value == 0 else int(math.log2(value)) + 1


def position_bin(value):
    return (
        "0–15" if value < 16 else ("16–63" if value < 64 else ("64–255" if value < 256 else "256+"))
    )


def active_bin(value):
    return (
        "64+"
        if value >= 64
        else (
            "32–63"
            if value >= 32
            else ("16–31" if value >= 16 else ("8–15" if value >= 8 else "<8"))
        )
    )


def annotate(events):
    histories = defaultdict(dict)
    last_same = {}
    response_events, prompt_events = defaultdict(int), defaultdict(int)
    response_y, prompt_y, all_y = defaultdict(int), defaultdict(int), defaultdict(int)
    seen, result = 0, []
    for boundary, batch in events.groupby("global_verification_batch_id", sort=True):
        pending, exposures = [], []
        for event in batch.to_dict("records"):
            outcome = event["is_miss"]
            if outcome is None or pd.isna(outcome):
                continue
            if event["termination_or_censor_reason"] != "ordinary":
                raise ValueError("Censored event has an ordinary miss label")
            y = int(event["target_selected_token_id"])
            response, prompt = int(event["response_id"]), event["prompt_id"]
            r = int(event["response_verification_round"])
            if r < 0:
                raise ValueError("An ordinary draft event has no verification round")
            history = histories[y]
            same = response in history
            sibling = any(other != response and p == prompt for other, p in history.items())
            different = any(p != prompt for p in history.values())
            source = (
                "same_response"
                if same
                else (
                    "sibling_response" if sibling else ("different_prompt" if different else "none")
                )
            )
            row = dict(
                event,
                y=y,
                is_miss=bool(outcome),
                token_same=same,
                token_group=same or sibling,
                token_sibling=sibling,
                token_different=different,
                token_source=source,
                distance=distance_bin(r - last_same[(response, y)]) if same else "none",
                round_bin=distance_bin(r + 1),
                position_bin=position_bin(event["token_position"]),
                active_bin=active_bin(event["active_response_count"]),
                exposure_bin=exposure_bin(seen),
            )
            opportunities = dict(
                same=response_events[response], group=prompt_events[prompt], batch=seen
            )
            observations = dict(
                same=response_y[(response, y)], group=prompt_y[(prompt, y)], batch=all_y[y]
            )
            for counts in (opportunities, observations):
                counts["sibling"] = counts["group"] - counts["same"]
                counts["different"] = counts["batch"] - counts["group"]
            for scope in opportunities:
                row[scope + "_events_exposure"] = exposure_bin(opportunities[scope])
                row[scope + "_y_exposure"] = exposure_bin(observations[scope])
            result.append(row)
            exposures.append((response, prompt, y))
            if outcome:
                pending.append((response, prompt, y, r))
        # Same-batch positions, including earlier slots of one response, are
        # deliberately unavailable until this entire verification completes.
        for response, prompt, y, r in pending:
            histories[y][response] = prompt
            last_same[(response, y)] = r
        for response, prompt, y in exposures:
            response_events[response] += 1
            prompt_events[prompt] += 1
            response_y[(response, y)] += 1
            prompt_y[(prompt, y)] += 1
            all_y[y] += 1
        seen += len(exposures)
    return pd.DataFrame(result)


class ClusterBootstrap:
    def __init__(self, clusters, samples=2000, seed=2026):
        self.clusters = sorted(set(clusters))
        self.index = {key: i for i, key in enumerate(self.clusters)}
        n = len(self.clusters)
        self.weights = (
            np.random.default_rng(seed).multinomial(n, np.full(n, 1 / n), size=samples)
            if n
            else np.zeros((samples, 0))
        )

    @staticmethod
    def interval(values):
        values = np.asarray(values)
        valid = values[np.isfinite(values)]
        return np.quantile(valid, [0.025, 0.975]).tolist() if len(valid) else [None, None]

    def totals(self, frame, values):
        totals = np.zeros(len(self.clusters))
        if len(frame):
            np.add.at(
                totals, frame["rollout_id"].map(self.index).to_numpy(int), np.asarray(values, float)
            )
        return totals

    def ratio(self, frame, numerator, denominator):
        sums, counts = self.totals(frame, numerator), self.totals(frame, denominator)
        total, num = counts.sum(), sums.sum()
        den, nums = self.weights @ counts, self.weights @ sums
        draws = np.divide(nums, den, out=np.full_like(nums, np.nan), where=den > 0)
        clusters = int((counts > 0).sum())
        return dict(
            estimate=float(num / total) if total else None,
            ci95=self.interval(draws) if clusters >= 2 else [None, None],
            numerator=int(num),
            denominator=int(total),
            clusters=clusters,
            bootstrap_valid_resamples=int(np.isfinite(draws).sum()),
            status="estimated"
            if total and clusters >= 2
            else ("insufficient_rollout_clusters" if total else "no_events"),
        )


def matched_comparison(frame, flag, boot, scope):
    strata = [
        "y",
        "round_bin",
        "position_bin",
        "block_slot_index",
        "draft_block_length",
        "valid_proposal_count",
        "active_bin",
        "exposure_bin",
        scope + "_events_exposure",
        scope + "_y_exposure",
    ]
    if "cross_exposure_bin" in frame:
        strata += ["cross_exposure_bin", "cross_y_exposure_bin"]
    empty = dict(
        exposed_risk=None,
        unexposed_risk=None,
        risk_ratio=None,
        risk_ratio_ci95=[None, None],
        exposed_ci95=[None, None],
        unexposed_ci95=[None, None],
        events=len(frame),
        matched_events=0,
        unsupported_events=len(frame),
        common_strata=0,
        unsupported_strata=0,
        strata=strata,
        bootstrap_valid_resamples=0,
        clusters=0,
        status="no_common_support",
    )
    if not len(frame):
        return empty
    codes, unique = pd.factorize(pd.MultiIndex.from_frame(frame[strata]))
    r, n = len(boot.clusters), len(unique)
    indices = frame["rollout_id"].map(boot.index).to_numpy(int)
    exposed = frame[flag].to_numpy(bool)
    miss = frame["is_miss"].to_numpy(float)
    matrices = [np.zeros((r, n)) for _ in range(4)]
    for matrix, values in zip(matrices, (miss * exposed, exposed, miss * ~exposed, ~exposed)):
        np.add.at(matrix, (indices, codes), values)
    a, b, c, d = matrices
    common = (b.sum(0) > 0) & (d.sum(0) > 0)
    empty["unsupported_strata"] = int((~common).sum())
    if not common.any():
        return empty
    a, b, c, d = [v[:, common] for v in matrices]
    population_weights = b.sum(0) + d.sum(0)
    population_weights /= population_weights.sum()
    p = float(np.sum(a.sum(0) / b.sum(0) * population_weights))
    q = float(np.sum(c.sum(0) / d.sum(0) * population_weights))
    exposed_draws, unexposed_draws, ratio_draws = [], [], []
    partial_support_draws = 0
    for start in range(0, len(boot.weights), 32):
        w = boot.weights[start : start + 32]
        aa, bb, cc, dd = [w @ v for v in (a, b, c, d)]
        valid = (bb > 0) & (dd > 0)
        partial_support_draws += int((~valid.all(1)).sum())
        normal = (valid * population_weights).sum(1)
        # Reference methodology: no missing risk is imputed; a resample uses
        # its common-support strata with fixed empirical weights renormalized.
        pp = (
            np.divide(aa, bb, out=np.zeros_like(aa), where=bb > 0) * population_weights * valid
        ).sum(1)
        qq = (
            np.divide(cc, dd, out=np.zeros_like(cc), where=dd > 0) * population_weights * valid
        ).sum(1)
        pp = np.divide(pp, normal, out=np.full_like(pp, np.nan), where=normal > 0)
        qq = np.divide(qq, normal, out=np.full_like(qq, np.nan), where=normal > 0)
        exposed_draws.extend(pp)
        unexposed_draws.extend(qq)
        ratio_draws.extend(np.divide(pp, qq, out=np.full_like(pp, np.nan), where=qq > 0))
    clusters = int(((b + d).sum(1) > 0).sum())
    matched = int((b + d).sum())
    return dict(
        empty,
        exposed_risk=p,
        unexposed_risk=q,
        risk_ratio=p / q if q > 0 else None,
        exposed_ci95=boot.interval(exposed_draws) if clusters >= 2 else [None, None],
        unexposed_ci95=boot.interval(unexposed_draws) if clusters >= 2 else [None, None],
        risk_ratio_ci95=boot.interval(ratio_draws) if clusters >= 2 else [None, None],
        matched_events=matched,
        unsupported_events=len(frame) - matched,
        common_strata=int(common.sum()),
        clusters=clusters,
        bootstrap_valid_resamples=int(np.isfinite(ratio_draws).sum()),
        bootstrap_partial_support_resamples=partial_support_draws,
        status="estimated" if clusters >= 2 and q > 0 else "insufficient_support",
    )


def comparison(frame, flag, boot, scope="same"):
    exp = frame[flag].to_numpy(bool) if len(frame) else np.array([], bool)
    miss = frame["is_miss"].to_numpy(float) if len(frame) else np.array([])
    exposed, unexposed = boot.ratio(frame, miss * exp, exp), boot.ratio(frame, miss * ~exp, ~exp)
    a, b = boot.totals(frame, miss * exp), boot.totals(frame, exp)
    c, d = boot.totals(frame, miss * ~exp), boot.totals(frame, ~exp)
    aa, bb, cc, dd = [boot.weights @ v for v in (a, b, c, d)]
    p = np.divide(aa, bb, out=np.full_like(aa, np.nan), where=bb > 0)
    q = np.divide(cc, dd, out=np.full_like(cc, np.nan), where=dd > 0)
    rr = np.divide(p, q, out=np.full_like(p, np.nan), where=q > 0)
    pe, pu = exposed["estimate"], unexposed["estimate"]
    enough = exposed["clusters"] >= 2 and unexposed["clusters"] >= 2
    return dict(
        raw=dict(
            exposed=exposed,
            unexposed=unexposed,
            risk_ratio=pe / pu if pe is not None and pu is not None and pu > 0 else None,
            risk_ratio_ci95=boot.interval(rr) if enough else [None, None],
            bootstrap_valid_resamples=int(np.isfinite(rr).sum()),
        ),
        matched=matched_comparison(frame, flag, boot, scope),
    )


def add_controls(frames):
    ids = sorted(frames)
    half = len(ids) // 2
    results, pairs = [], []
    for donor, recipient in zip(ids[:half], ids[half : 2 * half]):
        source, target = frames[donor], frames[recipient].copy()
        if not len(source) or not len(target):
            continue
        if set(source["question_sha256"]) & set(target["question_sha256"]):
            raise ValueError("Independent rollout control contains overlapping question hashes")
        first = (
            source[source["is_miss"]].groupby("y")["global_verification_batch_id"].min().to_dict()
        )
        prefix = source.groupby("global_verification_batch_id").size().sort_index().cumsum()
        token_prefix = {
            y: v.groupby("global_verification_batch_id").size().sort_index().cumsum()
            for y, v in source.groupby("y")
        }

        def prior(series, boundary):
            index = series.index.searchsorted(boundary, side="left")
            return int(series.iloc[index - 1]) if index else 0

        target["cross_prior"] = [
            first.get(y, math.inf) < boundary
            for y, boundary in zip(target["y"], target["global_verification_batch_id"])
        ]
        target["cross_exposure_bin"] = [
            exposure_bin(prior(prefix, boundary))
            for boundary in target["global_verification_batch_id"]
        ]
        target["cross_y_exposure_bin"] = [
            exposure_bin(prior(token_prefix[y], boundary)) if y in token_prefix else 0
            for y, boundary in zip(target["y"], target["global_verification_batch_id"])
        ]
        target["donor_rollout_id"] = donor
        results.append(target)
        pairs.append(dict(donor=donor, recipient=recipient))
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame(), pairs


def analyze_frames(frames, bootstrap_samples=2000, seed=2026):
    clusters = sorted(frames)
    nonempty = [f for f in frames.values() if len(f)]
    frame = pd.concat(nonempty, ignore_index=True) if nonempty else pd.DataFrame()
    boot = ClusterBootstrap(clusters, bootstrap_samples, seed)
    within = comparison(frame, "token_same", boot)
    distances = []
    for label in DISTANCES:
        subset = frame[frame["distance"].isin([label, "none"])].copy() if len(frame) else frame
        if len(subset):
            subset["distance_exposed"] = subset["distance"] == label
        distances.append(dict(distance=label, **comparison(subset, "distance_exposed", boot)))
    cross, pairs = add_controls(frames)
    cross_boot = ClusterBootstrap(
        sorted(cross["rollout_id"].unique()) if len(cross) else [], bootstrap_samples, seed
    )
    previous = comparison(cross, "cross_prior", cross_boot, "batch")
    previous.update(
        donor_recipient_pairs=pairs,
        comparison_protocol="donor global verification-completion ordinal strictly less than recipient ordinal",
        unused_rollouts=clusters[2 * (len(clusters) // 2) :],
    )
    exclusive = (
        frame[frame["token_source"].isin(["sibling_response", "none"])].copy()
        if len(frame)
        else frame
    )
    if len(exclusive):
        exclusive["source_exposed"] = exclusive["token_source"] == "sibling_response"
    sharing = comparison(exclusive, "source_exposed", boot, "sibling")
    inclusive = comparison(frame, "token_sibling", boot, "sibling")
    source_risks = {}
    for source in ("same_response", "sibling_response", "different_prompt", "none"):
        subset = frame[frame["token_source"] == source] if len(frame) else frame
        source_risks[source] = boot.ratio(
            subset, subset["is_miss"] if len(subset) else [], np.ones(len(subset))
        )
    misses = frame[frame["is_miss"]].copy() if len(frame) else frame
    availability = {}
    for label, flag in (
        ("within_response_prior_miss_share", "token_same"),
        ("within_prompt_prior_miss_share", "token_group"),
    ):
        availability[label] = boot.ratio(
            misses, misses[flag].to_numpy(float) if len(misses) else [], np.ones(len(misses))
        )
    by_progress = []
    if len(misses):
        for bin_name, subset in misses.groupby("round_bin", sort=False):
            by_progress.append(
                dict(
                    round_bin=bin_name,
                    within_response=boot.ratio(subset, subset["token_same"], np.ones(len(subset))),
                    within_prompt=boot.ratio(subset, subset["token_group"], np.ones(len(subset))),
                )
            )
    return dict(
        ordinary_events=len(frame),
        misses=len(misses),
        rollout_clusters=len(clusters),
        bootstrap_samples=bootstrap_samples,
        seed=seed,
        fig1a=dict(within_response=within, distances=distances, previous_rollout_control=previous),
        fig1b=dict(
            exclusive_sibling_vs_none=sharing,
            inclusive_sibling_vs_no_sibling=inclusive,
            exclusive_source_risks=source_risks,
            mutually_exclusive_sources=frame["token_source"].value_counts().to_dict()
            if len(frame)
            else {},
        ),
        fig1c=dict(availability, by_verification_progress=by_progress),
        bootstrap_unit="entire rollout; donor/recipient controls use independent pairs indexed by recipient",
        interpretation="Observational associations and available feedback; no causal correction claim",
    )
