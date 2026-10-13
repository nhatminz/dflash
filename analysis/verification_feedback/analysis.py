"""GPU-free analysis of committed event files; no runtime/model imports."""

from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from .collector import validate_rollout
from .io import atomic_bytes, atomic_json, digest, file_hash, read_json
from .statistics import analyze_frames, annotate


def flatten(row, prefix=""):
    result = {}
    for key, value in row.items():
        name = prefix + key
        if isinstance(value, dict):
            result.update(flatten(value, name + "_"))
        elif isinstance(value, list):
            if len(value) == 2 and all(v is None or isinstance(v, (int, float)) for v in value):
                result[name + "_low"], result[name + "_high"] = value
        else:
            result[name] = value
    return result


def write_report(root, results, validation=None):
    root = Path(root)
    lines = [
        "# Phân tích verification feedback của DFlash",
        "",
        "Baseline: DFlash thuần, Reflex tắt. Chỉ inference đóng băng; không chạy pretrain/GRPO hoặc optimizer step.",
        "Target và draft của mỗi mốc phải đến từ cùng một checkpoint resume gốc; export đặt trong cache phân tích riêng.",
        "",
        "Feedback chỉ có sẵn sau khi toàn bộ verification batch hoàn tất. Khoảng cách trong response dùng chỉ số verification riêng của response.",
        "Mẫu số miss chỉ gồm các proposal thực sự được thăm: accepted prefix và vị trí reject đầu tiên. Loại prefill, bonus, EOS, length cap và padding.",
        "Token đích là token sampler thực sự phát ra; không dùng target argmax. Với singleton greedy draft, recovery loại ID proposal.",
        "Các nguồn lịch sử loại trừ nhau: cùng response, nếu không thì sibling cùng prompt, nếu không thì prompt khác, nếu không thì chưa có.",
        "Figure 1(b) chính so sánh sibling với chưa có lịch sử; JSON còn có so sánh sibling bất kỳ với không có sibling.",
        "Figure 1(c) dùng lịch sử prompt bao gồm response hiện tại và các sibling; mẫu số là toàn bộ miss thông thường.",
        "",
        "Thống kê: raw risks và RR; matched risks dùng exact token, block slot/length, vị trí response, tiến độ verification, active batch và các bin exposure.",
        "Chỉ sử dụng common support. Không điền giá trị thiếu; JSON ghi số event/stratum bị loại. Bootstrap resample toàn rollout; dưới hai cluster có đóng góp thì CI không khả dụng.",
        "Trong mỗi resample, chỉ các strata có cả hai điều kiện mới đóng góp; giữ trọng số thực nghiệm và chuẩn hóa lại trên support đó, giống code E2/E3/E6 tham chiếu.",
        "Control: sort rollout ID, chia hai nửa donor/recipient rời nhau, ghép một-một không tái sử dụng; chỉ lấy donor boundary có ordinal nhỏ hơn recipient boundary.",
        "Đây là quan hệ quan sát và mức sẵn có của feedback, không phải bằng chứng hiệu quả nhân quả của adaptation.",
        "",
    ]
    if validation:
        lines += ["## Trạng thái kiểm tra", validation, ""]
    if not results:
        markers = sorted(root.glob("collection/step_*/rollout_*/complete.json"))
        if markers:
            counts = {}
            for marker in markers:
                row = read_json(marker)
                stage = counts.setdefault(
                    row["checkpoint_step"],
                    dict(rollouts=0, prompts=0, responses=0, rounds=0, ordinary_events=0, misses=0),
                )
                stage["rollouts"] += 1
                for name in stage:
                    if name != "rollouts":
                        stage[name] += row.get(name, 0)
            lines += [
                "Các complete marker hiện ghi các rollout sau; chưa tính thống kê/CI. "
                "Lệnh --mode analyze sẽ kiểm tra lại checksum và tính kết quả:"
            ]
            lines += [f"- Step {step}: {values}." for step, values in sorted(counts.items())]
        else:
            lines += [
                "Chưa có rollout thật đã hoàn tất. Chưa có số prompt/response/miss thực đo, CI hoặc figure kết quả."
            ]
        lines += ["Không thay thế bằng checkpoint giả hoặc kết quả FastGRPO.", ""]
    for step, result in results.items():
        lines += [
            f"## Step {step}",
            f"Rollout: {result['rollout_clusters']}; prompt: {result['prompts']}; response: {result['responses']}; "
            f"response verification rounds: {result['rounds']}; valid ordinary events: {result['ordinary_events']}; misses: {result['misses']}.",
            f"Bootstrap: {result['bootstrap_samples']} resamples.",
        ]
        for name, statistic in (
            ("1(a) trong response", result["fig1a"]["within_response"]),
            ("1(a) control rollout độc lập", result["fig1a"]["previous_rollout_control"]),
            ("1(b) sibling cùng prompt", result["fig1b"]["exclusive_sibling_vs_none"]),
        ):
            matched = statistic["matched"]
            lines.append(
                f"- {name}: matched RR={matched['risk_ratio']}, CI95={matched['risk_ratio_ci95']}; "
                f"matched events={matched['matched_events']}, unsupported events={matched['unsupported_events']}."
            )
        for name in ("within_response_prior_miss_share", "within_prompt_prior_miss_share"):
            value = result["fig1c"][name]
            percentage = (
                f"{100 * value['estimate']:.3f}%"
                if value["estimate"] is not None
                else "không khả dụng"
            )
            lines.append(
                f"- {name}: {value['numerator']}/{value['denominator']} = {percentage}; CI95 (tỷ lệ)={value['ci95']}."
            )
        lines += [
            f"Logging-disabled parity replays đã pass: {result['sampling_parity_replays_passed']}.",
            "Thời gian logging và hash weights không được tính là inference latency; chỉ replay tắt logging có timing riêng.",
            "",
        ]
    lines += [
        "## Đường dẫn output",
        str(root),
        "`statistics_step_<step>.json`, `summary.csv`, `summary.json`; các partition `collection/step_<step>/rollout_*/` chứa events, rounds, responses, audit và complete marker.",
        "`fig1a_dflash.pdf`, `fig1b_dflash.pdf`, `fig1c_dflash.pdf`, `fig1_dflash_combined.pdf`, `fig1_dflash_combined.png`.",
        "",
    ]
    atomic_bytes(root / "analysis_report.md", ("\n".join(lines) + "\n").encode())


def analyze(root, checkpoints, samples=2000, seed=2026):
    from tqdm.auto import tqdm

    from .plotting import figures

    root = Path(root)
    plan = read_json(root / "evaluation_manifest.json")
    results, summary, inputs = {}, [], {}
    for step in tqdm(checkpoints, desc="Offline statistics"):
        directory = root / f"collection/step_{step}"
        manifest = (
            read_json(directory / "manifest.json")
            if (directory / "manifest.json").is_file()
            else None
        )
        folders = sorted(directory.glob("rollout_*/complete.json"))
        if not folders:
            continue
        if manifest["evaluation_manifest_sha256"] != digest(plan):
            raise ValueError("Step manifest differs from shared held-out evaluation plan")
        cache_identity = digest(
            dict(
                commits={str(p): file_hash(p) for p in folders},
                samples=samples,
                seed=seed + step,
                statistics_source=file_hash(Path(__file__).with_name("statistics.py")),
                analysis_source=file_hash(Path(__file__)),
            )
        )
        saved_path, cache_path = (
            root / f"statistics_step_{step}.json",
            root / f".cache/statistics_step_{step}.json",
        )
        cached = read_json(cache_path) if cache_path.is_file() else {}
        if cached.get("identity") == cache_identity and saved_path.is_file():
            if file_hash(saved_path) != cached["result_sha256"]:
                raise ValueError(f"Statistical cache integrity failed: {saved_path}")
            for p in folders:
                validate_rollout(p.parent, manifest["identity"])
                inputs[str(p.relative_to(root))] = file_hash(p)
            results[str(step)] = read_json(saved_path)
            continue
        frames, count, labels, replays = {}, dict(prompts=0, responses=0, rounds=0), {}, 0
        for marker in folders:
            folder = marker.parent
            commit = validate_rollout(folder, manifest["identity"])
            if commit["checkpoint_step"] != step:
                raise ValueError("Step-specific events were mixed across checkpoints")
            rollout = commit["rollout_id"]
            if rollout in frames:
                raise ValueError("Duplicate rollout cluster")
            events = pq.read_table(folder / "events.parquet").to_pandas()
            if not events.empty and (events["checkpoint_step"] != step).any():
                raise ValueError("Event checkpoint identity mismatch")
            frames[rollout] = annotate(events)
            for key in count:
                count[key] += commit[key]
            for reason, n in events["termination_or_censor_reason"].value_counts().items():
                labels[reason] = labels.get(reason, 0) + int(n)
            audit = read_json(folder / "audit.json")
            replays += audit["parity"]["status"] == "passed"
            inputs[str(marker.relative_to(root))] = file_hash(marker)
        if replays < 1:
            raise ValueError(f"Step {step} has no successful genuine logging-disabled replay")
        result = analyze_frames(frames, samples, seed + step)
        result.update(
            checkpoint_step=step,
            **count,
            all_event_labels=labels,
            sampling_parity_replays_passed=replays,
            partial_collection=len(folders) < plan["eval_rollouts"],
        )
        atomic_json(root / f"statistics_step_{step}.json", result)
        atomic_json(cache_path, dict(identity=cache_identity, result_sha256=file_hash(saved_path)))
        results[str(step)] = result
    for step, result in results.items():
        for panel, stat in (
            ("fig1a_within", result["fig1a"]["within_response"]),
            ("fig1a_control", result["fig1a"]["previous_rollout_control"]),
            ("fig1b_sibling", result["fig1b"]["exclusive_sibling_vs_none"]),
        ):
            summary.append(flatten(dict(checkpoint_step=step, analysis=panel, **stat)))
        for key in ("within_response_prior_miss_share", "within_prompt_prior_miss_share"):
            summary.append(
                flatten(dict(checkpoint_step=step, analysis=key, **result["fig1c"][key]))
            )
    if not results:
        write_report(root, {}, "Chưa có complete marker hợp lệ để phân tích. Không khởi tạo model.")
        raise ValueError("No completed, validated rollouts available for the requested checkpoints")
    atomic_json(root / "summary.json", results)
    atomic_bytes(root / "summary.csv", pd.DataFrame(summary).to_csv(index=False).encode())
    atomic_json(
        root / "analysis_inputs.json",
        dict(input_commits=inputs, bootstrap_samples=samples, seed=seed),
    )
    figures(root, results)
    write_report(root, results)
    return results
