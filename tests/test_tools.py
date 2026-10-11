import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import torch

from dflash_reflex.metrics import Metrics

ROOT = Path(__file__).resolve().parents[1]


def test_actual_nemo_configuration_inheritance(tmp_path):
    process = subprocess.run(
        ["bash", str(ROOT / "train_qwen3_4b.sh")],
        check=True,
        capture_output=True,
        text=True,
        env=dict(os.environ, DRY_RUN="true", FAIRNESS_PROFILE="native_dflash"),
    )
    raw = json.loads(process.stdout[process.stdout.index("{\n") :])
    import yaml

    file = tmp_path / "resolved.yaml"
    file.write_text(yaml.safe_dump(raw))
    path = ROOT / "third_party/nemo-rl/nemo_rl/utils/config.py"
    spec = importlib.util.spec_from_file_location("nemo_config_composition", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.register_omegaconf_resolvers()
    from omegaconf import OmegaConf

    result = OmegaConf.to_container(module.load_config(file), resolve=True)
    assert result["policy"]["megatron_cfg"]["enabled"]
    assert result["policy"]["draft"]["speculator_type"] == "dflash"
    assert result["policy"]["tokenizer"]["chat_template"] == "default"
    assert result["data"]["train"]["dataset_name"] == "dflash_reflex.nemo_data.LocalMathDataset"
    assert not result["policy"]["sequence_packing"]["enabled"]


def test_reporting_commands_on_temporary_fixture(tmp_path):
    metrics = Metrics(tmp_path, "dflash", "simplelr")
    packet = dict(
        counters=torch.tensor([4, 2, 2, 5, 0, 0, 0, 0]),
        generated_tokens=4,
        generation_time_s=0.5,
        updates=0,
        profile_ms={},
    )
    metrics.record(0, 1, [packet], dict(policy_training=0.1), wall=1)
    for script, output in (("plot_aal.py", "aal.png"), ("plot_training_time.py", "time.png")):
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / script),
                str(tmp_path),
                "--output",
                str(tmp_path / output),
            ],
            check=True,
        )
        assert (tmp_path / output).stat().st_size > 1000
    result = subprocess.check_output(
        [sys.executable, str(ROOT / "scripts/summarize_run.py"), str(tmp_path)], text=True
    )
    assert json.loads(result)[0]["aal"] == 2
