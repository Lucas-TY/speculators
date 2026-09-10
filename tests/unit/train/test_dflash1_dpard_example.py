import os
import subprocess
import warnings
from pathlib import Path

from speculators.train.config import TrainConfig


def test_dflash1_offline_example_resolves(tmp_path: Path):
    repo = Path(__file__).resolve().parents[3]
    launcher = tmp_path / "torchrun"
    launcher.write_text('#!/bin/bash\nprintf "%s\\n" "$@"\n')
    launcher.chmod(0o755)
    result = subprocess.run(  # noqa: S603 - fixed repository example
        [
            "/bin/bash",
            str(repo / "examples/train/dflash_qwen3_4b_b16_dpard_offline.sh"),
        ],
        env={
            **os.environ,
            "PATH": f"{tmp_path}:{os.environ['PATH']}",
            "DATA_PATH": str(tmp_path / "data"),
        },
        capture_output=True,
        text=True,
        check=True,
    )
    argv = result.stdout.splitlines()
    with warnings.catch_warnings(record=True) as caught:
        cfg = TrainConfig.resolve(argv[argv.index("speculators.train") + 1 :])
    assert not any("--dpard-alpha" in str(item.message) for item in caught)
    assert cfg.speculator_type == "dflash"
    assert cfg.loss.loss_fn == "renyi_half"
    assert cfg.dflash.per_position_loss_weight == "dpard"
    assert cfg.dflash.dflash_loss_reduction == "sequence-mean-valid-anchor"
    assert cfg.dflash.block_size == 16
    assert cfg.dflash.sample_from_anchor is False
    assert cfg.draft.target_layer_ids == [1, 17, 33]
    assert cfg.trainer.gradient_accumulation_steps == 2
    assert cfg.generation.on_missing == "raise"
