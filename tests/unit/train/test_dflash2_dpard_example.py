import os
import subprocess
from pathlib import Path

import pytest

from speculators.train.config import TrainConfig


@pytest.mark.parametrize("offline", [False, True])
def test_dflash2_dpard_example_resolves(tmp_path: Path, offline: bool) -> None:
    repo = Path(__file__).resolve().parents[3]
    launcher = tmp_path / "torchrun"
    launcher.write_text('#!/bin/bash\nprintf "%s\\n" "$@"\n')
    launcher.chmod(0o755)
    name = (
        "dflash2_qwen3_4b_b16_dpard_offline.sh"
        if offline
        else "dflash2_qwen3_4b_dpard.sh"
    )
    result = subprocess.run(  # noqa: S603 - both example paths are fixed above
        ["/bin/bash", str(repo / "examples/train" / name)],
        env={
            **os.environ,
            "PATH": f"{tmp_path}:{os.environ['PATH']}",
            "DATA_PATH": str(tmp_path / "data"),
            "VLLM_ENDPOINT": "http://localhost:8000/v1",
            "OUTPUT_DIR": str(tmp_path / "output"),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    argv = result.stdout.splitlines()
    cfg = TrainConfig.resolve(argv[argv.index("speculators.train") + 1 :])
    assert cfg.speculator_type == "dflash2"
    assert cfg.loss.loss_fn == "renyi_half"
    assert cfg.dflash.per_position_loss_weight == "dpard"
    assert cfg.dflash2.dpard_alpha == 0.5
    assert cfg.dflash.block_size == (16 if offline else 8)
    assert cfg.draft.num_layers == (3 if offline else 5)
    assert cfg.draft.target_layer_ids == (
        [1, 17, 33] if offline else [1, 9, 17, 25, 33]
    )
    if offline:
        assert "--vllm-endpoint" not in argv
        assert cfg.generation.on_missing == "raise"
        assert cfg.trainer.epochs == 6
        assert cfg.trainer.fsdp_shard
        assert cfg.dflash.sample_from_anchor is False
        assert cfg.optimizer.weight_decay == 0.01
        assert cfg.scheduler.scheduler_type == "linear"
