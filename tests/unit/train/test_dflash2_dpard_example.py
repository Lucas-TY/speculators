import os
import subprocess
from pathlib import Path

from speculators.train.config import TrainConfig


def test_dflash2_dpard_example_resolves(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[3]
    launcher = tmp_path / "torchrun"
    launcher.write_text('#!/bin/bash\nprintf "%s\\n" "$@"\n')
    launcher.chmod(0o755)
    result = subprocess.run(
        ["/bin/bash", str(repo / "examples/train/dflash2_qwen3_4b_dpard.sh")],
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
    assert cfg.dflash.block_size == 8
    assert cfg.draft.num_layers == 5
    assert cfg.draft.target_layer_ids == [1, 9, 17, 25, 33]
