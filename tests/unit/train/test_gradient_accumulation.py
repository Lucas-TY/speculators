import math

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from speculators.train.config import TrainConfig
from speculators.train.trainer import Trainer, TrainerConfig, _resolve_scheduler_steps


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.25))
        self.seen = []

    def forward(self, x, y, **kwargs):
        self.seen.extend(x.flatten().tolist())
        loss = (self.weight * x - y).square().mean()
        return None, loss, {"loss_sum": loss.detach(), "loss_total": loss.new_tensor(1)}


@pytest.mark.parametrize(
    ("count", "accum", "max_steps"),
    [(4, 2, None), (3, 2, None), (4, 2, 1), (4, 1, None)],
)
def test_accumulation_matches_combined_batches(
    monkeypatch, tmp_path, count, accum, max_steps
):
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: None)
    batches = [
        {
            "x": torch.tensor([i / 10]),
            "y": torch.tensor([0.1]),
            "document_ids": torch.tensor([0]),
            "error_records": 0,
        }
        for i in range(1, count + 1)
    ]
    trainer = Trainer.__new__(Trainer)
    trainer.model = TinyModel()
    trainer.config = TrainerConfig(
        lr=0.1,
        num_epochs=1,
        save_path=str(tmp_path),
        gradient_accumulation_steps=accum,
        max_steps=max_steps,
    )
    trainer.rank, trainer.local_rank, trainer.device_type = 1, "cpu", "cpu"
    trainer.is_distributed = False
    trainer.global_step = 0
    trainer.train_loader = DataLoader(batches, batch_size=None)
    trainer.optimizers = [torch.optim.SGD(trainer.model.parameters(), lr=0.1)]
    trainer.schedulers = [
        torch.optim.lr_scheduler.StepLR(trainer.optimizers[0], step_size=1, gamma=0.9)
    ]
    trainer.train_epoch(0)

    reference = TinyModel()
    optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    updates = min(math.ceil(count / accum), max_steps or count)
    for start in range(0, min(count, updates * accum), accum):
        group = batches[start : start + accum]
        optimizer.zero_grad()
        _, loss, _ = reference(
            torch.cat([b["x"] for b in group]), torch.cat([b["y"] for b in group])
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
    assert trainer.global_step == updates
    assert len(trainer.model.seen) == min(count, updates * accum)
    torch.testing.assert_close(trainer.model.weight, reference.weight)
    assert trainer.schedulers[0].get_last_lr() == scheduler.get_last_lr()


def test_scheduler_counts_optimizer_updates():
    config = TrainerConfig(
        lr=0.1,
        num_epochs=6,
        save_path="unused",
        gradient_accumulation_steps=2,
        scheduler_warmup_ratio=0.1,
    )
    assert _resolve_scheduler_steps(config, 5) == (1, 18)


def test_accumulation_cli_and_validation():
    cfg = TrainConfig.resolve(
        [
            "--verifier-name-or-path",
            "Qwen/Qwen3-4B",
            "--gradient-accumulation-steps",
            "2",
        ]
    )
    assert cfg.trainer.gradient_accumulation_steps == 2
    with pytest.raises((ValueError, SystemExit)):
        TrainConfig.resolve(
            [
                "--verifier-name-or-path",
                "Qwen/Qwen3-4B",
                "--gradient-accumulation-steps",
                "0",
            ]
        )
