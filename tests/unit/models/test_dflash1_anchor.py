"""Pure unary objectives and sequence-balanced valid-anchor normalization."""

import pytest
import torch

from speculators.losses import dpace_loss_decay, dpard_loss_decay, resolve_loss_config
from speculators.losses.eager import ce_loss, renyi_half_loss, tv_loss
from speculators.models.dflash.core import DFlashDraftModel
from speculators.models.dflash.metrics import compute_metrics
from speculators.train.config.schema import TrainConfig


@pytest.mark.parametrize(("mode", "actor"), [("dpace", "ce"), ("dpard", "renyi_half")])
def test_sequence_balanced_anchor_loss_and_detached_backward(mode, actor):
    # Row 0 has two anchors (one partial); row 1 has one; row 2 is empty.
    q = torch.tensor([[[0.25, 0.75]] * 6, [[0.5, 0.5]] * 6, [[0.8, 0.2]] * 6])
    logits = q.log().requires_grad_()
    targets = torch.full_like(logits, -torch.inf)
    targets[..., 0] = 0
    mask = torch.tensor([[0, 1, 1, 0, 1, 0], [0, 1, 1, 0, 0, 0], [0] * 6])
    loss, metrics = compute_metrics(
        logits,
        targets,
        mask,
        3,
        loss_config=resolve_loss_config(actor, "eager"),
        tv_loss_fn=tv_loss,
        per_position_loss_weight=mode,
        dflash_loss_reduction="sequence-mean-valid-anchor",
    )
    # alpha=.5: row 0 credits [1.015625,.390625,.625]; row 1 [1.3125,.5625].
    expected = (
        (-torch.log(torch.tensor(0.25)) * 2.03125 / 2)
        + (-torch.log(torch.tensor(0.5)) * 1.875)
    ) / 2
    torch.testing.assert_close(loss, expected)
    frozen_credit = torch.tensor(
        [[0, 1.015625, 0.390625, 0, 0.625, 0], [0, 1.3125, 0.5625, 0, 0, 0], [0] * 6]
    )
    frozen_loss = (
        (ce_loss(logits, targets) * frozen_credit).sum(1) / torch.tensor([2, 1, 1])
    )[:2].mean()
    grad = torch.autograd.grad(loss, logits, retain_graph=True)[0]
    torch.testing.assert_close(grad, torch.autograd.grad(frozen_loss, logits)[0])
    assert torch.isfinite(grad).all()
    for key in ("acceptance_sum", "acceptance_total", "tau_sum", "tau_total"):
        assert key in metrics
        assert not metrics[key].requires_grad


def test_one_hot_actor_and_weights_reduce_to_dpace():
    torch.manual_seed(3)
    logits = torch.randn(2, 6, 7, requires_grad=True)
    targets = torch.full_like(logits, -torch.inf)
    targets[..., 2] = 0
    mask = torch.tensor([[0, 1, 1, 0, 1, 0], [0, 1, 1, 0, 1, 1]])
    ce = ce_loss(logits, targets)
    torch.testing.assert_close(renyi_half_loss(logits, targets), ce)
    acceptance = 1 - tv_loss(logits, targets)
    torch.testing.assert_close(acceptance, ce.neg().exp())
    pace = dpace_loss_decay(torch.zeros_like(mask), mask, 3, 0.5, ce)
    pard = dpard_loss_decay(acceptance, mask, 3, 0.5, start_pos=1)
    torch.testing.assert_close(pace, pard)
    assert not pace.requires_grad
    assert not pard.requires_grad


def test_soft_teacher_full_vocab_dpard_and_empty_batch():
    logits = torch.tensor([[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]], requires_grad=True)
    targets = torch.tensor([[[0.75, 0.25]] * 3]).log()
    mask = torch.tensor([[0, 1, 1]])
    loss, metrics = compute_metrics(
        logits,
        targets,
        mask,
        3,
        loss_config=resolve_loss_config("renyi_half", "eager"),
        tv_loss_fn=tv_loss,
        per_position_loss_weight="dpard",
        dflash_loss_reduction="sequence-mean-valid-anchor",
    )
    actor = -2 * torch.log(
        torch.sqrt(torch.tensor(0.375)) + torch.sqrt(torch.tensor(0.125))
    )
    torch.testing.assert_close(loss, actor * (0.875 + 2 * 0.875**2))
    torch.testing.assert_close(metrics["acceptance_sum"], torch.tensor(1.5))
    empty, _ = compute_metrics(
        logits,
        targets,
        mask * 0,
        3,
        loss_config=resolve_loss_config("renyi_half", "eager"),
        tv_loss_fn=tv_loss,
        per_position_loss_weight="dpard",
        dflash_loss_reduction="sequence-mean-valid-anchor",
    )
    empty.backward()
    assert empty.item() == 0
    assert torch.equal(logits.grad, torch.zeros_like(logits))


def test_default_still_uses_valid_token_denominator():
    logits = torch.zeros(1, 6, 2, requires_grad=True)
    targets = torch.zeros_like(logits)
    mask = torch.tensor([[0, 1, 1, 0, 1, 0]])
    loss, _ = compute_metrics(
        logits,
        targets,
        mask,
        3,
        loss_config=resolve_loss_config("ce", "eager"),
        per_position_loss_weight="dpace",
    )
    torch.testing.assert_close(loss, torch.log(torch.tensor(2.0)) * 2.625 / (3 + 1e-5))


def test_config_allows_pure_dflash_dpard_and_forwards_reduction():
    cfg = TrainConfig.from_flat(
        {
            "speculator_type": "dflash",
            "loss_fn": "renyi_half",
            "per_position_loss_weight": "dpard",
            "dflash_loss_reduction": "sequence-mean-valid-anchor",
        }
    )
    train, val = DFlashDraftModel.get_trainer_kwargs(**cfg.flatten())
    assert (
        train["dflash_loss_reduction"]
        == val["dflash_loss_reduction"]
        == "sequence-mean-valid-anchor"
    )
    assert train["dpard_alpha"] == 0.5


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(("mode", "actor"), [("dpace", "ce"), ("dpard", "renyi_half")])
def test_fused_bf16_matches_eager_anchor_objective(mode, actor):
    torch.manual_seed(31)
    logits = torch.randn(
        2, 16, 257, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    targets = torch.randn_like(logits)
    mask = torch.ones(2, 16, device="cuda")
    mask[:, ::8] = 0
    mask[1, 8:] = 0
    options = {
        "block_size": 8,
        "per_position_loss_weight": mode,
        "dflash_loss_reduction": "sequence-mean-valid-anchor",
    }
    loss, metrics = compute_metrics(
        logits,
        targets,
        mask,
        loss_config=resolve_loss_config(actor, "fused"),
        tv_loss_fn=resolve_loss_config("tv", "fused")["tv"][0],
        **options,
    )
    reference_logits = logits.detach().float().requires_grad_()
    reference, reference_metrics = compute_metrics(
        reference_logits,
        targets.float(),
        mask,
        loss_config=resolve_loss_config(actor, "eager"),
        tv_loss_fn=tv_loss,
        **options,
    )
    loss.backward()
    reference.backward()
    torch.testing.assert_close(loss, reference, atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(
        logits.grad.float(), reference_logits.grad, atol=2e-3, rtol=2e-2
    )
    assert torch.isfinite(logits.grad).all()
    for key in ("acceptance_sum", "tau_sum"):
        torch.testing.assert_close(
            metrics[key], reference_metrics[key], atol=2e-3, rtol=2e-3
        )
