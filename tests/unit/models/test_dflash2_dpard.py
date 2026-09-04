import pytest
import torch
from torch.nn import functional

from speculators.losses import dpard_loss_decay, resolve_loss_config
from speculators.losses.eager import renyi_half_loss
from speculators.models.dflash2.metrics import compute_metrics


def test_dpard_uses_selector_proposal_and_valid_position_mean() -> None:
    torch.manual_seed(23)
    unary_logits = torch.randn(1, 6, 7, requires_grad=True)
    targets = torch.randn(1, 6, 7)
    candidate_ids = unary_logits.detach().topk(3, dim=-1).indices
    training_candidate_ids = (candidate_ids + 1) % unary_logits.shape[-1]
    candidate_logits = torch.randn(1, 6, 3, requires_grad=True)
    runtime_candidate_logits = torch.randn(1, 6, 3)
    target_positions = torch.tensor([[0, 1, 2, 0, 1, 2]])
    mask = torch.tensor([[1, 1, 0, 1, 1, 1]], dtype=torch.float32)

    loss, metrics = compute_metrics(
        unary_logits=unary_logits,
        targets=targets,
        training_candidate_ids=training_candidate_ids,
        candidate_logits=candidate_logits,
        target_positions=target_positions,
        contains_target=torch.ones_like(mask, dtype=torch.bool),
        loss_mask=mask,
        block_size=3,
        top_k=3,
        loss_config=resolve_loss_config("renyi_half", "eager"),
        per_position_loss_weight="dpard",
        dpard_alpha=0.5,
        runtime_candidate_ids=candidate_ids,
        runtime_candidate_logits=runtime_candidate_logits,
    )

    target_prob = targets.softmax(dim=-1)
    candidate_target_prob = target_prob.gather(-1, candidate_ids)
    proposal_prob = runtime_candidate_logits.softmax(dim=-1)
    acceptance = torch.minimum(candidate_target_prob, proposal_prob).sum(dim=-1)
    credit = dpard_loss_decay(acceptance, mask, 3, 0.5, start_pos=1)
    unary_actor = renyi_half_loss(unary_logits, targets)
    selector_ce = functional.cross_entropy(
        candidate_logits.flatten(0, 1),
        target_positions.flatten(),
        reduction="none",
    ).view_as(target_positions)
    expected = (
        (unary_actor * credit * mask).sum()
        + (selector_ce * credit * mask).sum()
    ) / mask.sum()

    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(metrics["dpard_acceptance_sum"], (acceptance * mask).sum())
    torch.testing.assert_close(metrics["dpard_acceptance_total"], mask.sum())
    unary_grad, selector_grad = torch.autograd.grad(loss, (unary_logits, candidate_logits))
    assert torch.isfinite(unary_grad).all()
    assert torch.isfinite(selector_grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_dpard_b8_bf16_backward() -> None:
    torch.manual_seed(29)
    unary_logits = torch.randn(
        1, 16, 257, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    targets = torch.randn_like(unary_logits)
    candidate_ids = unary_logits.detach().topk(16, dim=-1).indices
    candidate_logits = torch.randn(
        1, 16, 16, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    target_positions = torch.randint(0, 16, (1, 16), device="cuda")
    mask = torch.ones(1, 16, device="cuda", dtype=torch.float32)
    loss, _ = compute_metrics(
        unary_logits=unary_logits,
        targets=targets,
        training_candidate_ids=candidate_ids,
        candidate_logits=candidate_logits,
        target_positions=target_positions,
        contains_target=torch.ones_like(mask, dtype=torch.bool),
        loss_mask=mask,
        block_size=8,
        top_k=16,
        loss_config=resolve_loss_config("renyi_half", "fused"),
        per_position_loss_weight="dpard",
        dpard_alpha=0.5,
        runtime_candidate_ids=candidate_ids,
        runtime_candidate_logits=candidate_logits,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(unary_logits.grad).all()
    assert torch.isfinite(candidate_logits.grad).all()
