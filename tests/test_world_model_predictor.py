"""Guards spec 6.3's per-frame prediction requirement.

Spec 6.3 says the predictor takes "the context encoder's pooled representation
(+ learned mask tokens for future positions)" and predicts the target
encoder's latent "for each future frame". Releases up to 0.1.0 instead
predicted a single vector and broadcast it across the whole future window —
an undocumented deviation that makes the predictor structurally incapable of
representing how the trajectory evolves over the horizon. These tests fail if
that regression comes back.
"""

from __future__ import annotations

import pytest
import torch

from worldgap.config import EncoderConfig, WorldModelConfig
from worldgap.models.encoders.landmark_encoder import LandmarkEncoder
from worldgap.models.world_model import WorldModel


def _build(predict_frames: int = 4) -> WorldModel:
    enc_cfg = EncoderConfig(d_model=16, n_layers=1, n_heads=2, dim_feedforward=32, dropout=0.0)
    wm_cfg = WorldModelConfig(context_frames=4, predict_frames=predict_frames, summary_dim=8)
    torch.manual_seed(0)
    return WorldModel(lambda: LandmarkEncoder(input_dim=12, config=enc_cfg), wm_cfg)


def test_predictor_emits_a_distinct_vector_per_future_frame():
    """The concrete anti-broadcast check: with mask tokens, two different
    future offsets MUST get different predictions from identical context.
    """
    model = _build(predict_frames=4)
    model.eval()
    ctx = torch.randn(2, 4, 12)
    ctx_mask = torch.ones(2, 4, 12)

    h = model.context_encoder(ctx, ctx_mask)
    pooled = h.mean(dim=1)
    tokens = model.future_mask_tokens  # (Tp, d)
    preds = model.predictor(
        torch.cat(
            [pooled.unsqueeze(1).expand(-1, tokens.shape[0], -1), tokens.unsqueeze(0).expand(2, -1, -1)],
            dim=-1,
        )
    )

    # frame 0's prediction must not equal frame 3's — broadcasting would make
    # every pairwise difference exactly zero.
    assert not torch.allclose(preds[:, 0, :], preds[:, -1, :])
    spread = (preds - preds.mean(dim=1, keepdim=True)).abs().max()
    assert spread > 1e-6


def test_learned_mask_tokens_are_trainable_parameters():
    model = _build()
    names = {n for n, p in model.named_parameters() if p.requires_grad}
    assert "future_mask_tokens" in names


def test_mask_tokens_receive_gradient_from_the_jepa_loss():
    model = _build(predict_frames=3)
    ctx = torch.randn(2, 4, 12)
    fut = torch.randn(2, 3, 12)
    loss, _ = model(ctx, torch.ones_like(ctx), fut, torch.ones_like(fut))
    loss.backward()
    assert model.future_mask_tokens.grad is not None
    assert model.future_mask_tokens.grad.abs().sum() > 0


def test_future_window_longer_than_predict_frames_raises_rather_than_guessing():
    """No silent token reuse/interpolation for offsets the model never learned."""
    model = _build(predict_frames=2)
    ctx = torch.randn(1, 4, 12)
    fut = torch.randn(1, 5, 12)  # longer than predict_frames
    with pytest.raises(ValueError, match="mask tokens"):
        model(ctx, torch.ones_like(ctx), fut, torch.ones_like(fut))


def test_target_branch_still_carries_no_gradient():
    """Spec 6.3: gradient MUST NOT flow into the target encoder directly."""
    model = _build(predict_frames=3)
    ctx = torch.randn(2, 4, 12)
    fut = torch.randn(2, 3, 12)
    loss, _ = model(ctx, torch.ones_like(ctx), fut, torch.ones_like(fut))
    loss.backward()
    assert all(p.grad is None for p in model.target_encoder.parameters())
