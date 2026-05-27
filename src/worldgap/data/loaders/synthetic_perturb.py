"""Synthetic perturbation pipeline, per TECHNICAL_SPEC.md Section 5.4.

No public dataset captures motor-impairment-like hand motion, so this MUST
exist as a documented, reproducible, seeded transform rather than an ad-hoc
one-off script (spec 5.4: 'every synthetic condition MUST be reproducible from
a stored seed + parameter set').

All three perturbations operate on a Rollout and return a new Rollout with
source='synthetic' and the perturbation parameters recorded in `condition`, so
downstream code (and the validation harness's pre-registered condition set)
can key off exactly what was applied.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from ..rollout import PERCEPTION_STATE_DIM, Rollout, perception_position_channel_mask


def _spatial_channel_mask(rollout: Rollout) -> np.ndarray:
    """Returns a boolean mask (n_channels,), True where a channel is a spatial
    (x/y/z) position channel a purely-spatial perturbation should touch.

    For real 258-dim perception rollouts this excludes the pose block's
    visibility columns (bugfix: earlier versions of inject_tremor/
    reduce_range_of_motion perturbed every column, including visibility, which
    contradicted their own docstrings and would have quietly corrupted
    MediaPipe-confidence-derived validation ground truth, spec 8.1). For any
    other shape (actuation state vectors, or small fixture/toy rollouts used
    in unit tests) there's no known visibility column to exclude, so every
    channel is treated as spatial — preserving prior behavior for those cases.
    """
    n_channels = rollout.states.shape[1]
    if rollout.modality == "perception" and n_channels == PERCEPTION_STATE_DIM:
        return perception_position_channel_mask()
    return np.ones(n_channels, dtype=bool)


def inject_tremor(
    rollout: Rollout,
    frequency_hz: float = 5.0,
    amplitude: float = 0.02,
    seed: int = 0,
) -> Rollout:
    """Adds band-limited sinusoidal-plus-noise motion at a configurable frequency
    to landmark positions, per spec 5.4 ('band-limited noise, 4-6 Hz, matching
    literature-typical pathological tremor frequency').

    Only perturbs spatial (x, y, z) channels, not visibility/presence — tremor
    doesn't change whether a landmark was detected, only where it appears (see
    `_spatial_channel_mask`).
    """
    rng = np.random.default_rng(seed)
    t = rollout.states.shape[0]
    dt = 1.0 / rollout.frame_rate_hz
    time = np.arange(t) * dt

    # A tremor signal with the target center frequency plus a little jitter in
    # phase/frequency across landmarks, so all landmarks don't move in lockstep
    # (which would look nothing like real tremor).
    n_channels = rollout.states.shape[1]
    phase = rng.uniform(0, 2 * np.pi, size=n_channels)
    freq_jitter = rng.normal(loc=frequency_hz, scale=0.3, size=n_channels)
    tremor = amplitude * np.sin(2 * np.pi * freq_jitter[None, :] * time[:, None] + phase[None, :])
    tremor = tremor * _spatial_channel_mask(rollout)[None, :]

    perturbed_states = rollout.states + tremor
    condition = {**rollout.condition, "motion_profile": f"tremor_{frequency_hz}hz_a{amplitude}"}
    return replace(
        rollout,
        states=perturbed_states,
        source="synthetic",
        condition=condition,
        rollout_id=None,  # force recompute — content changed
    )


def reduce_range_of_motion(rollout: Rollout, retain_fraction: float = 0.6, seed: int = 0) -> Rollout:
    """Clips each spatial channel's displacement around its mean to
    `retain_fraction` of its original range, per spec 5.4. Simulates reduced
    range of motion rather than adding noise — a systematically different
    failure mode from tremor.

    Only scales spatial (x, y, z) channels (see `_spatial_channel_mask`) —
    "range of motion" is a spatial-displacement concept, not something that
    applies to a visibility/confidence channel, so that channel is left
    untouched rather than compressed toward its mean.
    """
    if not (0.0 < retain_fraction <= 1.0):
        raise ValueError(f"retain_fraction must be in (0, 1], got {retain_fraction}")

    spatial_mask = _spatial_channel_mask(rollout)

    mean = rollout.states.mean(axis=0, keepdims=True)
    displacement = rollout.states - mean

    # Pass non-spatial (e.g. visibility) channels through byte-for-byte rather
    # than recomputing mean + displacement * 1.0 for them, which would
    # introduce harmless but needless floating-point rounding noise on values
    # this perturbation isn't meant to touch at all.
    clipped = rollout.states.copy()
    clipped[:, spatial_mask] = (
        mean[:, spatial_mask] + displacement[:, spatial_mask] * retain_fraction
    )

    condition = {**rollout.condition, "motion_profile": f"reduced_rom_{retain_fraction}"}
    return replace(
        rollout,
        states=clipped,
        source="synthetic",
        condition=condition,
        rollout_id=None,
    )


def simulate_occlusion(
    rollout: Rollout,
    landmark_group_slice: slice,
    occlusion_frac: float = 0.3,
    min_run_frames: int = 3,
    seed: int = 0,
) -> Rollout:
    """Zeroes out a contiguous window of frames for a chosen landmark subgroup
    (e.g. one hand) and sets presence_mask to 0 for that window — per spec 5.4
    and edge case 12.1: MUST mark the mask, MUST NOT just zero-fill and leave
    the mask claiming presence.
    """
    rng = np.random.default_rng(seed)
    t = rollout.states.shape[0]
    run_len = max(min_run_frames, int(t * occlusion_frac))
    run_len = min(run_len, t)
    start = int(rng.integers(0, max(1, t - run_len + 1)))

    new_states = rollout.states.copy()
    new_mask = rollout.presence_mask.copy()
    new_states[start : start + run_len, landmark_group_slice] = 0.0
    new_mask[start : start + run_len, landmark_group_slice] = 0.0

    condition = {
        **rollout.condition,
        "occlusion": f"frames_{start}-{start + run_len}_of_{t}",
    }
    return replace(
        rollout,
        states=new_states,
        presence_mask=new_mask,
        source="synthetic",
        condition=condition,
        rollout_id=None,
    )
