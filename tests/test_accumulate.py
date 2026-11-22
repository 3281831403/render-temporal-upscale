"""Contract tests for the temporal accumulation primitives.

The tests are intentionally deterministic and use only pytest plus the
standard library.  They exercise every public stub in
:mod:`tupscale.accumulate`; the repository is expected to start red.
"""

import pytest

from tupscale.accumulate import (
    accumulate,
    blend,
    converge_weight,
    jitter,
    reproject,
    reset_on_cut,
    validate_history,
    validate_state,
)

TOL = 1e-9


def rec(color=0.0, depth=0.0, weight=0.0):
    """Build one history record without sharing mutable state."""
    return {
        "color": float(color),
        "depth": float(depth),
        "weight": float(weight),
    }


def motion_grid(height, width, dx=0.0, dy=0.0):
    """Build a constant motion field."""
    return [[(float(dx), float(dy)) for _ in range(width)]
            for _ in range(height)]


def test_jitter_is_not_a_fixed_zero_offset():
    offsets = [jitter(i, 16) for i in range(4)]

    assert len(offsets) == 4
    assert any(abs(dx) > 1e-6 or abs(dy) > 1e-6
               for dx, dy in offsets), "jitter must sample sub-pixel phases"

    xs = [dx for dx, _ in offsets]
    ys = [dy for _, dy in offsets]
    assert max(xs) - min(xs) > 0.1
    assert max(ys) - min(ys) > 0.1


def test_reproject_samples_opposite_the_motion_vector():
    # One bright history sample at x = 1.  The scene moves one pixel to
    # the right, so x = 2 must fetch it from x - 1, not from x + 1.
    history = [[rec(0.0, 5.0, 1.0),
                rec(10.0, 5.0, 1.0),
                rec(0.0, 5.0, 1.0)]]
    motion = motion_grid(1, 3, dx=1.0)

    out = reproject(history, motion)

    assert out[0][2]["color"] == pytest.approx(10.0, abs=TOL)
    assert out[0][0]["color"] == pytest.approx(0.0, abs=TOL)
    assert out[0][1]["color"] == pytest.approx(0.0, abs=TOL)
    assert out[0][2]["weight"] == pytest.approx(1.0, abs=TOL)


def test_validate_history_uses_relative_depth_tolerance():
    # The near pair differs by 23% but only 0.03 units; the far pair
    # differs by 0.6 units but only 0.6%.  An absolute 0.5 threshold gets
    # both verdicts backwards.
    history = [[rec(0.0, 0.10, 1.0), rec(0.0, 100.0, 1.0)]]
    depth = [[0.13, 100.6]]
    motion = motion_grid(1, 2)

    usable = validate_history(history, depth, motion)

    assert usable[0][0] is False
    assert usable[0][1] is True


def test_blend_normalizes_pixel_weights():
    cur = [[10.0, 20.0, 10.0]]
    hist = [[rec(2.0), rec(2.0), rec(20.0)]]
    alpha = [[0.25, 0.5, 0.0]]

    out = blend(cur, hist, alpha)

    # 0.25 * current + 0.75 * history, per pixel; the two weights sum to 1.
    assert out[0][0] == pytest.approx(4.0, abs=TOL)
    assert out[0][1] == pytest.approx(11.0, abs=TOL)
    assert out[0][2] == pytest.approx(20.0, abs=TOL)


def test_accumulate_keeps_depth_consistent_with_blended_history():
    history = [[rec(0.0, 100.0, 1.0)]]
    cur = [[3.0]]
    depth = [[102.0]]
    motion = motion_grid(1, 1)
    config = {"frame_index": 3, "total_frames": 16}

    new_history, new_depth = accumulate(history, cur, motion, depth, config)

    # frame 3 has history weight 0.75: 0.75 * 100 + 0.25 * 102.
    assert new_depth[0][0] == pytest.approx(100.5, abs=TOL)
    assert new_history[0][0]["depth"] == pytest.approx(100.5, abs=TOL)
    assert new_history[0][0]["color"] == pytest.approx(0.75, abs=TOL)
    assert abs(new_depth[0][0] - 102.0) > 1e-6

    usable = validate_history(new_history, new_depth, motion)
    assert usable[0][0] is True


def test_converge_weight_is_an_accumulation_weight():
    weights = [converge_weight(i, 16) for i in range(17)]

    assert weights[0] == pytest.approx(0.0, abs=TOL)
    assert weights[1] == pytest.approx(0.5, abs=TOL)
    assert weights[3] == pytest.approx(0.75, abs=TOL)
    assert weights[15] == pytest.approx(15.0 / 16.0, abs=TOL)
    assert weights[16] == pytest.approx(1.0, abs=TOL)
    assert all(weights[i] < weights[i + 1] for i in range(16))
    assert converge_weight(100, 16) == pytest.approx(1.0, abs=TOL)


def test_reset_on_cut_clears_depth_as_well_as_color():
    history = [
        [rec(1.0, 2.0, 0.25), rec(3.0, 4.0, 0.5)],
        [rec(5.0, 6.0, 0.75), rec(7.0, 8.0, 1.0)],
    ]
    cut_flags = [[True, False], [False, True]]

    out = reset_on_cut(history, cut_flags)

    assert out[0][0] == rec(0.0, 0.0, 0.0)
    assert out[1][1] == rec(0.0, 0.0, 0.0)
    assert out[0][1] == rec(3.0, 4.0, 0.5)
    assert out[1][0] == rec(5.0, 6.0, 0.75)

    # The input state is not mutated.
    assert history[0][0]["depth"] == 2.0


def test_jitter_has_no_eight_frame_period():
    seq = [jitter(i, 16) for i in range(9)]

    assert len(set(seq[:8])) == 8
    assert seq[0] != seq[8], "an eight-frame cycle creates display beating"
    for dx, dy in seq:
        assert abs(dx) < 1.0
        assert abs(dy) < 1.0


def test_validate_state_rejects_bad_shape_and_non_finite_values():
    history = [[rec(0.0, 1.0, 1.0), rec(1.0, 2.0, 1.0)]]
    depth = [[1.0], [2.0]]

    errors = validate_state(history, depth)
    assert errors
    assert any("size" in error.lower() or "shape" in error.lower()
               for error in errors)

    nan_errors = validate_state([[rec(0.0, float("nan"), 1.0)]], [[1.0]])
    assert nan_errors
    assert any("nan" in error.lower() or "finite" in error.lower()
               for error in nan_errors)

    weight_errors = validate_state([[rec(0.0, 1.0, 1.25)]], [[1.0]])
    assert weight_errors
    assert any("weight" in error.lower() for error in weight_errors)


def test_accumulate_drops_invalid_history_pixels():
    # Historical depth 10 strongly disagrees with current depth 20.
    # Invalid history must contribute exactly zero, not a tiny ghost.
    history = [[rec(50.0, 10.0, 1.0), rec(50.0, 10.0, 1.0)]]
    cur = [[0.0, 0.0]]
    depth = [[20.0, 20.0]]
    motion = motion_grid(1, 2)
    config = {"frame_index": 3, "total_frames": 16}

    new_history, new_depth = accumulate(history, cur, motion, depth, config)

    for x in range(2):
        assert new_history[0][x]["color"] == pytest.approx(0.0, abs=TOL)
        assert new_history[0][x]["weight"] < 0.2
        assert new_history[0][x]["depth"] == pytest.approx(20.0, abs=TOL)
        assert new_depth[0][x] == pytest.approx(20.0, abs=TOL)


def test_static_sixteen_frames_converge_to_reference():
    # A static scene with deterministic sub-pixel exposure alternation
    # around the reference value 10.0.  Sixteen samples must average to
    # the reference, not drift toward the history or the latest sample.
    history = [[rec(0.0, 5.0, 0.0)]]
    motion = motion_grid(1, 1)
    new_depth = [[5.0]]

    for frame_index in range(16):
        value = 9.5 if frame_index % 2 == 0 else 10.5
        cur = [[value]]
        depth = [[5.0]]
        history, new_depth = accumulate(
            history,
            cur,
            motion,
            depth,
            {"frame_index": frame_index, "total_frames": 16},
        )

    assert history[0][0]["color"] == pytest.approx(10.0, abs=1e-6)
    assert new_depth[0][0] == pytest.approx(5.0, abs=1e-6)


def test_moving_region_does_not_keep_history_weight():
    history = [[rec(5.0, 10.0, 1.0), rec(5.0, 10.0, 1.0)]]
    cur = [[0.0, 0.0]]
    depth = [[10.0, 10.0]]
    motion = motion_grid(1, 2, dx=40.0)
    config = {"frame_index": 7, "total_frames": 16}

    new_history = accumulate(history, cur, motion, depth, config)[0]

    for x in range(2):
        assert new_history[0][x]["weight"] < 0.2
        assert new_history[0][x]["color"] == pytest.approx(0.0, abs=TOL)


def test_reproject_with_zero_motion_is_identity():
    history = [
        [rec(1.0, 5.0, 0.25), rec(2.0, 6.0, 0.5)],
        [rec(3.0, 7.0, 0.75), rec(4.0, 8.0, 1.0)],
    ]
    motion = motion_grid(2, 2)

    out = reproject(history, motion)

    for y in range(2):
        for x in range(2):
            assert out[y][x]["color"] == pytest.approx(
                history[y][x]["color"], abs=TOL)
            assert out[y][x]["depth"] == pytest.approx(
                history[y][x]["depth"], abs=TOL)
            assert out[y][x]["weight"] == pytest.approx(
                history[y][x]["weight"], abs=TOL)


def test_first_frame_after_cut_equals_current_frame():
    history = [[rec(7.0, 3.0, 1.0), rec(8.0, 4.0, 1.0)]]
    cut_flags = [[True, True]]
    cur = [[2.5, -1.0]]
    depth = [[4.0, 6.0]]
    motion = motion_grid(1, 2)

    reset = reset_on_cut(history, cut_flags)
    new_history, new_depth = accumulate(
        reset,
        cur,
        motion,
        depth,
        {"frame_index": 0, "total_frames": 16},
    )

    for x in range(2):
        assert new_history[0][x]["color"] == pytest.approx(
            cur[0][x], abs=TOL)
        assert new_history[0][x]["depth"] == pytest.approx(
            depth[0][x], abs=TOL)
        assert new_depth[0][x] == pytest.approx(depth[0][x], abs=TOL)


def test_jitter_is_subpixel_and_unique_within_eight_frames():
    seq = [jitter(i, 8) for i in range(8)]

    assert len(set(seq)) == 8
    for dx, dy in seq:
        assert abs(dx) < 1.0
        assert abs(dy) < 1.0
