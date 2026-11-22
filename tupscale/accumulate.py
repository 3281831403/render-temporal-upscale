"""Temporal accumulation primitives for a deterministic temporal upscaler.

Grid convention
---------------
Every image-like argument is a row-major ``list[list[...]]`` with shape
``(height, width)``.  Current colours are floats.  Depth is a grid of
positive floats.  Motion is a grid of ``(dx, dy)`` displacement pairs in
pixels.  History is a grid of records with exactly three float fields::

    {"color": ..., "depth": ..., "weight": ...}

``weight`` is a per-pixel confidence in the closed interval ``[0, 1]``.
Colours and depths are not premultiplied.  All functions are pure and do
not mutate their arguments.  Floating-point comparisons use tolerance
``1e-9`` unless a contract says otherwise.

Motion convention
-----------------
``motion[y][x] == (dx, dy)`` is the displacement of the scene point seen
at output pixel ``(x, y)`` from the previous frame to the current frame.
To bring history forward, sample the previous frame at ``(x - dx, y - dy)``
(bilinear reverse sampling), never at ``(x + dx, y + dy)``.
"""

import math

__all__ = [
    "jitter",
    "reproject",
    "validate_history",
    "blend",
    "accumulate",
    "reset_on_cut",
    "converge_weight",
    "validate_state",
]

# Validation thresholds shared by validate_history and accumulate.
DEPTH_REL_TOL = 0.05
MOTION_LIMIT = 32.0
WEIGHT_EPS = 1e-6
TOL = 1e-9


def jitter(index: int, samples: int) -> tuple:
    """Return one deterministic sub-pixel jitter offset.

    ``index`` is a zero-based frame index and ``samples`` is the number of
    jitter phases used by the caller (``samples >= 1``).  The offset is
    generated from the 2-D Halton sequence with bases 2 and 3, mapped to
    signed pixel units.  Its amplitude is strictly below one pixel:
    ``abs(dx) < 1`` and ``abs(dy) < 1``.  The sequence must not collapse
    to a fixed point and must not repeat with a period of eight frames.

    The result is a ``(dx, dy)`` tuple of floats.  Compare offsets with
    tolerance ``1e-9``.  Raise ``ValueError`` for ``index < 0`` or
    ``samples < 1``.
    """
    if index < 0:
        raise ValueError("index must be non-negative")
    if samples < 1:
        raise ValueError("samples must be positive")

    # Radical-inverse (van der Corput) sequences in bases 2 and 3.  The
    # global frame index is offset by one so that frame zero still gets a
    # non-degenerate phase.  Halton phases are aperiodic for all practical
    # frame counts, so there is no eight-frame cycle to beat against a
    # 60 Hz display, and the whole sequence is deterministic.
    def _van_der_corput(n: int, base: int) -> float:
        result = 0.0
        denom = 1.0
        while n > 0:
            n, digit = divmod(n, base)
            denom *= base
            result += digit / denom
        return result

    hx = _van_der_corput(index + 1, 2)
    hy = _van_der_corput(index + 1, 3)
    # Map (0, 1) onto (-1, 1); the endpoints are never reached, so the
    # amplitude is strictly below one pixel.
    return (2.0 * hx - 1.0, 2.0 * hy - 1.0)


def reproject(history: list, motion: list) -> list:
    """Move history forward by reverse-sampling motion vectors.

    For every output pixel ``(x, y)`` with ``motion[y][x] == (dx, dy)``,
    the source coordinate is ``(x - dx, y - dy)``.  Sample the four
    neighbouring history records bilinearly by their pixel centres.
    Neighbours outside the grid contribute nothing; do not clamp them.

    Each output record stores the bilinearly reconstructed colour and
    depth, while its weight is the bilinear sum of the source weights.
    Colour and depth are weighted by both the bilinear coefficient and
    the source weight, and are normalised by the reconstructed weight.
    An empty reconstruction is ``{"color": 0.0, "depth": 0.0,
    "weight": 0.0}``.  A zero motion field must reproduce the input
    history exactly to tolerance ``1e-9``.  All weights stay in ``[0, 1]``.
    """
    height = len(history)
    width = len(history[0]) if height else 0

    def _lookup(sx: float, sy: float) -> tuple:
        """Bilinearly sample history at (sx, sy); out-of-grid is empty."""
        x0 = math.floor(sx)
        y0 = math.floor(sy)
        fx = sx - x0
        fy = sy - y0
        color_acc = 0.0
        depth_acc = 0.0
        weight_acc = 0.0
        for oy, wy in ((y0, 1.0 - fy), (y0 + 1, fy)):
            if wy == 0.0:
                continue
            if oy < 0 or oy >= height:
                continue  # do not clamp: outside contributes nothing
            row = history[oy]
            if x0 >= 0 and x0 < width and (1.0 - fx) != 0.0:
                rec = row[x0]
                w = (1.0 - fx) * wy
                wc = rec["weight"]
                color_acc += w * wc * rec["color"]
                depth_acc += w * wc * rec["depth"]
                weight_acc += w * wc
            if fx != 0.0 and x0 + 1 >= 0 and x0 + 1 < width:
                rec = row[x0 + 1]
                w = fx * wy
                wc = rec["weight"]
                color_acc += w * wc * rec["color"]
                depth_acc += w * wc * rec["depth"]
                weight_acc += w * wc
        if weight_acc <= 0.0:
            return {"color": 0.0, "depth": 0.0, "weight": 0.0}
        return {
            "color": color_acc / weight_acc,
            "depth": depth_acc / weight_acc,
            "weight": weight_acc,
        }

    out = []
    for y in range(height):
        out_row = []
        for x in range(width):
            dx, dy = motion[y][x]
            # Reverse sampling: fetch the previous frame at (x - dx, y - dy).
            sample = _lookup(x - dx, y - dy)
            out_row.append({
                "color": sample["color"],
                "depth": sample["depth"],
                "weight": sample["weight"],
            })
        out.append(out_row)
    return out


def validate_history(history: list, depth: list, motion: list) -> list:
    """Return a boolean usability mask for reprojected history.

    Each output cell is ``True`` only when all of these hold:

    * the history weight is greater than ``1e-6``;
    * the history depth, current depth and motion are finite, and both
      depths are positive;
    * ``abs(history_depth - current_depth) <=
      0.05 * max(history_depth, current_depth) + 1e-6`` -- a relative,
      not absolute, depth test;
    * ``hypot(dx, dy) <= 32.0`` pixels (gross motion is unusable).

    The comparison tolerance is ``1e-9``.  Weights and booleans are not
    floating-point colours, so no extra normalisation is performed here.
    """
    out = []
    for y, hist_row in enumerate(history):
        row = []
        for x, rec in enumerate(hist_row):
            hw = rec["weight"]
            hd = rec["depth"]
            cd = depth[y][x]
            dx, dy = motion[y][x]
            ok = True
            if not (hw > WEIGHT_EPS):
                ok = False
            if not (math.isfinite(hw) and math.isfinite(hd)
                    and math.isfinite(cd) and math.isfinite(dx)
                    and math.isfinite(dy)):
                ok = False
            if not (hd > 0.0 and cd > 0.0):
                ok = False
            # Relative depth test: scales correctly for near and far pixels.
            if abs(hd - cd) > DEPTH_REL_TOL * max(hd, cd) + 1e-6:
                ok = False
            if math.hypot(dx, dy) > MOTION_LIMIT:
                ok = False
            row.append(ok)
        out.append(row)
    return out


def blend(cur: list, hist: list, alpha_map: list) -> list:
    """Blend current colours with history colours per pixel.

    ``alpha_map[y][x]`` is the current-frame weight in ``[0, 1]``; the
    history contributes exactly ``1 - alpha`` at that pixel.  The result
    is ``alpha * cur + (1 - alpha) * hist.color``.  The two weights must
    be normalised per pixel: never add both at full strength and never
    let their sum drift away from one.  Floats are compared with
    tolerance ``1e-9``.
    """
    out = []
    for y, cur_row in enumerate(cur):
        row = []
        for x, c in enumerate(cur_row):
            a = alpha_map[y][x]
            # Per-pixel normalisation: the two weights sum to exactly one.
            row.append(a * c + (1.0 - a) * hist[y][x]["color"])
        out.append(row)
    return out


def accumulate(history: list, cur: list, motion: list, depth: list,
               config: dict) -> tuple:
    """Accumulate one current frame into history and return a state pair.

    ``config`` contains ``"frame_index"`` (zero-based) and
    ``"total_frames"`` (positive).  The historical blend weight comes from
    :func:`converge_weight`; the current-frame weight is its complement.
    The pipeline is: reproject, validate, blend colours, blend depths,
    then write a new history record and the matching depth grid.

    A usable pixel keeps the reprojected depth in its history record and
    blends it with the current depth using the same weights as the colour,
    so colour, depth and weight always describe the same state.  Do not
    replace that accumulated depth with the raw current depth.  An
    unusable, already-initialised pixel is seeded from the current frame
    with zero history weight, so it cannot leak a ghost into the blend.
    The first frame initialises every pixel from the current frame.

    Returns ``(new_history, new_depth)``.  The depth stored inside each
    new history record must equal the corresponding entry of
    ``new_depth`` to tolerance ``1e-9``.  All weights are in ``[0, 1]``
    and the returned grids keep the input shape.
    """
    frame_index = config["frame_index"]
    total_frames = config["total_frames"]
    hist_weight = converge_weight(frame_index, total_frames)
    cur_weight = 1.0 - hist_weight

    reprojected = reproject(history, motion)
    usable = validate_history(reprojected, depth, motion)

    new_history = []
    new_depth = []
    for y in range(len(cur)):
        hist_row = []
        depth_row = []
        for x in range(len(cur[y])):
            c = cur[y][x]
            d = depth[y][x]
            h = reprojected[y][x]
            # A record with no stored weight has never been initialised
            # (first frame, or just after a camera cut).
            was_initialised = history[y][x]["weight"] > WEIGHT_EPS
            if usable[y][x]:
                # Same weights for colour, depth and weight so the three
                # pieces of history state stay in sync.
                new_color = cur_weight * c + hist_weight * h["color"]
                new_depth_value = cur_weight * d + hist_weight * h["depth"]
                hist_weight_out = 1.0
            elif not was_initialised:
                # Uninitialised pixel: seed it fully from the current frame
                # so the next frame has usable history to accumulate onto.
                new_color = c
                new_depth_value = d
                hist_weight_out = 1.0
            else:
                # Invalid, already-initialised history contributes exactly
                # zero: seed from the current frame with no history weight,
                # so no ghost survives into the blend.
                new_color = c
                new_depth_value = d
                hist_weight_out = 0.0
            hist_row.append({
                "color": new_color,
                "depth": new_depth_value,
                "weight": hist_weight_out,
            })
            depth_row.append(new_depth_value)
        new_history.append(hist_row)
        new_depth.append(depth_row)
    return new_history, new_depth


def reset_on_cut(history: list, cut_flags: list) -> list:
    """Reset complete pixel history at camera cuts.

    ``cut_flags`` is a boolean grid.  Where it is ``True``, the returned
    record is exactly ``{"color": 0.0, "depth": 0.0, "weight": 0.0}``:
    colour, depth and weight are reset together.  Everywhere else the
    record is preserved.  Return a new grid; do not mutate ``history``.
    Floating-point values use tolerance ``1e-9`` and weights stay in
    ``[0, 1]``.
    """
    out = []
    for y, hist_row in enumerate(history):
        row = []
        for x, rec in enumerate(hist_row):
            if cut_flags[y][x]:
                # Reset colour, depth and weight together: a record whose
                # depth survives a cut poisons the next frame's validation.
                row.append({"color": 0.0, "depth": 0.0, "weight": 0.0})
            else:
                row.append({
                    "color": rec["color"],
                    "depth": rec["depth"],
                    "weight": rec["weight"],
                })
        out.append(row)
    return out


def converge_weight(frame_index: int, total: int) -> float:
    """Return the accumulated history weight for a frame.

    This is the history weight used by :func:`accumulate`, not a factor
    to multiply directly into a colour.  It starts at ``0.0`` for frame
    zero and grows as ``1 - 1 / (frame_index + 1)`` up to the end of the
    convergence window; at ``frame_index == total`` it reaches ``1.0``
    and stays capped there.  The value is always in ``[0, 1]``; compare
    it with tolerance ``1e-9``.  Raise ``ValueError`` for a negative
    frame index or a non-positive total.
    """
    if frame_index < 0:
        raise ValueError("frame_index must be non-negative")
    if total <= 0:
        raise ValueError("total must be positive")
    if frame_index >= total:
        return 1.0
    return 1.0 - 1.0 / (frame_index + 1)


def validate_state(history: list, depth: list) -> list[str]:
    """Return a list of problems with a history/depth state.

    An empty list means the state is valid.  At minimum check that both
    grids are non-empty and rectangular, that their shapes are identical,
    that every colour/depth/weight is finite (no NaN or infinity), that
    every weight lies in ``[0, 1]``, and that the finite sum of weights
    does not exceed the number of pixels.  Each problem is a short
    human-readable string; size/shape errors contain ``"size"`` or
    ``"shape"``, non-finite errors contain ``"NaN"`` or ``"finite"``, and
    weight errors contain ``"weight"``.  Values are checked directly;
    the reported diagnostic is not a floating-point colour, so a
    tolerance of ``1e-9`` is only relevant at the weight boundaries.
    """
    problems = []

    if not history:
        problems.append("history is empty: size must be non-zero")
        return problems
    if not depth:
        problems.append("depth is empty: size must be non-zero")
        return problems

    hist_h = len(history)
    hist_w = len(history[0]) if hist_h else 0
    depth_h = len(depth)
    depth_w = len(depth[0]) if depth_h else 0

    for y, row in enumerate(history):
        if len(row) != hist_w:
            problems.append(f"history row {y} has wrong size")
            break
    for y, row in enumerate(depth):
        if len(row) != depth_w:
            problems.append(f"depth row {y} has wrong size")
            break
    if not problems and (hist_h, hist_w) != (depth_h, depth_w):
        problems.append(
            f"history shape {hist_h}x{hist_w} does not match "
            f"depth shape {depth_h}x{depth_w}: size/shape mismatch")

    for y, row in enumerate(history):
        for x, rec in enumerate(row):
            for key in ("color", "depth", "weight"):
                value = rec[key]
                if not math.isfinite(value):
                    problems.append(
                        f"history[{y}][{x}].{key} is not finite (NaN/inf)")
            w = rec["weight"]
            if math.isfinite(w) and not (-TOL <= w <= 1.0 + TOL):
                problems.append(
                    f"history[{y}][{x}] weight {w} outside [0, 1]")

    for y, row in enumerate(depth):
        for x, value in enumerate(row):
            if not math.isfinite(value):
                problems.append(f"depth[{y}][{x}] is not finite (NaN/inf)")

    if not problems:
        total_weight = sum(rec["weight"] for row in history for rec in row)
        pixels = hist_h * hist_w
        if total_weight > pixels + TOL:
            problems.append(
                f"weight sum {total_weight} exceeds pixel count {pixels}")

    return problems
