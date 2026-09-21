"""
Stage 1: track the shooter across every frame of each clip.

No persistent-ID tracker (ByteTrack etc.) here on purpose - in basketball,
players make contact constantly, and identity-tracking algorithms tend to
swap IDs across those contacts. Instead this mirrors nba-shot-quality's
actual approach: seed a single bbox at a known-good frame, then every
subsequent frame independently re-matches to whichever detected box has
the highest IoU with the last known position - no identity is assumed to
persist, it's just re-derived fresh each frame. If nothing plausible is
nearby (IoU < threshold), the last position is held rather than jumping to
a wrong box. This is inherently robust to the mid-clip contact/occlusion
that breaks ID trackers, because there's no ID to break.

Only detects/tracks within [release_frame - WINDOW_BEFORE, release_frame +
WINDOW_AFTER] (default 20/20, ~0.8s each side at 25fps) - outside that
window the player isn't doing the shooting motion anyway, so there's
nothing meaningful to track there, and it was the source of a real problem:
the detector locking onto some unrelated player or a spot on the court
during dead time, with no way to tell from the preview that it didn't
matter. Frames outside the window get bbox=None (no detection/tracking
attempted, nothing drawn in the preview).

Seed preference per clip:
  1. manual seed(s) from clips/clips_manifest.csv (review_seed.py), if set -
     human-verified, so it's the most trustworthy seed. Supports multiple
     anchors (a primary plus optional extra correction points) - see
     track_player().
  2. release_frame from the manifest - searches outward from that frame
     for the nearest one with a player detection, in case the exact
     release frame has none (motion blur etc.)
  3. the jump-shot action classifier's own highest-confidence detection
     within the window
  4. player detection nearest the window's center

Usage:
    python pipeline/track_clips.py

Output, written next to each clip:
    <clip>_tracking.json   per-frame shooter bbox (None outside the window)
                            + anchor frame
    <clip>_preview.mp4     clip annotated with just the tracked bbox (no
                            other detected players drawn); review this (or
                            run review_clips.py) and discard/retrack any
                            clip where the box visibly drifts onto someone
                            else within the window
"""
from __future__ import annotations
import argparse
import csv
import glob
import json
import os
import traceback

import cv2
from ultralytics import YOLO

WEIGHTS = "models/action_classifier/weights/best.pt"
CLIPS_DIRS = ["clips/pull_up", "clips/catch_and_shoot"]
TRACKS_DIRS = {"clips/pull_up": "tracks/pull_up", "clips/catch_and_shoot": "tracks/catch_and_shoot"}
MANIFEST = "clips/clips_manifest.csv"

JUMP_CLASS = 6
PLAYER_CLASS = 3
CONF_THRESH = 0.25
IOU_THRESH = 0.25

WINDOW_BEFORE = 20  # frames before release_frame to track (~0.8s @ 25fps)
WINDOW_AFTER = 20   # frames after release_frame to track

MAX_JUMP_DIAGONALS = 0.9  # max center movement per frame, in box-diagonals
                          # of the previous frame's box - rejects snapping
                          # onto a spatially separate person even if their
                          # box clears the IoU bar

MAX_INTERP_GAP = 3  # frames: a "held" run this short or shorter, bounded by
                    # real matches on both sides, gets linearly interpolated
                    # instead of frozen (basketball motion is smooth enough
                    # over ~3 frames / 0.12s for a straight line to be a
                    # good approximation regardless of the actual motion)


def iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0
    area = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / area


def scan_clip(clip_path: str, model: YOLO, window: tuple[int, int] | None = None):
    """Detect frames within `window` (inclusive), or the whole clip if
    window is None. Returns (frame_dets, fps, n_frames_total, (win_start, win_end))."""
    cap = cv2.VideoCapture(clip_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if window is not None:
        win_start = max(0, window[0])
        win_end = min(n_frames - 1, window[1])
    else:
        win_start, win_end = 0, n_frames - 1

    cap.set(cv2.CAP_PROP_POS_FRAMES, win_start)
    frame_dets = {}
    f = win_start
    while f <= win_end:
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame, conf=CONF_THRESH, verbose=False, device="mps")[0]
        players, jump_shot = [], None
        for box in results.boxes:
            cls = int(box.cls[0])
            bbox = list(map(int, box.xyxy[0]))
            conf = float(box.conf[0])
            if cls == PLAYER_CLASS:
                players.append(bbox)
            elif cls == JUMP_CLASS:
                if jump_shot is None or conf > jump_shot["conf"]:
                    jump_shot = {"bbox": bbox, "conf": round(conf, 3)}
        frame_dets[f] = {"players": players, "jump_shot": jump_shot}
        f += 1
    cap.release()
    return frame_dets, fps, n_frames, (win_start, win_end)


def nearest_player_detection(frame_dets: dict, center: int, win_start: int, win_end: int):
    """Search outward from `center` (clamped to the window) for the
    nearest frame with any player detection. Returns (frame, bbox) or (None, None)."""
    span = win_end - win_start
    for offset in range(span + 1):
        for f in (center - offset, center + offset):
            if f < win_start or f > win_end:
                continue
            d = frame_dets.get(f)
            if d and d["players"]:
                box = max(d["players"], key=lambda b: (b[2]-b[0])*(b[3]-b[1]))
                return f, box
    return None, None


def pick_anchor(frame_dets: dict, win_start: int, win_end: int,
                release_frame: int | None = None) -> tuple[int, list, str]:
    """Seed preference: (1) nearest player detection to release_frame, if
    given - release_frame is human-verified, so it's the most trustworthy
    seed point even if that exact frame has no detection (motion blur etc.);
    (2) best jump-shot classifier detection within the window; (3) player
    detection nearest the window's center."""
    if release_frame is not None:
        f, box = nearest_player_detection(frame_dets, release_frame, win_start, win_end)
        if f is not None:
            return f, box, "release_frame_seed"

    best_frame, best = None, None
    for f, d in frame_dets.items():
        js = d.get("jump_shot")
        if js and (best is None or js["conf"] > best["conf"]):
            best, best_frame = js, f
    if best is not None:
        return best_frame, best["bbox"], "classifier"

    center = (win_start + win_end) // 2
    f, box = nearest_player_detection(frame_dets, center, win_start, win_end)
    if f is not None:
        return f, box, "fallback_player_near_center"
    return center, [0, 0, 1, 1], "fallback_none_found"


def _candidates(frame_dets, f):
    d = frame_dets.get(f, {})
    c = list(d.get("players", []))
    js = d.get("jump_shot")
    if js:
        c.append(js["bbox"])
    return c


def _snap_to_anchor(frame_dets, anchor_frame, anchor_bbox):
    """Anchor frames snap to the closest real detection if one's close
    enough, same as any other frame - a manually-drawn box might not
    exactly match the detector's own box for the same person."""
    cands = _candidates(frame_dets, anchor_frame)
    if cands:
        best = max(cands, key=lambda b: iou(b, anchor_bbox))
        if iou(best, anchor_bbox) >= IOU_THRESH:
            return best
    return anchor_bbox


def _center(b):
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)


def _diagonal(b):
    return ((b[2] - b[0]) ** 2 + (b[3] - b[1]) ** 2) ** 0.5


def _center_dist(a, b):
    ax, ay = _center(a)
    bx, by = _center(b)
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def _propagate(frame_dets, start_bbox, frames_in_order):
    """Greedy IoU matching frame-by-frame starting from start_bbox, over
    frames_in_order (any direction/order). A candidate must both clear the
    IoU threshold AND not require an implausibly large jump from the last
    known position (capped at MAX_JUMP_DIAGONALS box-diagonals - scaled to
    the player's current apparent size so it works whether he's close to
    camera or far away) - this is what stops the box snapping onto a
    nearby-but-separate person (a coach, another player) just because
    their box happens to clear the IoU bar against a stale held position.
    Rejected candidates are treated the same as "no candidate": hold the
    last position rather than jump (short holds get linearly interpolated
    later in track_player, once both ends of the hold are known).

    Returns (tracked, is_direct) - is_direct[f] is True only for frames
    where a real detection was actually matched this frame, not held."""
    tracked = {}
    is_direct = {}
    last = start_bbox
    for f in frames_in_order:
        cands = _candidates(frame_dets, f)
        matched = False
        if cands:
            best = max(cands, key=lambda b: iou(b, last))
            if iou(best, last) >= IOU_THRESH:
                max_jump = _diagonal(last) * MAX_JUMP_DIAGONALS
                if _center_dist(best, last) <= max_jump:
                    last = best
                    matched = True
        tracked[f] = last
        is_direct[f] = matched
    return tracked, is_direct


def _interpolate_short_holds(tracked, is_direct, win_start, win_end, max_gap=MAX_INTERP_GAP):
    """A 'hold' is a run of consecutive frames where propagation couldn't
    find (or rejected) a match, so it just repeated the last known box -
    visually that's the box freezing while the player keeps moving, then
    snapping forward once a match resumes. Basketball motion is smooth
    enough that a short hold (<= max_gap frames) bounded by real matches on
    both sides is well approximated by a straight line between them, so
    replace those holds with a linear interpolation instead of a freeze.
    Holds at the window edge (no real match on one side) or longer than
    max_gap are left as-is - nothing reliable to interpolate towards."""
    f = win_start
    while f <= win_end:
        if is_direct.get(f, False):
            f += 1
            continue
        gap_start = f
        while f <= win_end and not is_direct.get(f, False):
            f += 1
        gap_end = f - 1
        gap_len = gap_end - gap_start + 1
        left, right = gap_start - 1, gap_end + 1
        if (gap_len <= max_gap and left >= win_start and right <= win_end
                and is_direct.get(left, False) and is_direct.get(right, False)):
            b_left, b_right = tracked[left], tracked[right]
            for i, gf in enumerate(range(gap_start, gap_end + 1), start=1):
                t = i / (gap_len + 1)
                tracked[gf] = [round(b_left[j] + t * (b_right[j] - b_left[j])) for j in range(4)]
    return tracked


def track_player(frame_dets: dict, anchors: list[tuple[int, list]],
                 win_start: int, win_end: int) -> tuple[dict, dict]:
    """Frame-independent greedy IoU matching - no identity tracker, every
    frame is re-derived fresh from whichever detected box has the highest
    IoU with the last known position (never assumed to persist an identity).
    Only tracks within [win_start, win_end]; anchors outside that range are
    ignored (nothing outside the window is tracked at all).

    Supports multiple anchors (a primary seed plus optional extra
    correction anchors added where a single seed was seen to drift):
    each frame between two consecutive anchors is propagated from
    whichever of the two is nearer, roughly halving the maximum
    propagation distance - and therefore the drift risk - within that
    span versus dragging one seed across the whole gap. Frames before the
    first anchor / after the last anchor propagate from that single
    nearest anchor, same as the original single-anchor behavior.

    Short held (undetected) stretches get linearly interpolated - see
    _interpolate_short_holds(). Returns (tracked, is_direct): is_direct[f]
    is False for frames whose bbox is a hold/interpolation rather than an
    actual anchor or matched detection.
    """
    anchors = sorted((a for a in anchors if win_start <= a[0] <= win_end), key=lambda a: a[0])
    if not anchors:
        return {}, {}

    tracked = {}
    is_direct = {}
    for f, bbox in anchors:
        tracked[f] = _snap_to_anchor(frame_dets, f, bbox)
        is_direct[f] = True  # anchors are human-verified, always trusted

    def merge(t, d):
        tracked.update(t)
        is_direct.update(d)

    first_f = anchors[0][0]
    merge(*_propagate(frame_dets, tracked[first_f], range(first_f - 1, win_start - 1, -1)))

    last_f = anchors[-1][0]
    merge(*_propagate(frame_dets, tracked[last_f], range(last_f + 1, win_end + 1)))

    for (f1, _), (f2, _) in zip(anchors, anchors[1:]):
        mid = (f1 + f2) // 2
        merge(*_propagate(frame_dets, tracked[f1], range(f1 + 1, mid + 1)))
        merge(*_propagate(frame_dets, tracked[f2], range(f2 - 1, mid, -1)))

    tracked = _interpolate_short_holds(tracked, is_direct, win_start, win_end)
    return tracked, is_direct


def process_clip(clip_path: str, model: YOLO, release_frame: int | None,
                 manual_seed: tuple[int, list] | None = None,
                 extra_anchors: list[tuple[int, list]] | None = None) -> None:
    stem = os.path.splitext(os.path.basename(clip_path))[0]
    clip_dir = os.path.dirname(clip_path)
    out_dir = TRACKS_DIRS.get(clip_dir, clip_dir)
    os.makedirs(out_dir, exist_ok=True)
    tracking_path = os.path.join(out_dir, f"{stem}_tracking.json")
    n_extra_now = len(extra_anchors or [])
    if os.path.exists(tracking_path):
        with open(tracking_path, encoding="utf-8") as fh:
            existing = json.load(fh)
        already_manual = existing.get("anchor_source") == "manual_seed"
        same_extras = existing.get("num_extra_anchors", 0) == n_extra_now
        if manual_seed is None or (already_manual and same_extras):
            print(f"[skip] {stem}: already tracked")
            return
        print(f"{stem}: re-tracking with manual seed ({n_extra_now} extra anchor(s))")
        os.remove(tracking_path)
        preview_path = os.path.join(out_dir, f"{stem}_preview.mp4")
        if os.path.exists(preview_path):
            os.remove(preview_path)

    preview_path = os.path.join(out_dir, f"{stem}_preview.mp4")
    cap = None
    writer = None
    try:
        center = release_frame if release_frame is not None else (manual_seed[0] if manual_seed else None)
        window = (center - WINDOW_BEFORE, center + WINDOW_AFTER) if center is not None else None

        frame_dets, fps, n_frames, (win_start, win_end) = scan_clip(clip_path, model, window)
        if n_frames == 0:
            raise RuntimeError("no frames could be read from this clip")

        if manual_seed is not None:
            anchor, anchor_bbox, anchor_source = manual_seed[0], manual_seed[1], "manual_seed"
            anchors = [(anchor, anchor_bbox)] + list(extra_anchors or [])
        else:
            anchor, anchor_bbox, anchor_source = pick_anchor(frame_dets, win_start, win_end, release_frame)
            anchors = [(anchor, anchor_bbox)]
        anchor = max(win_start, min(anchor, win_end))
        anchors = [(max(win_start, min(f, win_end)), b) for f, b in anchors]

        tracked, is_direct = track_player(frame_dets, anchors, win_start, win_end)
        if not tracked:
            # no anchor landed inside the window (e.g. all clicks were far
            # from release_frame) - fall back to a fresh in-window pick
            anchor, anchor_bbox, anchor_source = pick_anchor(frame_dets, win_start, win_end, release_frame)
            anchors = [(anchor, anchor_bbox)]
            tracked, is_direct = track_player(frame_dets, anchors, win_start, win_end)

        cap = cv2.VideoCapture(clip_path)
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(preview_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

        frames_meta = []
        f = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            preview = frame.copy()

            bbox = tracked.get(f)
            frame_is_direct = is_direct.get(f, False)
            is_shot = frame_dets.get(f, {}).get("jump_shot") is not None
            if bbox:
                is_anchor = f == anchor or any(f == af for af, _ in anchors)
                if is_anchor:
                    color = (0, 255, 0)
                elif frame_is_direct:
                    color = (0, 165, 255)
                else:
                    color = (255, 0, 255)  # magenta = interpolated/held, not a real detection this frame
                cv2.rectangle(preview, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 3)
                label = "JUMP SHOT" if is_shot else ("tracking" if frame_is_direct else "interpolated")
                cv2.putText(preview, label, (bbox[0], max(bbox[1] - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            if f == anchor:
                marker = " <-- ANCHOR"
            elif any(f == af for af, _ in anchors):
                marker = " <-- EXTRA ANCHOR"
            elif win_start <= f <= win_end:
                marker = ""
            else:
                marker = " (outside window)"
            cv2.putText(preview, f"{stem}  frame {f}{marker}",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            writer.write(preview)
            frames_meta.append({
                "frame": f, "sec": round(f / fps, 3),
                "bbox": bbox, "jump_shot_detected": is_shot,
                "bbox_is_direct": frame_is_direct,
            })
            f += 1
        cap.release()
        writer.release()
        cap = writer = None

        tracking = {
            "clip": os.path.basename(clip_path),
            "fps": fps,
            "clip_start_frame": 0,
            "clip_end_frame": n_frames - 1,
            "window_start": win_start,
            "window_end": win_end,
            "anchor_frame": anchor,
            "anchor_sec": round(anchor / fps, 2),
            "anchor_source": anchor_source,
            "release_frame": release_frame,
            "num_extra_anchors": len(anchors) - 1,
            "extra_anchor_frames": [af for af, _ in anchors if af != anchor],
            "frames": frames_meta,
        }
        with open(tracking_path, "w", encoding="utf-8") as fh:
            json.dump(tracking, fh, indent=2)

        print(f"{stem}: anchor=frame {anchor} ({anchor_source}), "
              f"window=[{win_start},{win_end}] of {n_frames} frames tracked")

    except Exception as e:
        print(f"[ERROR] {stem}: {type(e).__name__}: {e} -- skipping this clip")
        traceback.print_exc()
        if cap is not None:
            cap.release()
        if writer is not None:
            writer.release()
        for p in (tracking_path, preview_path):
            if os.path.exists(p):
                os.remove(p)


def load_release_frames(manifest_path: str) -> dict[str, int]:
    if not os.path.exists(manifest_path):
        return {}
    with open(manifest_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out = {}
    for r in rows:
        rf = r.get("release_frame", "").strip()
        if rf.isdigit():
            out[r["clip_filename"]] = int(rf)
    return out


def load_manual_seeds(manifest_path: str) -> dict[str, tuple[int, list]]:
    """Per-clip manual seed bbox from review_seed.py, if set - takes
    priority over everything else since it's human-verified to be the
    correct player, not just a heuristic guess."""
    if not os.path.exists(manifest_path):
        return {}
    with open(manifest_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out = {}
    for r in rows:
        x1 = r.get("seed_x1", "").strip()
        if not x1:
            continue
        frame = int(r["seed_frame"])
        bbox = [int(r["seed_x1"]), int(r["seed_y1"]), int(r["seed_x2"]), int(r["seed_y2"])]
        out[r["clip_filename"]] = (frame, bbox)
    return out


def load_extra_anchors(manifest_path: str) -> dict[str, list[tuple[int, list]]]:
    """Extra correction anchors from review_seed.py, per clip - see
    track_player() for how these reduce mid-clip drift."""
    if not os.path.exists(manifest_path):
        return {}
    with open(manifest_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out = {}
    for r in rows:
        raw = r.get("extra_seeds", "").strip()
        if not raw:
            continue
        parsed = json.loads(raw)
        if parsed:
            out[r["clip_filename"]] = [(e[0], e[1:]) for e in parsed]
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=WEIGHTS)
    parser.add_argument("--manifest", default=MANIFEST)
    args = parser.parse_args()

    model = YOLO(args.weights)
    model.to("mps")

    release_frames = load_release_frames(args.manifest)
    manual_seeds = load_manual_seeds(args.manifest)
    extra_anchors_all = load_extra_anchors(args.manifest)
    print(f"{len(release_frames)} clips have a release_frame in {args.manifest}")
    print(f"{len(manual_seeds)} clips have a manual seed bbox (from review_seed.py)")
    print(f"{len(extra_anchors_all)} clips have extra correction anchors")
    print(f"tracking window: release_frame -{WINDOW_BEFORE}/+{WINDOW_AFTER} frames")

    clip_paths = sorted(
        p for d in CLIPS_DIRS for p in glob.glob(os.path.join(d, "*.mp4"))
        if "_preview" not in p and "_pose" not in p
    )
    print(f"{len(clip_paths)} clips to track\n")

    failed = []
    for clip_path in clip_paths:
        fn = os.path.basename(clip_path)
        try:
            release_frame = release_frames.get(fn)
            manual_seed = manual_seeds.get(fn)
            extra_anchors = extra_anchors_all.get(fn)
            process_clip(clip_path, model, release_frame, manual_seed, extra_anchors)
        except Exception as e:
            print(f"[ERROR] {fn}: {type(e).__name__}: {e} -- skipping this clip")
            traceback.print_exc()
            failed.append(fn)

    print("\nDone.")
    if failed:
        print(f"{len(failed)} clip(s) failed and were skipped:")
        for fn in failed:
            print(" ", fn)


if __name__ == "__main__":
    main()
