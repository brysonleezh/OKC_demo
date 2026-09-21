"""
Stage 2.5: review tracked clips before spending pose-estimation time on
them. Plays each <clip>_preview.mp4 (bbox overlay) in a window.

Controls:
  k          keep, advance to next
  d          delete (clip is bad - wrong angle, unusable), advance to next
  t          mark for retrack (clip is fine, tracking just went wrong -
             wipes this clip's tracking output + manifest seed so
             review_seed.py will prompt for it again, then track_clips.py
             will retrack it; the source clip itself is kept)
  u          undo last mark (delete or retrack), stays on current clip
  r          replay from start
  q / ESC    quit (progress saved, resumes next run)

Skips already-reviewed clips automatically. Use --reset to start fresh.

Usage:
    python pipeline/review_clips.py
    python pipeline/review_clips.py --reset
"""
from __future__ import annotations
import argparse
import csv
import glob
import json
import os
import subprocess

import cv2
import numpy as np

CLIPS_DIRS = ["clips/pull_up", "clips/catch_and_shoot"]
TRACKS_DIRS = {"clips/pull_up": "tracks/pull_up", "clips/catch_and_shoot": "tracks/catch_and_shoot"}
REVIEW_STATE = "clips/review_state.json"
MANIFEST = "clips/clips_manifest.csv"
SEED_FIELDS = ["seed_frame", "seed_x1", "seed_y1", "seed_x2", "seed_y2", "extra_seeds"]

DISPLAY_W, DISPLAY_H = 960, 540
WINDOW = "Clip Review"
FONT = cv2.FONT_HERSHEY_SIMPLEX
RED, ORANGE, WHITE, GRAY = (50, 50, 220), (0, 140, 255), (255, 255, 255), (160, 160, 160)


def load_state() -> dict:
    if os.path.exists(REVIEW_STATE):
        return json.load(open(REVIEW_STATE, encoding="utf-8"))
    return {"reviewed_clips": []}


def save_state(state: dict) -> None:
    with open(REVIEW_STATE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def load_frames(path: str) -> tuple[list[np.ndarray], int]:
    probe = subprocess.run(
        ["ffprobe", "-v", "quiet", "-select_streams", "v:0",
         "-show_entries", "stream=r_frame_rate", "-of", "default=noprint_wrappers=1:nokey=1", path],
        capture_output=True, text=True,
    )
    try:
        num, den = probe.stdout.strip().split("/")
        fps = int(num) // int(den)
    except Exception:
        fps = 25

    proc = subprocess.run(
        ["ffmpeg", "-i", path, "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-vf", f"scale={DISPLAY_W}:{DISPLAY_H}", "-loglevel", "quiet", "pipe:1"],
        capture_output=True,
    )
    raw = proc.stdout
    size = DISPLAY_W * DISPLAY_H * 3
    frames = [
        np.frombuffer(raw[off:off + size], dtype=np.uint8).reshape(DISPLAY_H, DISPLAY_W, 3).copy()
        for off in range(0, len(raw) - size + 1, size)
    ]
    return frames, fps


def draw_overlay(frame, i, total, clip_stem, mark, n_deleted, n_retrack):
    h, w = frame.shape[:2]
    out = frame.copy()
    cv2.rectangle(out, (0, 0), (w, 52), (20, 20, 20), -1)
    cv2.putText(out, f"[{i+1}/{total}]", (10, 36), FONT, 0.9, GRAY, 2)
    cv2.putText(out, clip_stem, (110, 36), FONT, 0.6, WHITE, 1)
    if mark is not None:
        color = RED if mark == 'delete' else ORANGE
        badge = f" DELETE ({n_deleted}) " if mark == 'delete' else f" RETRACK ({n_retrack}) "
        (bw, _), _ = cv2.getTextSize(badge, FONT, 0.65, 2)
        cv2.rectangle(out, (w - bw - 16, 10), (w - 6, 46), color, -1)
        cv2.putText(out, badge, (w - bw - 12, 36), FONT, 0.65, WHITE, 2)
    cv2.rectangle(out, (0, h - 38), (w, h), (20, 20, 20), -1)
    cv2.putText(out, "k=keep  d=delete  t=retrack  u=undo  r=replay  q=quit", (10, h - 10), FONT, 0.55, GRAY, 1)
    return out


def confirm_in_window(to_delete: list[str], to_retrack: list[str]) -> bool:
    bg = np.zeros((DISPLAY_H, DISPLAY_W, 3), dtype=np.uint8)
    if not to_delete and not to_retrack:
        for j, line in enumerate(["Review complete - nothing marked.", "", "Press any key to exit."]):
            cv2.putText(bg, line, (40, 180 + j * 50), FONT, 0.8, WHITE, 2)
        cv2.imshow(WINDOW, bg)
        cv2.waitKey(0)
        return False
    lines = [f"Delete {len(to_delete)}, retrack {len(to_retrack)} clip(s)?", ""]
    if to_delete:
        lines += ["DELETE:"] + to_delete[:8]
        if len(to_delete) > 8:
            lines.append(f"  ... and {len(to_delete) - 8} more")
    if to_retrack:
        lines += ["", "RETRACK:"] + to_retrack[:8]
        if len(to_retrack) > 8:
            lines.append(f"  ... and {len(to_retrack) - 8} more")
    lines += ["", "y = confirm    n / ESC = cancel"]
    for j, line in enumerate(lines):
        cv2.putText(bg, line, (40, 60 + j * 34), FONT, 0.6, WHITE, 1)
    cv2.imshow(WINDOW, bg)
    while True:
        key = cv2.waitKey(33) & 0xFF
        if key in (ord('y'), ord('Y')):
            return True
        if key in (ord('n'), ord('N'), 27):
            return False


def review_clip(frames, fps, i, total, clip_stem, to_delete, to_retrack) -> str:
    delay = max(1, int(1000 / fps))
    mark = 'delete' if clip_stem in to_delete else ('retrack' if clip_stem in to_retrack else None)
    fi = 0
    while True:
        frame = draw_overlay(frames[fi % len(frames)], i, total, clip_stem, mark, len(to_delete), len(to_retrack))
        cv2.imshow(WINDOW, frame)
        fi += 1
        key = cv2.waitKey(delay) & 0xFF
        if key in (ord('k'), ord('K')):
            return 'k'
        if key in (ord('d'), ord('D')):
            return 'd'
        if key in (ord('t'), ord('T')):
            return 't'
        if key in (ord('u'), ord('U')):
            return 'u'
        if key in (ord('r'), ord('R')):
            fi = 0
        if key in (ord('q'), ord('Q'), 27):
            return 'q'


def clear_manifest_seed(stems: set[str]) -> None:
    if not stems or not os.path.exists(MANIFEST):
        return
    with open(MANIFEST, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames
    for col in SEED_FIELDS:
        if col not in fieldnames:
            fieldnames = fieldnames + [col]
    for r in rows:
        for col in SEED_FIELDS:
            r.setdefault(col, "")
        if os.path.splitext(r["clip_filename"])[0] in stems:
            for col in SEED_FIELDS:
                r[col] = ""
    with open(MANIFEST, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()

    state = load_state()
    reviewed: set[str] = set() if args.reset else set(state.get("reviewed_clips", []))

    all_previews = sorted(
        p for d in TRACKS_DIRS.values() for p in glob.glob(os.path.join(d, "*_preview.mp4"))
        if "pose" not in p
    )
    previews = [p for p in all_previews if os.path.basename(p)[:-len("_preview.mp4")] not in reviewed]

    skipped = len(all_previews) - len(previews)
    if skipped:
        print(f"Skipping {skipped} already-reviewed clips.")
    if not previews:
        print("All clips reviewed. Use --reset to start over.")
        return

    total = len(previews)
    print(f"\n{total} clips to review")
    print("Focus the video window, then: k=keep  d=delete  t=retrack  u=undo  r=replay  q=quit\n")

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 960, 600)

    marks: list[tuple[str, str]] = []  # [(clip_stem, 'delete'|'retrack'), ...] in mark order

    def to_list(kind):
        return [s for s, a in marks if a == kind]

    i = 0
    while i < len(previews):
        preview = previews[i]
        clip_stem = os.path.basename(preview)[:-len("_preview.mp4")]

        print(f"  loading [{i+1}/{total}] ...", end="\r", flush=True)
        frames, fps = load_frames(preview)
        if not frames:
            print(f"  [skip] {clip_stem} - could not decode")
            i += 1
            continue

        action = review_clip(frames, fps, i, total, clip_stem, to_list('delete'), to_list('retrack'))

        if action == 'k':
            marks = [(s, a) for s, a in marks if s != clip_stem]
            reviewed.add(clip_stem)
            state["reviewed_clips"] = sorted(reviewed)
            save_state(state)
            print(f"[{i+1}/{total}]  {clip_stem}  -> kept")
            i += 1
        elif action in ('d', 't'):
            kind = 'delete' if action == 'd' else 'retrack'
            marks = [(s, a) for s, a in marks if s != clip_stem]
            marks.append((clip_stem, kind))
            reviewed.add(clip_stem)
            state["reviewed_clips"] = sorted(reviewed)
            save_state(state)
            print(f"[{i+1}/{total}]  {clip_stem}  -> {kind.upper()} ({len(to_list(kind))} marked)")
            i += 1
        elif action == 'u':
            if marks:
                undone, _ = marks.pop()
                reviewed.discard(undone)
                state["reviewed_clips"] = sorted(reviewed)
                save_state(state)
                undone_idx = next((j for j, p in enumerate(previews)
                                   if os.path.basename(p)[:-len("_preview.mp4")] == undone), None)
                if undone_idx is not None:
                    print(f"  <- undo  {undone}  (going back)")
                    i = undone_idx
            else:
                print("  nothing to undo")
        elif action == 'q':
            print(f"\nQuit early. Progress saved ({len(reviewed)} reviewed).")
            break

    to_delete = to_list('delete')
    to_retrack = to_list('retrack')
    confirmed = confirm_in_window(to_delete, to_retrack)
    cv2.destroyAllWindows()
    cv2.waitKey(1)

    if not confirmed:
        if to_delete or to_retrack:
            print("Cancelled. Marks saved - they'll show on next run.")
        else:
            print("\nNothing marked. Done.")
        return

    for stem in to_delete:
        for d in CLIPS_DIRS:
            p = os.path.join(d, f"{stem}.mp4")
            if os.path.exists(p):
                os.remove(p)
                print(f"  deleted  {p}")
        for d in TRACKS_DIRS.values():
            for suffix in ("_preview.mp4", "_tracking.json"):
                p = os.path.join(d, f"{stem}{suffix}")
                if os.path.exists(p):
                    os.remove(p)
                    print(f"  deleted  {p}")
        reviewed.discard(stem)

    for stem in to_retrack:
        for d in TRACKS_DIRS.values():
            for suffix in ("_preview.mp4", "_tracking.json"):
                p = os.path.join(d, f"{stem}{suffix}")
                if os.path.exists(p):
                    os.remove(p)
                    print(f"  cleared  {p}")
        reviewed.discard(stem)
    clear_manifest_seed(set(to_retrack))
    if to_retrack:
        print(f"  cleared manifest seed for {len(to_retrack)} clip(s) - "
              f"run review_seed.py to re-pick, then track_clips.py to retrack")

    state["reviewed_clips"] = sorted(reviewed)
    save_state(state)
    print(f"Done. Deleted {len(to_delete)}, marked {len(to_retrack)} for retrack.")


if __name__ == "__main__":
    main()
