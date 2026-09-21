"""
Stage 4: review pose results — frame-by-frame, since bad pose frames (wrong
person locked in, garbled skeleton) are easier to spot paused than at full
playback speed.

Controls:
  Space          pause / resume
  left / right   previous / next frame (works while paused or playing)
  k              keep (clears any issue flag too), advance to next clip
  m              mark as having an issue (writes pose_issue=yes into
                 clips_manifest.csv - non-destructive, files are kept;
                 use this for "mostly fine but something's off" rather
                 than deleting outright), advance to next clip
  d              delete (clip and all its files), advance to next clip
  u              undo last keep/mark/delete, stays on current clip
  r              replay from start
  q / ESC        quit (progress saved, resumes next run)

Usage:
    python pipeline/review_pose.py
    python pipeline/review_pose.py --reset
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
POSE_ISSUE_FIELD = "pose_issue"

DISPLAY_W, DISPLAY_H = 960, 540
WINDOW = "Pose Review"
FONT = cv2.FONT_HERSHEY_SIMPLEX
RED, ORANGE, WHITE, GRAY = (50, 50, 220), (0, 140, 255), (255, 255, 255), (160, 160, 160)

TRACKS_SUFFIXES = ["_preview.mp4", "_tracking.json",
                   "_pose.json", "_pose_preview.mp4", "_pose_preview_web.mp4"]


def load_state() -> dict:
    if os.path.exists(REVIEW_STATE):
        return json.load(open(REVIEW_STATE, encoding="utf-8"))
    return {}


def save_state(state: dict) -> None:
    with open(REVIEW_STATE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def set_pose_issue(stem: str, flagged: bool) -> None:
    """Non-destructive: writes pose_issue=yes/blank into clips_manifest.csv
    for this clip. Matched by clip_filename stem, same clip naming used
    everywhere else in the pipeline."""
    if not os.path.exists(MANIFEST):
        return
    with open(MANIFEST, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames
    if POSE_ISSUE_FIELD not in fieldnames:
        fieldnames = fieldnames + [POSE_ISSUE_FIELD]
    for r in rows:
        r.setdefault(POSE_ISSUE_FIELD, "")
        if os.path.splitext(r["clip_filename"])[0] == stem:
            r[POSE_ISSUE_FIELD] = "yes" if flagged else ""
    with open(MANIFEST, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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


def draw_overlay(frame, i, total, clip_stem, fi, n_frames, paused, mark, n_deleted, n_flagged):
    h, w = frame.shape[:2]
    out = frame.copy()
    cv2.rectangle(out, (0, 0), (w, 52), (20, 20, 20), -1)
    cv2.putText(out, f"[{i+1}/{total}]", (10, 36), FONT, 0.9, GRAY, 2)
    cv2.putText(out, clip_stem, (110, 36), FONT, 0.55, WHITE, 1)
    state_txt = "PAUSED" if paused else "playing"
    cv2.putText(out, f"frame {fi+1}/{n_frames}  {state_txt}", (10, h - 44), FONT, 0.55, GRAY, 1)
    if mark is not None:
        color = RED if mark == 'delete' else ORANGE
        badge = f" DELETE ({n_deleted}) " if mark == 'delete' else f" ISSUE ({n_flagged}) "
        (bw, _), _ = cv2.getTextSize(badge, FONT, 0.65, 2)
        cv2.rectangle(out, (w - bw - 16, 10), (w - 6, 46), color, -1)
        cv2.putText(out, badge, (w - bw - 12, 36), FONT, 0.65, WHITE, 2)
    cv2.rectangle(out, (0, h - 32), (w, h), (20, 20, 20), -1)
    cv2.putText(out, "space=pause  </>=frame  k=keep  m=mark issue  d=delete  u=undo  r=replay  q=quit",
                (10, h - 10), FONT, 0.47, GRAY, 1)
    return out


def confirm_in_window(to_delete: list[str]) -> bool:
    bg = np.zeros((DISPLAY_H, DISPLAY_W, 3), dtype=np.uint8)
    if not to_delete:
        for j, line in enumerate(["Nothing marked for deletion.", "", "Press any key to exit."]):
            cv2.putText(bg, line, (40, 180 + j * 50), FONT, 0.8, WHITE, 2)
        cv2.imshow(WINDOW, bg)
        cv2.waitKey(0)
        return False
    lines = [f"Delete {len(to_delete)} clip(s)?", ""] + to_delete[:12]
    if len(to_delete) > 12:
        lines.append(f"  ... and {len(to_delete) - 12} more")
    lines += ["", "y = confirm delete    n / ESC = cancel"]
    for j, line in enumerate(lines):
        cv2.putText(bg, line, (40, 60 + j * 38), FONT, 0.65, WHITE, 1)
    cv2.imshow(WINDOW, bg)
    while True:
        key = cv2.waitKey(33) & 0xFF
        if key in (ord('y'), ord('Y')):
            return True
        if key in (ord('n'), ord('N'), 27):
            return False


def review_clip(frames, fps, i, total, clip_stem, to_delete, to_flag) -> str:
    delay = max(1, int(1000 / fps))
    mark = 'delete' if clip_stem in to_delete else ('flag' if clip_stem in to_flag else None)
    n = len(frames)
    fi = 0
    paused = False

    while True:
        frame = draw_overlay(frames[fi], i, total, clip_stem, fi, n, paused, mark, len(to_delete), len(to_flag))
        cv2.imshow(WINDOW, frame)
        key = cv2.waitKey(1 if paused else delay) & 0xFF

        if key == ord(' '):
            paused = not paused
        elif key in (ord('k'), ord('K')):
            return 'k'
        elif key in (ord('m'), ord('M')):
            return 'm'
        elif key in (ord('d'), ord('D')):
            return 'd'
        elif key in (ord('u'), ord('U')):
            return 'u'
        elif key in (ord('r'), ord('R')):
            fi, paused = 0, True
        elif key in (ord('q'), ord('Q'), 27):
            return 'q'
        elif key in (83, 3, ord('.')):   # right arrow / d-alt
            paused = True
            fi = min(fi + 1, n - 1)
        elif key in (81, 2, ord(',')):   # left arrow
            paused = True
            fi = max(fi - 1, 0)
        elif not paused:
            fi = (fi + 1) % n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()

    state = load_state()
    reviewed: set[str] = set() if args.reset else set(state.get("reviewed_pose_clips", []))

    all_previews = sorted(p for d in TRACKS_DIRS.values() for p in glob.glob(os.path.join(d, "*_pose_preview.mp4")))
    previews = [p for p in all_previews if os.path.basename(p)[:-len("_pose_preview.mp4")] not in reviewed]

    skipped = len(all_previews) - len(previews)
    if skipped:
        print(f"Skipping {skipped} already-reviewed clips.")
    if not previews:
        print("All pose results reviewed. Use --reset to start over.")
        return

    total = len(previews)
    print(f"\n{total} clips to review")
    print("space=pause  </>=frame step  k=keep  m=mark issue  d=delete  u=undo  r=replay  q=quit\n")

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 960, 600)

    marks: list[tuple[str, str]] = []  # [(clip_stem, 'delete'|'flag'), ...] in mark order

    def to_list(kind):
        return [s for s, a in marks if a == kind]

    i = 0
    while i < len(previews):
        preview = previews[i]
        clip_stem = os.path.basename(preview)[:-len("_pose_preview.mp4")]

        print(f"  loading [{i+1}/{total}] ...", end="\r", flush=True)
        frames, fps = load_frames(preview)
        if not frames:
            print(f"  [skip] {clip_stem} - could not decode")
            i += 1
            continue

        action = review_clip(frames, fps, i, total, clip_stem, to_list('delete'), to_list('flag'))

        if action == 'k':
            was_flagged = clip_stem in to_list('flag')
            marks = [(s, a) for s, a in marks if s != clip_stem]
            if was_flagged:
                set_pose_issue(clip_stem, False)
            reviewed.add(clip_stem)
            state["reviewed_pose_clips"] = sorted(reviewed)
            save_state(state)
            print(f"[{i+1}/{total}]  {clip_stem}  -> kept")
            i += 1
        elif action in ('d', 'm'):
            kind = 'delete' if action == 'd' else 'flag'
            marks = [(s, a) for s, a in marks if s != clip_stem]
            marks.append((clip_stem, kind))
            if kind == 'flag':
                set_pose_issue(clip_stem, True)
            reviewed.add(clip_stem)
            state["reviewed_pose_clips"] = sorted(reviewed)
            save_state(state)
            label = "DELETE" if kind == 'delete' else "ISSUE flagged"
            print(f"[{i+1}/{total}]  {clip_stem}  -> {label} ({len(to_list(kind))} marked)")
            i += 1
        elif action == 'u':
            if marks:
                undone, kind = marks.pop()
                if kind == 'flag':
                    set_pose_issue(undone, False)
                reviewed.discard(undone)
                state["reviewed_pose_clips"] = sorted(reviewed)
                save_state(state)
                undone_idx = next((j for j, p in enumerate(previews)
                                   if os.path.basename(p)[:-len("_pose_preview.mp4")] == undone), None)
                if undone_idx is not None:
                    print(f"  <- undo  {undone}  (going back)")
                    i = undone_idx
            else:
                print("  nothing to undo")
        elif action == 'q':
            print(f"\nQuit early. Progress saved ({len(reviewed)} reviewed).")
            break

    to_delete = to_list('delete')
    to_flag = to_list('flag')
    confirmed = confirm_in_window(to_delete)
    cv2.destroyAllWindows()
    cv2.waitKey(1)

    if to_flag:
        print(f"\n{len(to_flag)} clip(s) flagged with pose_issue=yes in {MANIFEST} (files kept):")
        for s in to_flag:
            print("  ", s)

    if not confirmed:
        if to_delete:
            print("Delete marks saved - they'll show on next run.")
        return

    for stem in to_delete:
        for d in CLIPS_DIRS:
            p = os.path.join(d, f"{stem}.mp4")
            if os.path.exists(p):
                os.remove(p)
                print(f"  deleted  {p}")
        for d in TRACKS_DIRS.values():
            for suffix in TRACKS_SUFFIXES:
                p = os.path.join(d, f"{stem}{suffix}")
                if os.path.exists(p):
                    os.remove(p)
                    print(f"  deleted  {p}")
        reviewed.discard(stem)

    state["reviewed_pose_clips"] = sorted(reviewed)
    save_state(state)
    print(f"Done. Deleted {len(to_delete)} clip(s).")


if __name__ == "__main__":
    main()
