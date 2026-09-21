"""
Manually mark the exact release frame for clips that don't have one yet
(needed before review_seed.py / track_clips.py can do anything useful with
them - review_seed.py only lists clips that already have a release_frame).

Controls:
  a / left-arrow   previous frame
  d / right-arrow  next frame
  A / D            jump 5 frames
  r                mark current frame as release
  n                save and next clip
  q                quit

Usage:
    python pipeline/annotate_release.py
    python pipeline/annotate_release.py --reset   # re-annotate clips that
                                                   # already have one too
"""
from __future__ import annotations
import argparse
import csv
import json
import os

import cv2
import numpy as np

CLIPS_DIR = "clips"
TRACKS_DIRS = {"pull_up": "tracks/pull_up", "catch_and_shoot": "tracks/catch_and_shoot"}
MANIFEST = "clips/clips_manifest.csv"
FPS_DEFAULT = 25.0


def annotate_clip(video_path: str, fps: float, start_frame: int) -> tuple[str, int | None]:
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cur = max(0, min(start_frame, total - 1))
    marked = None
    cache = {}

    def get_frame(idx):
        if idx not in cache:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, f = cap.read()
            cache[idx] = f if ret else np.zeros((H, W, 3), np.uint8)
        return cache[idx].copy()

    win = "annotate_release"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, min(W, 1280), min(H, 720))

    result = "skip"
    while True:
        frame = get_frame(cur)
        t = cur / fps

        cv2.putText(frame, f"frame {cur}/{total-1}   t={t:.2f}s",
                    (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        if marked is not None:
            mt = marked / fps
            cv2.putText(frame, f"release marked: frame {marked}  ({mt:.2f}s)",
                        (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
            if cur == marked:
                cv2.rectangle(frame, (0, 0), (W - 1, H - 1), (0, 255, 0), 5)

        cv2.putText(frame, "a/d=frame  A/D=jump5  r=mark  n=save+next  q=quit",
                    (15, H - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

        cv2.imshow(win, frame)
        key = cv2.waitKey(30) & 0xFF

        if key == 255:
            continue
        if key in (ord('d'), 3, 83):
            cur = min(cur + 1, total - 1)
        elif key in (ord('a'), 2, 81):
            cur = max(cur - 1, 0)
        elif key == ord('D'):
            cur = min(cur + 5, total - 1)
        elif key == ord('A'):
            cur = max(cur - 5, 0)
        elif key == ord('r'):
            marked = cur
        elif key == ord('n'):
            if marked is not None:
                result = "saved"
            else:
                result = "skip"
            break
        elif key in (ord('q'), 27):
            result = "quit"
            break

    cap.release()
    cv2.destroyWindow(win)
    cv2.waitKey(1)
    return result, marked


def starting_frame_for(folder: str, stem: str) -> int:
    """Use the existing tracking.json's anchor_frame as a starting point if
    one exists (from the automatic classifier/fallback pass), else the
    clip's midpoint."""
    tj_path = os.path.join(TRACKS_DIRS[folder], f"{stem}_tracking.json")
    if os.path.exists(tj_path):
        meta = json.load(open(tj_path, encoding="utf-8"))
        return int(meta.get("anchor_frame", 50))
    return 50


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true",
                        help="re-annotate clips that already have a release_frame too")
    args = parser.parse_args()

    with open(MANIFEST, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames

    def save():
        with open(MANIFEST, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    todo = [r for r in rows if args.reset or not r["release_frame"].strip()]
    print(f"{len(todo)} clip(s) to annotate")
    print("a/d = frame   A/D = jump 5   r = mark release   n = save+next   q = quit\n")

    for i, row in enumerate(todo):
        stem = os.path.splitext(row["clip_filename"])[0]
        clip_path = os.path.join(CLIPS_DIR, row["folder"], row["clip_filename"])
        preview_path = os.path.join(TRACKS_DIRS[row["folder"]], f"{stem}_preview.mp4")
        video = preview_path if os.path.exists(preview_path) else clip_path

        if not os.path.exists(video):
            print(f"[{i+1}/{len(todo)}] skipping {row['clip_filename']} — clip not found")
            continue

        start_frame = starting_frame_for(row["folder"], stem)
        existing = row["release_frame"].strip()
        label = f"  [currently: frame {existing}]" if existing else "  [not annotated]"
        print(f"[{i+1}/{len(todo)}]  {row['clip_filename']}{label}")

        status, marked = annotate_clip(video, FPS_DEFAULT, start_frame)
        if status == "saved":
            row["release_frame"] = marked
            save()
            print(f"  -> saved frame {marked} ({marked/FPS_DEFAULT:.2f}s)")
        elif status == "skip":
            print("  -> skipped")
        elif status == "quit":
            print("\nQuit.")
            break

    print("\nDone.")


if __name__ == "__main__":
    main()
