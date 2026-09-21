#!/usr/bin/env python3
"""
Interactive release-frame picker.

For each clip in clips_manifest.csv, opens a window showing one frame at a
time (from the pre-extracted images in clips/frames/<clip>/). Step through
frames with the keyboard, press ENTER on the frame where the ball leaves the
shooter's fingertips, and it gets written straight into clips_manifest.csv's
`release_frame` column. Re-running the script resumes at the first clip that
still has an empty release_frame.

Controls (click the window once first so it has keyboard focus):
  a / Left arrow   -1 frame
  d / Right arrow  +1 frame
  [                -5 frames
  ]                +5 frames
  ENTER            mark current frame as release_frame, save, go to next clip
  n                skip this clip (leave release_frame blank), go to next
  p                go back to the previous clip
  q / ESC          quit (progress already saved after every ENTER)

Run:
  python3 scripts/pick_release_frame.py
"""
import cv2
import csv
import os

CLIPS_DIR = "/Users/brycelee/Desktop/OKC/clips"
FRAMES_DIR = os.path.join(CLIPS_DIR, "frames")
MANIFEST = os.path.join(CLIPS_DIR, "clips_manifest.csv")

LEFT_KEYS = {ord('a'), 2, 63234, 65361}
RIGHT_KEYS = {ord('d'), 3, 63235, 65363}
JUMP_FWD_KEYS = {ord(']'), 63232, 65362}
JUMP_BACK_KEYS = {ord('['), 63233, 65364}
ENTER_KEYS = {13, 10}
SKIP_KEYS = {ord('n'), ord('N')}
PREV_KEYS = {ord('p'), ord('P')}
QUIT_KEYS = {ord('q'), ord('Q'), 27}


def load_manifest():
    with open(MANIFEST, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader), reader.fieldnames


def save_manifest(rows, fieldnames):
    with open(MANIFEST, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def frame_paths_for(clip_stem):
    d = os.path.join(FRAMES_DIR, clip_stem)
    if not os.path.isdir(d):
        return []
    files = sorted(fn for fn in os.listdir(d) if fn.startswith("frame_") and fn.endswith(".jpg"))
    return [os.path.join(d, fn) for fn in files]


def main():
    rows, fieldnames = load_manifest()

    start = 0
    for k, r in enumerate(rows):
        if not r.get("release_frame", "").strip():
            start = k
            break

    win = "release frame picker"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    i = start
    while 0 <= i < len(rows):
        row = rows[i]
        clip_stem = os.path.splitext(row["clip_filename"])[0]
        paths = frame_paths_for(clip_stem)
        n = len(paths)
        if n == 0:
            print(f"[skip] no extracted frames for {clip_stem}")
            i += 1
            continue

        existing = row.get("release_frame", "").strip()
        idx = int(existing) if existing.isdigit() else min(50, n - 1)
        idx = max(0, min(idx, n - 1))

        action = None
        while True:
            img = cv2.imread(paths[idx])
            disp = img.copy()
            h, w = disp.shape[:2]
            cv2.rectangle(disp, (0, 0), (w, 34), (0, 0, 0), -1)
            cv2.putText(
                disp,
                f"[{i+1}/{len(rows)}] {clip_stem}  frame {idx}/{n-1}  {row['shot_type']}/{row['result']}",
                (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA,
            )
            cv2.rectangle(disp, (0, h - 28), (w, h), (0, 0, 0), -1)
            cv2.putText(
                disp,
                "a/d:-1/+1  [/]:-5/+5  ENTER:mark+next  n:skip  p:prev clip  q:quit",
                (8, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA,
            )
            cv2.imshow(win, disp)
            key = cv2.waitKeyEx(0)

            if key in LEFT_KEYS:
                idx = max(0, idx - 1)
            elif key in RIGHT_KEYS:
                idx = min(n - 1, idx + 1)
            elif key in JUMP_FWD_KEYS:
                idx = min(n - 1, idx + 5)
            elif key in JUMP_BACK_KEYS:
                idx = max(0, idx - 5)
            elif key in ENTER_KEYS:
                row["release_frame"] = str(idx)
                save_manifest(rows, fieldnames)
                print(f"{clip_stem}: release_frame = {idx}  [saved]")
                action = "next"
                break
            elif key in SKIP_KEYS:
                print(f"{clip_stem}: skipped (left blank)")
                action = "next"
                break
            elif key in PREV_KEYS:
                action = "prev"
                break
            elif key in QUIT_KEYS:
                action = "quit"
                break

        if action == "quit":
            break
        elif action == "prev":
            i = max(0, i - 1)
        else:
            i += 1

    cv2.destroyAllWindows()
    print("done — progress saved to", MANIFEST)


if __name__ == "__main__":
    main()
