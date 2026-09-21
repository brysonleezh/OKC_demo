#!/usr/bin/env python3
"""
Stage B step 2: for each clip, show release_frame with every ByteTrack
track_id present at that frame drawn + numbered. Click the box that's Ajay
Mitchell. Because track_players.py already tracked everyone through the
whole clip, picking a track_id here immediately gives you that player's
complete bbox-per-frame record — no separate propagation step needed.

Tracks are color-coded by how long they persist in the clip: green = lasts
>=50% of frames (reliable pick), orange = shorter-lived (flickery, pick with
caution — may need the manual-draw fallback instead).

Controls (click the window once first so it has keyboard focus):
  left click inside a numbered box   select that track_id, save, next clip
  m + left-click-drag                draw a custom box; we then search this
                                      frame (then +-5 frames) for whichever
                                      track_id best overlaps it, and use that
  n                                  skip this clip (leave blank), next clip
  p                                  go back to the previous clip
  q / ESC                            quit (progress already saved)

Run:
  .venv/bin/python3 scripts/select_player_track.py
"""
import csv
import json
import os

import cv2

CLIPS_DIR = "/Users/brycelee/Desktop/OKC/clips"
FRAMES_DIR = os.path.join(CLIPS_DIR, "frames")
TRACKS_DIR = os.path.join(CLIPS_DIR, "tracks")
MANIFEST = os.path.join(CLIPS_DIR, "clips_manifest.csv")
SEED_CSV = os.path.join(CLIPS_DIR, "player_track_seed.csv")
SEED_FIELDS = ["clip_filename", "release_frame", "track_id"]
LONG_TRACK_FRACTION = 0.5

state = {
    "boxes": [],       # list of (track_id, x1,y1,x2,y2, is_long)
    "manual_mode": False,
    "drag_start": None,
    "drag_cur": None,
    "selected_track_id": None,
    "action": None,
}


def load_manifest_rows():
    with open(MANIFEST, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_seed_map():
    if not os.path.exists(SEED_CSV):
        return {}
    with open(SEED_CSV, newline="", encoding="utf-8") as f:
        return {r["clip_filename"]: r for r in csv.DictReader(f)}


def save_seed_map(seed_map):
    with open(SEED_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SEED_FIELDS)
        writer.writeheader()
        for r in seed_map.values():
            writer.writerow(r)


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter)


def box_at_point(boxes, x, y):
    candidates = [b for b in boxes if b[1] <= x <= b[3] and b[2] <= y <= b[4]]
    if not candidates:
        return None
    return min(candidates, key=lambda b: (b[3] - b[1]) * (b[4] - b[2]))


def find_best_track_for_manual_box(track_json, release_frame, manual_box, window=5):
    best_tid, best_iou = None, 0.0
    lo, hi = max(0, release_frame - window), min(len(track_json["frames"]) - 1, release_frame + window)
    for fi in range(lo, hi + 1):
        for t in track_json["frames"][fi]["tracks"]:
            v = iou(manual_box, t["bbox"])
            if v > best_iou:
                best_iou, best_tid = v, t["track_id"]
    return best_tid, best_iou


def on_mouse(event, x, y, flags, param):
    if state["manual_mode"]:
        if event == cv2.EVENT_LBUTTONDOWN:
            state["drag_start"] = (x, y)
            state["drag_cur"] = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and state["drag_start"] is not None:
            state["drag_cur"] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and state["drag_start"] is not None:
            x1, y1 = state["drag_start"]
            x2, y2 = x, y
            param["manual_box"] = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
            state["action"] = "manual_select"
    else:
        if event == cv2.EVENT_LBUTTONDOWN:
            hit = box_at_point(state["boxes"], x, y)
            if hit is not None:
                state["selected_track_id"] = hit[0]
                state["action"] = "select"


def render(img, clip_stem, i, total, release_frame):
    disp = img.copy()
    for track_id, x1, y1, x2, y2, is_long in state["boxes"]:
        color = (0, 255, 0) if is_long else (0, 140, 255)
        cv2.rectangle(disp, (x1, y1), (x2, y2), color, 1)
        cv2.putText(disp, f"id={track_id}", (x1, max(0, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    if state["manual_mode"] and state["drag_start"] and state["drag_cur"]:
        cv2.rectangle(disp, state["drag_start"], state["drag_cur"], (255, 0, 255), 2)

    h, w = disp.shape[:2]
    cv2.rectangle(disp, (0, 0), (w, 30), (0, 0, 0), -1)
    mode = "MANUAL DRAW (drag a box)" if state["manual_mode"] else "click a track box"
    cv2.putText(disp, f"[{i+1}/{total}] {clip_stem}  release_frame={release_frame}  {mode}",
                (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.rectangle(disp, (0, h - 24), (w, h), (0, 0, 0), -1)
    cv2.putText(disp, "green=long-lived track  orange=short  click:select  m:manual-draw  n:skip  p:prev  q:quit",
                (8, h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)
    return disp


def main():
    rows = load_manifest_rows()
    seed_map = load_seed_map()

    win = "select target player track"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    mouse_param = {"manual_box": None}
    cv2.setMouseCallback(win, on_mouse, mouse_param)

    start = 0
    for k, r in enumerate(rows):
        if r["clip_filename"] not in seed_map:
            start = k
            break

    i = start
    while 0 <= i < len(rows):
        row = rows[i]
        clip_stem = os.path.splitext(row["clip_filename"])[0]
        rf = row.get("release_frame", "").strip()
        if not rf.isdigit():
            print(f"[skip] {clip_stem}: no release_frame set yet")
            i += 1
            continue
        rf = int(rf)

        track_path = os.path.join(TRACKS_DIR, f"{clip_stem}.json")
        if not os.path.exists(track_path):
            print(f"[skip] {clip_stem}: run track_players.py first")
            i += 1
            continue
        with open(track_path, encoding="utf-8") as f:
            track_json = json.load(f)

        frame_path = os.path.join(FRAMES_DIR, clip_stem, f"frame_{rf:03d}.jpg")
        img = cv2.imread(frame_path)

        total_frames = track_json["num_frames"]
        track_lengths = {}
        for fr in track_json["frames"]:
            for t in fr["tracks"]:
                track_lengths[t["track_id"]] = track_lengths.get(t["track_id"], 0) + 1

        this_frame_tracks = track_json["frames"][rf]["tracks"] if rf < len(track_json["frames"]) else []
        state["boxes"] = [
            (t["track_id"], int(t["bbox"][0]), int(t["bbox"][1]), int(t["bbox"][2]), int(t["bbox"][3]),
             track_lengths.get(t["track_id"], 0) >= LONG_TRACK_FRACTION * total_frames)
            for t in this_frame_tracks
        ]
        state["manual_mode"] = False
        state["drag_start"] = None
        state["drag_cur"] = None
        state["selected_track_id"] = None
        state["action"] = None
        mouse_param["manual_box"] = None

        while state["action"] is None:
            disp = render(img, clip_stem, i, len(rows), rf)
            cv2.imshow(win, disp)
            key = cv2.waitKeyEx(20)
            if key == -1:
                continue
            if key in (ord("m"), ord("M")):
                state["manual_mode"] = not state["manual_mode"]
                state["drag_start"] = None
                state["drag_cur"] = None
            elif key in (ord("n"), ord("N")):
                state["action"] = "skip"
            elif key in (ord("p"), ord("P")):
                state["action"] = "prev"
            elif key in (ord("q"), ord("Q"), 27):
                state["action"] = "quit"

        action = state["action"]
        if action == "select":
            tid = state["selected_track_id"]
            seed_map[row["clip_filename"]] = {
                "clip_filename": row["clip_filename"], "release_frame": rf, "track_id": tid,
            }
            save_seed_map(seed_map)
            print(f"{clip_stem}: track_id={tid} ({track_lengths.get(tid,0)}/{total_frames} frames)  [saved]")
            i += 1
        elif action == "manual_select":
            manual_box = mouse_param["manual_box"]
            tid, best_iou = find_best_track_for_manual_box(track_json, rf, manual_box)
            if tid is None:
                print(f"{clip_stem}: manual box matched no track within +-5 frames, try again")
                continue
            seed_map[row["clip_filename"]] = {
                "clip_filename": row["clip_filename"], "release_frame": rf, "track_id": tid,
            }
            save_seed_map(seed_map)
            print(f"{clip_stem}: manual box -> matched track_id={tid} (iou={best_iou:.2f}, "
                  f"{track_lengths.get(tid,0)}/{total_frames} frames)  [saved]")
            i += 1
        elif action == "skip":
            print(f"{clip_stem}: skipped")
            i += 1
        elif action == "prev":
            i = max(0, i - 1)
        elif action == "quit":
            break

    cv2.destroyAllWindows()
    print("done — progress saved to", SEED_CSV)


if __name__ == "__main__":
    main()
