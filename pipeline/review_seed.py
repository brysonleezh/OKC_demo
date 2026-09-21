"""
Review the seed bbox track_clips.py would pick for each clip, and correct
it by hand when it's wrong (e.g. it grabbed a bystander instead of the
shooter, or fused the shooter with a contesting defender - both real
failure modes with the "biggest box near release_frame" heuristic).

Default flow is multi-anchor: scrub frames with a/d, click the correct
player's box on however many frames you want (typically once near
release_frame, plus one more wherever a contest/occlusion happens), then
press ENTER to save all of them and jump to the next clip. The earliest
frame you clicked becomes the primary seed; any others become extra
correction anchors - track_clips.py propagates each frame from whichever
anchor is nearest to it, instead of dragging one seed across the whole
clip (which is what lets it drift onto the wrong player mid-clip).

Saves directly into clips/clips_manifest.csv:
  seed_frame, seed_x1..y2   the primary (earliest-frame) anchor
  extra_seeds               JSON list of [frame, x1, y1, x2, y2] for the rest

Controls (click the window once first so it has keyboard focus):
  a / Left arrow    previous frame
  d / Right arrow   next frame
  click a box        add it as an anchor on this frame (doesn't advance -
                      keep scrubbing and clicking as many as you need)
  m + click-drag      add a hand-drawn box as an anchor (detection missed him)
  c                   add the current (yellow) auto-pick as an anchor
  BACKSPACE           undo the most recently added anchor
  ENTER               save all anchors added this session, next clip
  n                   skip this clip (discard anything added, don't save)
  p                   previous clip
  q / ESC             quit (already-saved clips are unaffected, resumable)

Usage:
    python pipeline/review_seed.py          # only clips not yet seeded
    python pipeline/review_seed.py --all     # every clip, from the start,
                                              # including already-seeded ones,
                                              # pre-loaded with their saved
                                              # anchors (orange=primary,
                                              # cyan=extra) so you can add
                                              # more or backspace to remove
"""
from __future__ import annotations
import csv
import json
import os

import cv2
from ultralytics import YOLO

from track_clips import WEIGHTS, PLAYER_CLASS, CONF_THRESH

MANIFEST = "clips/clips_manifest.csv"
SEED_FIELDS = ["seed_frame", "seed_x1", "seed_y1", "seed_x2", "seed_y2", "extra_seeds"]

state = {
    "boxes": [], "auto_idx": None, "manual_mode": False,
    "drag_start": None, "drag_cur": None, "action": None, "chosen_bbox": None,
}


def detect_frame(img, model):
    results = model(img, conf=CONF_THRESH, verbose=False, device="mps")[0]
    players = []
    for box in results.boxes:
        if int(box.cls[0]) == PLAYER_CLASS:
            players.append(list(map(int, box.xyxy[0])))
    return players


def box_at_point(boxes, x, y):
    candidates = [b for b in boxes if b[0] <= x <= b[2] and b[1] <= y <= b[3]]
    if not candidates:
        return None
    return min(candidates, key=lambda b: (b[2]-b[0])*(b[3]-b[1]))


def on_mouse(event, x, y, flags, param):
    if state["manual_mode"]:
        if event == cv2.EVENT_LBUTTONDOWN:
            state["drag_start"] = (x, y)
            state["drag_cur"] = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and state["drag_start"] is not None:
            state["drag_cur"] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and state["drag_start"] is not None:
            x1, y1 = state["drag_start"]
            state["chosen_bbox"] = [min(x1, x), min(y1, y), max(x1, x), max(y1, y)]
            state["action"] = "add_anchor"
    else:
        if event == cv2.EVENT_LBUTTONDOWN:
            hit = box_at_point(state["boxes"], x, y)
            if hit is not None:
                state["chosen_bbox"] = hit
                state["action"] = "add_anchor"


def render(img, clip_stem, i, total, cur_frame, release_frame, anchors):
    disp = img.copy()
    for idx, b in enumerate(state["boxes"]):
        is_auto = (idx == state["auto_idx"])
        color = (0, 255, 255) if is_auto else (140, 140, 140)
        thickness = 3 if is_auto else 1
        cv2.rectangle(disp, (b[0], b[1]), (b[2], b[3]), color, thickness)
        label = "AUTO PICK" if is_auto else str(idx)
        cv2.putText(disp, label, (b[0], max(0, b[1] - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    on_this_frame = [(n, f, b) for n, (f, b) in enumerate(anchors) if f == cur_frame]
    for n, f, b in on_this_frame:
        is_primary = n == 0
        color = (255, 128, 0) if is_primary else (255, 255, 0)
        label = f"ANCHOR #{n+1}" + (" (primary)" if is_primary else "")
        cv2.rectangle(disp, (b[0], b[1]), (b[2], b[3]), color, 2)
        cv2.putText(disp, label, (b[0], min(disp.shape[0]-4, b[3] + 18)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    if state["manual_mode"] and state["drag_start"] and state["drag_cur"]:
        cv2.rectangle(disp, state["drag_start"], state["drag_cur"], (255, 0, 255), 2)

    h, w = disp.shape[:2]
    cv2.rectangle(disp, (0, 0), (w, 30), (0, 0, 0), -1)
    tag = "  (release_frame)" if cur_frame == release_frame else ""
    mode = " MANUAL DRAW" if state["manual_mode"] else ""
    n_anchor_txt = f"  [{len(anchors)} anchor(s) pending]" if anchors else "  [0 anchors - click a box]"
    cv2.putText(disp, f"[{i+1}/{total}] {clip_stem}  frame {cur_frame}{tag}{mode}{n_anchor_txt}",
                (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.rectangle(disp, (0, h - 24), (w, h), (0, 0, 0), -1)
    cv2.putText(disp, "a/d:frame  click:add anchor  m:manual-draw  c:add auto-pick  BKSP:undo  ENTER:save+next  n:skip  p:prev  q:quit",
                (8, h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1, cv2.LINE_AA)
    return disp


def review_one(clip_path, release_frame, model, i, total, mouse_param, initial_anchors=None):
    cap = cv2.VideoCapture(clip_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    anchors = list(initial_anchors) if initial_anchors else []
    cur_frame = anchors[0][0] if anchors else max(0, release_frame - 10)
    clip_stem = os.path.splitext(os.path.basename(clip_path))[0]
    frame_cache = {}

    def get_frame(idx):
        if idx not in frame_cache:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, f = cap.read()
            frame_cache[idx] = f if ret else None
        return frame_cache[idx]

    state["manual_mode"] = False
    state["drag_start"] = state["drag_cur"] = None
    state["action"] = None
    state["chosen_bbox"] = None

    result_action = None
    while result_action is None:
        img = get_frame(cur_frame)
        if img is None:
            cap.release()
            return "skip", []

        state["boxes"] = detect_frame(img, model)
        state["auto_idx"] = None
        if state["boxes"]:
            areas = [(b[2]-b[0])*(b[3]-b[1]) for b in state["boxes"]]
            state["auto_idx"] = areas.index(max(areas))

        disp = render(img, clip_stem, i, total, cur_frame, release_frame, anchors)
        cv2.imshow("review seed bbox", disp)
        key = cv2.waitKeyEx(20)

        if state["action"] == "add_anchor":
            anchors.append((cur_frame, state["chosen_bbox"]))
            print(f"  + anchor #{len(anchors)} at frame {cur_frame}: {state['chosen_bbox']}")
            state["action"] = None
            state["chosen_bbox"] = None
            state["manual_mode"] = False
            state["drag_start"] = state["drag_cur"] = None
            continue

        if key == -1:
            continue
        if key in (ord("a"), 2, 81, 63234):
            cur_frame = max(0, cur_frame - 1)
        elif key in (ord("d"), 3, 83, 63235):
            cur_frame = min(total_frames - 1, cur_frame + 1)
        elif key in (ord("m"), ord("M")):
            state["manual_mode"] = not state["manual_mode"]
            state["drag_start"] = state["drag_cur"] = None
        elif key == ord("c"):
            if state["auto_idx"] is not None:
                anchors.append((cur_frame, state["boxes"][state["auto_idx"]]))
                print(f"  + anchor #{len(anchors)} (auto-pick) at frame {cur_frame}")
        elif key in (8, 127):  # backspace / delete
            if anchors:
                removed = anchors.pop()
                print(f"  - undid anchor at frame {removed[0]}")
        elif key in (13, 10):  # enter/return
            if anchors:
                result_action = "save"
            else:
                print("  no anchors added yet - click a box first, or press n to skip this clip")
        elif key in (ord("n"), ord("N")):
            result_action = "skip"
        elif key in (ord("p"), ord("P")):
            result_action = "prev"
        elif key in (ord("q"), ord("Q"), 27):
            result_action = "quit"

    cap.release()
    return result_action, anchors


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true",
                        help="review every clip from the start, including ones already seeded, "
                             "so you can fix wrong picks or add/remove anchors")
    args = parser.parse_args()

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

    def save():
        with open(MANIFEST, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    model = YOLO(WEIGHTS)
    model.to("mps")

    cv2.namedWindow("review seed bbox", cv2.WINDOW_NORMAL)
    mouse_param = {}
    cv2.setMouseCallback("review seed bbox", on_mouse, mouse_param)

    todo = [r for r in rows if r["release_frame"].strip().isdigit()]
    print(f"{len(todo)} clips with a release_frame")
    if args.all:
        print("--all: reviewing every clip, including already-seeded ones\n")
    else:
        print("(skipping already-seeded clips - pass --all to re-review everything)\n")

    i = 0
    while 0 <= i < len(todo):
        row = todo[i]
        if row["seed_x1"].strip() and not args.all:
            i += 1
            continue
        clip_path = os.path.join("clips", row["folder"], row["clip_filename"])
        release_frame = int(row["release_frame"])

        initial_anchors = []
        if row["seed_x1"].strip():
            initial_anchors.append((int(row["seed_frame"]),
                [int(row["seed_x1"]), int(row["seed_y1"]), int(row["seed_x2"]), int(row["seed_y2"])]))
        if row["extra_seeds"].strip():
            initial_anchors += [(e[0], e[1:]) for e in json.loads(row["extra_seeds"])]

        action, anchors = review_one(clip_path, release_frame, model, i, len(todo), mouse_param, initial_anchors)

        if action == "save":
            anchors = sorted(anchors, key=lambda a: a[0])
            primary_f, primary_b = anchors[0]
            extras = anchors[1:]
            row["seed_frame"] = primary_f
            row["seed_x1"], row["seed_y1"], row["seed_x2"], row["seed_y2"] = primary_b
            row["extra_seeds"] = json.dumps([[f, *b] for f, b in extras])
            save()
            print(f"{row['clip_filename']}: primary=frame {primary_f}, {len(extras)} extra anchor(s)  [saved]")
            i += 1
        elif action == "skip":
            print(f"{row['clip_filename']}: skipped")
            i += 1
        elif action == "prev":
            i = max(0, i - 1)
        elif action == "quit":
            break

    cv2.destroyAllWindows()
    print("\nDone.")


if __name__ == "__main__":
    main()
