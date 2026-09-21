#!/usr/bin/env python3
"""
Stage B step 3: for each clip with a selected track_id (select_player_track.py),
pull that track's bbox in every frame straight out of the saved tracking
record (clips/tracks/<clip>.json) — gaps where ByteTrack briefly lost the id
are linear-interpolated (edge gaps hold the nearest known box) — then run
RTMPose (17 COCO keypoints) on that bbox per frame.

Output per clip:
  clips/keypoints/<clip>.json         frame_idx, frame_idx_rel_release,
                                       bbox, whether the bbox was a direct
                                       detection or interpolated, keypoints,
                                       scores
  clips/pose_overlay/<clip>_overlay.mp4   skeleton+bbox QC video (bbox drawn
                                       orange when interpolated, blue when
                                       from a direct detection)

Also back-fills `pose_quality_flag` (mean keypoint confidence near
release_frame) and `track_completeness` (fraction of frames that were a
direct detection, not interpolated) into clips_manifest.csv.

Run:
  SSL_CERT_FILE=$(.venv/bin/python3 -m certifi) .venv/bin/python3 scripts/run_pose_pipeline.py
"""
import csv
import json
import os

import cv2
import numpy as np
from rtmlib import Body, draw_skeleton

CLIPS_DIR = "/Users/brycelee/Desktop/OKC/clips"
FRAMES_DIR = os.path.join(CLIPS_DIR, "frames")
TRACKS_DIR = os.path.join(CLIPS_DIR, "tracks")
MANIFEST = os.path.join(CLIPS_DIR, "clips_manifest.csv")
SEED_CSV = os.path.join(CLIPS_DIR, "player_track_seed.csv")
KEYPOINTS_DIR = os.path.join(CLIPS_DIR, "keypoints")
OVERLAY_DIR = os.path.join(CLIPS_DIR, "pose_overlay")

FPS = 25
QUALITY_WINDOW = 5
QUALITY_LOW_THR = 0.4


def build_bbox_series(track_json, track_id):
    """Return (bboxes[n], is_direct[n]) for track_id across every frame,
    filling gaps by linear interpolation (edges hold nearest known box)."""
    n = track_json["num_frames"]
    known = {}
    for fr in track_json["frames"]:
        for t in fr["tracks"]:
            if t["track_id"] == track_id:
                known[fr["frame_idx"]] = t["bbox"]

    bboxes = [None] * n
    is_direct = [False] * n
    for idx, bbox in known.items():
        bboxes[idx] = bbox
        is_direct[idx] = True

    known_idxs = sorted(known.keys())
    if not known_idxs:
        return None, None

    # fill leading/trailing edges by holding nearest known box
    for idx in range(0, known_idxs[0]):
        bboxes[idx] = known[known_idxs[0]]
    for idx in range(known_idxs[-1] + 1, n):
        bboxes[idx] = known[known_idxs[-1]]

    # linear-interpolate internal gaps
    for a, b in zip(known_idxs, known_idxs[1:]):
        if b - a <= 1:
            continue
        ba, bb = np.array(known[a]), np.array(known[b])
        for idx in range(a + 1, b):
            frac = (idx - a) / (b - a)
            bboxes[idx] = (ba + frac * (bb - ba)).tolist()

    return bboxes, is_direct


def main():
    os.makedirs(KEYPOINTS_DIR, exist_ok=True)
    os.makedirs(OVERLAY_DIR, exist_ok=True)

    with open(MANIFEST, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames
    for col in ("pose_quality_flag", "track_completeness"):
        if col not in fieldnames:
            fieldnames = fieldnames + [col]

    with open(SEED_CSV, newline="", encoding="utf-8") as f:
        seed_map = {r["clip_filename"]: r for r in csv.DictReader(f)}

    body = Body(mode="balanced", backend="onnxruntime", device="cpu")

    for row in rows:
        clip_filename = row["clip_filename"]
        seed = seed_map.get(clip_filename)
        if seed is None:
            print(f"[skip] {clip_filename}: no selected track_id")
            continue

        clip_stem = os.path.splitext(clip_filename)[0]
        out_json = os.path.join(KEYPOINTS_DIR, f"{clip_stem}.json")
        if os.path.exists(out_json):
            print(f"[skip] {clip_filename}: already processed")
            continue

        track_path = os.path.join(TRACKS_DIR, f"{clip_stem}.json")
        if not os.path.exists(track_path):
            print(f"[skip] {clip_filename}: run track_players.py first")
            continue
        with open(track_path, encoding="utf-8") as f:
            track_json = json.load(f)

        track_id = int(seed["track_id"])
        release_frame = int(seed["release_frame"])
        bboxes, is_direct = build_bbox_series(track_json, track_id)
        if bboxes is None:
            print(f"[skip] {clip_filename}: track_id {track_id} not found in track record")
            continue

        n = len(bboxes)
        records = []
        overlay_writer = None
        near_release_scores = []

        for idx in range(n):
            frame_path = os.path.join(FRAMES_DIR, clip_stem, f"frame_{idx:03d}.jpg")
            img = cv2.imread(frame_path)
            bbox = bboxes[idx]
            keypoints, scores = body.pose_model(img, bboxes=[bbox])

            records.append({
                "frame_idx": idx,
                "frame_idx_rel_release": idx - release_frame,
                "bbox": [round(v, 1) for v in bbox],
                "bbox_is_direct_detection": is_direct[idx],
                "keypoints": np.round(keypoints[0], 1).tolist(),
                "scores": np.round(scores[0], 3).tolist(),
            })

            if abs(idx - release_frame) <= QUALITY_WINDOW:
                near_release_scores.append(float(scores[0].mean()))

            disp = img.copy()
            x1, y1, x2, y2 = map(int, bbox)
            box_color = (255, 128, 0) if is_direct[idx] else (0, 165, 255)
            cv2.rectangle(disp, (x1, y1), (x2, y2), box_color, 2)
            disp = draw_skeleton(disp, keypoints, scores, kpt_thr=0.3)
            tag = "direct" if is_direct[idx] else "interp"
            cv2.putText(disp, f"frame {idx} (rel {idx - release_frame:+d}) [{tag}]",
                        (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
            if overlay_writer is None:
                h, w = disp.shape[:2]
                overlay_writer = cv2.VideoWriter(
                    os.path.join(OVERLAY_DIR, f"{clip_stem}_overlay.mp4"),
                    cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
            overlay_writer.write(disp)

        if overlay_writer is not None:
            overlay_writer.release()

        with open(out_json, "w", encoding="utf-8") as f:
            json.dump({
                "clip_filename": clip_filename,
                "track_id": track_id,
                "release_frame": release_frame,
                "num_frames": n,
                "frames": records,
            }, f)

        completeness = sum(is_direct) / n
        quality = "low" if np.mean(near_release_scores) < QUALITY_LOW_THR else "ok"
        row["pose_quality_flag"] = quality
        row["track_completeness"] = round(completeness, 3)

        print(f"{clip_filename}: {n} frames, track_completeness={completeness:.2f}, "
              f"near-release conf={np.mean(near_release_scores):.3f} ({quality})")

    with open(MANIFEST, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            r.setdefault("pose_quality_flag", "")
            r.setdefault("track_completeness", "")
            writer.writerow(r)

    print("done — keypoints in", KEYPOINTS_DIR, "| overlays in", OVERLAY_DIR)


if __name__ == "__main__":
    main()
