#!/usr/bin/env python3
"""
Stage B step 1: run YOLO+ByteTrack multi-object tracking over each clip,
independent of who the target player is. Saves the full per-frame,
per-track bbox record to clips/tracks/<clip>.json so player selection and
pose estimation can both just read from it later.

Run:
  .venv/bin/python3 scripts/track_players.py
"""
import glob
import json
import os

from ultralytics import YOLO

CLIPS_DIR = "/Users/brycelee/Desktop/OKC/clips"
TRACKS_DIR = os.path.join(CLIPS_DIR, "tracks")
YOLO_WEIGHTS = "/Users/brycelee/Desktop/OKC/scripts/.model_cache/yolov8n.pt"
DET_CONF = 0.25
LONG_TRACK_MIN_FRAMES = 20  # just for the printed sanity summary


def main():
    os.makedirs(TRACKS_DIR, exist_ok=True)
    yolo = YOLO(YOLO_WEIGHTS)

    clip_paths = sorted(
        glob.glob(os.path.join(CLIPS_DIR, "pull_up", "*.mp4"))
        + glob.glob(os.path.join(CLIPS_DIR, "catch_and_shoot", "*.mp4"))
    )

    for clip_path in clip_paths:
        clip_filename = os.path.basename(clip_path)
        clip_stem = os.path.splitext(clip_filename)[0]
        out_path = os.path.join(TRACKS_DIR, f"{clip_stem}.json")
        if os.path.exists(out_path):
            print(f"[skip] {clip_filename}: already tracked")
            continue

        results = yolo.track(
            source=clip_path, classes=[0], conf=DET_CONF,
            persist=False, tracker="bytetrack.yaml", stream=True, verbose=False,
        )

        frames = []
        track_frame_count = {}
        for fi, r in enumerate(results):
            tracks = []
            if r.boxes.id is not None:
                ids = r.boxes.id.cpu().numpy().astype(int)
                xyxy = r.boxes.xyxy.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()
                for tid, box, conf in zip(ids, xyxy, confs):
                    tracks.append({
                        "track_id": int(tid),
                        "bbox": [round(float(v), 1) for v in box],
                        "conf": round(float(conf), 3),
                    })
                    track_frame_count[int(tid)] = track_frame_count.get(int(tid), 0) + 1
            frames.append({"frame_idx": fi, "tracks": tracks})

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({
                "clip_filename": clip_filename,
                "num_frames": len(frames),
                "frames": frames,
            }, f)

        n_long = sum(1 for c in track_frame_count.values() if c >= LONG_TRACK_MIN_FRAMES)
        print(f"{clip_filename}: {len(frames)} frames, "
              f"{len(track_frame_count)} unique track ids, "
              f"{n_long} tracks lasting >={LONG_TRACK_MIN_FRAMES} frames -> {out_path}")

    print("done —", TRACKS_DIR)


if __name__ == "__main__":
    main()
