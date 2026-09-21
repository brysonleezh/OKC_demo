"""
Stage 1: raw game film -> jump-shot clips + per-frame player tracking.

Scans each source video with a jump-shot action classifier (YOLOv11n,
trained separately — see nba-shot-quality/models/action_classifier),
clusters nearby detections into discrete shot events, and for each event
extracts a clip window with a single-target player tracker seeded at the
detection bbox.

Usage:
    python pipeline/extract_shots.py

Output (per shot event), written to jump_shot_clips/:
    clip_TIMESTAMP.mp4            clean clip
    clip_TIMESTAMP_preview.mp4    annotated with tracked bbox
    clip_TIMESTAMP_tracking.json  per-frame player bbox + anchor frame
    index.json                    running index of all extracted clips
"""
from __future__ import annotations
import argparse
import json
from datetime import datetime
from pathlib import Path

import cv2
from ultralytics import YOLO

WEIGHTS           = Path("models/action_classifier/weights/best.pt")
SOURCE_DIR        = Path("noaudio")
OUT_DIR           = Path("jump_shot_clips")

JUMP_CLASS        = 6      # player-jump-shot
PLAYER_CLASS      = 3      # player
CONF_THRESH       = 0.25
SAMPLE_EVERY       = 2
GAP_TOLERANCE_SEC = 2.0    # detections within this gap -> same shot event
MIN_SEG_SEC       = 0.15
CLIP_BEFORE_SEC   = 1.0    # clip window start, relative to anchor
CLIP_AFTER_SEC    = 2.0    # clip window end, relative to anchor
IOU_THRESH        = 0.25   # min IoU to accept a tracking match


def iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0
    area = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / area


def scan_video(video_path: Path, model: YOLO) -> tuple[list, float, int]:
    """Sample every Nth frame, record every jump-shot classifier hit."""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    detections = []
    frame_idx = 0

    print(f"  scanning {video_path.name}  ({fps:.0f}fps, {total} frames)")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % SAMPLE_EVERY == 0:
            results = model(frame, conf=CONF_THRESH, verbose=False)[0]
            for box in results.boxes:
                if int(box.cls[0]) == JUMP_CLASS:
                    detections.append({
                        "frame": frame_idx,
                        "conf": round(float(box.conf[0]), 3),
                        "bbox": list(map(int, box.xyxy[0])),
                    })
        frame_idx += 1
        if frame_idx % 2000 == 0:
            print(f"    {frame_idx}/{total}  detections so far: {len(detections)}")

    cap.release()
    print(f"    done: {len(detections)} jump-shot detections")
    return detections, fps, total


def cluster_detections(detections: list, fps: float) -> list[dict]:
    """Merge detections within GAP_TOLERANCE_SEC into one shot event each."""
    if not detections:
        return []

    gap_frames = int(GAP_TOLERANCE_SEC * fps)
    min_frames = int(MIN_SEG_SEC * fps)

    frames = sorted(set(d["frame"] for d in detections))
    seg_start = seg_end = frames[0]
    raw_clusters = []
    for f in frames[1:]:
        if f - seg_end <= gap_frames:
            seg_end = f
        else:
            raw_clusters.append((seg_start, seg_end))
            seg_start = seg_end = f
    raw_clusters.append((seg_start, seg_end))

    clusters = []
    for s, e in raw_clusters:
        if e - s < min_frames:
            continue
        dets = [d for d in detections if s <= d["frame"] <= e]
        best = max(dets, key=lambda d: d["conf"])
        clusters.append({
            "anchor_frame": s,
            "anchor_bbox": best["bbox"],
            "anchor_conf": best["conf"],
        })

    print(f"    {len(clusters)} shot event(s) after clustering")
    return clusters


def track_player(frame_dets: dict, anchor: int, anchor_bbox: list) -> dict:
    """Single-target IoU tracker seeded at the anchor bbox, propagated
    forward and backward across sampled frames, then linear-interpolated
    to fill the skipped (non-sampled) frames in between."""
    sampled = sorted(frame_dets.keys())
    tracked = {}

    def best_candidate(f):
        candidates = list(frame_dets[f].get("players", []))
        js = frame_dets[f].get("jump_shot")
        if js:
            candidates.append(js["bbox"])
        return candidates

    candidates = best_candidate(anchor) if anchor in frame_dets else []
    if candidates:
        best = max(candidates, key=lambda b: iou(b, anchor_bbox))
        tracked[anchor] = best if iou(best, anchor_bbox) >= IOU_THRESH else anchor_bbox
    else:
        tracked[anchor] = anchor_bbox

    last = tracked[anchor]
    for f in (x for x in sampled if x > anchor):
        candidates = best_candidate(f)
        if candidates:
            best = max(candidates, key=lambda b: iou(b, last))
            if iou(best, last) >= IOU_THRESH:
                last = best
        tracked[f] = last

    last = tracked[anchor]
    for f in (x for x in reversed(sampled) if x < anchor):
        candidates = best_candidate(f)
        if candidates:
            best = max(candidates, key=lambda b: iou(b, last))
            if iou(best, last) >= IOU_THRESH:
                last = best
        tracked[f] = last

    full = {}
    keys = sorted(tracked)
    for i in range(len(keys) - 1):
        f1, f2 = keys[i], keys[i + 1]
        b1, b2 = tracked[f1], tracked[f2]
        for f in range(f1, f2 + 1):
            t = (f - f1) / (f2 - f1) if f2 > f1 else 0
            full[f] = [int(b1[j] + t * (b2[j] - b1[j])) for j in range(4)]
    if keys:
        full[keys[-1]] = tracked[keys[-1]]
    return full


def extract_clip(video_path: Path, cluster: dict, fps: float, total_frames: int,
                 model: YOLO, out_dir: Path) -> dict:
    anchor = cluster["anchor_frame"]
    anchor_bbox = cluster["anchor_bbox"]
    clip_start = max(0, anchor - int(CLIP_BEFORE_SEC * fps))
    clip_end = min(total_frames - 1, anchor + int(CLIP_AFTER_SEC * fps))

    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, clip_start)
    frame_dets = {}
    f = clip_start
    while f <= clip_end:
        ret, frame = cap.read()
        if not ret:
            break
        if (f - clip_start) % SAMPLE_EVERY == 0:
            results = model(frame, conf=CONF_THRESH, verbose=False)[0]
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

    full_tracking = track_player(frame_dets, anchor, anchor_bbox)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    stem = f"clip_{ts}"
    clip_path = out_dir / f"{stem}.mp4"
    preview_path = out_dir / f"{stem}_preview.mp4"

    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, clip_start)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    w_clean = cv2.VideoWriter(str(clip_path), fourcc, fps, (W, H))
    w_preview = cv2.VideoWriter(str(preview_path), fourcc, fps, (W, H))

    frames_meta = []
    f = clip_start
    while f <= clip_end:
        ret, frame = cap.read()
        if not ret:
            break
        w_clean.write(frame)

        preview = frame.copy()
        bbox = full_tracking.get(f)
        is_shot = f in frame_dets and frame_dets[f].get("jump_shot") is not None
        color = (0, 255, 0) if is_shot else (0, 165, 255)
        if bbox:
            cv2.rectangle(preview, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 3)
            label = "JUMP SHOT" if is_shot else "tracking"
            cv2.putText(preview, label, (bbox[0], max(bbox[1] - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        sec = (f - clip_start) / fps
        cv2.putText(preview, f"{stem}  {sec:.1f}s",
                    (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        w_preview.write(preview)

        frames_meta.append({
            "frame": f, "sec": round(sec, 3),
            "bbox": bbox, "jump_shot_detected": is_shot,
        })
        f += 1

    cap.release()
    w_clean.release()
    w_preview.release()

    tracking = {
        "clip": clip_path.name,
        "source_video": video_path.name,
        "fps": fps,
        "clip_start_frame": clip_start,
        "clip_end_frame": clip_end,
        "anchor_frame": anchor,
        "anchor_sec": round((anchor - clip_start) / fps, 2),
        "frames": frames_meta,
    }
    (out_dir / f"{stem}_tracking.json").write_text(json.dumps(tracking, indent=2))

    print(f"  {stem}  {clip_start/fps:.1f}s-{clip_end/fps:.1f}s  ({video_path.stem[:35]})")
    return {"clip": clip_path.name, "source_video": video_path.name}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", default=str(SOURCE_DIR))
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--weights", default=str(WEIGHTS))
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    index_path = out_dir / "index.json"
    if index_path.exists():
        existing = json.loads(index_path.read_text())
        all_info = existing.get("clips", [])
        already_processed = set(existing.get("processed_sources", []))
    else:
        all_info = []
        already_processed = set()

    videos = sorted(source_dir.glob("*.mp4"))
    new_videos = [v for v in videos if v.name not in already_processed]
    print(f"{len(videos)} source videos ({len(new_videos)} new, "
          f"{len(videos)-len(new_videos)} already processed)\n")

    if not new_videos:
        print("No new videos to process.")
        return

    model = YOLO(args.weights)
    new_info = []

    for video_path in new_videos:
        dets, fps, total = scan_video(video_path, model)
        clusters = cluster_detections(dets, fps)
        for cluster in clusters:
            info = extract_clip(video_path, cluster, fps, total, model, out_dir)
            new_info.append(info)
        already_processed.add(video_path.name)

    all_info.extend(new_info)
    index_path.write_text(json.dumps({
        "total_clips": len(all_info),
        "processed_sources": sorted(already_processed),
        "clips": all_info,
    }, indent=2))
    print(f"\nDone. +{len(new_info)} new clips ({len(all_info)} total) -> {out_dir}")


if __name__ == "__main__":
    main()
