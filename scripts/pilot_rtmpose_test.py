#!/usr/bin/env python3
"""
Stage A pilot: run RTMPose (via rtmlib) on a handful of representative clips
and render a skeleton-overlay video so we can eyeball whether pose quality
is usable on this broadcast footage before building the full pipeline.

Broadcast frames contain many people (crowd, bench, refs) — rtmlib's bundled
detector returns all of them. For this pilot we don't yet have the manual
player-selection step (that's Stage B), so as a rough stand-in we draw only
the single highest mean-keypoint-confidence person per frame (a proxy for
"the clearest, closest subject", which is often but not always the shooter).

Run:
  SSL_CERT_FILE=$(.venv/bin/python3 -m certifi) .venv/bin/python3 scripts/pilot_rtmpose_test.py
"""
import glob
import os
import time

import cv2
import numpy as np
from rtmlib import Body, draw_skeleton

CLIPS_DIR = "/Users/brycelee/Desktop/OKC/clips"
FRAMES_DIR = os.path.join(CLIPS_DIR, "frames")
OUT_DIR = os.path.join(CLIPS_DIR, "pose_pilot")

PILOT_CLIPS = [
    "OKC_DEN_0309_1_0409_pull_up_2011",
    "OKC_WSH_1030_3_0520_catch_and_shoot_1431",
    "OKC_PHI_1228_2_0745_catch_and_shoot_1137",
]

FPS = 25
KPT_THR = 0.3


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    body = Body(mode="balanced", backend="onnxruntime", device="cpu")

    for clip_stem in PILOT_CLIPS:
        frame_paths = sorted(glob.glob(os.path.join(FRAMES_DIR, clip_stem, "frame_*.jpg")))
        if not frame_paths:
            print(f"[skip] no frames for {clip_stem}")
            continue

        out_path = os.path.join(OUT_DIR, f"{clip_stem}_overlay.mp4")
        writer = None
        confidences = []
        person_counts = []
        t_clip0 = time.time()

        for fi, fp in enumerate(frame_paths):
            img = cv2.imread(fp)
            keypoints, scores = body(img)

            n_persons = keypoints.shape[0]
            person_counts.append(n_persons)

            disp = img.copy()
            if n_persons > 0:
                mean_scores = scores.mean(axis=1)
                top = int(np.argmax(mean_scores))
                top_score = float(mean_scores[top])
                confidences.append(top_score)

                disp = draw_skeleton(
                    disp,
                    keypoints[top:top + 1],
                    scores[top:top + 1],
                    kpt_thr=KPT_THR,
                )
                label = f"frame {fi}  top_person_conf={top_score:.2f}  detected={n_persons}"
            else:
                confidences.append(0.0)
                label = f"frame {fi}  no person detected"

            cv2.rectangle(disp, (0, 0), (disp.shape[1], 30), (0, 0, 0), -1)
            cv2.putText(disp, label, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)

            if writer is None:
                h, w = disp.shape[:2]
                writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
            writer.write(disp)

        if writer is not None:
            writer.release()

        dt = time.time() - t_clip0
        print(
            f"{clip_stem}: {len(frame_paths)} frames in {dt:.1f}s "
            f"({dt/len(frame_paths):.2f}s/frame)  "
            f"avg_top_conf={np.mean(confidences):.3f}  "
            f"avg_detected_persons={np.mean(person_counts):.1f}  "
            f"-> {out_path}"
        )


if __name__ == "__main__":
    main()
