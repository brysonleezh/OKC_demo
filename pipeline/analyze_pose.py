"""
Stage 3: RTMPose analysis on tracked clips.
Uses the per-frame bbox from <clip>_tracking.json; RTMPose runs top-down on
the full frame with that bbox as a hint — it handles its own crop + 1.25x
padding internally, no manual crop/uncrop needed.

Unlike nba-shot-quality's version (which restricts to a +-0.5s window
around a rough classifier anchor inside a longer raw-quarter clip), our
clips are already tightly cut to ~4s around the shot, so this analyzes
every frame of the clip and aligns to release_frame (from
annotate_release.py) rather than the classifier's anchor_frame.

Usage:
    python pipeline/analyze_pose.py
    python pipeline/analyze_pose.py --complexity 2
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import subprocess

import cv2
import numpy as np
from rtmlib import RTMPose

CLIPS_DIRS = ["clips/pull_up", "clips/catch_and_shoot"]
TRACKS_DIRS = {"clips/pull_up": "tracks/pull_up", "clips/catch_and_shoot": "tracks/catch_and_shoot"}

CONF_THRESH = 0.30
CROP_PAD = 0.25
MIN_STABILITY_IOU = 0.30
MIN_POSE_FRAMES = 10
MODEL_COMPLEXITY = 1

_MODEL_URLS = {
    0: "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-s_simcc-body7_pt-body7_420e-256x192-acd4a1ef_20230504.zip",
    1: "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip",
    2: "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-x_simcc-body7_pt-body7_700e-384x288-71d7b7e9_20230629.zip",
}
_MODEL_INPUT_SIZES = {0: (192, 256), 1: (192, 256), 2: (288, 384)}
_MODEL_LABELS = {0: "rtmpose-s (lightweight)", 1: "rtmpose-m (balanced)", 2: "rtmpose-x (performance)"}

KP = dict(
    nose=0,
    l_shoulder=5, r_shoulder=6,
    l_elbow=7, r_elbow=8,
    l_wrist=9, r_wrist=10,
    l_hip=11, r_hip=12,
    l_knee=13, r_knee=14,
    l_ankle=15, r_ankle=16,
)

SKELETON = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
]


class _Enc(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.integer): return int(o)
        if isinstance(o, np.floating): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        return super().default(o)


def _dump(obj, path, **kw):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, cls=_Enc, **kw)


def _get_model(complexity: int) -> RTMPose:
    print(f"Loading {_MODEL_LABELS[complexity]} (auto-download on first run)...")
    return RTMPose(
        onnx_model=_MODEL_URLS[complexity],
        model_input_size=_MODEL_INPUT_SIZES[complexity],
        backend="onnxruntime", device="cpu",
    )


def crop_region(bbox, W, H, pad=CROP_PAD):
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    return (max(0, int(x1 - bw * pad)), max(0, int(y1 - bh * pad)),
            min(W, int(x2 + bw * pad)), min(H, int(y2 + bh * pad)))


def bbox_iou(a, b) -> float:
    xi1, yi1 = max(a[0], b[0]), max(a[1], b[1])
    xi2, yi2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / union if union > 0 else 0.0


def angle_at(a, b, c) -> float | None:
    a, b, c = np.array(a, float), np.array(b, float), np.array(c, float)
    ba, bc = a - b, c - b
    n1, n2 = np.linalg.norm(ba), np.linalg.norm(bc)
    if n1 < 1e-6 or n2 < 1e-6:
        return None
    return float(np.degrees(np.arccos(np.clip(np.dot(ba, bc) / (n1 * n2), -1, 1))))


def body_lean_deg(sh, hp) -> float | None:
    dx, dy = sh[0] - hp[0], hp[1] - sh[1]
    return float(np.degrees(np.arctan2(dx, dy))) if abs(dy) > 1e-6 else None


def draw_skeleton(img, kps, vis, shoot_kps):
    for i, j in SKELETON:
        if vis[i] > CONF_THRESH and vis[j] > CONF_THRESH:
            color = (0, 255, 255) if {i, j} <= shoot_kps else (0, 200, 0)
            cv2.line(img, (int(kps[i][0]), int(kps[i][1])), (int(kps[j][0]), int(kps[j][1])), color, 2)
    for i in range(len(kps)):
        if vis[i] > CONF_THRESH:
            cv2.circle(img, (int(kps[i][0]), int(kps[i][1])), 4,
                       (0, 0, 255) if i in shoot_kps else (0, 255, 0), -1)


def _delete_clip(clip_path: str, tracking_json: str, out_dir: str) -> None:
    stem = os.path.splitext(os.path.basename(clip_path))[0]
    for f in [clip_path, tracking_json,
              os.path.join(out_dir, f"{stem}_preview.mp4"),
              os.path.join(out_dir, f"{stem}_pose.json"),
              os.path.join(out_dir, f"{stem}_pose_preview.mp4"),
              os.path.join(out_dir, f"{stem}_pose_preview_web.mp4")]:
        if os.path.exists(f):
            os.remove(f)


def analyze_clip(clip_path: str, tracking_json: str, out_dir: str, hand: str,
                 pose_estimator: RTMPose, min_pose_frames: int = MIN_POSE_FRAMES) -> dict | None:
    meta = json.load(open(tracking_json, encoding="utf-8"))
    fps = meta["fps"]
    release_frame = meta.get("release_frame", meta["anchor_frame"])

    if hand == "r":
        s_sh, s_el, s_wr = KP["r_shoulder"], KP["r_elbow"], KP["r_wrist"]
        g_wr = KP["l_wrist"]
        s_hip, s_kn, s_an = KP["r_hip"], KP["r_knee"], KP["r_ankle"]
    else:
        s_sh, s_el, s_wr = KP["l_shoulder"], KP["l_elbow"], KP["l_wrist"]
        g_wr = KP["r_wrist"]
        s_hip, s_kn, s_an = KP["l_hip"], KP["l_knee"], KP["l_ankle"]
    shoot_kps = {s_sh, s_el, s_wr}

    cap = cv2.VideoCapture(clip_path)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fmeta = meta["frames"]
    n_total = len(fmeta)

    print(f"  {os.path.basename(clip_path)}  release_frame={release_frame}({release_frame/fps:.2f}s)  frames=0..{n_total-1}")

    trajectory: list[dict] = []
    prev_bbox = None
    prev_kps = None
    prev_vis = None

    for f_idx in range(n_total):
        ret, frame = cap.read()
        if not ret:
            break

        bbox = fmeta[f_idx].get("bbox")
        if bbox is None:
            continue
        if prev_bbox is not None and bbox_iou(prev_bbox, bbox) < MIN_STABILITY_IOU:
            continue
        prev_bbox = bbox

        bx1, by1, bx2, by2 = [int(v) for v in bbox]
        try:
            kps_all, vis_all = pose_estimator(frame, bboxes=[[bx1, by1, bx2, by2]])
        except Exception:
            continue
        if len(kps_all) == 0:
            continue

        kps = kps_all[0].copy()
        vis = vis_all[0].copy()

        torso_idx = [KP["l_shoulder"], KP["r_shoulder"], KP["l_hip"], KP["r_hip"]]

        torso_pts = [kps[i] for i in torso_idx if vis[i] > CONF_THRESH]
        if torso_pts:
            tc_x = float(np.mean([p[0] for p in torso_pts]))
            tc_y = float(np.mean([p[1] for p in torso_pts]))
            mx, my = (bx2 - bx1) * 0.20, (by2 - by1) * 0.20
            if not (bx1 - mx <= tc_x <= bx2 + mx and by1 - my <= tc_y <= by2 + my):
                prev_kps = prev_vis = None
                continue

        shoot_joints = [s_sh, s_el, s_wr, s_hip, s_kn]
        if sum(1 for i in shoot_joints if vis[i] > CONF_THRESH) < 3:
            prev_kps = prev_vis = None
            continue

        anat_ok = True
        for sh_i, hp_i in [(KP["l_shoulder"], KP["l_hip"]), (KP["r_shoulder"], KP["r_hip"])]:
            if vis[hp_i] > CONF_THRESH and vis[sh_i] > CONF_THRESH and kps[hp_i][1] < kps[sh_i][1] - 5:
                anat_ok = False
        if vis[s_kn] > CONF_THRESH and vis[s_hip] > CONF_THRESH and kps[s_kn][1] < kps[s_hip][1] - 5:
            anat_ok = False
        if not anat_ok:
            prev_kps = prev_vis = None
            continue

        if prev_kps is not None and prev_vis is not None:
            max_jump = (bx2 - bx1) * 0.40
            jumps = [np.linalg.norm(kps[i] - prev_kps[i]) for i in torso_idx
                     if vis[i] > CONF_THRESH and prev_vis[i] > CONF_THRESH]
            if jumps and float(np.mean(jumps)) > max_jump:
                prev_kps = prev_vis = None
                continue

        prev_kps, prev_vis = kps.copy(), vis.copy()

        def v(idx):
            return kps[idx].tolist() if vis[idx] > CONF_THRESH else None

        pts = {i: v(i) for i in [s_sh, s_el, s_wr, g_wr, s_hip, s_kn, s_an,
                                  KP["l_shoulder"], KP["r_shoulder"], KP["l_hip"], KP["r_hip"]]}

        elbow = angle_at(pts[s_sh], pts[s_el], pts[s_wr]) if all(pts[k] for k in [s_sh, s_el, s_wr]) else None
        knee = angle_at(pts[s_hip], pts[s_kn], pts[s_an]) if all(pts[k] for k in [s_hip, s_kn, s_an]) else None
        wrist_norm = (1.0 - float(kps[s_wr][1]) / H) if vis[s_wr] > CONF_THRESH else None
        sh_l, sh_r, hp_l, hp_r = pts[KP["l_shoulder"]], pts[KP["r_shoulder"]], pts[KP["l_hip"]], pts[KP["r_hip"]]
        lean = body_lean_deg([(sh_l[0]+sh_r[0])/2, (sh_l[1]+sh_r[1])/2],
                              [(hp_l[0]+hp_r[0])/2, (hp_l[1]+hp_r[1])/2]) if sh_l and sh_r and hp_l and hp_r else None
        g, s_pt = pts[g_wr], pts[s_wr]
        guide_sep = float(np.linalg.norm(np.array(s_pt) - np.array(g))) / W if g and s_pt else None

        trajectory.append({
            "frame_idx": f_idx,
            "rel_frame": f_idx - release_frame,
            "sec": round(fmeta[f_idx]["sec"], 3),
            "elbow_angle": round(elbow, 1) if elbow else None,
            "knee_angle": round(knee, 1) if knee else None,
            "wrist_height": round(wrist_norm, 3) if wrist_norm else None,
            "body_lean": round(lean, 1) if lean else None,
            "guide_sep": round(guide_sep, 3) if guide_sep else None,
            "keypoints": [[float(x), float(y)] for x, y in kps],
            "kp_conf": [float(c) for c in vis],
        })

    cap.release()

    # ── outlier block removal (player-switch detection) ──
    OUTLIER_ABS_PX, OUTLIER_STEP_MULT, MAX_BLOCK_FRAMES = 35, 3.0, 25
    n_traj = len(trajectory)
    if n_traj >= 5:
        torso_track = [KP["l_shoulder"], KP["r_shoulder"], KP["l_hip"], KP["r_hip"]]
        kps_tmp = np.array([r["keypoints"] for r in trajectory])
        steps = [float(np.mean(np.linalg.norm(kps_tmp[i][torso_track] - kps_tmp[i-1][torso_track], axis=1)))
                 for i in range(1, n_traj)]
        med_step = float(np.median(steps))
        thresh = max(OUTLIER_ABS_PX, med_step * OUTLIER_STEP_MULT)
        jump_idxs = [i for i, s in enumerate(steps) if s > thresh]
        to_remove: set = set()
        for k in range(len(jump_idxs) - 1):
            j_in, j_out = jump_idxs[k], jump_idxs[k + 1]
            block = range(j_in + 1, j_out + 1)
            if 0 < len(block) <= MAX_BLOCK_FRAMES:
                to_remove.update(block)
        if to_remove:
            print(f"    outlier removal: dropped {len(to_remove)} frame(s)")
            trajectory = [r for i, r in enumerate(trajectory) if i not in to_remove]
            n_traj = len(trajectory)

    if n_traj < min_pose_frames:
        print(f"  -> auto-deleted ({n_traj} pose frames < min {min_pose_frames})")
        return None

    # ── 3-frame median smoothing ──
    if n_traj >= 3:
        kps_arr = np.array([r["keypoints"] for r in trajectory])
        kps_smooth = kps_arr.copy()
        for i in range(1, n_traj - 1):
            kps_smooth[i] = np.median(np.stack([kps_arr[i-1], kps_arr[i], kps_arr[i+1]]), axis=0)
        for i, r in enumerate(trajectory):
            r["keypoints"] = [[float(x), float(y)] for x, y in kps_smooth[i]]
        for key in ["elbow_angle", "knee_angle", "body_lean"]:
            idxs = [i for i, r in enumerate(trajectory) if r.get(key) is not None]
            if len(idxs) < 3:
                continue
            raw = [trajectory[i][key] for i in idxs]
            for j in range(1, len(idxs) - 1):
                trajectory[idxs[j]][key] = round(float(np.median([raw[j-1], raw[j], raw[j+1]])), 1)

    # ── render pose preview ──
    traj_by_frame = {r["frame_idx"]: r for r in trajectory}
    stem = os.path.splitext(os.path.basename(clip_path))[0]
    preview_path = os.path.join(out_dir, f"{stem}_pose_preview.mp4")
    cap2 = cv2.VideoCapture(clip_path)
    writer = cv2.VideoWriter(preview_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    for f_idx in range(n_total):
        ret, frame = cap2.read()
        if not ret:
            break
        preview = frame.copy()
        r = traj_by_frame.get(f_idx)
        if r:
            kps, vis = np.array(r["keypoints"]), np.array(r["kp_conf"])
            bbox = fmeta[f_idx].get("bbox")
            if bbox:
                cx1, cy1, cx2, cy2 = crop_region(bbox, W, H, CROP_PAD)
                cv2.rectangle(preview, (cx1, cy1), (cx2, cy2), (200, 200, 0), 1)
            draw_skeleton(preview, kps, vis, shoot_kps)
            y0 = 50
            for txt in filter(None, [
                f"Elbow: {r['elbow_angle']:.0f}" if r.get("elbow_angle") else None,
                f"Knee:  {r['knee_angle']:.0f}" if r.get("knee_angle") else None,
                f"Lean:  {r['body_lean']:+.1f}" if r.get("body_lean") else None,
            ]):
                cv2.putText(preview, txt, (20, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                y0 += 32
        marker = " <-- RELEASE" if f_idx == release_frame else ""
        cv2.putText(preview, f"{os.path.basename(stem)}  frame {f_idx}{marker}",
                    (20, H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        writer.write(preview)

    cap2.release()
    writer.release()

    def vals(k):
        return [r[k] for r in trajectory if r.get(k) is not None]

    ev, kv, lv, wv = vals("elbow_angle"), vals("knee_angle"), vals("body_lean"), vals("wrist_height")
    summary = {
        "clip": os.path.basename(clip_path),
        "fps": fps,
        "release_frame": release_frame,
        "release_sec": round(release_frame / fps, 3),
        "num_frames_total": n_total,
        "num_pose_frames": len(trajectory),
        "trajectory": trajectory,
        "stats": {
            "elbow": {"min": round(float(min(ev)), 1), "max": round(float(max(ev)), 1),
                      "mean": round(float(np.mean(ev)), 1)} if ev else None,
            "knee_min": round(float(min(kv)), 1) if kv else None,
            "wrist_max": round(float(max(wv)), 3) if wv else None,
            "lean": {"mean": round(float(np.mean(lv)), 1), "std": round(float(np.std(lv)), 1)} if lv else None,
        },
    }
    _dump(summary, os.path.join(out_dir, f"{stem}_pose.json"), indent=2)

    web_path = os.path.join(out_dir, f"{stem}_pose_preview_web.mp4")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", preview_path, "-vcodec", "libx264", "-crf", "23",
             "-preset", "fast", "-movflags", "+faststart", "-an", web_path],
            check=True, capture_output=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    print(f"  {os.path.basename(stem)}  {len(trajectory)} pose frames  elbow={summary['stats']['elbow']}  lean={summary['stats']['lean']}")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hand", default="r", choices=["r", "l"])
    parser.add_argument("--complexity", type=int, default=MODEL_COMPLEXITY, choices=[0, 1, 2])
    parser.add_argument("--min-frames", type=int, default=MIN_POSE_FRAMES)
    args = parser.parse_args()

    pose_estimator = _get_model(args.complexity)

    clip_paths = sorted(
        p for d in CLIPS_DIRS for p in glob.glob(os.path.join(d, "*.mp4"))
        if "preview" not in p and "pose" not in p
    )
    print(f"{len(clip_paths)} clips | hand={args.hand} | complexity={args.complexity} ({_MODEL_LABELS[args.complexity]})\n")

    summaries = []
    for clip_path in clip_paths:
        clip_dir = os.path.dirname(clip_path)
        out_dir = TRACKS_DIRS.get(clip_dir, clip_dir)
        stem = os.path.splitext(os.path.basename(clip_path))[0]
        tj = os.path.join(out_dir, f"{stem}_tracking.json")
        if not os.path.exists(tj):
            print(f"  missing tracking.json for {os.path.basename(clip_path)}, skipping")
            continue
        if os.path.exists(os.path.join(out_dir, f"{stem}_pose_preview.mp4")):
            print(f"  {os.path.basename(clip_path)}  already done, skipping")
            continue
        result = analyze_clip(clip_path, tj, out_dir, args.hand, pose_estimator, args.min_frames)
        if result is None:
            _delete_clip(clip_path, tj, out_dir)
        else:
            summaries.append(result)

    if summaries:
        print(f"\nDone. {len(summaries)} clips analyzed.")


if __name__ == "__main__":
    main()
