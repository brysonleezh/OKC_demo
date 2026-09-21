"""
Aggregate the 30 reviewed clips (manifest + pose.json) into portal/data.json
for the local shot-mechanics dashboard. Read-only against the pipeline outputs;
does not touch clips/, tracks/, or the manifest.

Run with: /Users/brycelee/miniconda3/envs/nba/bin/python3 pipeline/build_portal_data.py
"""

import csv
import glob
import json
import os
import statistics as stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(ROOT, "clips", "clips_manifest.csv")
TRACKS_GLOB = os.path.join(ROOT, "tracks", "*", "*_pose.json")
OUT_PATH = os.path.join(ROOT, "portal", "data.json")

METRICS = ["elbow_angle", "knee_angle", "wrist_height", "body_lean", "guide_sep"]
METRIC_LABELS = {
    "elbow_angle": "Elbow angle",
    "knee_angle": "Knee angle",
    "wrist_height": "Wrist height",
    "body_lean": "Body lean",
    "guide_sep": "Guide separation",
}
# decimal precision per metric — angles are degrees (1dp is plenty), but
# wrist_height/guide_sep are normalized 0-1 fractions with small absolute
# ranges (guide_sep is ~0.001-0.04), so 1dp would round every value to 0.0
METRIC_DECIMALS = {
    "elbow_angle": 1,
    "knee_angle": 1,
    "wrist_height": 3,
    "body_lean": 1,
    "guide_sep": 4,
}


def group_shot_type(shot_type):
    return "Pull-up" if shot_type.startswith("Pull-up") else "Catch-and-shoot"


def result_en(result):
    return "Make" if result == "命中" else "Miss"


def value_at_release(trajectory, metric):
    """Value at rel_frame == 0, or nearest available frame if that one is missing/None."""
    candidates = sorted(trajectory, key=lambda t: abs(t["rel_frame"]))
    for t in candidates:
        v = t.get(metric)
        if v is not None:
            return v
    return None


def mean_std(values, decimals=1):
    values = [v for v in values if v is not None]
    if not values:
        return {"n": 0, "mean": None, "std": None}
    if len(values) == 1:
        return {"n": 1, "mean": round(values[0], decimals), "std": 0.0}
    return {
        "n": len(values),
        "mean": round(stats.mean(values), decimals),
        "std": round(stats.pstdev(values), decimals),
    }


def main():
    rows = {os.path.splitext(r["clip_filename"])[0]: r for r in csv.DictReader(open(MANIFEST))}

    pose_files = sorted(glob.glob(TRACKS_GLOB))
    clips = []
    for pose_path in pose_files:
        stem = os.path.basename(pose_path).replace("_pose.json", "")
        row = rows.get(stem)
        if row is None:
            print(f"WARNING: {stem} has a pose.json but no manifest row, skipping")
            continue

        pose = json.load(open(pose_path))
        folder = row["folder"]
        video_rel = f"tracks/{folder}/{stem}_pose_preview_web.mp4"

        def r(v, m):
            return round(v, METRIC_DECIMALS[m]) if v is not None else None

        trajectory = [
            {
                "rel_frame": t["rel_frame"],
                "elbow_angle": r(t.get("elbow_angle"), "elbow_angle"),
                "knee_angle": r(t.get("knee_angle"), "knee_angle"),
                "wrist_height": r(t.get("wrist_height"), "wrist_height"),
                "body_lean": r(t.get("body_lean"), "body_lean"),
                "guide_sep": r(t.get("guide_sep"), "guide_sep"),
            }
            for t in pose["trajectory"]
        ]

        release_values = {m: value_at_release(pose["trajectory"], m) for m in METRICS}

        clips.append(
            {
                "id": stem,
                "clip_filename": row["clip_filename"],
                "folder": folder,
                "game_id": row["game_id"],
                "game_clock": row["game_clock"],
                "distance_ft": int(row["distance_ft"]) if row["distance_ft"] else None,
                "result": row["result"],
                "result_en": result_en(row["result"]),
                "shot_type": row["shot_type"],
                "shot_type_group": group_shot_type(row["shot_type"]),
                "video_time": row["video_time"],
                "note": row["note"],
                "video_url": video_rel,
                "fps": pose["fps"],
                "release_frame": pose["release_frame"],
                "stats": pose["stats"],
                "release_values": {
                    m: (round(v, METRIC_DECIMALS[m]) if v is not None else None)
                    for m, v in release_values.items()
                },
                "trajectory": trajectory,
            }
        )

    clips.sort(key=lambda c: (c["shot_type_group"], c["result_en"], c["id"]))

    # --- 2x2 grid: shot_type_group x result_en, mean/std of each metric at release ---
    grid = {}
    for shot_group in ("Pull-up", "Catch-and-shoot"):
        for result in ("Make", "Miss"):
            cell_clips = [c for c in clips if c["shot_type_group"] == shot_group and c["result_en"] == result]
            grid[f"{shot_group}|{result}"] = {
                "shot_type_group": shot_group,
                "result_en": result,
                "n": len(cell_clips),
                "clip_ids": [c["id"] for c in cell_clips],
                "metrics": {
                    m: mean_std([c["release_values"][m] for c in cell_clips], METRIC_DECIMALS[m])
                    for m in METRICS
                },
            }

    n_total = len(clips)
    n_make = sum(1 for c in clips if c["result_en"] == "Make")
    by_shot_type = {}
    for shot_group in ("Pull-up", "Catch-and-shoot"):
        group_clips = [c for c in clips if c["shot_type_group"] == shot_group]
        by_shot_type[shot_group] = {
            "n": len(group_clips),
            "n_make": sum(1 for c in group_clips if c["result_en"] == "Make"),
        }

    data = {
        "metric_labels": METRIC_LABELS,
        "metrics": METRICS,
        "summary": {
            "total": n_total,
            "n_make": n_make,
            "n_miss": n_total - n_make,
            "make_pct": round(100 * n_make / n_total, 1) if n_total else None,
            "by_shot_type": by_shot_type,
        },
        "grid": grid,
        "clips": clips,
    }

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Wrote {OUT_PATH}: {n_total} clips")
    for key, cell in grid.items():
        print(f"  {key}: n={cell['n']}")


if __name__ == "__main__":
    main()
