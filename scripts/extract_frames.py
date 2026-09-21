#!/usr/bin/env python3
"""Extract every frame of each clip as a numbered JPG so you can flip through
them in Finder/Quick Look to spot the release frame."""
import subprocess
import os
import glob

CLIPS_DIR = "/Users/brycelee/Desktop/OKC/clips"
OUT_ROOT = os.path.join(CLIPS_DIR, "frames")

def main():
    clip_paths = sorted(
        glob.glob(os.path.join(CLIPS_DIR, "pull_up", "*.mp4"))
        + glob.glob(os.path.join(CLIPS_DIR, "catch_and_shoot", "*.mp4"))
    )

    for src in clip_paths:
        clip_name = os.path.splitext(os.path.basename(src))[0]
        out_dir = os.path.join(OUT_ROOT, clip_name)
        os.makedirs(out_dir, exist_ok=True)

        cmd = [
            "ffmpeg", "-y",
            "-i", src,
            "-start_number", "0",
            "-vsync", "0",
            "-q:v", "2",
            os.path.join(out_dir, "frame_%03d.jpg"),
        ]
        print(f"Extracting {clip_name} ...")
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if result.returncode != 0:
            print(f"  FAILED: {result.stdout.decode(errors='replace')[-1000:]}")
            continue

        n_frames = len(glob.glob(os.path.join(out_dir, "frame_*.jpg")))
        print(f"  -> {n_frames} frames in {out_dir}")

if __name__ == "__main__":
    main()
