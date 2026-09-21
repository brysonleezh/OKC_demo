# OKC_demo — Ajay Mitchell Shot Mechanics Portal

A shot-mechanics analysis pipeline and results dashboard built from broadcast game
film of OKC Thunder guard Ajay Mitchell. Manually curated jump-shot clips are
tracked frame-by-frame, run through pose estimation, and reduced to five release
mechanics metrics, then browsed in a local/static dashboard.

**Live demo:** `https://<github-username>.github.io/OKC_demo/portal/`
(enable GitHub Pages on this repo — see below)

## What's here

- `pipeline/` — the analysis pipeline (Python):
  - clip curation against `clips/clips_manifest.csv`
  - `track_clips.py` — ID-free, IoU-based player tracking seeded from a
    human-verified bounding box (no persistent-ID tracker, since basketball
    contact causes frequent ID switches), windowed to `release_frame ± 20`,
    with a max-jump cap and short-gap interpolation
  - `analyze_pose.py` — RTMPose keypoint extraction restricted to the tracked
    target player, with stability/continuity QC and outlier removal
  - `build_portal_data.py` — aggregates the manifest + pose output into
    `portal/data.json`
  - `review_seed.py`, `review_clips.py`, `review_pose.py`,
    `annotate_release.py` — the human-in-the-loop review tools used to seed,
    QC, and correct tracking/pose output
- `portal/` — the static dashboard (`index.html` / `style.css` / `app.js`),
  themed in OKC Thunder colors, showing per-shot skeleton-overlay video +
  mechanics charts, a shot-type × outcome comparison, and a sortable clip table
- `models/action_classifier/` — trained YOLO action-classifier weights (12
  classes incl. `player-jump-shot`) used to auto-pick tracking anchors
- `mitchell_shot_checklist.md` — the source-of-truth shot list

## The 5 mechanics metrics

| Metric | Definition |
|---|---|
| Elbow angle | Shoulder → Elbow → Wrist |
| Knee angle | Hip → Knee → Ankle |
| Wrist height | Normalized 0–1 by frame height (inverted: higher = higher release) |
| Body lean | Torso angle from vertical |
| Guide separation | Shooting wrist to guide wrist distance, normalized by frame width |

All trajectories are indexed to `rel_frame = frame_idx - release_frame` so shots
line up on a common release point.

## Dataset

40 shots were manually clipped and annotated; 30 passed tracking + pose QC and
are the ones shown in the portal (6 dropped for tracking issues on curation,
4 more for pose-quality issues after QC). Small-n caveat: some shot-type ×
outcome cells are as few as n=4 — see the portal's own discussion of this for
which patterns are suggestive vs. which aren't statistically supported.

## What's excluded from this repo (and why)

Raw broadcast footage (`noaudio/`, cut clips, extracted frames, and the
non-web-optimized preview renders) is excluded — it's copyrighted NBA/broadcast
video, much of it well past GitHub's file-size limits, and none of it is needed
to run the portal, which only plays the small `*_pose_preview_web.mp4`
skeleton-overlay clips already included under `tracks/`. Tracking/pose output
(`*_tracking.json`, `*_pose.json`) — plain numeric coordinates, not video — is
included.

## Running locally

```bash
python3 -m http.server 8000
# open http://localhost:8000/portal/index.html
```

To regenerate `portal/data.json` after re-running the pipeline:

```bash
python3 pipeline/build_portal_data.py
```

## Deploying (GitHub Pages)

This repo's relative paths (`portal/index.html` referencing `../tracks/...`)
are already laid out to work with GitHub Pages serving straight from the repo
root — no build step or restructuring needed. Enable it under
**Settings → Pages → Source: Deploy from a branch → `main` / `/(root)`**.
