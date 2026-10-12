# V2D Challenge — Team ASAP

NVIDIA V2D Challenge (https://nvidia-isaac.github.io/video_to_data/v2d_challenge) entries.

## Track 2 · Tier 1 — static-object baseline (v1, 2026-10-12)

First leaderboard entry. **No policy, no simulator rollout**: the predictor holds the
object at its first visible pose from the released Tier 1 multiview-caption data for the
whole episode. This exists to validate the submission toolchain end-to-end and to put an
honest baseline number on the four Tier 1 boards.

- Generator: `gen_static_baseline.py` — reads
  `track_2/tier_1_multiview_caption/data/chunk-000/episode_*.parquet`, keeps
  `observation.object.visible == true` frames, and writes one
  `flash_chord_object_trajectories_v1` parquet per episode with
  `achieved = first visible released pose (constant)` and `reference = released poses`.
- Provenance: `eval_script_sha256 = eval_code_sha256 = sha256(gen_static_baseline.py)`;
  `checkpoint_sha256 = sha256(JSON manifest describing the constant-pose predictor for
  that episode)` — there is no trained model.
- Packed with the official `v2d_submission_kit` `tools/pack_submission.py`
  (`--track track_2_tier1`).

### Reproduce

```bash
# dataset: huggingface.co/datasets/nvidia/video_to_data_challenge (track_2/tier_1_multiview_caption)
python gen_static_baseline.py --dataset-root <dataset> --out results/
python tools/pack_submission.py --track track_2_tier1 --results results/ \
  --dataset-root <dataset> --code-commit <this commit url> -o track2_tier1.csv
```

Local scores (clean reference = released Tier 1 object tracks):
AUC=0.2663 · SP-SR=0 · MP-SR=0 · MPPE=49.12 cm.

### Plan

v2: SOMA→G1 retargeting with the RoboMemoClaw pipeline (fit → retarget_g1) feeding
FlashCHORD reference tracking + FlashSAC training per episode.
