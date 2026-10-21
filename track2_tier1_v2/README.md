# Track 2 v2 producing scripts (replay-based trajectories)

- `batch_retarget.sh` - drives the official `soma_to_g1.py` retargeter (inside the FlashCHORD
  container) over all 30 Tier-1 episodes; computes per-hand contact detection.
- `record_replay.py` - runs the FlashCHORD kinematic reference replay per episode and records
  the controller-driven rollout.
- `convert_replay_to_results.py` - converts replay recordings into the kit's
  `flash_chord_object_trajectories_v1` episode parquets (results_v2mix).
