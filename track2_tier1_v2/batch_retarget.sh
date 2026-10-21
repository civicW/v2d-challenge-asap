#!/bin/bash
# Batch: tier_1 episodes -> adapted folders -> 2-pass retarget (palm-calibrated).
# Runs on the A100 HOST; file ops use host paths, container paths stay inside docker exec.
set -u
T1_HOST=$HOME/work/v2d/v2d_challenge_data/track_2/tier_1_multiview_caption
WORK_HOST=$HOME/work/v2d/v2d_work
REPO_CONT=/hosthome/work/v2d/video_to_data/robotic_grounding
T1_CONT=/big/v2d/v2d_challenge_data/track_2/tier_1_multiview_caption
WORK_CONT=/big/v2d/v2d_work

dx() { docker exec -e PYTHONPATH=$REPO_CONT/source/robotic_grounding -e MPLBACKEND=Agg gr00t_env "$@"; }

for EP in $(seq 1 29); do
  TAG=$(printf "ep%04d" $EP)
  D_HOST=$WORK_HOST/soma_folders/$TAG
  D_CONT=$WORK_CONT/soma_folders/$TAG
  [ -f "$D_HOST/.done" ] && { echo "skip $TAG"; continue; }
  mkdir -p "$D_HOST"
  echo "=== $TAG $(date +%H:%M:%S) ==="

  dx python /big/v2d/tier1_to_soma_folder.py --tier1-root $T1_CONT --episode $EP --out $D_CONT \
      --align-robot-world --rebase-to-hands > /tmp/retarget_$TAG.log 2>&1 \
      || { echo "FAIL adapter $TAG"; tail -3 /tmp/retarget_$TAG.log; continue; }

  dx python $REPO_CONT/scripts/retarget/soma_to_g1.py $D_CONT --save --motion-root $WORK_CONT/motion \
      --soma-subdir tier1_pass1 >> /tmp/retarget_$TAG.log 2>&1 \
      || { echo "FAIL retarget1 $TAG"; tail -3 /tmp/retarget_$TAG.log; continue; }

  dx python /big/v2d/tier1_to_soma_folder.py --tier1-root $T1_CONT --episode $EP --out $D_CONT \
      --align-robot-world --rebase-to-hands \
      --rebase-from-parquet $WORK_CONT/motion/whole_body/tier1_pass1/sequence_id=$TAG/robot_name=g1/data.parquet \
      >> /tmp/retarget_$TAG.log 2>&1 \
      || { echo "FAIL rebase2 $TAG"; tail -3 /tmp/retarget_$TAG.log; continue; }

  dx python $REPO_CONT/scripts/retarget/soma_to_g1.py $D_CONT --save --motion-root $WORK_CONT/motion \
      --soma-subdir tier1_final >> /tmp/retarget_$TAG.log 2>&1 \
      || { echo "FAIL retarget2 $TAG"; tail -3 /tmp/retarget_$TAG.log; continue; }

  grep -h "contact_active" /tmp/retarget_$TAG.log | tail -2
  touch "$D_HOST/.done"
  echo "DONE $TAG"
done
echo "ALL DONE $(date +%H:%M:%S)"
