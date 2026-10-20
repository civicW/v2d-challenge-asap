#!/bin/bash
cd /workspace
HYDRA_FULL_ERROR=1 python scripts/train_flash_sac.py \
  experiment=g1_recon_body_flash_sac \
  "task.parquet='/v2d/assets/human_motion_data/whole_body/tier1_r2/sequence_id=ep0000_r2/robot_name=g1/data.parquet'" \
  training.total_environment_steps=30720000 \
  logging.mode=disabled \
  > /tmp/train.log 2>&1
echo "train exit=$?"
grep -vE "onnxruntime|pthread" /tmp/train.log | tail -20
