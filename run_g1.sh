
# H2O Train Sim2Real Policy (8point tracking, no history, MLP, with linear velocity) with RL directly
python -m legged_gym.scripts.train_hydra \
  --config-name=config_teleop_g1 \
  task=g1:teleop \
  run_name=H2O_Policy \
  motion.teleop_obs_version=v-teleop-extend-max \
  motion.extend_head=False \
  motion.extend_hand=False \
  num_envs=2048 \
  asset.zero_out_far=False \
  asset.termination_scales.max_ref_motion_distance=1.5 \
  sim_device=cuda:0 \
  motion.motion_file=data/g1/test.pkl \
  rewards.penalty_curriculum=True \
  rewards.penalty_scale=0.5 \
  env.add_short_history=False
