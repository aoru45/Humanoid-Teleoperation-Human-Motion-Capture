
# H2O Train Sim2Real Policy (8point tracking, no history, MLP, with linear velocity) with RL directly
python -m legged_gym.scripts.train_hydra \
  --config-name=config_teleop \
  task=h1:teleop \
  run_name=H2O_Policy \
  env.num_observations=138 \
  env.num_privileged_obs=215 \
  motion.teleop_obs_version=v-teleop-extend-max \
  motion.teleop_selected_keypoints_names=[left_ankle_link,right_ankle_link,left_shoulder_pitch_link,right_shoulder_pitch_link,left_elbow_link,right_elbow_link] \
  motion.extend_head=False \
  num_envs=4096 \
  asset.zero_out_far=False \
  asset.termination_scales.max_ref_motion_distance=1.5 \
  sim_device=cuda:0 \
  motion.motion_file=resources/motions/h1/amass_phc_filtered.pkl \
  rewards=rewards_teleop_omnih2o_teacher \
  rewards.penalty_curriculum=True \
  rewards.penalty_scale=0.5 \
  env.add_short_history=False
