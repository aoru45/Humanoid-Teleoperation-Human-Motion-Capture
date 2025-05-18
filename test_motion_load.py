import numpy as np
from phc.utils.motion_lib_g1 import MotionLibG1
from smpl_sim.poselib.skeleton.skeleton3d import SkeletonTree
import torch
import joblib

print("########################### H1")
# test Motion G1
device = "cpu"
motion_lib = MotionLibG1(motion_file="data/g1/test.pkl",
    device=device,
    masterfoot_conifg=None,
    fix_height=False,
    multi_thread=False,
    extend_head=False,
    mjcf_file="resources/unitree_robots/g1/g1_29dof_lock_waist_rev_1_0.xml") 

motion_lib.load_motions(skeleton_trees=[SkeletonTree.from_mjcf("resources/unitree_robots/g1/g1_29dof_lock_waist_rev_1_0.xml")] * 10,
 gender_betas=[torch.zeros(17)] * 10,
  limb_weights=[np.zeros(10)] * 10,
   random_sample=True)

motion_ids = torch.arange(1).to(device)
motion_times = torch.ones(1, device=device, dtype=torch.long)# * 1/10.

offset = torch.zeros(1,3).to(device)
motion_res = motion_lib.get_motion_state(motion_ids, motion_times, offset=offset)

for k,v in motion_res.items():
    if hasattr(v, "shape"):
        print(k, v.shape)
    else:
        print(k, v)
    motion_res[k] = motion_res[k].cpu()
print(motion_lib._motion_dt)
print(motion_lib.mesh_parsers.model_names)
# joblib.dump(motion_res, "motion_res.pkl")
exit()