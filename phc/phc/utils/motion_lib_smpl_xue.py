
import roma

import numpy as np
import os
import yaml
from tqdm import tqdm
import os.path as osp

from phc.utils import torch_utils
import joblib
import torch
from smpl_sim.poselib.skeleton.skeleton3d import SkeletonMotion, SkeletonState
#import torch.multiprocessing as mp
import copy
import gc
from smpl_sim.smpllib.smpl_joint_names import SMPL_MUJOCO_NAMES, SMPL_BONE_ORDER_NAMES
from smpl_sim.smpllib.smpl_parser import (
    SMPL_Parser,
    SMPLH_Parser,
    SMPLX_Parser,
)
from scipy.spatial.transform import Rotation as sRot
import random
from phc.utils.flags import flags
from phc.utils.motion_lib_base import MotionLibBase, DeviceCache, compute_motion_dof_vels, FixHeightMode
import msgpack
import socket
import multiprocessing as mp
import time


def to_torch(tensor):
    if torch.is_tensor(tensor):
        return tensor
    else:
        return torch.from_numpy(tensor)
USE_CACHE = False
print("MOVING MOTION DATA TO GPU, USING CACHE:", USE_CACHE)

if not USE_CACHE:
    old_numpy = torch.Tensor.numpy
    
    class Patch:

        def numpy(self):
            if self.is_cuda:
                return self.to("cpu").numpy()
            else:
                return old_numpy(self)

    torch.Tensor.numpy = Patch.numpy






def receive_data(sock):
    header = b''
    while len(header) < 4:
        chunk = sock.recv(4 - len(header))
        if not chunk:
            raise ConnectionError("Connection closed unexpectedly")
        header += chunk
    msg_length = int.from_bytes(header, byteorder='little')
    payload = b''
    while len(payload) < msg_length:
        chunk = sock.recv(msg_length - len(payload))
        if not chunk:
            raise ConnectionError("Connection closed during data transfer")
        payload += chunk
    try:
        unpacked_data = msgpack.unpackb(payload, raw=False)
    except msgpack.UnpackException as e:
        raise ValueError("Failed to unpackpack msg data") from e
    datas = unpacked_data.get("data", [])
    processed_data = []
    for data in datas:
        kp_pos = torch.tensor(data["kp_pos"]).float().reshape(-1, 3)
        kp_vel = torch.tensor(data["kp_vel"]).float().reshape(-1, 3)
        kp_rot = torch.tensor(data["kp_rot"]).float().reshape(-1, 3)
        kp_rot_vel = torch.tensor(data["kp_rot_vel"]).float().reshape(-1, 3)

        processed_data.append({
            #"id": data["id"],
            "kp_pos": kp_pos,
            "kp_vel": kp_vel,
            "kp_rot": kp_rot,
            "kp_rot_vel": kp_rot_vel
        })
    return processed_data[0]
    # return processed_data[0]["kp_pos"], \
    #         processed_data[0]["kp_vel"], \
    #         processed_data[0]["kp_rot"], \
    #         processed_data[0]["kp_rot_vel"]
    # return processed_data

def recv_data(conn, buffer, lock):
    while True:
        frame_data = receive_data(conn)
        with lock:
            try:
                if not buffer.full():
                    buffer.put(frame_data, block=False)
            except Exception as e:
                print(e)
        time.sleep(0.001)


class MotionLibSMPLXUE(MotionLibBase):

    def __init__(self, device, fix_height=FixHeightMode.full_fix, masterfoot_conifg=None, min_length=-1, im_eval=False, multi_thread=True):
        self.buffer = mp.Queue(1)
        self.lock = mp.Lock()
        super().__init__(motion_file=None, device=device, fix_height=fix_height, masterfoot_conifg=masterfoot_conifg, min_length=min_length, im_eval=im_eval, multi_thread=multi_thread)
        
        data_dir = "data/smpl"
        smpl_parser_n = SMPL_Parser(model_path=data_dir, gender="neutral")
        smpl_parser_m = SMPL_Parser(model_path=data_dir, gender="male")
        smpl_parser_f = SMPL_Parser(model_path=data_dir, gender="female")
        self.mesh_parsers = {0: smpl_parser_n, 1: smpl_parser_m, 2: smpl_parser_f}
        self.last_motion_frame = None
    
        return
    def load_data(self, motion_file,  min_length=-1, im_eval = False):
        if motion_file is not None:
            return super().load_data(motion_file, min_length, im_eval)
        server_ip = '127.0.0.1'
        server_port = 10086
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind((server_ip, server_port))
        server_socket.listen(1)
        print('Waiting for connect...')
        conn, addr = server_socket.accept()
        print("Connected from ", addr)
        self.task_data = mp.Process(target=recv_data, args = (conn, self.buffer, self.lock))
        self.task_data.start()

    def setup_constants(self, fix_height = FixHeightMode.full_fix, masterfoot_conifg=None, multi_thread = True):
        self._masterfoot_conifg = masterfoot_conifg
        self.fix_height = fix_height
        self.multi_thread = multi_thread
        
        #### Termination history
        self._curr_motion_ids = None

    def load_motions(self, skeleton_trees, gender_betas, limb_weights, random_sample=True, start_idx=0, max_len=-1, target_heading = None):
        self.num_joints = len(skeleton_trees[0].node_names)
        self._motion_dt = 1 / 30.
        self.skeleton_trees = skeleton_trees

    @staticmethod
    def fix_trans_height(pose_aa, trans, curr_gender_betas, mesh_parsers, fix_height_mode):
        if fix_height_mode == FixHeightMode.no_fix:
            return trans, 0
        with torch.no_grad():
            frame_check = 30
            gender = curr_gender_betas[0]
            betas = curr_gender_betas[1:]
            mesh_parser = mesh_parsers[gender.item()]
            height_tolorance = 0.0
            vertices_curr, joints_curr = mesh_parser.get_joints_verts(pose_aa[:frame_check], betas[None,], trans[:frame_check])
            offset = joints_curr[:, 0] - trans[:frame_check] # account for SMPL root offset. since the root trans we pass in has been processed, we have to "add it back".
            
            if fix_height_mode == FixHeightMode.ankle_fix:
                assignment_indexes = mesh_parser.lbs_weights.argmax(axis=1)
                pick = (((assignment_indexes != mesh_parser.joint_names.index("L_Toe")).int() + (assignment_indexes != mesh_parser.joint_names.index("R_Toe")).int() 
                    + (assignment_indexes != mesh_parser.joint_names.index("R_Hand")).int() + + (assignment_indexes != mesh_parser.joint_names.index("L_Hand")).int()) == 4).nonzero().squeeze()
                diff_fix = ((vertices_curr[:, pick] - offset[:, None])[:frame_check, ..., -1].min(dim=-1).values - height_tolorance).min()  # Only acount the first 30 frames, which usually is a calibration phase.
            elif fix_height_mode == FixHeightMode.full_fix:
                
                diff_fix = ((vertices_curr - offset[:, None])[:frame_check, ..., -1].min(dim=-1).values - height_tolorance).min()  # Only acount the first 30 frames, which usually is a calibration phase.
            
            
            
            trans[..., -1] -= diff_fix
            return trans, diff_fix

    @staticmethod
    def load_motion_with_skeleton(ids, motion_data_list, skeleton_trees, gender_betas, fix_height, mesh_parsers, masterfoot_config, target_heading, max_len, queue, pid):
        # ZL: loading motion with the specified skeleton. Perfoming forward kinematics to get the joint positions
        np.random.seed(np.random.randint(5000)* pid)
        res = {}
        assert (len(ids) == len(motion_data_list))
        for f in range(len(motion_data_list)):
            curr_id = ids[f]  # id for this datasample
            curr_file = motion_data_list[f]
            if not isinstance(curr_file, dict) and osp.isfile(curr_file):
                key = motion_data_list[f].split("/")[-1].split(".")[0]
                curr_file = joblib.load(curr_file)[key]
            curr_gender_beta = gender_betas[f]

            seq_len = curr_file['root_trans_offset'].shape[0]
            if max_len == -1 or seq_len < max_len:
                start, end = 0, seq_len
            else:
                start = random.randint(0, seq_len - max_len)
                end = start + max_len

            trans = curr_file['root_trans_offset'].clone()[start:end]
            pose_aa = to_torch(curr_file['pose_aa'][start:end])
            pose_quat_global = curr_file['pose_quat_global'][start:end]

            B, J, N = pose_quat_global.shape

            ##### ZL: randomize the heading ######
            # if (not flags.im_eval) and (not flags.test):
            #     # if True:
            #     random_rot = np.zeros(3)
            #     random_rot[2] = np.pi * (2 * np.random.random() - 1.0)
            #     random_heading_rot = sRot.from_euler("xyz", random_rot)
            #     pose_aa[:, :3] = torch.tensor((random_heading_rot * sRot.from_rotvec(pose_aa[:, :3])).as_rotvec())
            #     pose_quat_global = (random_heading_rot * sRot.from_quat(pose_quat_global.reshape(-1, 4))).as_quat().reshape(B, J, N)
            #     trans = torch.matmul(trans, torch.from_numpy(random_heading_rot.as_matrix().T).float())
            ##### ZL: randomize the heading ######

            trans, trans_fix = MotionLibSMPL.fix_trans_height(pose_aa, trans, curr_gender_beta, mesh_parsers, fix_height_mode = fix_height)

            if not masterfoot_config is None:
                num_bodies = len(masterfoot_config['body_names'])
                pose_quat_holder = np.zeros([B, num_bodies, N])
                pose_quat_holder[..., -1] = 1
                pose_quat_holder[...,masterfoot_config['body_to_orig_without_toe'], :] \
                    = pose_quat_global[..., masterfoot_config['orig_to_orig_without_toe'], :]

                pose_quat_holder[..., [masterfoot_config['body_names'].index(name) for name in ["L_Toe", "L_Toe_1", "L_Toe_1_1", "L_Toe_2"]], :] = pose_quat_holder[..., [masterfoot_config['body_names'].index(name) for name in ["L_Ankle"]], :]
                pose_quat_holder[..., [masterfoot_config['body_names'].index(name) for name in ["R_Toe", "R_Toe_1", "R_Toe_1_1", "R_Toe_2"]], :] = pose_quat_holder[..., [masterfoot_config['body_names'].index(name) for name in ["R_Ankle"]], :]

                pose_quat_global = pose_quat_holder

            pose_quat_global = to_torch(pose_quat_global)
            sk_state = SkeletonState.from_rotation_and_root_translation(skeleton_trees[f], pose_quat_global, trans, is_local=False)

            curr_motion = SkeletonMotion.from_skeleton_state(sk_state, curr_file.get("fps", 30))
            curr_dof_vels = compute_motion_dof_vels(curr_motion)
            
            if flags.real_traj:
                quest_sensor_data = to_torch(curr_file['quest_sensor_data'])
                quest_trans = quest_sensor_data[..., :3]
                quest_rot = quest_sensor_data[..., 3:]
                
                quest_trans[..., -1] -= trans_fix # Fix trans
                
                global_angular_vel = SkeletonMotion._compute_angular_velocity(quest_rot, time_delta=1 / curr_file['fps'])
                linear_vel = SkeletonMotion._compute_velocity(quest_trans, time_delta=1 / curr_file['fps'])
                quest_motion = {
                    "global_angular_vel": global_angular_vel,
                    "linear_vel": linear_vel,
                    "quest_trans": quest_trans,
                    "quest_rot": quest_rot}
                curr_motion.quest_motion = quest_motion

            curr_motion.dof_vels = curr_dof_vels
            curr_motion.gender_beta = curr_gender_beta
            res[curr_id] = (curr_file, curr_motion)
            
            

        if not queue is None:
            queue.put(res)
        else:
            return res

    def get_motion_state(self, motion_ids, motion_times, offset=None):
        if self.last_motion_frame is None:
            frame_data = None
            while frame_data is None:
                try:
                    with self.lock:
                        if not self.buffer.empty():
                            frame_data = self.buffer.get(block=False)
                except Exception as e:
                    pass
        else:
            frame_data = None
            try:
                with self.lock:
                    if not self.buffer.empty():
                        frame_data = self.buffer.get(block=False)
            except Exception as e:
                pass
            if frame_data is None:
                frame_data = self.last_motion_frame
        self.last_motion_frame = frame_data

        # match_idx = [0, 1, 1, 1, 2, 3, 5,5,5, 6, 7, 9, 15, 15,15,16,20,20,20,21]
        match_idx = [0, 1, 1, 1, 4, 7,7, 2, 2, 2, 5, 8,8, 3, 16,16,16,18,17,17,17,19]
        kp_pos = frame_data["kp_pos"].reshape(-1,24,3) # 24 joints
        kp_rot = frame_data["kp_rot"].reshape(-1,24,3) # 24 rotations
        kp_vel = frame_data["kp_vel"].reshape(-1,24,3)
        kp_rot_vel = frame_data["kp_rot_vel"].reshape(-1,24,3)

        # smpl_2_mujoco = [SMPL_BONE_ORDER_NAMES.index(q) for q in SMPL_MUJOCO_NAMES if q in SMPL_BONE_ORDER_NAMES]
        smpl_2_mujoco = [i for i in range(24)]
        pose_aa_mj = kp_rot.reshape(-1, 24, 3)[:, smpl_2_mujoco]
        kp_pos = kp_pos[:, smpl_2_mujoco]
        kp_vel = kp_vel[:, smpl_2_mujoco]
        kp_rot = pose_aa_mj
        kp_rot_vel = kp_rot_vel[:, smpl_2_mujoco]
        pose_quat = sRot.from_rotvec(pose_aa_mj.reshape(-1, 3)).as_quat().reshape(-1, 24, 4)

        # root_trans_offset = kp_pos[:, 0] + self.skeleton_trees[0].local_translation[0]
        # new_sk_state = SkeletonState.from_rotation_and_root_translation(
        #         self.skeleton_trees[0],  # This is the wrong skeleton tree (location wise) here, but it's fine since we only use the parent relationship here. 
        #         torch.from_numpy(pose_quat).float(),
        #         root_trans_offset,
        #         is_local=True)
        # pose_quat_global = new_sk_state.global_rotation.to(self._device)
        # pose_quat = new_sk_state.local_rotation

        kp_rot_quat = torch.from_numpy(pose_quat).float().to(self._device)

        kp_pos = kp_pos.to(self._device)
        kp_vel = kp_vel.to(self._device)
        kp_rot_vel = kp_rot_vel.to(self._device)
        kp_rot_quat = kp_rot_quat.to(self._device)
        kp_rot = kp_rot.to(self._device)
        


        return {
            "root_pos": kp_pos[:, 0].clone(),
            "root_rot": kp_rot_quat[:, 0].clone(),
            "dof_pos": torch.zeros(1, 72).float().to(self._device), 
            "root_vel": kp_vel[..., 0, :].clone(),
            "root_ang_vel": kp_rot_vel[..., 0, :].clone(),
            "dof_vel": torch.zeros(1, 72).float().to(self._device),
            "motion_aa": kp_rot.reshape(-1,72),
            "rg_pos": kp_pos[:, match_idx],
            "rb_rot": kp_rot_quat[:, match_idx],
            "body_vel": kp_vel[:, match_idx],
            "body_ang_vel": kp_rot_vel[:, match_idx],
            "motion_bodies": torch.zeros(1, 17).float().to(self._device),
            "motion_limb_weights": torch.zeros(1, 10).float().to(self._device),
            
            
            "rg_pos_t": kp_pos[:, match_idx + [20, 21, 15]].clone(),
            "rg_rot_t": kp_rot_quat[:, match_idx + [20, 21, 15], :].clone(),
            "body_vel_t": kp_vel[:, match_idx + [20, 21, 15]].clone(),
            "body_ang_vel_t": kp_rot_vel[:, match_idx + [20, 21, 15]].clone(),
        } 