import torch 
import numpy as np
import phc.utils.rotation_conversions as tRot
import xml.etree.ElementTree as ETree
from easydict import EasyDict
import scipy.ndimage.filters as filters
import smpl_sim.poselib.core.rotation3d as pRot


class Humanoid_Batch:

    def __init__(self, mjcf_file = f"resources/robots/h1/h1.xml", extend_hand = True, extend_head = False, device = torch.device("cpu")):
        self.mjcf_data = mjcf_data = self.from_mjcf(mjcf_file)
        self.extend_hand = True # extend_hand
        self.extend_head = True # extend_head
        print(mjcf_data['node_names'] )
        if self.extend_hand:
            self.model_names = mjcf_data['node_names'] + ["left_hand_link", "right_hand_link"]
            self._parents = torch.cat((mjcf_data['parent_indices'], torch.tensor([19, 26]))).to(device) # Adding the hands joints
            arm_length = 0.3
            self._offsets = torch.cat((mjcf_data['local_translation'], torch.tensor([[arm_length, 0, 0], [arm_length, 0, 0]])), dim = 0)[None, ].to(device)
            self._local_rotation = torch.cat((mjcf_data['local_rotation'], torch.tensor([[1, 0, 0, 0], [1, 0, 0, 0]])), dim = 0)[None, ].to(device)
            self._remove_idx = 2
        else:
            self._parents = mjcf_data['parent_indices']
            self.model_names = mjcf_data['node_names']
            self._offsets = mjcf_data['local_translation'][None, ].to(device)
            self._local_rotation = mjcf_data['local_rotation'][None, ].to(device)
            
        if self.extend_head:
            if extend_hand:
                self._remove_idx = 3
            else:
                self._remove_idx = 1

            self.model_names = self.model_names + ["head_link_new"]
            self._parents = torch.cat((self._parents, torch.tensor([0]).to(device))).to(device) # Adding the hands joints
            head_length = 0.4
            self._offsets = torch.cat((self._offsets, torch.tensor([[[0, 0, head_length]]]).to(device)), dim = 1).to(device)
            self._local_rotation = torch.cat((self._local_rotation, torch.tensor([[[1, 0, 0, 0]]]).to(device)), dim = 1).to(device)

        
        self._local_rotation_mat = tRot.quaternion_to_matrix(self._local_rotation).float() # w, x, y ,z
        
    def from_mjcf(self, path):
        # function from Poselib: 
        tree = ETree.parse(path)
        xml_doc_root = tree.getroot()
        xml_world_body = xml_doc_root.find("worldbody")
        if xml_world_body is None:
            raise ValueError("MJCF parsed incorrectly please verify it.")
        # assume this is the root
        xml_body_root = xml_world_body.find("body")
        if xml_body_root is None:
            raise ValueError("MJCF parsed incorrectly please verify it.")
            
        xml_joint_root = xml_body_root.find("joint")
        
        node_names = []
        parent_indices = []
        local_translation = []
        local_rotation = []

        # recursively adding all nodes into the skel_tree
        def _add_xml_node(xml_node, parent_index, node_index):
            node_name = xml_node.attrib.get("name")
            # parse the local translation into float list
            pos = np.fromstring(xml_node.attrib.get("pos", "0 0 0"), dtype=float, sep=" ")
            quat = np.fromstring(xml_node.attrib.get("quat", "1 0 0 0"), dtype=float, sep=" ")
            all_joints = xml_node.findall("joint")


            node_names.append(node_name)
            parent_indices.append(parent_index)
            local_translation.append(pos)
            local_rotation.append(quat)
            curr_index = node_index
            node_index += 1
            
            for next_node in xml_node.findall("body"):
                node_index = _add_xml_node(next_node, curr_index, node_index)
            return node_index
        
        _add_xml_node(xml_body_root, -1, 0)
        return {
            "node_names": node_names,
            "parent_indices": torch.from_numpy(np.array(parent_indices, dtype=np.int32)),
            "local_translation": torch.from_numpy(np.array(local_translation, dtype=np.float32)),
            "local_rotation": torch.from_numpy(np.array(local_rotation, dtype=np.float32)),
        }

        
    def fk_batch(self, pose, trans, convert_to_mat=True, return_full = False, dt=1/30):
        """
        dof_names
        ['left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint', 'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint', 'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint', 'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint', 'waist_yaw_joint', 'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 'left_elbow_joint', 'left_wrist_roll_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint', 'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint', 'right_elbow_joint', 'right_wrist_roll_joint', 'right_wrist_pitch_joint', 'right_wrist_yaw_joint']
        model_names - pelvis
        ['left_hip_pitch_link', 'left_hip_roll_link', 'left_hip_yaw_link', 'left_knee_link', 'left_ankle_pitch_link', 'left_ankle_roll_link', 'right_hip_pitch_link', 'right_hip_roll_link', 'right_hip_yaw_link', 'right_knee_link', 'right_ankle_pitch_link', 'right_ankle_roll_link', 'waist_yaw_link', 'left_shoulder_pitch_link', 'left_shoulder_roll_link', 'left_shoulder_yaw_link', 'left_elbow_link', 'left_wrist_roll_link', 'left_wrist_pitch_link', 'left_wrist_yaw_link', 'right_shoulder_pitch_link', 'right_shoulder_roll_link', 'right_shoulder_yaw_link', 'right_elbow_link', 'right_wrist_roll_link', 'right_wrist_pitch_link', 'right_wrist_yaw_link']
 

        """
        device, dtype = pose.device, pose.dtype
        pose_input = pose.clone()
        B, seq_len = pose.shape[:2]

        #if self.extend_hand and self.extend_head:
        pose = torch.cat([pose, torch.zeros(B, seq_len, 3, 3).to(device).type(dtype)], dim = -2) 

        if convert_to_mat:
            pose_quat = tRot.axis_angle_to_quaternion(pose)
            pose_mat = tRot.quaternion_to_matrix(pose_quat)
        else:
            pose_mat = pose
        if pose_mat.shape != 5:
            pose_mat = pose_mat.reshape(B, seq_len, -1, 3, 3)
        
        wbody_pos, wbody_mat = self.forward_kinematics_batch(pose_mat[:, :, 1:], pose_mat[:, :, 0:1], trans)
        # from vis_tools import Visualizer
        # vis = Visualizer()
        # data = wbody_pos.reshape(-1, 33, 3)
        # vis.show_points([data.cpu()],fix_axis=True)
        #print(wbody_pos.shape)
        
        return_dict = EasyDict()
        
        
        wbody_rot = tRot.wxyz_to_xyzw(tRot.matrix_to_quaternion(wbody_mat))

        select_no_pelvis = [0,1,2,3,4,5,6,7,8,9,10,11,12,15,16,17,18,22,23,24,25] + [29,30,31]
        select_with_pelvis = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,16,17,18,19,23,24,25,26] + [30,31,32]
        #print(len(select_with_pelvis))
        wbody_pos = wbody_pos[..., select_with_pelvis, :]
        wbody_mat = wbody_mat[..., select_with_pelvis, :, :]
        wbody_rot = wbody_rot[..., select_with_pelvis, :]
        if return_full:
            return_dict.global_velocity_extend = self._compute_velocity(wbody_pos, dt) 
            return_dict.global_angular_velocity_extend = self._compute_angular_velocity(wbody_rot, dt)

        return_dict.global_translation_extend = wbody_pos.clone()
        return_dict.global_rotation_mat_extend = wbody_mat.clone()
        return_dict.global_rotation_extend = wbody_rot.clone()

        wbody_pos = wbody_pos[..., :-3, :]
        wbody_mat = wbody_mat[..., :-3, :, :]
        wbody_rot = wbody_rot[..., :-3, :]
        
        return_dict.global_translation = wbody_pos.clone()
        return_dict.global_rotation_mat = wbody_mat.clone()
        return_dict.global_rotation = wbody_rot.clone()
            
        if return_full:
            rigidbody_linear_velocity = self._compute_velocity(wbody_pos, dt)  # Isaac gym is [x, y, z, w]. All the previous functions are [w, x, y, z]
            rigidbody_angular_velocity = self._compute_angular_velocity(wbody_rot, dt)
            #print(tRot.wxyz_to_xyzw(pose_quat))
            return_dict.local_rotation = tRot.wxyz_to_xyzw(pose_quat)[..., select_with_pelvis, :][..., :-3, :]
            return_dict.global_root_velocity = rigidbody_linear_velocity[..., 0, :]
            return_dict.global_root_angular_velocity = rigidbody_angular_velocity[..., 0, :]
            return_dict.global_angular_velocity = rigidbody_angular_velocity
            return_dict.global_velocity = rigidbody_linear_velocity
            
            return_dict.dof_pos = pose.sum(dim = -1)[..., 1:][..., select_no_pelvis][..., :-3] # you can sum it up since unitree's each joint has 1 dof. Last two are for hands. doesn't really matter. 
            
            dof_vel = ((return_dict.dof_pos[:, 1:] - return_dict.dof_pos[:, :-1] )/dt)
            return_dict.dof_vels = torch.cat([dof_vel, dof_vel[:, -2:-1]], dim = 1)
            return_dict.fps = int(1/dt)

        return return_dict
    

    def forward_kinematics_batch(self, rotations, root_rotations, root_positions):
        """
        Perform forward kinematics using the given trajectory and local rotations.
        Arguments (where B = batch size, J = number of joints):
         -- rotations: (B, J, 4) tensor of unit quaternions describing the local rotations of each joint.
         -- root_positions: (B, 3) tensor describing the root joint positions.
        Output: joint positions (B, J, 3)
        """
        
        device, dtype = root_rotations.device, root_rotations.dtype
        B, seq_len = rotations.size()[0:2]
        J = self._offsets.shape[1]
        positions_world = []
        rotations_world = []

        expanded_offsets = (self._offsets[:, None].expand(B, seq_len, J, 3).to(device).type(dtype))
        # print(expanded_offsets.shape, J)

        for i in range(J):
            if self._parents[i] == -1:
                positions_world.append(root_positions)
                rotations_world.append(root_rotations)
            else:
                jpos = (torch.matmul(rotations_world[self._parents[i]][:, :, 0], expanded_offsets[:, :, i, :, None]).squeeze(-1) + positions_world[self._parents[i]])
                rot_mat = torch.matmul(rotations_world[self._parents[i]], torch.matmul(self._local_rotation_mat[:,  (i):(i + 1)], rotations[:, :, (i - 1):i, :]))
                # rot_mat = torch.matmul(rotations_world[self._parents[i]], rotations[:, :, (i - 1):i, :])
                # print(rotations[:, :, (i - 1):i, :].shape, self._local_rotation_mat.shape)
                
                positions_world.append(jpos)
                rotations_world.append(rot_mat)
        
        positions_world = torch.stack(positions_world, dim=2)
        rotations_world = torch.cat(rotations_world, dim=2)
        return positions_world, rotations_world
    
    @staticmethod
    def _compute_velocity(p, time_delta, guassian_filter=True):
        velocity = np.gradient(p.numpy(), axis=-3) / time_delta
        if guassian_filter:
            velocity = torch.from_numpy(filters.gaussian_filter1d(velocity, 2, axis=-3, mode="nearest")).to(p)
        else:
            velocity = torch.from_numpy(velocity).to(p)
        
        return velocity
    
    @staticmethod
    def _compute_angular_velocity(r, time_delta: float, guassian_filter=True):
        # assume the second last dimension is the time axis
        diff_quat_data = pRot.quat_identity_like(r).to(r)
        diff_quat_data[..., :-1, :, :] = pRot.quat_mul_norm(r[..., 1:, :, :], pRot.quat_inverse(r[..., :-1, :, :]))
        diff_angle, diff_axis = pRot.quat_angle_axis(diff_quat_data)
        angular_velocity = diff_axis * diff_angle.unsqueeze(-1) / time_delta
        if guassian_filter:
            angular_velocity = torch.from_numpy(filters.gaussian_filter1d(angular_velocity.numpy(), 2, axis=-3, mode="nearest"),)
        return angular_velocity  
