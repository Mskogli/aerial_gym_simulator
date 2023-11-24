# Copyright (c) 2023, Autonomous Robots Lab, Norwegian University of Science and Technology
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
import numpy as np
import os
import torch
import sys

from matplotlib import pyplot as plt
import cv2 as cv

from sevae.inference.scripts.VAENetworkInterface import VAENetworkInterface

from aerial_gym import AERIAL_GYM_ROOT_DIR, AERIAL_GYM_ROOT_DIR

from isaacgym import gymutil, gymtorch, gymapi
from isaacgym.torch_utils import *

from aerial_gym.envs.base.base_task import BaseTask
from .aerial_robot_with_obstacles_config import AerialRobotWithObstaclesCfg

from aerial_gym.envs.controllers.controller import Controller

from aerial_gym.utils.asset_manager import AssetManager

from aerial_gym.utils.helpers import asset_class_to_AssetOptions
import time


class AerialRobotWithObstacles(BaseTask):

    def __init__(self, cfg: AerialRobotWithObstaclesCfg, sim_params, physics_engine, sim_device, headless):
        self.cfg = cfg

        self.max_episode_length = int(self.cfg.env.episode_length_s / self.cfg.sim.dt)
        self.debug_viz = False

        self.sim_params = sim_params
        self.physics_engine = physics_engine
        self.sim_device_id = sim_device
        self.headless = headless
        self.saved_frame = False

        self.enable_onboard_cameras = self.cfg.env.enable_onboard_cameras

        self.env_asset_manager = AssetManager(self.cfg, sim_device)
        self.cam_resolution = (480,270)

        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)
        self.root_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        self.num_obstacles = self.env_asset_manager.get_env_actor_count() 
    
        num_actors = self.num_obstacles + 1 # Number of obstacles in the environment + one robot
        bodies_per_env = self.env_asset_manager.get_env_link_count() + self.robot_num_bodies # Number of links in the environment + robot
        self.seVAE = VAENetworkInterface(device=self.sim_device)

        self.vec_root_tensor = gymtorch.wrap_tensor(
            self.root_tensor).view(self.num_envs, num_actors, 13)

        self.root_states = self.vec_root_tensor[:, 0, :]
        self.root_positions = self.root_states[..., 0:3]
        self.root_quats = self.root_states[..., 3:7]
        self.root_linvels = self.root_states[..., 7:10]
        self.root_angvels = self.root_states[..., 10:13]

        self.env_asset_root_states = self.vec_root_tensor[:, 1:, :]
        
        self.privileged_obs_buf = None
        if self.vec_root_tensor.shape[1] > 1:
            if self.get_privileged_obs:
                self.privileged_obs_buf = self.env_asset_root_states.clone()


        self.contact_forces = gymtorch.wrap_tensor(self.contact_force_tensor).view(self.num_envs, bodies_per_env, 3)[:, 0]

        self.collisions = torch.zeros(self.num_envs, device=self.device)

        self.goal_positions = torch.zeros(self.num_envs, 3,  device=self.device, dtype=torch.float32)
        self.travel_distances = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)

        self.initial_root_states = self.root_states.clone()
        self.counter = 0

        self.action_upper_limits = torch.tensor(
            [1, 1, 1, 1], device=self.device, dtype=torch.float32)
        self.action_lower_limits = torch.tensor(
            [-1, -1, -1, -1], device=self.device, dtype=torch.float32)

        # control tensors
        self.action_input = torch.zeros(
            (self.num_envs, 4), dtype=torch.float32, device=self.device, requires_grad=False)
        self.forces = torch.zeros((self.num_envs, bodies_per_env, 3),
                                  dtype=torch.float32, device=self.device, requires_grad=False)
        self.torques = torch.zeros((self.num_envs, bodies_per_env, 3),
                                   dtype=torch.float32, device=self.device, requires_grad=False)
        
        self.controller = Controller(self.cfg.control, self.device)

        # Getting environment bounds
        self.env_lower_bound = torch.zeros(
            (self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.env_upper_bound = torch.zeros(
            (self.num_envs, 3), dtype=torch.float32, device=self.device)


        if self.cfg.env.enable_onboard_cameras:
            self.full_camera_array = torch.zeros((self.num_envs, 270, 480), device=self.device)
            self.latents = torch.zeros((self.num_envs, 128), device=self.device)
        

        if self.viewer:
            cam_pos_x, cam_pos_y, cam_pos_z = self.cfg.viewer.pos[0], self.cfg.viewer.pos[1], self.cfg.viewer.pos[2]
            cam_target_x, cam_target_y, cam_target_z = self.cfg.viewer.lookat[0], self.cfg.viewer.lookat[1], self.cfg.viewer.lookat[2]
            cam_pos = gymapi.Vec3(cam_pos_x, cam_pos_y, cam_pos_z)
            cam_target = gymapi.Vec3(cam_target_x, cam_target_y, cam_target_z)
            cam_ref_env = self.cfg.viewer.ref_env
            
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    def create_sim(self):
        self.sim = self.gym.create_sim(
            self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        if self.cfg.env.create_ground_plane:
            self._create_ground_plane()
        self._create_envs()
        self.progress_buf = torch.zeros(
            self.cfg.env.num_envs, device=self.sim_device, dtype=torch.long)

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)
        return

    def _create_envs(self):
        print("\n\n\n\n\n CREATING ENVIRONMENT \n\n\n\n\n\n")
        asset_path = self.cfg.robot_asset.file.format(
            AERIAL_GYM_ROOT_DIR=AERIAL_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = asset_class_to_AssetOptions(self.cfg.robot_asset)

        robot_asset = self.gym.load_asset(
            self.sim, asset_root, asset_file, asset_options)

        self.robot_num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)

        start_pose = gymapi.Transform()
        # create env instance
        pos = torch.tensor([0, 0, 0], device=self.device)
        start_pose.p = gymapi.Vec3(*pos)
        self.env_spacing = self.cfg.env.env_spacing
        env_lower = gymapi.Vec3(-self.env_spacing, -
                                self.env_spacing, -self.env_spacing)
        env_upper = gymapi.Vec3(
            self.env_spacing, self.env_spacing, self.env_spacing)
        self.actor_handles = []
        self.env_asset_handles = []
        self.envs = []
        self.camera_handles = []
        self.camera_tensors = []


        # Set Camera Properties
        camera_props = gymapi.CameraProperties()
        camera_props.enable_tensors = True
        camera_props.width = self.cam_resolution[0]
        camera_props.height = self.cam_resolution[1]
        camera_props.far_plane = 15.0
        camera_props.horizontal_fov = 87.0
        # local camera transform
        local_transform = gymapi.Transform()
        # position of the camera relative to the body
        local_transform.p = gymapi.Vec3(0.15, 0.00, 0.05)
        # orientation of the camera relative to the body
        local_transform.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)


        self.segmentation_counter = 0


        for i in range(self.num_envs):
            # create environment
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            # insert robot asset
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, "robot", i, self.cfg.robot_asset.collision_mask, 0)
            # append to lists
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

            if self.enable_onboard_cameras:
                cam_handle = self.gym.create_camera_sensor(env_handle, camera_props)
                self.gym.attach_camera_to_body(cam_handle, env_handle, actor_handle, local_transform, gymapi.FOLLOW_TRANSFORM)
                self.camera_handles.append(cam_handle)
                camera_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env_handle, cam_handle, gymapi.IMAGE_DEPTH)
                torch_cam_tensor = gymtorch.wrap_tensor(camera_tensor)
                self.camera_tensors.append(torch_cam_tensor)

            env_asset_list = self.env_asset_manager.prepare_assets_for_simulation(self.gym, self.sim)
            asset_counter = 0

            # have the segmentation counter be the max defined semantic id + 1. Use this to set the semantic mask of objects that are
            # do not have a defined semantic id in the config file, but still requre one. Increment for every instance in the next snippet
            for dict_item in env_asset_list:
                self.segmentation_counter = max(self.segmentation_counter, int(dict_item["semantic_id"])+1)

            for dict_item in env_asset_list:
                folder_path = dict_item["asset_folder_path"]
                filename = dict_item["asset_file_name"]
                asset_options = dict_item["asset_options"]
                whole_body_semantic = dict_item["body_semantic_label"]
                per_link_semantic = dict_item["link_semantic_label"]
                semantic_masked_links = dict_item["semantic_masked_links"]
                semantic_id = dict_item["semantic_id"]
                color = dict_item["color"]
                collision_mask = dict_item["collision_mask"]

                loaded_asset = self.gym.load_asset(self.sim, folder_path, filename, asset_options)


                assert not (whole_body_semantic and per_link_semantic)
                if semantic_id < 0:
                    object_segmentation_id = self.segmentation_counter
                    self.segmentation_counter += 1
                else:
                    object_segmentation_id = semantic_id

                asset_counter += 1

                env_asset_handle = self.gym.create_actor(env_handle, loaded_asset, start_pose, "env_asset_"+str(asset_counter), i, collision_mask, object_segmentation_id)
                self.env_asset_handles.append(env_asset_handle)
                if len(self.gym.get_actor_rigid_body_names(env_handle, env_asset_handle)) > 1:
                    print("Env asset has rigid body with more than 1 link: ", len(self.gym.get_actor_rigid_body_names(env_handle, env_asset_handle)))
                    sys.exit(0)

                if per_link_semantic:
                    rigid_body_names = None
                    if len(semantic_masked_links) == 0:
                        rigid_body_names = self.gym.get_actor_rigid_body_names(env_handle, env_asset_handle)
                    else:
                        rigid_body_names = semantic_masked_links
                    for rb_index in range(len(rigid_body_names)):
                        self.segmentation_counter += 1
                        self.gym.set_rigid_body_segmentation_id(env_handle, env_asset_handle, rb_index, self.segmentation_counter)
            
                if color is None:
                    color = np.random.randint(low=50,high=200,size=3)

                self.gym.set_rigid_body_color(env_handle, env_asset_handle, 0, gymapi.MESH_VISUAL,
                        gymapi.Vec3(color[0]/255,color[1]/255,color[2]/255))

            if self.enable_onboard_cameras:
                self.camera_handles.append(cam_handle)
                camera_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env_handle, cam_handle, gymapi.IMAGE_DEPTH)
                torch_cam_tensor = gymtorch.wrap_tensor(camera_tensor)
                self.camera_tensors.append(torch_cam_tensor)

            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)


        self.robot_body_props = self.gym.get_actor_rigid_body_properties(self.envs[0],self.actor_handles[0])
        self.robot_mass = 0
        for prop in self.robot_body_props:
            self.robot_mass += prop.mass
        print("Total robot mass: ", self.robot_mass)
        
        print("\n\n\n\n\n ENVIRONMENT CREATED \n\n\n\n\n\n")


    def step(self, actions):
        # step physics and render each frame
        for i in range(self.cfg.env.num_control_steps_per_env_step):
            self.pre_physics_step(actions)
            self.gym.simulate(self.sim)
            # NOTE: as per the isaacgym docs, self.gym.fetch_results must be called after self.gym.simulate, but not having it here seems to work fine
            # it is called in the render function.
            self.post_physics_step()

        self.render(sync_frame_time=False)
        if self.enable_onboard_cameras:
            self.render_cameras()
        
        self.progress_buf += 1

        self.check_collisions()
        self._compute_travel_distances()
        self.compute_observations()
        self.compute_reward()

        if self.cfg.env.reset_on_collision:
            ones = torch.ones_like(self.reset_buf)
            self.reset_buf = torch.where(self.collisions > 0, ones, self.reset_buf)

        self.extras["resets"] = self.reset_buf # Record resets in order to facilitate data logging
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        self.time_out_buf = self.progress_buf > self.max_episode_length
        self.extras["time_outs"] = self.time_out_buf
        self.extras["actions"] = actions  
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def reset_idx(self, env_ids):
        num_resets = len(env_ids)
        if 0 in env_ids:
            print("\n\n\n RESETTING ENV 0 \n\n\n")

        self.env_asset_manager.randomize_pose()
        self.env_asset_root_states[env_ids, :, 0:3] = self.env_asset_manager.asset_pose_tensor[env_ids, :, 0:3]
        euler_angles = self.env_asset_manager.asset_pose_tensor[env_ids, :, 3:6]
        self.env_asset_root_states[env_ids, :, 3:7] = quat_from_euler_xyz(euler_angles[..., 0], euler_angles[..., 1], euler_angles[..., 2])
        self.env_asset_root_states[env_ids, :, 
                         7:10] = torch_rand_float_3(-0.1, 0.1, (num_resets, self.num_obstacles, 3), self.device)
        self.env_asset_root_states[env_ids, :,
                         10:13] = torch_rand_float_3(-0.1, 0.1, (num_resets, self.num_obstacles, 3), self.device)
        
        self.env_asset_root_states[env_ids, :, 7:10] = torch.tensor([0, 0, -1], device=self.device, dtype=torch.float32)
        self.env_asset_root_states[env_ids, :, 10:13] = 0

        env_asset_vel_lower_bound = torch.tensor([-0.1, -0.1,-0.1], device=self.device, dtype=torch.float32)
        env_asset_vel_upper_bound = torch.tensor([0.1, 0.1, 0.1], device=self.device, dtype=torch.float32)
        asset_lin_vel_rand_sample = torch.rand((num_resets, self.num_obstacles, 3), device=self.device)
        asset_rot_vel_rand_sample = torch.rand((num_resets, self.num_obstacles, 3), device=self.device)

        #self.env_asset_root_states[env_ids, :, 7:10] =  (env_asset_vel_lower_bound - env_asset_vel_upper_bound)*asset_lin_vel_rand_sample + env_asset_vel_lower_bound
        #self.env_asset_root_states[env_ids, :, 10:13] =  (env_asset_vel_lower_bound - env_asset_vel_upper_bound)*asset_rot_vel_rand_sample + env_asset_vel_lower_bound


        # get environment lower and upper bounds
        self.env_lower_bound[env_ids] = torch.tensor([-5, -5, 0], device=self.device, dtype=torch.float32)
        self.env_upper_bound[env_ids] = torch.tensor([5, 5, 5], device=self.device, dtype=torch.float32)

        drone_pos_rand_sample = torch.rand((num_resets, 3), device=self.device)
        drone_positions = (self.env_upper_bound[env_ids] - self.env_lower_bound[env_ids] -
                           0.50)*drone_pos_rand_sample + (self.env_lower_bound[env_ids]+ 0.25)

        goal_pos_rand_sample = torch.rand((num_resets, 3), device=self.device)
        self.goal_positions[env_ids] = (self.env_upper_bound[env_ids] - self.env_lower_bound[env_ids] -
                           0.50)*goal_pos_rand_sample + (self.env_lower_bound[env_ids]+ 0.25)
          
        self.travel_distances[env_ids] = 0 

        # set drone positions that are sampled within environment bounds
        # x = torch.tensor([0, 0, 2.5], device=self.device) 
        self.root_states[env_ids,
                         0:3] = drone_positions
        self.root_states[env_ids,
                         7:10] = 0.0*torch_rand_float(-1.0, 1.0, (num_resets, 3), self.device)
        self.root_states[env_ids,
                         10:13] = 0.0*torch_rand_float(-1.0, 1.0, (num_resets, 3), self.device)

        self.root_states[env_ids, 3:6] = 0 # standard orientation, can be randomized
        self.root_states[env_ids, 6] = 1

        self.compute_observations() # Reset the observation buffer

        self.gym.set_actor_root_state_tensor(self.sim, self.root_tensor)
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1

    def pre_physics_step(self, _actions):
        # resets
        if self.counter % 250 == 0:
            print("self.counter:", self.counter)
        self.counter += 1

        
        actions = _actions.to(self.device)
        actions = tensor_clamp(
            actions, self.action_lower_limits, self.action_upper_limits)
        self.action_input[:] = actions

        # clear actions for reset envs
        self.forces[:] = 0.0
        self.torques[:, :] = 0.0

        output_thrusts_mass_normalized, output_torques_inertia_normalized = self.controller(self.root_states, self.action_input)
        self.forces[:, 0, 2] = self.robot_mass * (-self.sim_params.gravity.z) * output_thrusts_mass_normalized
        self.torques[:, 0] = output_torques_inertia_normalized
        self.forces = torch.where(self.forces < 0, torch.zeros_like(self.forces), self.forces)

        # apply actions
        self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(
            self.forces), gymtorch.unwrap_tensor(self.torques), gymapi.LOCAL_SPACE)

    def render_cameras(self):        
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)
        self.dump_images()
        self._process_depth_images()
        self._compute_latent_representation()
        self.gym.end_access_image_tensors(self.sim)
        return

    def post_physics_step(self): 
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
    
    def check_collisions(self):
        ones = torch.ones((self.num_envs), device=self.device)
        zeros = torch.zeros((self.num_envs), device=self.device)
        self.collisions[:] = 0
        self.collisions = torch.where(torch.norm(self.contact_forces, dim=1) > 0.01, ones, zeros)


    def dump_images(self):
        for env_id in range(self.num_envs):
            # the depth values are in -ve z axis, so we need to flip it to positive
            self.full_camera_array[env_id] = -self.camera_tensors[env_id]
    
    # Post process depth images before computing the latent representation
    def _process_depth_images(self):
        IMAGE_MAX_DEPTH = 10

        self.full_camera_array[torch.isnan(self.full_camera_array)] = IMAGE_MAX_DEPTH
        self.full_camera_array[self.full_camera_array > IMAGE_MAX_DEPTH] = IMAGE_MAX_DEPTH
        self.full_camera_array[self.full_camera_array < 0.20] = -1.0
        
        self.full_camera_array = self.full_camera_array/IMAGE_MAX_DEPTH
        self.full_camera_array[self.full_camera_array < 0.2/IMAGE_MAX_DEPTH] = -1.0
    
    # Forward pass of the seVAE encoder
    def _compute_latent_representation(self):
        self.latents = self.seVAE.forward_torch(self.full_camera_array).clone().to(self.device)

    def _compute_travel_distances(self):
        self.travel_distances += (self.obs_buf[..., 131:134] - self.root_positions).pow(2).sqrt().sum(1)

    def compute_observations(self):
        self.obs_buf[..., :128] = self.latents
        self.obs_buf[..., 128:131] = self.goal_positions
        self.obs_buf[..., 131:134] = self.root_positions
        self.obs_buf[..., 134:138] = self.root_quats
        self.obs_buf[..., 138:141] = self.root_linvels
        self.obs_buf[..., 141:144] = self.root_angvels
        return self.obs_buf

    def compute_reward(self):
        self.rew_buf[:], self.reset_buf[:] = compute_quadcopter_reward(
            self.root_positions,
            self.goal_positions,
            self.root_quats,
            self.root_linvels,
            self.root_angvels,
            self.reset_buf, self.progress_buf, self.collisions, self.travel_distances, self.max_episode_length
        )


###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def quat_rotate(q, v):
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * \
        torch.bmm(q_vec.view(shape[0], 1, 3), v.view(
            shape[0], 3, 1)).squeeze(-1) * 2.0
    return a + b + c

@torch.jit.script
def quat_axis(q, axis=0):
    # type: (Tensor, int) -> Tensor
    basis_vec = torch.zeros(q.shape[0], 3, device=q.device)
    basis_vec[:, axis] = 1
    return quat_rotate(q, basis_vec)

@torch.jit.script
def compute_quadcopter_reward(root_positions, goal_position, root_quats, root_linvels, root_angvels, reset_buf, progress_buf, collisions, distances, max_episode_length):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float) -> Tuple[Tensor, Tensor]

    R_OBST = -30.0
    R_GOAL = 10
    R_TO = 0.0
    R_DIST = -0.5
    R_MB = -30

    alpha_tiltage = -0.3
    alpha_spinnage = -0.3
    alpha_pos = 1.0
    alpha_vels = -0.1

    # distance to target
    target_dist = (root_positions - goal_position).pow(2).sqrt().sum(1)
    pos_reward = 1.0 / (1.0 + target_dist * target_dist)

    # uprightness
    ups = quat_axis(root_quats, 2)
    tiltage = torch.abs(1 - ups[..., 2])

    # spinning
    up_reward = tiltage*tiltage

    # spinning
    spinnage = torch.abs(root_angvels[..., 2])
    spinnage_reward = spinnage*spinnage

    vel_reward = torch.abs(root_linvels[..., 2]).pow(2)

    tilt_spin_scaling = torch.where(pos_reward > 1.0, pos_reward, torch.ones_like(pos_reward))

    # combined reward
    # uprigness and spinning only matter when close to the target
    reward = alpha_pos*pos_reward + tilt_spin_scaling*(alpha_tiltage*up_reward + alpha_spinnage*spinnage_reward) + alpha_vels*vel_reward

    # resets due to misbehavior
    ones = torch.ones_like(reset_buf)
    die = torch.zeros_like(reset_buf)

    resets_timeouts = torch.where(progress_buf >= max_episode_length - 1, ones, die)
    resets_misbehaviour = torch.where(torch.norm(root_positions, dim=1) > 20, ones, die)
    resets_collisions = torch.where(collisions > 0, ones, die)
    resets_goal = torch.where(target_dist < 1.0, ones, die)

    reset = resets_timeouts + resets_misbehaviour + resets_collisions + resets_goal
    
    # terminal rewards incurred at the end of the episode
    reward = reward + R_OBST*resets_collisions + R_TO*resets_timeouts + torch.mul(R_DIST*distances, reset) + R_GOAL*resets_goal + R_MB*resets_misbehaviour

    return reward, reset
