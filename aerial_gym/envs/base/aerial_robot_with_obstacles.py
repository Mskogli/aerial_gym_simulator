# Copyright (c) 2023, Autonomous Robots Lab, Norwegian University of Science and Technology
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import os
import torch
import sys


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
    def __init__(
        self,
        cfg: AerialRobotWithObstaclesCfg,
        sim_params,
        physics_engine,
        sim_device,
        headless,
    ):
        self.cfg = cfg

        self.max_episode_length = int(self.cfg.env.episode_length_s / self.cfg.sim.dt)
        self.debug_viz = False

        self.sim_params = sim_params
        self.physics_engine = physics_engine
        self.sim_device_id = sim_device
        self.headless = headless
        self.saved_frame = False
        self.dynamic_assets = cfg.dynamic_assets
        self.prediction_horizon = 10

        self.enable_onboard_cameras = self.cfg.env.enable_onboard_cameras

        self.env_asset_manager = AssetManager(self.cfg, sim_device)
        self.cam_resolution = (135, 240)
        self.max_depth_value = 20
        self.min_depth_value = 0
        self.pred_horizon = 10

        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)
        self.root_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        num_actors = (
            self.env_asset_manager.get_env_actor_count() + 1
        )  # Number of obstacles in the environment + one robot
        bodies_per_env = (
            self.env_asset_manager.get_env_link_count() + self.robot_num_bodies
        )  # Number of links in the environment + robot

        self.vec_root_tensor = gymtorch.wrap_tensor(self.root_tensor).view(
            self.num_envs, num_actors, 13
        )

        self.root_states = self.vec_root_tensor[:, 0, :]
        self.root_positions = self.root_states[..., 0:3]
        self.root_quats = self.root_states[..., 3:7]
        self.root_linvels = self.root_states[..., 7:10]
        self.root_angvels = self.root_states[..., 10:13]

        self.prev_root_positions = torch.zeros_like(self.root_positions)

        self.env_asset_root_states = self.vec_root_tensor[:, 1:, :]

        self.privileged_obs_buf = None
        if self.vec_root_tensor.shape[1] > 1:
            if self.get_privileged_obs:
                self.privileged_obs_buf = self.env_asset_root_states.clone()

        self.contact_forces = gymtorch.wrap_tensor(self.contact_force_tensor).view(
            self.num_envs, bodies_per_env, 3
        )[:, 0]

        self.collisions = torch.zeros(self.num_envs, device=self.device)

        self.goal_positions = torch.zeros(
            self.num_envs, 3, device=self.device, dtype=torch.float32
        )

        self.initial_root_states = self.root_states.clone()
        self.counter = 0

        self.action_upper_limits = torch.tensor(
            [1, 1, 1, 1], device=self.device, dtype=torch.float32
        )
        self.action_lower_limits = torch.tensor(
            [-1, -1, -1, -1], device=self.device, dtype=torch.float32
        )

        self.trajectory = torch.zeros(
            (self.num_envs, self.pred_horizon, 4),
            dtype=torch.float32,
            device=self.device,
        )

        # control tensors
        self.action_input = torch.zeros(
            self.num_envs,
            4,
            dtype=torch.float32,
            device=self.device,
            requires_grad=False,
        )

        self.forces = torch.zeros(
            (self.num_envs, bodies_per_env, 3),
            dtype=torch.float32,
            device=self.device,
            requires_grad=False,
        )
        self.torques = torch.zeros(
            (self.num_envs, bodies_per_env, 3),
            dtype=torch.float32,
            device=self.device,
            requires_grad=False,
        )

        self.controller = Controller(self.cfg.control, self.device)

        # Getting environment bounds
        self.env_lower_bound = torch.zeros(
            (self.num_envs, 3), dtype=torch.float32, device=self.device
        )
        self.env_upper_bound = torch.zeros(
            (self.num_envs, 3), dtype=torch.float32, device=self.device
        )

        if self.cfg.env.enable_onboard_cameras:

            self.full_camera_array = torch.zeros(
                (self.num_envs, self.cam_resolution[0], self.cam_resolution[1]),
                device=self.device,
            )

        if self.viewer:
            cam_pos_x, cam_pos_y, cam_pos_z = (
                self.cfg.viewer.pos[0],
                self.cfg.viewer.pos[1],
                self.cfg.viewer.pos[2],
            )
            cam_target_x, cam_target_y, cam_target_z = (
                self.cfg.viewer.lookat[0],
                self.cfg.viewer.lookat[1],
                self.cfg.viewer.lookat[2],
            )
            cam_pos = gymapi.Vec3(cam_pos_x, cam_pos_y, cam_pos_z)
            cam_target = gymapi.Vec3(cam_target_x, cam_target_y, cam_target_z)
            cam_ref_env = self.cfg.viewer.ref_env

            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    def create_sim(self):
        self.sim = self.gym.create_sim(
            self.sim_device_id,
            self.graphics_device_id,
            self.physics_engine,
            self.sim_params,
        )
        if self.cfg.env.create_ground_plane:
            self._create_ground_plane()
        self._create_envs()
        self.progress_buf = torch.zeros(
            self.cfg.env.num_envs, device=self.sim_device, dtype=torch.long
        )

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)
        return

    def _create_envs(self):
        print("\n\n\n\n\n CREATING ENVIRONMENT \n\n\n\n\n\n")
        asset_path = self.cfg.robot_asset.file.format(
            AERIAL_GYM_ROOT_DIR=AERIAL_GYM_ROOT_DIR
        )
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = asset_class_to_AssetOptions(self.cfg.robot_asset)

        robot_asset = self.gym.load_asset(
            self.sim, asset_root, asset_file, asset_options
        )

        self.robot_num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)

        start_pose = gymapi.Transform()
        # create env instance
        pos = torch.tensor([0, 0, 0], device=self.device)
        start_pose.p = gymapi.Vec3(*pos)
        self.env_spacing = self.cfg.env.env_spacing
        env_lower = gymapi.Vec3(-self.env_spacing, -self.env_spacing, -self.env_spacing)
        env_upper = gymapi.Vec3(self.env_spacing, self.env_spacing, self.env_spacing)
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
            env_handle = self.gym.create_env(
                self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs))
            )
            # insert robot asset
            actor_handle = self.gym.create_actor(
                env_handle,
                robot_asset,
                start_pose,
                "robot",
                i,
                self.cfg.robot_asset.collision_mask,
                0,
            )
            # append to lists
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

            if self.enable_onboard_cameras:
                cam_handle = self.gym.create_camera_sensor(env_handle, camera_props)
                self.gym.attach_camera_to_body(
                    cam_handle,
                    env_handle,
                    actor_handle,
                    local_transform,
                    gymapi.FOLLOW_TRANSFORM,
                )
                self.camera_handles.append(cam_handle)
                camera_tensor = self.gym.get_camera_image_gpu_tensor(
                    self.sim, env_handle, cam_handle, gymapi.IMAGE_DEPTH
                )
                torch_cam_tensor = gymtorch.wrap_tensor(camera_tensor)
                self.camera_tensors.append(torch_cam_tensor)
                print("Batman")

            env_asset_list = self.env_asset_manager.prepare_assets_for_simulation(
                self.gym, self.sim
            )

            asset_counter = 0

            # have the segmentation counter be the max defined semantic id + 1. Use this to set the semantic mask of objects that are
            # do not have a defined semantic id in the config file, but still requre one. Increment for every instance in the next snippet
            for dict_item in env_asset_list:
                self.segmentation_counter = max(
                    self.segmentation_counter, int(dict_item["semantic_id"]) + 1
                )

            for i, dict_item in enumerate(env_asset_list):
                folder_path = dict_item["asset_folder_path"]
                filename = dict_item["asset_file_name"]
                asset_options = dict_item["asset_options"]
                whole_body_semantic = dict_item["body_semantic_label"]
                per_link_semantic = dict_item["link_semantic_label"]
                semantic_masked_links = dict_item["semantic_masked_links"]
                semantic_id = dict_item["semantic_id"]
                color = dict_item["color"]
                collision_mask = dict_item["collision_mask"]

                loaded_asset = self.gym.load_asset(
                    self.sim, folder_path, filename, asset_options
                )

                assert not (whole_body_semantic and per_link_semantic)
                if semantic_id < 0:
                    object_segmentation_id = self.segmentation_counter
                    self.segmentation_counter += 1
                else:
                    object_segmentation_id = semantic_id

                asset_counter += 1

                env_asset_handle = self.gym.create_actor(
                    env_handle,
                    loaded_asset,
                    start_pose,
                    "env_asset_" + str(asset_counter),
                    i,
                    collision_mask,
                    object_segmentation_id,
                )
                self.env_asset_handles.append(env_asset_handle)
                if (
                    len(
                        self.gym.get_actor_rigid_body_names(
                            env_handle, env_asset_handle
                        )
                    )
                    > 1
                ):
                    print(
                        "Env asset has rigid body with more than 1 link: ",
                        len(
                            self.gym.get_actor_rigid_body_names(
                                env_handle, env_asset_handle
                            )
                        ),
                    )
                    sys.exit(0)

                if per_link_semantic:
                    rigid_body_names = None
                    if len(semantic_masked_links) == 0:
                        rigid_body_names = self.gym.get_actor_rigid_body_names(
                            env_handle, env_asset_handle
                        )
                    else:
                        rigid_body_names = semantic_masked_links
                    for rb_index in range(len(rigid_body_names)):
                        self.segmentation_counter += 1
                        self.gym.set_rigid_body_segmentation_id(
                            env_handle,
                            env_asset_handle,
                            rb_index,
                            self.segmentation_counter,
                        )

                if color is None:
                    color = np.random.randint(low=50, high=200, size=3)
                self.gym.set_rigid_body_color(
                    env_handle,
                    env_asset_handle,
                    0,
                    gymapi.MESH_VISUAL,
                    gymapi.Vec3(color[0] / 255, color[1] / 255, color[2] / 255),
                )

            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

        self.robot_body_props = self.gym.get_actor_rigid_body_properties(
            self.envs[0], self.actor_handles[0]
        )
        self.robot_mass = 0
        for prop in self.robot_body_props:
            self.robot_mass += prop.mass
        print("Total robot mass: ", self.robot_mass)

        print("\n\n\n\n\n ENVIRONMENT CREATED \n\n\n\n\n\n")

    def step(self, trajectory):
        # step physics and render each frame
        self.trajectory = transform_actions(trajectory.to(self.device))
        for i in range(self.cfg.env.num_control_steps_per_env_step):
            self.pre_physics_step(self.trajectory[:, i])
            self.gym.simulate(self.sim)
            self.post_physics_step()

        self.render(sync_frame_time=False)
        if self.enable_onboard_cameras:
            self.render_cameras()

        self.progress_buf += 1

        self.check_collisions()
        self.compute_observations()
        self.compute_reward()

        if self.cfg.env.reset_on_collision:
            ones = torch.ones_like(self.reset_buf)
            self.reset_buf = torch.where(self.collisions > 0, ones, self.reset_buf)

        self.extras["resets"] = (
            self.reset_buf
        )  # Record resets in order to facilitate data logging
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        self.time_out_buf = self.progress_buf > self.max_episode_length
        self.extras["time_outs"] = self.time_out_buf

        return (
            self.obs_buf,
            self.privileged_obs_buf,
            self.rew_buf,
            self.reset_buf,
            self.extras,
        )

    def reset_idx(self, env_ids):
        num_resets = len(env_ids)
        if 0 in env_ids:
            print("\n\n\n RESETTING ENV 0 \n\n\n")

        self.env_asset_manager.randomize_pose()

        self.env_asset_root_states[env_ids, :, 0:3] = (
            self.env_asset_manager.asset_pose_tensor[env_ids, :, 0:3]
        )

        euler_angles = self.env_asset_manager.asset_pose_tensor[env_ids, :, 3:6]
        self.env_asset_root_states[env_ids, :, 3:7] = quat_from_euler_xyz(
            euler_angles[..., 0], euler_angles[..., 1], euler_angles[..., 2]
        )

        self.env_asset_root_states[env_ids, :, 7:13] = 0.0

        # get environment lower and upper bounds
        self.env_lower_bound[env_ids] = self.env_asset_manager.env_lower_bound.diagonal(
            dim1=-2, dim2=-1
        )
        self.env_upper_bound[env_ids] = self.env_asset_manager.env_upper_bound.diagonal(
            dim1=-2, dim2=-1
        )
        drone_pos_rand_sample = torch.rand((num_resets, 3), device=self.device)

        drone_positions = (
            self.env_upper_bound[env_ids] - self.env_lower_bound[env_ids] - 0.50
        ) * drone_pos_rand_sample + (self.env_lower_bound[env_ids] + 0.25)

        # set drone positions that are sampled within environment bounds

        self.root_states[env_ids, 0:3] = drone_positions
        self.root_states[env_ids, 7:10] = 0.2 * torch_rand_float(
            -1.0, 1.0, (num_resets, 3), self.device
        )
        self.root_states[env_ids, 10:13] = 0.2 * torch_rand_float(
            -1.0, 1.0, (num_resets, 3), self.device
        )

        self.root_states[env_ids, 3:6] = 0  # standard orientation, can be randomized
        self.root_states[env_ids, 6] = 1

        self.compute_observations()  # Reset the observation buffer

        self.gym.set_actor_root_state_tensor(self.sim, self.root_tensor)
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1

    def pre_physics_step(self, actions):
        # resets
        if self.counter % 250 == 0:
            print("self.counter:", self.counter)
        self.counter += 1

        self.action_input[:] = actions

        # clear actions for reset envs
        self.forces[:] = 0.0
        self.torques[:, :] = 0.0

        (
            quad_thrusts_mass_normalized,
            quad_torques_inertia_normalized,
        ) = self.controller(self.root_states, self.action_input)
        self.forces[:, 0, 2] = (
            self.robot_mass
            * (-self.sim_params.gravity.z)
            * quad_thrusts_mass_normalized
        )

        self.torques[:, 0] = quad_torques_inertia_normalized
        self.forces = torch.where(
            self.forces < 0, torch.zeros_like(self.forces), self.forces
        )

        if self.dynamic_assets:
            dynamic_asset_forces, dynamic_asset_torques = (
                self.env_asset_manager.compute_dyn_asset_forces(
                    self.env_asset_root_states, self.counter
                )
            )

            self.forces[:, 5:, :][
                :, self.env_asset_manager.dynamic_asset_ids, :
            ] = dynamic_asset_forces

            self.torques[:, 5:][
                :, self.env_asset_manager.dynamic_asset_ids
            ] = dynamic_asset_torques

        self.gym.apply_rigid_body_force_tensors(
            self.sim,
            gymtorch.unwrap_tensor(self.forces),
            gymtorch.unwrap_tensor(self.torques),
            gymapi.LOCAL_SPACE,
        )

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
        self.collisions[:]
        self.collisions = torch.where(
            torch.norm(self.contact_forces, dim=1) > 0.1, ones, zeros
        )

    def dump_images(self):
        for env_id in range(self.num_envs):
            # the depth values are in -ve z axis, so we need to flip it to positive
            self.full_camera_array[env_id] = -self.camera_tensors[env_id]

    def _process_depth_images(self):
        imgs[torch.isinf(imgs)] = self.max_depth_value
        imgs[imgs > self.max_depth_value] = self.max_depth_value
        imgs[imgs < self.min_depth_value] = self.min_depth_value
        imgs = (imgs - self.min_depth_value) / (
            self.max_depth_value - self.min_depth_value
        )

    def compute_observations(self):
        self.obs_buf[..., :128] = self.latents
        self.obs_buf[..., 128:131] = self.goal_positions
        self.obs_buf[..., 131:134] = self.root_positions
        self.obs_buf[..., 134:138] = self.root_quats
        self.obs_buf[..., 138:141] = self.root_linvels
        self.obs_buf[..., 141:144] = self.root_angvels
        return self.obs_buf

    # Call into the torch wrapper, step the world model, compute the current latent state and hidden state
    def step_world_model(self):
        pass

    # Compute rewards and environment resets
    def compute_reward(self):
        pass


###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def quat_rotate(q, v):
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w**2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = (
        q_vec
        * torch.bmm(q_vec.view(shape[0], 1, 3), v.view(shape[0], 3, 1)).squeeze(-1)
        * 2.0
    )
    return a + b + c


@torch.jit.script
def quat_axis(q, axis=0):
    # type: (Tensor, int) -> Tensor
    basis_vec = torch.zeros(q.shape[0], 3, device=q.device)
    basis_vec[:, axis] = 1
    return quat_rotate(q, basis_vec)


def transform_actions(actions):
    # type (Tensor) -> Tensor
    s_max = 1
    i_max = 1
    omega_max = 1

    v_x = torch.unsqueeze(
        s_max * ((actions[..., 0] + 1) / 2 * torch.cos(i_max * actions[..., 1])), -1
    )
    v_y = torch.zeros_like(v_x)
    v_z = torch.unsqueeze(
        s_max * (((actions[..., 0] + 1) / 2 * torch.sin(i_max * actions[..., 1]))), -1
    )
    omega_z = torch.unsqueeze(omega_max * actions[..., 2], -1)
    return torch.cat((v_x, v_y, v_z, omega_z), dim=-1)


def calculate_rewards(
    positions,
    prev_positions,
    goal_positions,
    trajectory,
    collisions,
    progress_buffer,
    max_episode_length,
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float) -> Tuple[Tensor, Tensor]

    # Tuning parameters
    nu_1 = 1
    nu_2 = 1
    nu_3 = 1

    nu_4_v = torch.tensor([0.1, 0.1, 0.1, 0.1], device=trajectory.device)
    nu_4 = 0.1
    nu_5_v = torch.tensor([0.1, 0.1, 0.1, 0.1], device=trajectory.device)
    nu_5 = 0.1
    nu_6 = 1

    distance_to_target = torch.linalg.vector_norm(goal_positions - positions)
    prev_distance_to_target = torch.linalg.vector_norm(goal_positions - prev_positions)
    distance_diff = prev_distance_to_target - distance_to_target
    trajectory_diff = trajectory[:, :-1] - trajectory[:, 1:]

    ones = torch.ones_like(progress_buffer)
    zeros = torch.zeros_like(progress_buffer)

    # Reward Terms
    r_1 = torch.exp(-(torch.square(distance_to_target) / nu_1))
    r_2 = torch.exp(-(torch.square(distance_to_target) / nu_2))
    r_3 = nu_3 * (distance_diff)
    reward = r_1 + r_2 + r_3

    # Penalty
    p_1 = nu_4 * torch.sum(
        torch.exp(-torch.square(trajectory) / nu_4_v) - 1, dim=(-1, -2)
    )
    p_2 = nu_5 * torch.sum(
        torch.exp(-torch.square(trajectory_diff) / nu_5_v) - 1, dim=(-1, -2)
    )
    p_3 = -nu_6 * torch.where(collisions > 0, ones, zeros)
    penalty = p_1 + p_2 + p_3

    resets_timeout = torch.where(progress_buffer >= max_episode_length - 1, ones, zeros)
    resets = resets_timeout + p_3

    return reward + penalty, resets
