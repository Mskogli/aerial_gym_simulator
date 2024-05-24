# Copyright (c) 2023, Autonomous Robots Lab, Norwegian University of Science and Technology
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
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
from s4wm.nn.s4_wm import S4WMTorchWrapper


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

        self.max_episode_length = int(
            self.cfg.env.episode_length_s
            / (self.cfg.env.num_control_steps_per_env_step * self.cfg.sim.dt)
        )
        self.debug_viz = False

        self.sim_params = sim_params
        self.physics_engine = physics_engine
        self.sim_device_id = sim_device
        self.headless = headless

        self.enable_onboard_cameras = self.cfg.env.enable_onboard_cameras

        self.env_asset_manager = AssetManager(self.cfg, sim_device)
        self.cam_resolution = (240, 135)

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
        self.applied_control_actions = []

        self.goals = 0

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
        self.timeouts = torch.zeros(self.num_envs, device=self.device)

        self.initial_root_states = self.root_states.clone()
        self.counter = 0

        self.action_upper_limits = torch.tensor(
            [1, 1, 1, 1], device=self.device, dtype=torch.float32
        )
        self.action_lower_limits = torch.tensor(
            [-1, -1, -1, -1], device=self.device, dtype=torch.float32
        )

        # control tensors
        self.action_input = torch.zeros(
            (self.num_envs, 4),
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

        self.successful = 0
        self.crash = 0

        # S4RL

        self.latent_dim = cfg.env.latent_dim
        self.hidden_dim = cfg.env.hidden_dim
        self.prediction_horizon = cfg.env.prediction_horizon

        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

        self.S4WM = S4WMTorchWrapper(
            self.num_envs,
            "/home/mathias/dev/rl_checkpoints/gaussian_128",
            d_latent=self.latent_dim * 2,
            d_pssm_blocks=self.hidden_dim,
            num_pssm_blocks=3,
            d_ssm=128,
            sample_mean=True,
        )

        self.left_env_bounds = False

        self.latent = torch.zeros(
            (self.num_envs, self.latent_dim),
            dtype=torch.float32,
            device=self.device,
            requires_grad=False,
        )
        self.hidden = torch.zeros(
            (self.num_envs, self.hidden_dim),
            dtype=torch.float32,
            device=self.device,
            requires_grad=False,
        )

        self.vehicle_frame_euler_angles = torch.zeros(
            (self.num_envs, 3), device=self.device, requires_grad=False
        )
        self.root_euler_angles = torch.zeros(
            (self.num_envs, 3), device=self.device, requires_grad=False
        )
        self.vehicle_frame_quats = torch.zeros(
            (self.num_envs, 4), device=self.device, requires_grad=False
        )
        self.goal_dir_vehicle_frame = torch.zeros(
            (self.num_envs, 3), device=self.device, requires_grad=False
        )
        self.angvels_body_frame = torch.zeros(
            (self.num_envs, 3), device=self.device, requires_grad=False
        )
        self.linvels_body_frame = torch.zeros(
            (self.num_envs, 3), device=self.device, requires_grad=False
        )
        self.linvels_vehicle_frame = torch.zeros(
            (self.num_envs, 3), device=self.device, requires_grad=False
        )
        self.unit_goal_dir_vehicle_frame = torch.zeros(
            (self.num_envs, 3), device=self.device, requires_grad=False
        )

        self.distances_to_target = torch.zeros(
            (self.num_envs),
            dtype=torch.float32,
            device=self.device,
            requires_grad=False,
        )
        self.prev_distances_to_target = torch.zeros_like(
            self.distances_to_target,
            dtype=torch.float32,
            device=self.device,
            requires_grad=False,
        )
        self.goal_positions = torch.zeros(
            (self.num_envs, 3),
            dtype=torch.float32,
            device=self.device,
            requires_grad=False,
        )

        self.num_rollouts = 0
        self.traj = []
        self.collided = False

        self.goal_spawning_offset = torch.tensor(
            self.cfg.goal_spawning_config.offset, device=self.device
        ).expand(self.num_envs, -1)
        self.goal_spawning_pos_min = torch.tensor(
            self.cfg.goal_spawning_config.min_position_ratio, device=self.device
        ).expand(self.num_envs, -1)
        self.goal_spawning_pos_max = torch.tensor(
            self.cfg.goal_spawning_config.max_position_ratio, device=self.device
        ).expand(self.num_envs, -1)

        self.robot_spawning_offset = torch.tensor(
            self.cfg.robot_spawning_config.offset, device=self.device
        ).expand(self.num_envs, -1)
        self.robot_spawning_pos_min = torch.tensor(
            self.cfg.robot_spawning_config.min_position_ratio, device=self.device
        ).expand(self.num_envs, -1)
        self.robot_spawning_pos_max = torch.tensor(
            self.cfg.robot_spawning_config.max_position_ratio, device=self.device
        ).expand(self.num_envs, -1)

        self.zeros_3d = torch.zeros(
            (self.num_envs, 3), device=self.device, requires_grad=False
        )

        self.max_depth_value = 10
        self.min_depth_value = 0

        self.ones = torch.ones_like(self.reset_buf, device=self.reset_buf.device)
        self.zeros = torch.zeros_like(self.reset_buf, device=self.reset_buf.device)

        if self.cfg.env.enable_onboard_cameras:
            self.full_camera_array = torch.zeros(
                (self.num_envs, 135, 240), device=self.device
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
                    color = np.random.randint(low=155, high=190, size=3)

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

    def step(self, actions):
        # step physics and render each frame
        if self.goals == 8 or self.left_env_bounds:
            actions = torch.zeros_like(actions)

        self.applied_control_actions.append(
            [
                actions[0][0].item(),
                actions[0][1].item(),
                actions[0][2].item(),
                actions[0][3].item(),
            ]
        )
        self.prev_root_positions = self.root_positions.detach().clone()
        for _ in range(self.prediction_horizon):
            for i in range(self.cfg.env.num_control_steps_per_env_step):
                self.pre_physics_step(actions)
                self.gym.simulate(self.sim)
                self.post_physics_step()

            self.gym.fetch_results(self.sim, True)
            self.progress_buf += 1
            self.check_collisions()
            self.timeouts = torch.where(
                self.progress_buf >= self.max_episode_length - 1, self.ones, self.zeros
            )

        self.compute_reward()
        self.prev_distances_to_target[:] = self.distances_to_target

        self.reset_buf = torch.where(self.collisions > 0, self.ones, self.zeros)
        self.reset_buf = torch.where(self.timeouts > 0, self.ones, self.reset_buf)
        # self.reset_buf = torch.where(
        #     self.prev_distances_to_target < 0.2, self.ones, self.reset_buf
        # )

        if self.prev_distances_to_target < 1.0 and self.prev_distances_to_target > 0:
            if self.goal_positions[0][1] == 14:
                self.goal_positions[0][1] = -14
            else:
                self.goal_positions[0][1] = 14
            # self.S4WM.reset_cache(torch.tensor([0], device=self.device))
            # self.hidden[0] = 0
            self.goals += 1

        if self.goals == 8:
            self.reset_buf = torch.ones_like(self.reset_buf)

        if self.root_positions[0][2] > 12:
            self.left_env_bounds = True
            self.reset_buf = torch.ones_like(self.reset_buf)

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        self.render(sync_frame_time=False)
        self.render_cameras()

        line = [
            self.prev_root_positions[0][0].item(),
            self.prev_root_positions[0][1].item(),
            self.prev_root_positions[0][2].item(),
            self.root_positions[0][0].item(),
            self.root_positions[0][1].item(),
            self.root_positions[0][2].item(),
        ]
        self.traj.append(line)
        # self.gym.add_lines(self.viewer, self.envs[0], 1, line, [1, 0, 0])

        self.latent, self.hidden = self.S4WM.forward(
            self.full_camera_array.view(self.num_envs, 1, 135, 240, 1),
            self.action_input.view(self.num_envs, 1, 4),
            self.latent.view(self.num_envs, 1, self.latent_dim),
        )
        self.latent = self.latent.squeeze()
        self.hidden = self.hidden.squeeze()
        self.S4WM.reset_cache(reset_env_ids)
        self.hidden[reset_env_ids] = 0

        self.compute_observations()

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

        if self.num_rollouts == 0:
            self.env_asset_manager.randomize_pose(reset_envs=env_ids)

        self.num_rollouts += 1

        self.env_asset_root_states[env_ids, :, 0:3] = (
            self.env_asset_manager.asset_pose_tensor[env_ids, :, 0:3]
        )

        euler_angles = self.env_asset_manager.asset_pose_tensor[env_ids, :, 3:6]
        self.env_asset_root_states[env_ids, :, 3:7] = quat_from_euler_xyz(
            euler_angles[..., 0], euler_angles[..., 1], euler_angles[..., 2]
        )
        self.env_asset_root_states[env_ids, :, 7:13] = 0.0

        # get environment lower and upper bounds
        self.env_lower_bound[env_ids] = self.env_asset_manager.env_lower_bound[env_ids]
        self.env_upper_bound[env_ids] = self.env_asset_manager.env_upper_bound[env_ids]

        # Randomize drone starting positions
        drone_pos_rand_sample = torch.rand((num_resets, 3), device=self.device)
        drone_spawning_min_bounds = (
            self.env_lower_bound[env_ids] + self.robot_spawning_offset[env_ids]
        )
        drone_spawning_max_bounds = (
            self.env_upper_bound[env_ids] - self.robot_spawning_offset[env_ids]
        )
        drone_spawning_ratio_in_env_bound = (
            drone_pos_rand_sample
            * (
                self.robot_spawning_pos_max[env_ids]
                - self.robot_spawning_pos_min[env_ids]
            )
            + self.robot_spawning_pos_min[env_ids]
        )
        drone_positions = (
            drone_spawning_min_bounds
            + drone_spawning_ratio_in_env_bound
            * (drone_spawning_max_bounds - drone_spawning_min_bounds)
        )

        # Randomize starting orientation
        robot_euler_angles = torch_rand_float(-1.0, 1.0, (num_resets, 3), self.device)
        robot_euler_angles[:, 0] = 0
        robot_euler_angles[:, 1] = 0
        robot_euler_angles[:, 2] = 90

        self.root_states[env_ids, 0:3] = drone_positions
        self.root_states[env_ids, 3:7] = quat_from_euler_xyz(
            robot_euler_angles[..., 0],
            robot_euler_angles[..., 1],
            robot_euler_angles[..., 2],
        )

        self.root_states[env_ids, 7:13] = 0
        self.root_states[env_ids, 6] = 1
        # self.root_states[env_ids, 7:10] = 0.3 * torch_rand_float(
        #     -1.0, 1.0, (num_resets, 3), self.device
        # )
        # self.root_states[env_ids, 10:13] = 0.3 * torch_rand_float(
        #     -1.0, 1.0, (num_resets, 3), self.device
        # )

        self.gym.set_actor_root_state_tensor(self.sim, self.root_tensor)

        # Randomize goal positions
        goal_pos_rand_sample = torch.rand((num_resets, 3), device=self.device)
        goal_spawning_min_bounds = (
            self.env_lower_bound[env_ids] + self.goal_spawning_offset[env_ids]
        )
        goal_spawning_max_bounds = (
            self.env_upper_bound[env_ids] - self.goal_spawning_offset[env_ids]
        )
        goal_spawning_ratio_in_env_bound = (
            goal_pos_rand_sample
            * (
                self.goal_spawning_pos_max[env_ids]
                - self.goal_spawning_pos_min[env_ids]
            )
            + self.goal_spawning_pos_min[env_ids]
        )
        self.goal_positions[env_ids] = (
            goal_spawning_min_bounds
            + goal_spawning_ratio_in_env_bound
            * (goal_spawning_max_bounds - goal_spawning_min_bounds)
        )

        center_x = self.goal_positions[0][0].item()
        center_y = self.goal_positions[0][1].item()
        center_z = self.goal_positions[0][2].item()

        center = (center_x, center_y, center_z)

        # Zero progress and reset buffers
        self.progress_buf[env_ids] = 0
        self.timeouts[env_ids] = 0
        self.hidden[env_ids] = 0
        self.reset_buf[env_ids] = 0
        # self.gym.clear_lines(self.viewer)
        self.prev_root_positions[:] = drone_positions

        lines = generate_wireframe_sphere_lines(center, 0.13, 40)
        center_2 = (center_x, -center_y, center_z)
        lines_2 = generate_wireframe_sphere_lines(center_2, 0.13, 40)
        lines += lines_2

        for line in lines:
            spehere_line = [
                line[0][0],
                line[0][1],
                line[0][2],
                line[1][0],
                line[1][1],
                line[1][2],
            ]
            self.gym.add_lines(self.viewer, self.envs[0], 1, spehere_line, [1, 0, 0])

        for line in self.traj:
            color = (
                [0.9, 0.0, 0.0]
                if self.collisions[0] or self.left_env_bounds
                else [0.0, 0.9, 0.0]
            )
            self.gym.add_lines(self.viewer, self.envs[0], 1, line, color)

        if self.collisions[0]:
            self.crash += 1
        else:
            self.successful += 1

        if self.num_rollouts > 1:
            with open("test.npy", "wb") as f:
                np.save(f, np.array(self.applied_control_actions))
        self.applied_control_actions = []

        self.collisions[env_ids] = 0

        print(self.successful)
        print(self.crash)
        self.traj = []

    def pre_physics_step(self, _actions):
        # resets
        # resets
        self.counter += 1

        actions = _actions.to(self.device)
        actions = tensor_clamp(
            actions, self.action_lower_limits, self.action_upper_limits
        )
        self.action_input[:] = actions

        # clear actions for reset envs
        self.forces[:] = 0.0
        self.torques[:, :] = 0.0

        output_thrusts_mass_normalized, output_torques_inertia_normalized = (
            self.controller(self.root_states, self.action_input)
        )
        self.forces[:, 0, 2] = (
            self.robot_mass
            * (-self.sim_params.gravity.z)
            * output_thrusts_mass_normalized
        )
        self.torques[:, 0] = output_torques_inertia_normalized

        self.forces = torch.where(
            self.forces < 0, torch.zeros_like(self.forces), self.forces
        )

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

        # apply actions
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
        self.gym.end_access_image_tensors(self.sim)
        self._process_depth_images()
        return

    def _process_depth_images(self):
        self.full_camera_array[torch.isinf(self.full_camera_array)] = (
            self.max_depth_value
        )
        self.full_camera_array[self.full_camera_array > self.max_depth_value] = (
            self.max_depth_value
        )
        self.full_camera_array[self.full_camera_array < self.min_depth_value] = (
            self.min_depth_value
        )
        self.full_camera_array = (self.full_camera_array - self.min_depth_value) / (
            self.max_depth_value - self.min_depth_value
        )

    def post_physics_step(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

    def check_collisions(self):
        ones = torch.ones((self.num_envs), device=self.device)
        zeros = torch.zeros((self.num_envs), device=self.device)
        self.collisions[:] = 0
        self.collisions = torch.where(
            torch.norm(self.contact_forces, dim=1) > 0.1, ones, zeros
        )

    def dump_images(self):
        for env_id in range(self.num_envs):
            # the depth values are in -ve z axis, so we need to flip it to positive
            self.full_camera_array[env_id] = -self.camera_tensors[env_id]

    def compute_vehicle_frame_states(self):
        r, p, y = get_euler_xyz(self.root_quats)
        r = ssa(r)
        p = ssa(p)
        y = ssa(y)
        self.root_euler_angles[:, 0] = r
        self.root_euler_angles[:, 1] = p
        self.root_euler_angles[:, 2] = y

        # vehicle frame is the same but with 0 roll and pitch
        self.vehicle_frame_euler_angles[:] = self.zeros_3d
        self.vehicle_frame_euler_angles[:, 2] = self.root_euler_angles[:, 2]

        # vehicle frame quats
        self.vehicle_frame_quats[:] = quat_from_euler_xyz(
            self.vehicle_frame_euler_angles[:, 0],
            self.vehicle_frame_euler_angles[:, 1],
            self.vehicle_frame_euler_angles[:, 2],
        )

        # goal dir vector in vehicle frame
        self.goal_dir_vehicle_frame[:] = quat_rotate_inverse(
            self.vehicle_frame_quats, (self.goal_positions - self.root_positions)
        )
        self.angvels_body_frame[:] = quat_rotate_inverse(
            self.root_quats, self.root_angvels
        )
        self.linvels_body_frame[:] = quat_rotate_inverse(
            self.root_quats, self.root_linvels
        )
        self.linvels_vehicle_frame[:] = quat_rotate_inverse(
            self.vehicle_frame_quats, self.root_linvels
        )
        self.unit_goal_dir_vehicle_frame[:] = self.goal_dir_vehicle_frame / torch.norm(
            self.goal_dir_vehicle_frame, dim=1
        ).unsqueeze(-1)

    def compute_observations(self):
        self.distances_to_target[:] = torch.norm(
            self.goal_positions - self.root_positions, dim=1
        )
        self.compute_vehicle_frame_states()

        self.obs_buf[..., :3] = self.unit_goal_dir_vehicle_frame
        self.obs_buf[..., 3] = self.distances_to_target
        self.obs_buf[..., 4:6] = self.root_euler_angles[:, 0:2]
        self.obs_buf[..., 6:9] = self.linvels_body_frame
        self.obs_buf[..., 10:13] = self.angvels_body_frame
        self.obs_buf[..., 13 : 13 + self.latent_dim] = self.latent
        self.obs_buf[
            ..., 13 + self.latent_dim : 13 + self.latent_dim + self.hidden_dim
        ] = self.hidden

        return self.obs_buf

    def compute_reward(self):
        self.rew_buf[:] = compute_quadcopter_reward(
            self.distances_to_target,
            self.prev_distances_to_target,
            self.action_input,
            self.collisions,
            self.timeouts,
            self.progress_buf,
        )


def generate_wireframe_sphere_lines(center, radius, num_segments):
    lines = []
    # Longitude lines
    for phi in np.linspace(-0.5 * np.pi, 0.5 * np.pi, num_segments):
        points = []
        for theta in np.linspace(0, 2 * np.pi, num_segments, endpoint=False):
            x = center[0] + radius * np.cos(phi) * np.cos(theta)
            y = center[1] + radius * np.cos(phi) * np.sin(theta)
            z = center[2] + radius * np.sin(phi)
            points.append((x, y, z))
        # Add lines for current circle
        for i in range(len(points)):
            next_index = (i + 1) % num_segments  # Ensures wrapping around
            lines.append((points[i], points[next_index]))

    # Latitude lines
    for theta in np.linspace(0, 2 * np.pi, num_segments):
        points = []
        for phi in np.linspace(-0.5 * np.pi, 0.5 * np.pi, num_segments, endpoint=False):
            x = center[0] + radius * np.sin(phi) * np.cos(theta)
            y = center[1] + radius * np.sin(phi) * np.sin(theta)
            z = center[2] + radius * np.cos(phi)
            points.append((x, y, z))
        # Add lines for current circle
        for i in range(len(points)):
            next_index = (i + 1) % num_segments  # Ensures wrapping around
            lines.append((points[i], points[next_index]))

    return lines


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


@torch.jit.script
def ssa(a: torch.Tensor) -> torch.Tensor:
    """Smallest signed angle"""
    return torch.remainder(a + np.pi, 2 * np.pi) - np.pi


@torch.jit.script
def exponential_penalty_function(
    magnitude: float, base_width: float, value: torch.Tensor
) -> torch.Tensor:
    """Exponential reward function"""
    return magnitude * (torch.exp(-(value * value) / base_width) - 1.0)


@torch.jit.script
def exponential_reward_function(
    magnitude: float, base_width: float, value: torch.Tensor
) -> torch.Tensor:
    """Exponential reward function"""
    return magnitude * torch.exp(-(value * value) / base_width)


@torch.jit.script
def compute_quadcopter_reward(
    distances_to_goal,
    prev_distances_to_goal,
    action_input,
    collisions,
    timeouts,
    progress_buf,
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) -> Tensor

    ## The reward function set here is arbitrary and the user is encouraged to modify this as per their need to achieve collision avoidance.
    ones = torch.ones_like(collisions, device=collisions.device, dtype=torch.float32)
    zeros = torch.zeros_like(collisions, device=collisions.device, dtype=torch.float32)

    r1 = exponential_reward_function(5.0, 3.5, distances_to_goal)
    r2 = exponential_reward_function(2.5, 0.75, distances_to_goal)
    r3 = 3 * ((20 - distances_to_goal) / 20)

    rewards = r1 + r2 + r3

    x_absolute_penalty = exponential_penalty_function(1.8, 2.9, action_input[:, 0])
    y_absolute_penalty = exponential_penalty_function(3.0, 1.0, action_input[:, 1])
    z_absolute_penalty = exponential_penalty_function(3.0, 1.0, action_input[:, 2])
    yawrate_absolute_penalty = exponential_penalty_function(
        0.0, 2.0, action_input[:, 3]
    )

    ones = torch.ones_like(collisions, device=collisions.device, dtype=torch.float32)
    zeros = torch.zeros_like(collisions, device=collisions.device, dtype=torch.float32)
    p1 = (
        x_absolute_penalty
        + y_absolute_penalty
        + z_absolute_penalty
        + yawrate_absolute_penalty
    )

    p2 = -700 * torch.where(collisions > 0, ones, zeros)
    penalties = p1 + p2

    return rewards + penalties
