from isaacgym import gymtorch
from isaacgym import torch_utils
from isaacgym import gymapi
from contexts.terrain_utils import *
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from contexts.terrain_context import TerrainContextSpace, TerrainContext
import numpy as np
from utils import *
from cfg.base_config import *
import glob
import torchvision.transforms as transforms
import sys
from PIL import Image
import random
import time
import open3d
from simsense import DepthSensor
Image.MAX_IMAGE_PIXELS = 933120000


class PIDController:
    def __init__(self, p_gain, i_gain, d_gain, out_max, out_min) -> None:
        self.p_gain = p_gain
        self.i_gain = i_gain
        self.d_gain = d_gain
        self.out_min = out_min
        self.out_max = out_max
        self.prev_error = None 
        self.integral = 0.0
        self.prev_input = None
    
    def __call__(self, inpt, setpoint, dt):
        error = setpoint - inpt 
        self.proportional = self.p_gain * error
        self.derivative = self.d_gain * (error - (self.prev_error if self.prev_error is not None else error)) / dt
        
        self.integral += error.clone() * dt
        self.prev_error = error.clone()
        self.prev_input = inpt.clone()
        control = self.proportional + self.derivative 
        return torch.clamp(control, self.out_min, self.out_max)

    def reset(self):
        self.integral = torch.zeros_like(self.integral)
        self.prev_error = torch.zeros_like(self.prev_error)


class OffRoadEnv:
    """
    Creates a simulation for a ground vehicle moving on unstructured terrains.
    """
    def __init__(self, cfg: BaseVehicleCfg, mode, render=True, terrain_path=None):
        self.device = DEVICE
        self.cfg = cfg
        self.mode = mode
        self.render = render
        self.envs = []
        self.actor_handles = []
        self.context_sampler = TerrainContext(cfg.context)
        self.terrain_path = terrain_path

        self.termination_copy = None

        self.use_globalmap = self.cfg.state_space.use_globalmap
        self.use_localmap = self.cfg.state_space.use_localmap

        self.max_goal_dis = 3.
        self.max_goal_ang = np.pi / 3. * 2.

        self._prepare_reward()
        self._configure_sim_params()
        self._load_asset()
        self._create_envs()
        self._create_terrain()
        self._create_camera()
        # self._create_ground_plane()
        self._init_buffers()
        # we use two simple P controllers to further smooth out the control signals 
        self.v_controller = PIDController(5, 0.0, 0.0, 3.0, -3.0)
        self.w_controller = PIDController(3, 0.0, 0.0, 6.0, -6.0)
        
        if self.render:
            cam_props = gymapi.CameraProperties()
            self.viewer = self.gym.create_viewer(self.sim, cam_props)
        # Feel free to use this to develop your own algorithm
        if self.use_localmap:
            size = 0.6
            resolution = 0.1
            side_length = round(size * 2. / resolution) + 1

            # Precompute the dx and dy grid (Tensor with shape (side_length * side_length, 2))
            dx, dy = torch.meshgrid(torch.arange(-size, size + resolution * 0.1, resolution),
                                    torch.arange(-size, size + resolution * 0.1, resolution))
            dxdy_grid = torch.stack([dx.flatten(), dy.flatten()], dim=1).to(self.device)
            self.localmap_grid = dxdy_grid.unsqueeze(0).repeat(self.num_agents, 1, 1)
                
        
    def step(self, actions):
        self.action = actions.clone()
        for _ in range(self.cfg.env.action_repeat):
            wheel_torques = self._compute_wheel_torques(actions)
            self.gym.set_dof_velocity_target_tensor(self.sim, gymtorch.unwrap_tensor(wheel_torques))
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self.gym.refresh_actor_root_state_tensor(self.sim)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.gym.refresh_net_contact_force_tensor(self.sim)
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            self.gym.refresh_force_sensor_tensor(self.sim)
            if self.render or self.use_camera:
                self.gym.step_graphics(self.sim)
            if self.render:
                self._draw_lines(np.arange(self.num_agents))
                self.gym.draw_viewer(self.viewer, self.sim, True)
                self.gym.sync_frame_time(self.sim)
                self.gym.clear_lines(self.viewer) 
            
            has_contact = torch.any(self.net_cf_wheels[..., 2] >= 1e-5, dim=1)
            self.has_contact = torch.logical_or(has_contact, self.has_contact)

        if self.use_camera:
            depth_images = self._render_depth()
        
        self._compute_reward_and_reset()
        self.compute_accelerations()
        self._compute_state()

        self.prev_action = self.applied_actions.clone()
        self.prev_base_position = self.base_position.clone()
        self.prev_euler_angles = self.euler_angles.clone()
        self.prev_base_net_force = self.base_net_force.clone()
        self.prev_base_net_torque = self.base_net_torque.clone()

        if not self.use_camera:
            return self.state, self.rewards, self.terminations, self.truncation
        else: # if camera is used, we return both the teacher's state and the student's state
            return self.state, self.student_state, depth_images, self.rewards, self.terminations, self.truncation

    def reset(self, env_ids=None):
        """
        The reset function performs domain randomization and resets the robots' physical states.
        """
        self._reset(env_ids)
        
        if not self.use_camera:
            return self.state
        else:
            depth_images = self._render_depth()
            return self.state, self.student_state, depth_images

    def _reset(self, env_ids):
        if env_ids == None:
            env_ids = torch.arange(self.num_agents, device=self.device)

        # resets the actions computed by the controller 
        self.applied_actions[env_ids] = torch.zeros_like(self.applied_actions[env_ids])
        # reset the wheel contact flag
        self.has_contact = torch.zeros_like(self.has_contact)

        # reset the counter
        self.step_index[env_ids] = 0
        # reset the action tensors
        self.prev_action[env_ids] = torch.zeros_like(self.prev_action[env_ids])
        self.action[env_ids] = torch.zeros_like(self.action[env_ids])

        # note: after randomizing the rigid body properties, we need to do a simulation step for state randomization to be effective
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)

        # reset the previous force sensor readings 
        self.prev_base_net_force = self.base_net_force.clone()
        self.prev_base_net_torque = self.base_net_torque.clone()

        self.sample_state(env_ids)
        self.compute_accelerations()
        self._compute_state() 
        self._compute_reward_and_reset()
        return self.state

    def sample_state(self, env_ids):
        """
        Randomizes the robot's state based on the range specified in the configuration.
        """
        num_envs_to_randomize = env_ids.size(0)
        # initialize the base position to the "safe zone"
        boundary = [self.cfg.context.environment.num_rows_per_env * self.cfg.context.environment.y_res,
                    self.cfg.context.environment.num_cols_per_env * self.cfg.context.environment.x_res]
        patch = 2.8
        self.base_position[env_ids, 0] = uniform(patch, boundary[1]-patch, num_envs_to_randomize)
        self.base_position[env_ids, 1] = uniform(patch, boundary[0]-patch, num_envs_to_randomize)
        
        # TODO: a simplified constractive sampler
        restore = False
        # if self.termination_copy is not None:
        #     random_float = np.random.rand()
        #     if random_float > 0.5:
        #         restore_id = torch.where(self.termination_copy == True)[0]
        #         common_ids = torch.intersect1d(restore_id, env_ids)
        #         if restore_id.size(dim=0) > 0:
        #             self.base_position[restore_id, :2] = self.origin_pose[restore_id, :2]
        #             restore = True
        #         else:
        #             self.termination_copy = None
        self.origin_pose[env_ids, :2] = self.base_position[env_ids, :2]

        # sample z to prevent immediate failure
        row_id = torch.floor((self.base_position[env_ids, 0]) / self.context_sampler.x_res).int()
        col_id = torch.floor(self.base_position[env_ids, 1] / self.context_sampler.y_res).int()
        terrain_idx = env_ids // self.cfg.env.num_agents_per_terrain
        z = self.context_sampler.heightfields[terrain_idx, row_id, col_id] * self.vertical_scale - 0.01
        z = z.to(device=self.device).float()
        self.base_position[env_ids, 2] = self.cfg.robot.base_height + z

        # todo: we currently only randomly rotate the robot's base around the z axis
        zeros = torch.zeros(num_envs_to_randomize, device=self.device)
        yaws = torch.rand(num_envs_to_randomize, device=self.device) * 2 * torch.pi
        quaternions = torch_utils.quat_from_euler_xyz(zeros, zeros, yaws)
        self.base_orientation[env_ids] = quaternions
        self.base_yaw = torch.zeros(self.num_agents, device=self.device)
        self.base_yaw[env_ids] = yaws

        # we first randomize the velocities in the robot's frame and then project them back to the global frame
        if self.cfg.state_space.lin_velocity.randomize:
            projected_base_linear_vel = uniform(self.state_space_lin_vel_min,
                                            self.state_space_lin_vel_max,
                                            (num_envs_to_randomize, 3))  
        else:
            projected_base_linear_vel = torch.zeros((num_envs_to_randomize, 3), device=self.device)
        self.base_linear_vel[env_ids] = torch_utils.quat_rotate(self.base_orientation[env_ids], projected_base_linear_vel)                
        
        # we first randomize the angular velocities in the robot's frame and then project them back to the global frame
        if self.cfg.state_space.ang_velocity.randomize:
            projected_base_ang_vel = uniform(self.state_space_ang_vel_min,
                                         self.state_space_ang_vel_max,
                                         (num_envs_to_randomize, 3))  
        else:
            projected_base_ang_vel = torch.zeros((num_envs_to_randomize, 3), device=self.device)
        self.base_angular_vel[env_ids] = torch_utils.quat_rotate(self.base_orientation[env_ids], projected_base_ang_vel)

        # Keep within bounds
        mask2d = torch.zeros(self.num_agents, dtype=torch.bool, device=self.device)
        mask2d[env_ids] = True
        if restore:
            mask2d[restore_id] = False
        while torch.any(mask2d):
            # discretize goal into 0.1m and 1degree
            base_goal_ranges = uniform(1., self.max_goal_dis, torch.sum(mask2d))
            base_goal_ranges = (base_goal_ranges / 0.1).int().float() * 0.1
            base_goal_yaw = uniform(-self.max_goal_ang/2., +self.max_goal_ang/2., torch.sum(mask2d))
            base_goal_yaw = (base_goal_yaw / (np.pi / 180.)).int().float() * (np.pi / 180.)

            base_goal_x = base_goal_ranges * torch.cos(base_goal_yaw)
            base_goal_y = base_goal_ranges * torch.sin(base_goal_yaw)

            self.goal[mask2d, 0] = self.base_position[mask2d, 0] + torch.cos(self.base_yaw[mask2d]) * base_goal_x - torch.sin(self.base_yaw[mask2d]) * base_goal_y
            self.goal[mask2d, 1] = self.base_position[mask2d, 1] + torch.sin(self.base_yaw[mask2d]) * base_goal_x + torch.cos(self.base_yaw[mask2d]) * base_goal_y
            self.goal_yaw[mask2d] = (self.base_yaw[mask2d] + base_goal_yaw).unsqueeze(1)

            # Reapply masks for updated values
            mask_x = (self.goal[:, 0] < 0.8) | (self.goal[:, 0] > 12.0)
            mask_y = (self.goal[:, 1] < 0.8) | (self.goal[:, 1] > 12.0)
            mask2d = (mask_x | mask_y) & mask2d
        normalize_angle(self.goal_yaw[env_ids])
        
        # shift the robot's goal position to each terrain's frame
        self.base_position[env_ids, :2] = self._shift_positions(env_ids, self.base_position[env_ids, :2])
        self.goal[env_ids] = self._shift_positions(env_ids, self.goal[env_ids])

        # log the previous state
        self.prev_base_position = self.base_position.clone()
        self.prev_euler_angles = self._base_orientation_euler()
    
        # set robot's base state
        self.gym.set_actor_root_state_tensor(self.sim, self._root_tensor)

    def _load_asset(self):
        """
        Loads the robot's assets (3D model) and store the names for the rigid body and dof of the assets (3D model).
        """
        asset_options = gymapi.AssetOptions()
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints 
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule 
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode 
        asset_options.override_inertia = True 
        asset_options.angular_damping = self.cfg.asset.angular_damping 
        asset_options.linear_damping = self.cfg.asset.linear_damping 
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.thickness = self.cfg.asset.thickness
        self.asset = self.gym.load_asset(self.sim, self.cfg.asset.asset_root, self.cfg.asset.asset_file, asset_options)
        self.body_names = self.gym.get_asset_rigid_body_names(self.asset)
        self.rigid_body_dict = self.gym.get_asset_rigid_body_dict(self.asset)
        self.num_bodies = len(self.body_names)
        self.dofs = self.gym.get_asset_dof_dict(self.asset)
        self.num_dofs = len(self.dofs)
        self._create_force_sensors()

    def _base_orientation_euler(self):
        roll, pitch, yaw = torch_utils.get_euler_xyz(self.base_orientation)
        roll = roll.view(-1, 1)
        pitch = pitch.view(-1, 1)
        yaw = yaw.view(-1, 1)
        # make 2 pi same as 0
        euler_angles = torch.cat([roll, pitch, yaw], dim=-1)
        euler_angles[torch.where(torch.isclose(euler_angles, torch.zeros_like(euler_angles)+(2*torch.pi)))] = 0
        return euler_angles 
    
    def _configure_sim_params(self):
        """
        Configures the simulation parameters based on the cfg class.
        """
        self.num_agents = self.cfg.env.num_agents_per_terrain * self.context_sampler.num_terrains
        self.agent_ids = torch.arange(0, self.num_agents, device=self.device)
        self.gym = gymapi.acquire_gym()
        self.sim_params = gymapi.SimParams()
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.gravity_vec = torch.tensor([[self.cfg.sim.gravity[0], 
                                          self.cfg.sim.gravity[1], 
                                          self.cfg.sim.gravity[2]]], device=self.device).repeat(self.num_agents, 1)
        self.sim_params.gravity = gymapi.Vec3(self.cfg.sim.gravity[0], 
                                              self.cfg.sim.gravity[1], 
                                              self.cfg.sim.gravity[2])
        self.sim_params.use_gpu_pipeline = True
        self.sim_params.physx.use_gpu = True
        self.sim_params.substeps = self.cfg.sim.substeps
        self.dt = self.cfg.sim.dt
        self.sim_params.dt = self.dt
        self.sim_params.physx.num_threads = self.cfg.sim.physx.num_threads 
        self.sim_params.physx.contact_collection = gymapi.ContactCollection.CC_LAST_SUBSTEP # CC_NEVER : Don’t collect any contacts (value = 0). # CC_LAST_SUBSTEP : Collect contacts for last substep only (value = 1). CC_ALL_SUBSTEPS : Collect contacts for all substeps (value = 2) (default).
        self.sim_params.physx.solver_type = self.cfg.sim.physx.solver_type
        self.sim_params.physx.rest_offset = self.cfg.sim.physx.rest_offset
        self.sim_params.physx.num_position_iterations = self.cfg.sim.physx.num_position_iterations
        self.sim_params.physx.max_gpu_contact_pairs = self.cfg.sim.physx.max_gpu_contact_pairs
        self.sim_params.physx.contact_offset = self.cfg.sim.physx.contact_offset
        self.sim_params.physx.max_depenetration_velocity = 1.0 # 0.0 m/s means that if the robot is initialized inside an object, it will remain inside
        self.sim = self.gym.create_sim(0, 0, gymapi.SIM_PHYSX, self.sim_params)

        # For manipulation
        # object_asset_options = gymapi.AssetOptions()
        # object_asset_options.use_mesh_materials = True  
        # object_asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX 
        # object_asset_options.override_com = True 
        # object_asset_options.override_inertia = True 
        # object_asset_options.vhacd_enabled = True 
        # object_asset_options.vhacd_params = gymapi.VhacdParams() 
        # object_asset_options.vhacd_params.resolution = 200000 
        # object_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

    def _compute_wheel_torques(self, action):
        """
        Converts the action to wheel torques. There are four modes:
            1. Base control: the action is the desired linear and angular velocities of the base. 
            2. Position control: the action is the desired wheel position;
            3. Velocity control: the action is the desired wheel velocity;
            4. Torque control: the action is directly the wheel torque.
        """
        # clip action
        clipped_action = torch.clamp(action, min=self.action_lower, max=self.action_upper)
        # use PID controller to compute the desired wheel velocities 
        pid_v = self.v_controller(self.applied_actions[:, 0], clipped_action[:, 0], self.dt)
        pid_w = self.w_controller(self.applied_actions[:, 1], clipped_action[:, 1], self.dt)
        self.applied_actions[:, 0] = self.applied_actions[:, 0] + pid_v * self.dt
        self.applied_actions[:, 1] = self.applied_actions[:, 1] + pid_w * self.dt
        self.applied_actions = torch.clamp(self.applied_actions, min=self.action_lower, max=self.action_upper)
        # the velocities of one side of the wheels are the same
        linear_vel = self.applied_actions[:, 0]
        angular_vel = self.applied_actions[:, 1]
        vel_left = (linear_vel - 1.5 * angular_vel * self.cfg.robot.base_width) / (self.cfg.robot.wheel_radius)
        vel_right = (linear_vel + 1.5 * angular_vel * self.cfg.robot.base_width) / (self.cfg.robot.wheel_radius)
        desired_wheel_vels = torch.cat([vel_left.view(-1, 1), 
                                        vel_right.view(-1, 1),
                                        vel_left.view(-1, 1),
                                        vel_right.view(-1, 1), 
                                        ], dim=1)
        return desired_wheel_vels

    def _create_envs(self):
        """Creates the environment and agents.
        """
        lower = gymapi.Vec3(0.0, 0.0, 0.0)
        upper = gymapi.Vec3(0.0, 0.0, 0.0)
        self.env_x_len = self.context_sampler.num_rows_per_env * self.context_sampler.x_res
        self.env_y_len = self.context_sampler.num_cols_per_env * self.context_sampler.y_res
        self.envs = []
        self.actor_handles = []

        # https://forums.developer.nvidia.com/t/how-to-randomize-ground-plane-friction/187389
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(self.asset)
        for i in range(1, len(rigid_shape_props_asset)):
            rigid_shape_props_asset[i].friction = 0.01
            rigid_shape_props_asset[i].restitution = 1.
            rigid_shape_props_asset[i].torsion_friction = 0.01
        self.gym.set_asset_rigid_shape_properties(self.asset, rigid_shape_props_asset)
        
        for i in range(self.num_agents):
            env = self.gym.create_env(self.sim, lower, upper, int(sqrt(self.num_agents)))
            pose = gymapi.Transform()
            init_x = np.random.uniform(0, self.env_x_len)
            init_y = np.random.uniform(0, self.env_y_len)
            init_z = self.cfg.robot.base_height
            pose.p = gymapi.Vec3(init_x, init_y, init_z)
            pose.r = gymapi.Quat(0, 0.0, 0.0, 0.707107)

            actor_handle = self.gym.create_actor(env, self.asset, pose, self.cfg.robot.name, i, 1)
            # rigid body properties
            rigid_body_props = self.gym.get_actor_rigid_body_properties(env, actor_handle)
            base_rigid_body_prop = rigid_body_props[0] 
            
            # every robot is the same
            self.base_mass = base_rigid_body_prop.mass
            xx = base_rigid_body_prop.inertia.x
            yy = base_rigid_body_prop.inertia.y
            zz = base_rigid_body_prop.inertia.z
            self.base_inertia_matrix = torch.tensor([ 
                xx.x, xx.y, xx.z,
                yy.x, yy.y, yy.z,
                zz.x, zz.y, zz.z
            ], device=self.device).view(1, 3, 3)

            # dof properties
            dof_props = self.gym.get_actor_dof_properties(env, actor_handle)
            self._process_dof_props(dof_props)
            self.gym.set_actor_dof_properties(env, actor_handle, dof_props)
            self.envs.append(env)
            self.actor_handles.append(actor_handle)
        self.base_inertia_matrix_inv = torch.inverse(self.base_inertia_matrix)
        self.base_inertia_matrix = self.base_inertia_matrix.repeat(self.num_agents, 1, 1)
        self.base_inertia_matrix_inv = self.base_inertia_matrix_inv.repeat(self.num_agents, 1, 1)

    def _create_terrain(self):
        self.horizontal_scale = self.cfg.context.environment.x_res
        self.vertical_scale = self.cfg.context.environment.z_scale
        
        row_terrain = self.context_sampler.num_rows * self.context_sampler.num_rows_per_env
        col_terrain = self.context_sampler.num_cols * self.context_sampler.num_cols_per_env
        if self.terrain_path is not None:
            if self.terrain_path[-4:] == ".tif":
                self.elevation_data = np.array(Image.open(self.terrain_path), dtype=np.float32)
            elif self.terrain_path[-4:] == ".txt":
                self.elevation_data = np.loadtxt(self.terrain_path, delimiter=',', dtype=np.float32)
            row_img = len(self.elevation_data)
            col_img = len(self.elevation_data[0])
            rand_row_start = 0 * row_terrain
            rand_col_start = 0 * col_terrain
            
            self.heightmap_np = self.elevation_data[rand_row_start:rand_row_start+row_terrain, rand_col_start:rand_col_start+col_terrain].copy()
        
        friction_range = self.cfg.context.ground.friction
        restitution_range = self.cfg.context.ground.restitution
        self.context_sampler.frictions = uniform(friction_range[0], friction_range[1], self.context_sampler.num_terrains)
        self.context_sampler.restitutions = uniform(restitution_range[0], restitution_range[1], self.context_sampler.num_terrains)

        for i in range(self.context_sampler.num_rows):
            for j in range(self.context_sampler.num_cols):
                if self.terrain_path is not None:
                    height_raw = self.heightmap_np[i * self.context_sampler.num_rows_per_env: (i + 1) * self.context_sampler.num_rows_per_env,
                                    j * self.context_sampler.num_cols_per_env: (j + 1) * self.context_sampler.num_cols_per_env]
                    
                else:
                    height_raw = self.context_sampler.procedural_sample(np.arange(0, self.context_sampler.num_terrains, 1))
                
                self.context_sampler.heightfields.append(height_raw)

                vertices, triangles = convert_heightfield_to_trimesh(height_raw, 
                                                                     horizontal_scale=self.horizontal_scale, 
                                                                     vertical_scale=self.vertical_scale)

                tm_params = gymapi.TriangleMeshParams()
                tm_params.nb_vertices = vertices.shape[0]
                tm_params.nb_triangles = triangles.shape[0]
                tm_params.transform.p.x = i * self.context_sampler.num_cols_per_env * self.horizontal_scale
                tm_params.transform.p.y = j * self.context_sampler.num_rows_per_env * self.horizontal_scale
                tm_params.transform.p.z = 0
                
                friction_range = self.cfg.context.ground.friction
                restitution_range = self.cfg.context.ground.restitution
                tm_params.dynamic_friction = self.context_sampler.frictions[i*self.context_sampler.num_cols + j].item()
                tm_params.static_friction = tm_params.dynamic_friction
                tm_params.restitution = self.context_sampler.restitutions[i*self.context_sampler.num_cols + j].item()
                self.gym.add_triangle_mesh(self.sim, vertices.flatten(), triangles.flatten(), tm_params)

        self.context_sampler.heightfields = torch.from_numpy(np.array(self.context_sampler.heightfields)).to(device=self.device)
        self.context_sampler.frictions = self.context_sampler.frictions.repeat_interleave(self.cfg.env.num_agents_per_terrain).unsqueeze(1)
        self.context_sampler.restitutions = self.context_sampler.restitutions.repeat_interleave(self.cfg.env.num_agents_per_terrain).unsqueeze(1)
        if self.use_globalmap:
            self.context_sampler.encoder(self.cfg.env.num_agents_per_terrain)

        # Sample code to import a HOME
        # Download obj at https://drive.google.com/file/d/1x57aZfAebg6lpGXVrxFE49QGFBK-BRMm/view?usp=sharing
        # mesh = open3d.io.read_triangle_mesh('assets/mesh/lts3d_combined_1_2.obj')
        # mesh_vertices = np.asarray(mesh.vertices).astype(np.float32)
        # mesh_triangles = np.asarray(mesh.triangles).astype(np.uint32)

        # tm_params = gymapi.TriangleMeshParams()
        # tm_params.nb_vertices = mesh_vertices.shape[0]
        # tm_params.nb_triangles = mesh_triangles.shape[0]
        # self.gym.add_triangle_mesh(self.sim, mesh_vertices.flatten(order='C'),
        #                             mesh_triangles.flatten(order='C'),
        #                             tm_params)

    def _create_force_sensors(self):
        """Attach a force sensor to each robot's base_link body to simulate the IMU which measures the linear and angular acceleration of the robot.
        """
        body_idx = self.gym.find_asset_rigid_body_index(self.asset, "base_link")
        sensor_props = gymapi.ForceSensorProperties()
        sensor_props.enable_forward_dynamics_forces = True
        sensor_props.enable_constraint_solver_forces = True
        sensor_props.use_world_frame = False    
        self.imu_sensor = self.gym.create_asset_force_sensor(self.asset, body_idx, gymapi.Transform(gymapi.Vec3(0.0, 0.0, 0.0)), sensor_props)
        
    def _create_camera(self):
        """
        Creates a depth camera for each actor.
        """
        self.camera_cfg = self.cfg.robot.camera
        self.use_camera = self.camera_cfg.use_camera
        if self.use_camera:
            self.rigid_body_dict = self.gym.get_actor_rigid_body_dict(self.envs[0], self.actor_handles[0])
            self.camera_handles = []
            camera_props = gymapi.CameraProperties()
            camera_props.width = self.camera_cfg.img_size[0]
            camera_props.height = self.camera_cfg.img_size[1]
            camera_props.enable_tensors = True
            camera_props.horizontal_fov = self.camera_cfg.horizontal_fov
            camera_props.far_plane = self.camera_cfg.far_plane
            for i in range(self.num_agents):
                if self.cfg.robot.camera.type  == 'depth':
                    camera_handle = self.gym.create_camera_sensor(self.envs[i], camera_props)
                    self.camera_handles.append(camera_handle)
                    local_transform = gymapi.Transform()
                    # Attention: Change the camera position based on where it will be mounted on the robot
                    local_transform.p = gymapi.Vec3(0.215, 0.0, 0.225)
                    rigid_body_handle = self.gym.get_actor_rigid_body_handle(self.envs[i], self.actor_handles[i], self.rigid_body_dict['base_link'])
                    self.gym.attach_camera_to_body(camera_handle, self.envs[i], rigid_body_handle, local_transform, gymapi.FOLLOW_TRANSFORM)
                elif self.cfg.robot.camera.type  == 'mono':
                    camera_handle = self.gym.create_camera_sensor(self.envs[i], camera_props)
                    self.camera_handles.append(camera_handle)
                    local_transform = gymapi.Transform()
                    # RGB, See https://github.com/IntelRealSense/realsense-ros/blob/ros2-master/realsense2_description/urdf/_d435.urdf.xacro
                    local_transform.p = gymapi.Vec3(0.215, 0.0 + 0.015, 0.225)
                    rigid_body_handle = self.gym.get_actor_rigid_body_handle(self.envs[i], self.actor_handles[i], self.rigid_body_dict['base_link'])
                    self.gym.attach_camera_to_body(camera_handle, self.envs[i], rigid_body_handle, local_transform, gymapi.FOLLOW_TRANSFORM)
                elif self.cfg.robot.camera.type == 'stereo':
                    camera_handle = self.gym.create_camera_sensor(self.envs[i], camera_props)
                    self.camera_handles.append(camera_handle)
                    local_transform = gymapi.Transform()
                    # IR Left
                    local_transform.p = gymapi.Vec3(0.215, 0.0, 0.225)
                    rigid_body_handle = self.gym.get_actor_rigid_body_handle(self.envs[i], self.actor_handles[i], self.rigid_body_dict['base_link'])
                    self.gym.attach_camera_to_body(camera_handle, self.envs[i], rigid_body_handle, local_transform, gymapi.FOLLOW_TRANSFORM)

                    camera_handle = self.gym.create_camera_sensor(self.envs[i], camera_props)
                    self.camera_handles.append(camera_handle)
                    local_transform = gymapi.Transform()
                    # IR Right
                    local_transform.p = gymapi.Vec3(0.215, 0.0 - 0.05, 0.225)
                    rigid_body_handle = self.gym.get_actor_rigid_body_handle(self.envs[i], self.actor_handles[i], self.rigid_body_dict['base_link'])
                    self.gym.attach_camera_to_body(camera_handle, self.envs[i], rigid_body_handle, local_transform, gymapi.FOLLOW_TRANSFORM)
            
            self.proj = self.gym.get_camera_proj_matrix(self.sim, self.envs[0], self.camera_handles[0])

            # Option 1: https://github.com/ZiwenZhuang/parkour
            # self.contour_detection_kernel = torch.zeros(
            #     (8, 1, 3, 3),
            #     dtype = torch.float32,
            #     device = self.device
            # )
            # self.contour_detection_kernel[0, :, 1, 1] = 0.5
            # self.contour_detection_kernel[0, :, 0, 0] = -0.5
            # self.contour_detection_kernel[1, :, 1, 1] = 0.1
            # self.contour_detection_kernel[1, :, 0, 1] = -0.1
            # self.contour_detection_kernel[2, :, 1, 1] = 0.5
            # self.contour_detection_kernel[2, :, 0, 2] = -0.5
            # self.contour_detection_kernel[3, :, 1, 1] = 1.2
            # self.contour_detection_kernel[3, :, 1, 0] = -1.2
            # self.contour_detection_kernel[4, :, 1, 1] = 1.2
            # self.contour_detection_kernel[4, :, 1, 2] = -1.2
            # self.contour_detection_kernel[5, :, 1, 1] = 0.5
            # self.contour_detection_kernel[5, :, 2, 0] = -0.5
            # self.contour_detection_kernel[6, :, 1, 1] = 0.1
            # self.contour_detection_kernel[6, :, 2, 1] = -0.1
            # self.contour_detection_kernel[7, :, 1, 1] = 0.5
            # self.contour_detection_kernel[7, :, 2, 2] = -0.5

            # Option 2: SGBM
            horizontal_fov = self.camera_cfg.horizontal_fov * np.pi / 180
            vertical_fov = 58 * np.pi / 180
            image_width = self.camera_cfg.img_size[0]
            image_heigth = self.camera_cfg.img_size[1]
            f_x = (image_width / 2.0) / np.tan(horizontal_fov / 2.0)
            f_y = (image_heigth / 2.0) / np.tan(vertical_fov / 2.0)
            lr_size = (image_width, image_heigth)
            k_l = np.array([
                [f_x, 0., image_width / 2.0],
                [0., f_y, image_heigth / 2.0],
                [0., 0., 1.]
            ])
            k_r = k_l
            l2r = np.array([
                [1., 0, 0, -0.05],
                [0, 1., 0, 0],
                [0, 0, 1., 0],
                [0, 0, 0, 1.]
            ])
            self.sgbm = DepthSensor(lr_size, k_l, k_r, l2r, rectified=True, 
                                    min_depth=self.camera_cfg.near_plane, max_depth=self.camera_cfg.far_plane, lr_max_diff=255,)
    
    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1) # z-up!
        plane_params.distance = 0
        # default value is 1.0
        plane_params.static_friction = 1.
        plane_params.dynamic_friction = 1.
        plane_params.restitution = 0.
        self.gym.add_ground(self.sim, plane_params)

    def _draw_lines(self, env_ids):
        for env_id in env_ids:
            env = self.envs[env_id]
            robot_position = self.base_position[env_id].detach().cpu().numpy()
            goal = self.goal[env_id].detach().cpu().numpy()
            p1 = gymapi.Vec3(robot_position[0], robot_position[1], robot_position[2] + 0.1)
            p2 = gymapi.Vec3(goal[0], goal[1], robot_position[2] + 0.1)
            color = gymapi.Vec3(1, 0, 0)
            gymutil.draw_line(p1, p2, color, self.gym, self.viewer, env)
            
            # For Debug
            if self.use_localmap:
                valid_heights = self.localmaps[env_id]
                for point_id in range(0, valid_heights.shape[0], 1):
                    # Retrieve a single scalar from the batched mx, my, and height values
                    mx_val = self.mx[env_id, point_id].item()
                    my_val = self.my[env_id, point_id].item()
                    height_val = valid_heights[point_id].item()
                    # Create the start and end points for the line visualization
                    p1 = gymapi.Vec3(mx_val, my_val, valid_heights[point_id])
                    p2 = gymapi.Vec3(mx_val, my_val, valid_heights[point_id] + 0.1)
                    color = gymapi.Vec3(0, 1, 0)
                    gymutil.draw_line(p1, p2, color, self.gym, self.viewer, env)

    def _init_buffers(self):
        """
        Initializes the all the variables used in the simulation.
        """
        self.action_dim = self.cfg.action_space.dim

        self.state_dim =  self.cfg.state_space.angle.dim +\
                          self.cfg.state_space.lin_velocity.dim +\
                          self.cfg.state_space.ang_velocity.dim +\
                          self.cfg.state_space.goal.dim +\
                          self.action_dim +\
                          self.use_globalmap * (1000 + 1 + 1 + 3 + 12)

        self.gym.prepare_sim(self.sim)
        self.step_index = torch.zeros(self.num_agents, device=self.device) 

        # initialize the robot's rigid body buffer
        self._rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rigid_body_tensor = gymtorch.wrap_tensor(self._rigid_body_tensor).view(self.num_agents, -1, 13)

        # initialize robot's base buffers
        self._root_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.root_tensor = gymtorch.wrap_tensor(self._root_tensor).view(self.num_agents, -1, 13)
        # split the root tensor into position, orientation, and velocity
        self.base_position = self.root_tensor[:, 0, 0:3]
        
        self.origin_pose = self.base_position.clone()
        self.prev_base_position = self.base_position.clone()
        # quaternion
        self.base_orientation = self.root_tensor[:, 0, 3:7]
        # linear velocity
        self.base_linear_vel = self.root_tensor[:, 0, 7:10]
        # angular velocity
        self.base_angular_vel = self.root_tensor[:, 0, 10:13]
        # goal in global frame
        self.goal = torch.zeros(self.num_agents, 2, device=self.device)
        self.goal_yaw = torch.zeros(self.num_agents, 1, device=self.device)

        # initialize the imu buffer
        self._imu_data = self.gym.acquire_force_sensor_tensor(self.sim)
        self.imu_data = gymtorch.wrap_tensor(self._imu_data)
        # f_t - f_{t-1} defines the jerk of the position 
        # t_t - t_{t-1} defines the jerk of the orientation
        self.base_net_force = self.imu_data[..., :3]
        self.base_net_torque = self.imu_data[..., 3:]
        self.prev_base_net_force = self.base_net_force.clone()
        self.prev_base_net_torque = self.base_net_torque.clone()
        # a_v = f/m, a_w = torque/I 
        self.compute_accelerations()

        # contacts
        self.has_contact = torch.zeros(self.num_agents, device=self.device)

        # state space boundary
        # todo: the position limits need to be inside the configuration file
        self.min_position = torch.tensor([0.6, 0.6], device=self.device)
        self.max_position = torch.tensor([12.2, 12.2], device=self.device)
        self.state_space_angle_min = self.cfg.state_space.angle.angle_min
        self.state_space_angle_max = self.cfg.state_space.angle.angle_max
        self.state_space_lin_vel_min = self.cfg.state_space.lin_velocity.min_lin_vel
        self.state_space_lin_vel_max = self.cfg.state_space.lin_velocity.max_lin_vel
        self.state_space_ang_vel_min = self.cfg.state_space.ang_velocity.min_ang_vel
        self.state_space_ang_vel_max = self.cfg.state_space.ang_velocity.max_ang_vel
        self.min_angle_reward = torch.full((self.num_agents,), torch.pi / 50., device=self.device)
        self.max_angle_reward = torch.full((self.num_agents,), -torch.pi / 50., device=self.device)
        self.min_angle_terminal = torch.full((self.num_agents,), torch.pi / 9., device=self.device)
        self.max_angle_terminal = torch.full((self.num_agents,), -torch.pi / 9., device=self.device)

        # actions
        self.action = torch.zeros(self.num_agents, self.action_dim, device=self.device)
        # for the actual action applied to the robot, which are computed by the p-controllers
        self.applied_actions = torch.zeros_like(self.action)
        self.prev_action = torch.zeros_like(self.action)
        self.action_lower = self.cfg.action_space.action_lower
        self.action_upper = self.cfg.action_space.action_upper 
        
        # initialize dof state
        self._dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        self.dof_state_tensor = gymtorch.wrap_tensor(self._dof_state_tensor).view(self.num_agents, -1, 2)

        # initialize contact forces
        self._contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)
        self.contact_force_tensor = gymtorch.wrap_tensor(self._contact_force_tensor)
        self.contact_force_tensor = self.contact_force_tensor.view(self.num_agents, -1, 3)

        # action scaling
        self.action_scale = torch.zeros(self.num_agents, self.action_dim, device=self.device)
        self.action_scale[:, 0] = self.cfg.robot.action_scale_v 
        self.action_scale[:, 1] = self.cfg.robot.action_scale_w 

        # logged statistics for individual returns over one episode
        self.individual_returns = torch.zeros(self.num_agents, len(self.reward_fns), device=self.device)

        # contact forces
        self._net_cf = self.gym.acquire_net_contact_force_tensor(self.sim)
        self.net_cf = gymtorch.wrap_tensor(self._net_cf)
        # resize the contact forces to (num_agents, num_rigid_bodies, 3) 
        self.net_cf = self.net_cf.view(self.num_agents, self.num_bodies, 3)
        self.net_cf_base = self.net_cf[:, 0, :]
        self.net_cf_wheels = self.net_cf[:, 1:, :]

        # the translational shift of each terrain's origin to the global origin
        self.terrain_origins = torch.zeros(self.context_sampler.num_rows, self.context_sampler.num_cols, 2, device=self.device)
        for iy in range(self.context_sampler.num_cols):
            x = torch.arange(self.context_sampler.num_rows, device=self.device) * self.context_sampler.terrain_width
            y = torch.ones(self.context_sampler.num_rows, device=self.device) * iy * self.context_sampler.terrain_height
            self.terrain_origins[:, iy] = torch.cat([x.view(-1, 1), y.view(-1, 1)], dim=1)
        
        # pre-compute (all) position relative to each environment's origin
        terrain_ids = (torch.arange(self.num_agents, device=self.device) / self.cfg.env.num_agents_per_terrain).int()
        xi = (terrain_ids / self.context_sampler.num_cols).int()
        yi = (terrain_ids % self.context_sampler.num_cols).int()
        self.terrain_origins_all = self.terrain_origins[xi, yi].view(self.num_agents, 2)
        
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)

    def _prepare_reward(self):
        """
        Initializes the reward functions indicated in the config class.
        """
        self.reward_fns = []
        self.reward_scales = {}
        for reward_fn_names in self.cfg.reward.reward_scales.keys():
            self.reward_fns.append(self.__getattribute__(reward_fn_names))
            self.reward_scales[reward_fn_names] = self.cfg.reward.reward_scales[reward_fn_names]

    def _process_rigid_body_props(self, props):
        """
        Reads (and optionally change) the default value of the rigid body.
        """
        total_mass = 0.0
        for i, prop in enumerate(props):
            total_mass += prop.mass
        # todo: we currently only focus on the base properties and do not change any values 
        base_prop = props[0] 
        self.default_base_mass = base_prop.mass
        self.mass_min = self.cfg.context.rigid_body_base.base_mass_displacement[0] + self.default_base_mass
        self.mass_max = self.cfg.context.rigid_body_base.base_mass_displacement[1] + self.default_base_mass
        xx = base_prop.inertia.x
        yy = base_prop.inertia.y
        zz = base_prop.inertia.z
        self.default_base_inertia = torch.tensor([xx.x, yy.y, zz.z], device=self.device)
        base_inertia_displacement_min = torch.tensor([self.cfg.context.rigid_body_base.base_inertia_displacement_x[0],
                                                      self.cfg.context.rigid_body_base.base_inertia_displacement_y[0],
                                                      self.cfg.context.rigid_body_base.base_inertia_displacement_z[0]], device=self.device)
        base_inertia_displacement_max = torch.tensor([self.cfg.context.rigid_body_base.base_inertia_displacement_x[1],
                                                      self.cfg.context.rigid_body_base.base_inertia_displacement_y[1],
                                                      self.cfg.context.rigid_body_base.base_inertia_displacement_z[1]], device=self.device)
        base_inertia_eps = torch.tensor([self.cfg.context.rigid_body_base.inertia_lower_bound] * 3, device=self.device)
        self.base_inertia_min = torch.max(base_inertia_displacement_min + self.default_base_inertia, base_inertia_eps)
        self.base_inertia_max = base_inertia_displacement_max + self.default_base_inertia

    def _process_dof_props(self, props):
        """
        Reads (and optionally change) the default value of the dof.
        """
        # https://forums.developer.nvidia.com/t/damping-and-stiffness-parameters-effect-on-drive-mode-effort/204259
        props["driveMode"].fill(gymapi.DOF_MODE_VEL)
        props["stiffness"].fill(0.0)
        props["damping"].fill(500)
    
    def _compute_state(self):
        """
        Compute the state variables
        """
        # increment the step counter
        self.step_index += 1
        self.euler_angles = self._base_orientation_euler()

        # compute position relative to each environment's origin
        self.shifted_base_position = self.base_position[:, :3].clone()
        self.shifted_base_position[:, :2] = self.shifted_base_position[:, :2] - self.terrain_origins_all
        self.shifted_prev_base_position = self.prev_base_position[:, :3].clone()
        self.shifted_prev_base_position[:, :2] = self.shifted_prev_base_position[:, :2] - self.terrain_origins_all

        self.projected_lin_vel = torch_utils.quat_rotate_inverse(self.base_orientation, self.base_linear_vel)
        self.projected_ang_vel = torch_utils.quat_rotate_inverse(self.base_orientation, self.base_angular_vel)
        self.projected_gravity = torch_utils.quat_rotate_inverse(self.base_orientation, self.gravity_vec)
        self.goal_dist = (torch_utils.quat_rotate_inverse(self.base_orientation, 
                            torch.cat((self.goal - self.base_position[:, :2], torch.zeros(self.goal.size(dim=0), 1).to(self.device)), 1)))[:, :2]

        _, _, robot_yaw = torch_utils.get_euler_xyz(self.base_orientation)
        robot_yaw = robot_yaw.view(-1, 1)
        self.goal_yaw_dist = normalize_angle(self.goal_yaw - robot_yaw)
        self.goal_yaw_dist = torch.min(self.goal_yaw_dist, 2 * np.pi - self.goal_yaw_dist)

        if self.use_globalmap:
            self.state = torch.cat([
                self.base_orientation,
                self.projected_lin_vel / self.state_space_lin_vel_max,
                self.projected_ang_vel / self.state_space_ang_vel_max,
                self.goal_dist,
                self.prev_action / self.action_upper,
                self.context_sampler.encoded_heightfields,
                self.context_sampler.frictions,
                self.context_sampler.restitutions,
                self.gravity_vec,
                self.net_cf_wheels.view(self.num_agents, -1)
            ], dim=1)
        elif self.use_localmap:
            # Precompute cos and sin of yaw angles for all agents (shape: (num_agents,))
            cos_yaw = torch.cos(robot_yaw)
            sin_yaw = torch.sin(robot_yaw)

            # Get x, y, z positions for all agents
            x_i = self.base_position[:, 0].unsqueeze(1)  # (num_agents, 1)
            y_i = self.base_position[:, 1].unsqueeze(1)  # (num_agents, 1)
            z_i = self.base_position[:, 2].unsqueeze(1)  # (num_agents, 1)

            # Convert world coordinates (mx, my) to grid indices (gx, gy) for all agents
            self.mx = cos_yaw * self.localmap_grid[:, :, 0] - sin_yaw * self.localmap_grid[:, :, 1] + x_i
            self.my = sin_yaw * self.localmap_grid[:, :, 0] + cos_yaw * self.localmap_grid[:, :, 1] + y_i

            # Retrieve the height for all agents based on the grid indices
            gx = ((self.mx - self.terrain_origins_all[:, 0].unsqueeze(-1)) / self.horizontal_scale).int()
            gy = ((self.my - self.terrain_origins_all[:, 1].unsqueeze(-1)) / self.horizontal_scale).int()

            heightfield_indices = (torch.arange(self.num_agents) // (self.cfg.env.num_agents_per_terrain)).unsqueeze(1)
            heightfield_indices = heightfield_indices.expand(-1, gx.shape[1])

            self.localmaps = self.context_sampler.heightfields[heightfield_indices, gx, gy] / 10
        else:
            self.state = torch.cat([
                self.base_orientation,
                self.projected_lin_vel / self.state_space_lin_vel_max,
                self.projected_ang_vel / self.state_space_ang_vel_max,
                self.goal_dist,
                self.prev_action / self.action_upper,
            ], dim=1)

        # Student State
        if self.cfg.state_space.use_noise:
            noisy_orientation = self.base_orientation + torch.normal(0., 0.01, size=(self.num_agents, 4)).to(self.device).float()
            noisy_orientation = noisy_orientation / torch.norm(noisy_orientation, dim=-1).unsqueeze(-1)

            noisy_goal_dist = self.goal_dist + torch.normal(0., 0.05, size=(self.num_agents, 2)).to(self.device).float()
            noisy_projected_lin_vel = self.projected_lin_vel + torch.normal(0., 0.1, size=(self.num_agents, 3)).to(self.device).float()
            noisy_projected_ang_vel = self.projected_ang_vel + torch.normal(0., 0.1, size=(self.num_agents, 3)).to(self.device).float()
            # compute the student's state
            self.student_state = torch.cat([
                noisy_orientation,
                noisy_projected_lin_vel / self.state_space_lin_vel_max,
                noisy_projected_ang_vel / self.state_space_ang_vel_max,
                noisy_goal_dist,
                self.prev_action / self.action_upper
            ], dim=1)
        else:
            self.student_state = torch.cat([
                self.base_orientation,
                self.projected_lin_vel / self.state_space_lin_vel_max,
                self.projected_ang_vel / self.state_space_ang_vel_max,
                self.goal_dist,
                self.prev_action / self.action_upper,
            ], dim=1)
    
    def _compute_reward_and_reset(self):
        """Computes the reward, termination conditions, and truncation conditions for each environment.
        """
        # truncations
        self.episode_truncation = self.step_index >= self.cfg.env.max_episode_length
        self.position_truncation = out_of_bounds(self.shifted_base_position[:, :2], self.min_position, self.max_position)
        self.z_out_of_bound_truncation = self.z_terminal()

        # terminations
        self.goal_termination = self.goal_terminal() 
        self.collision_termination = self.collision_terminal()
        # self.wheel_in_air_termination = self.wheel_in_air_terminal()
        # self.collision_termination = torch.logical_or(self.collision_termination, self.wheel_in_air_termination)

        # ****** terminal when orientation is above 20 degrees
        self.orientation_termination = self.orientation_terminal()
        self.collision_termination = torch.logical_or(self.collision_termination, self.orientation_termination)
        
        # ****** if terminated within goal, then resample
        # self.episode_greater_100 = (self.step_index >= 100)
        # self.termination_copy = torch.logical_and(self.collision_termination, self.episode_greater_100)
        
        # compute truncations and terminations
        self.truncation = torch.logical_or(self.episode_truncation, self.position_truncation) 
        self.truncation = torch.logical_or(self.truncation, self.z_out_of_bound_truncation)

        self.terminations = torch.logical_or(self.goal_termination, self.collision_termination)

        # the rewards are added
        self.rewards = torch.zeros(self.num_agents, device=self.device) 
        for i, reward_fn in enumerate(self.reward_fns):
            reward = reward_fn()
            self.rewards += reward * self.reward_scales[reward_fn.__name__] 
            self.individual_returns[:, i] += reward * self.reward_scales[reward_fn.__name__] 
        
        # termination rewards
        self.rewards[torch.where(self.goal_termination == 1)[0]] += 1000.0
        # self.rewards[torch.where(self.collision_termination == 1)[0]] = 0. # this reward should be admissible, i.e., it should be less than the cumulative reward of the shortest path to the goal
    
    def _shift_positions(self, env_ids, positions):
        """Shifts the position of each agent to the respective terrain frame.
        """
        terrain_ids = (env_ids / self.cfg.env.num_agents_per_terrain).int()
        xi = (terrain_ids / self.context_sampler.num_rows).int()
        yi = (terrain_ids % self.context_sampler.num_rows).int()
        shifted_positions = positions + self.terrain_origins[xi, yi].resize(env_ids.size(0), 2)
        return shifted_positions
    
    def compute_accelerations(self):
        """Computes the accelerations based on the current and previous velocities.
        """
        return
        # linear acceleration = a = f/m
        self.base_lin_acc = self.base_net_force / self.base_mass
        self.base_ang_acc = self.base_inertia_matrix_inv @ (self.base_net_torque.view(self.num_agents, 3, 1) \
            - torch.cross(self.base_angular_vel.view(self.num_agents, 3, 1), self.base_inertia_matrix @ self.base_angular_vel.view(self.num_agents, 3, 1), dim=1))
        self.base_ang_acc = self.base_ang_acc.view(self.num_agents, 3)
    
    def _render_depth(self):
        # render camera images
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)

        depth_images = torch.empty(
            (self.num_agents, self.cfg.robot.camera.img_size[1], self.cfg.robot.camera.img_size[0]), 
            device=self.device, dtype=torch.float32)
        
        if self.cfg.robot.camera.type == 'mono':
            rgb_images = torch.empty(
                (self.num_agents, self.cfg.robot.camera.img_size[1], self.cfg.robot.camera.img_size[0]), 
                device=self.device, dtype=torch.uint8)
            
        for i in range(self.num_agents):
            if self.cfg.robot.camera.type == 'depth':
                depth_image = self.gym.get_camera_image_gpu_tensor(self.sim, self.envs[i], self.camera_handles[i], gymapi.IMAGE_DEPTH)
                depth_images[i] = -gymtorch.wrap_tensor(depth_image)
            elif self.cfg.robot.camera.type == 'mono':
                rgb_image = self.gym.get_camera_image_gpu_tensor(self.sim, self.envs[i], self.camera_handles[i], gymapi.IMAGE_COLOR)
                depth_images[i] = gymtorch.wrap_tensor(rgb_image)
            elif self.cfg.robot.camera.type == 'stereo':
                ir_left_image = self.gym.get_camera_image_gpu_tensor(self.sim, self.envs[i], self.camera_handles[i*2], gymapi.IMAGE_COLOR)
                # Since rgb are same, we don't take mean to save (some) time
                ir_left_image = gymtorch.wrap_tensor(ir_left_image)[:, :, 0].cpu().numpy()
                ir_right_image = self.gym.get_camera_image_gpu_tensor(self.sim, self.envs[i], self.camera_handles[i*2+1], gymapi.IMAGE_COLOR)
                ir_right_image = gymtorch.wrap_tensor(ir_right_image)[:, :, 0].cpu().numpy()

                depth_image = self.gym.get_camera_image_gpu_tensor(self.sim, self.envs[i], self.camera_handles[i*2], gymapi.IMAGE_DEPTH)
                depth_image = -gymtorch.wrap_tensor(depth_image)

                # set the invalid pixels to 0
                inf_indices = torch.where(depth_image == torch.inf)
                max_indices = torch.where(depth_image > self.cfg.robot.camera.far_plane)
                depth_image[max_indices[0], max_indices[1]] = 0.0
                depth_image[inf_indices[0], inf_indices[1]] = 0.0

                sgm_res = self.sgbm.compute(ir_left_image, ir_right_image)
                sgm_res = torch.from_numpy(sgm_res).to(device=self.device)
                sgm_res[depth_image < 0.05] = 0

                depth_images[i] = sgm_res

                # image_name = '/home/youwyu/Downloads/test/images/depth_' + "{}.png".format(i)
                # depth_image[depth_image < self.cfg.robot.camera.near_plane] = 0
                # depth_image[depth_image > self.cfg.robot.camera.far_plane] = 0
                # _image = (depth_image / self.cfg.robot.camera.far_plane * 255.).cpu().numpy()
                # _image = Image.fromarray(_image.astype(np.uint8), mode='L')
                # _image.save(image_name)

                # image_name = '/home/youwyu/Downloads/test/images/left_' + "{}.png".format(i)
                # _image = (ir_images[i][0]).cpu().numpy().astype(np.uint8)
                # _image = Image.fromarray(_image, mode='L')
                # _image.save(image_name)

                # image_name = '/home/youwyu/Downloads/test/images/right_' + "{}.png".format(i)
                # _image = (ir_images[i][1]).cpu().numpy().astype(np.uint8)
                # _image = Image.fromarray(_image, mode='L')
                # _image.save(image_name)
        self.gym.end_access_image_tensors(self.sim)
        
        # if self.cfg.state_space.use_noise:
        #     depth_images = self.noisify_depth(depth_images)
        #     for i in range(self.num_agents):
        #         image_name = '/home/youwyu/Downloads/test/images/noisy_' + "{}.png".format(i)
        #         _image = (depth_images[i] / self.cfg.robot.camera.far_plane * 255.).cpu().numpy()
        #         _image = Image.fromarray(_image.astype(np.uint8), mode='L')
        #         _image.save(image_name)

        depth_images = depth_images / self.cfg.robot.camera.far_plane

        return depth_images

    # https://github.com/ZiwenZhuang/parkour
    def noisify_depth(self, depth_images):
        # mask =  F.max_pool2d(
        #     torch.abs(F.conv2d(depth_images, self.contour_detection_kernel, padding= 1)).max(dim= -3, keepdim= True)[0],
        #     kernel_size= self.cfg.noise.forward_depth.contour_detection_kernel_size,
        #     stride= 1,
        #     padding= int(self.cfg.noise.forward_depth.contour_detection_kernel_size / 2),
        # ) > self.cfg.noise.forward_depth.contour_threshold
        # depth_images[mask] = 0.

        """ Simulate the noise from the depth limit of the stereo camera. """
        N, H, W = depth_images.shape
        far_mask = depth_images > 2.0 #self.cfg.noise.forward_depth.stereo_far_distance
        too_close_mask = depth_images < 0.12 #self.cfg.noise.forward_depth.stereo_min_distance
        near_mask = (~far_mask) & (~too_close_mask)

        # add noise to the far points
        far_noise = uniform(
            0., 0.08, #self.cfg.noise.forward_depth.stereo_far_noise_std,
            (N, H, W))
        far_noise = far_noise * far_mask
        depth_images += far_noise

        # add noise to the near points
        near_noise = uniform(
            0., 0.02,#self.cfg.noise.forward_depth.stereo_near_noise_std,
            (N, H, W))
        near_noise = near_noise * near_mask
        depth_images += near_noise

        # add artifacts to the too close points
        vertical_block_mask = too_close_mask.sum(dim= -2, keepdim= True) > (too_close_mask.shape[-2] * 0.6)
        full_block_mask = vertical_block_mask & too_close_mask
        half_block_mask = (~vertical_block_mask) & too_close_mask
        # add artifacts where vertical pixels are all too close
        # stereo_full_block_values = [0.0, 0.25, 0.5, 1., 3.]
        # for pixel_value in random.sample(
        #         stereo_full_block_values,
        #         len(stereo_full_block_values),
        #     ):
        #     artifacts_buffer = torch.ones_like(depth_images)
        #     artifacts_buffer = self._add_depth_artifacts(artifacts_buffer,
        #         0.004,#self.cfg.noise.forward_depth.stereo_full_block_artifacts_prob,
        #         [62, 1.5],#self.cfg.noise.forward_depth.stereo_full_block_height_mean_std,
        #         [3, 0.01],#self.cfg.noise.forward_depth.stereo_full_block_width_mean_std,
        #     )
        #     depth_images[full_block_mask] = ((1 - artifacts_buffer) * pixel_value)[full_block_mask]
        # add artifacts where not all the same vertical pixels are too close
        half_block_spark = uniform(
            0., 1.,
            (N, H, W)) < 0.02#self.cfg.noise.forward_depth.stereo_half_block_spark_prob
        depth_images[half_block_mask] = (half_block_spark.to(torch.float32) * 3000)[half_block_mask]

        return depth_images

    # https://github.com/ZiwenZhuang/parkour
    def _add_depth_artifacts(self, depth_images,
            artifacts_prob,
            artifacts_height_mean_std,
            artifacts_width_mean_std,
        ):
        """ Simulate artifacts from stereo depth camera. In the final artifacts_mask, where there
        should be an artifacts, the mask is 1.
        """
        N, H, W = depth_images.shape
        def _clip(x, dim):
            return torch.clip(x, 0., (H, W)[dim])

        # random patched artifacts
        artifacts_mask = uniform(
            0., 1.,
            (N, H, W)) < artifacts_prob

        artifacts_mask = artifacts_mask & (depth_images > 0.)
        artifacts_coord = torch.nonzero(artifacts_mask).to(torch.float32) # (n_, 3) n_ <= N * H * W
        artifcats_size = (
            torch.clip(
                artifacts_height_mean_std[0] + torch.randn(
                    (artifacts_coord.shape[0],),
                    device= self.device,
                ) * artifacts_height_mean_std[1],
                0., H,
            ),
            torch.clip(
                artifacts_width_mean_std[0] + torch.randn(
                    (artifacts_coord.shape[0],),
                    device= self.device,
                ) * artifacts_width_mean_std[1],
                0., W,
            ),
        ) # (n_,), (n_,)
        artifacts_top_left = (
            _clip(artifacts_coord[:, 1] - artifcats_size[0] / 2, 0),
            _clip(artifacts_coord[:, 2] - artifcats_size[1] / 2, 1),
        )
        artifacts_bottom_right = (
            _clip(artifacts_coord[:, 1] + artifcats_size[0] / 2, 0),
            _clip(artifacts_coord[:, 2] + artifcats_size[1] / 2, 1),
        )
        for i in range(N):
            # NOTE: make sure the artifacts points are as few as possible
            artifacts_mask = self.form_artifacts(
                H, W,
                artifacts_top_left[0][artifacts_coord[:, 0] == i],
                artifacts_bottom_right[0][artifacts_coord[:, 0] == i],
                artifacts_top_left[1][artifacts_coord[:, 0] == i],
                artifacts_bottom_right[1][artifacts_coord[:, 0] == i],
            )
            depth_images[i] *= (1 - artifacts_mask)

        return depth_images
    
    @torch.no_grad()
    def form_artifacts(self,
            H, W, # image resolution
            tops, bottoms, # artifacts positions (in pixel) shape (n_,)
            lefts, rights,
        ):
        """ Paste an artifact to the depth image.
        NOTE: Using the paradigm of spatial transformer network to build the artifacts of the
        entire depth image.
        """
        batch_size = tops.shape[0]
        tops, bottoms = tops[:, None, None], bottoms[:, None, None]
        lefts, rights = lefts[:, None, None], rights[:, None, None]

        # build the source patch
        source_patch = torch.zeros((batch_size, 1, 25, 25), device= self.device)
        source_patch[:, :, 1:24, 1:24] = 1.

        # build the grid
        grid = torch.zeros((batch_size, H, W, 2), device= self.device)
        grid[..., 0] = torch.linspace(-1, 1, W, device= self.device).view(1, 1, W)
        grid[..., 1] = torch.linspace(-1, 1, H, device= self.device).view(1, H, 1)
        grid[..., 0] = (grid[..., 0] * W + W - rights - lefts) / (rights - lefts)
        grid[..., 1] = (grid[..., 1] * H + H - bottoms - tops) / (bottoms - tops)

        # sample using the grid and form the artifacts for the entire depth image
        artifacts = torch.clip(
            F.grid_sample(
                source_patch,
                grid,
                mode= "bilinear",
                padding_mode= "zeros",
                align_corners= False,
            ).sum(dim= 0).view(H, W),
            0, 1,
        )

        return artifacts

    ## reward functions ###
    def goal_reward(self):
        # solution 1: progress to the goal, i.e., d_{t-1} - d_t - slack
        # current timestep goal distance
        d_t1 = torch.norm(self.goal - self.base_position[:, :2], dim=1)
        # previous timestep goal distance
        d_t0 = torch.norm(self.goal - self.prev_base_position[:, :2], dim=1)
        return d_t0 - d_t1 - 0.001 # -0.001 is a slack variable penalizing longer path 

        # solution 2: absolute distance to the goal, i.e., |g-s|
        # return -torch.sum(torch.square(self.goal_dist), dim=1)

    def forward_vel_reward(self):
        """
        Rewards the forward velocity.
        """
        lin_vel = self.projected_lin_vel[:, 0]
        return (lin_vel > 0.0).float()

    def orientation_reward(self):
        return -torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def lin_vel_reward(self):
        """
        Penalizes the linear velocity in y and z direction.
        """
        return -torch.sum(torch.square(self.projected_lin_vel[:, 1:]), dim=1)
    
    def ang_vel_reward(self):
        """
        Penalizes the angular velocity in x and y direction.
        """
        return -torch.sum(torch.abs(self.projected_ang_vel[:, :2]), dim=1)

    def action_mag_reward(self):
        """"
        Penalizes the action magnitude.
        """
        return -torch.sum(torch.square(self.action), dim=1)

    def action_limits_reward(self):
        """
        Penalizes actions that are out of action limits.
        """
        action_out_of_limits = - (self.action - self.action_lower).clip(max=0.0)
        action_out_of_limits += (self.action - self.action_upper).clip(min=0.0)
        return -torch.sum(action_out_of_limits, dim=1)

    def action_smoothness_reward(self):
        return -torch.sum(torch.square(self.action - self.prev_action), dim=1)

    def backward_vel_reward(self):
        """
        Penalizes backward velocity -- if it moves backward, gives a -1 reward.
        """
        lin_vel_x = self.projected_lin_vel[:, 0]
        backward_vel = (lin_vel_x < 0.0).float()
        return -backward_vel 
    
    def orientation_limit_reward(self):
        # orientation
        roll, pitch = self.euler_angles[:, 0], self.euler_angles[:, 1]
        # roll and pitch should be within [-pi/6, pi/6]
        roll_in_range = angle_in_between(roll, self.min_angle_reward, self.max_angle_reward)
        pitch_in_range = angle_in_between(pitch, self.min_angle_reward, self.max_angle_reward)
        orientation_out_of_bound = (~torch.logical_and(roll_in_range, pitch_in_range)).float()
        return -orientation_out_of_bound

    def velocity_limit_reward(self):
        """
        Penalizes the higher-order state dimensions that are out of bounds:
        1. Roll and Pitch orientations
        2. Linear and angular velocities
        """
        # velocities
        lin_vel_out_of_limits = - (self.projected_lin_vel - self.state_space_lin_vel_min).clip(max=0.0)
        lin_vel_out_of_limits += (self.projected_lin_vel- self.state_space_lin_vel_max).clip(min=0.0)
        ang_vel_out_of_limits = - (self.projected_ang_vel - self.state_space_ang_vel_min).clip(max=0.0)
        ang_vel_out_of_limits += (self.projected_ang_vel- self.state_space_ang_vel_max).clip(min=0.0)
        return -torch.sum(torch.square(lin_vel_out_of_limits), dim=1) - torch.sum(torch.square(ang_vel_out_of_limits), dim=1)
    
    def acceleration_limit_reward(self):
        acceleration = torch.abs(self.action - self.prev_action) / self.dt
        valid_linear_acceleration = (acceleration[:, 0] > self.cfg.state_space.lin_acc.lin_acc_mag_limit)
        valid_angular_acceleration = (acceleration[:, 0] > self.cfg.state_space.ang_acc.ang_acc_mag_limit)
        return -torch.logical_or(valid_linear_acceleration, valid_angular_acceleration).float()

    def wheel_air_reward(self):
        """
        Penalizes the air time of the wheels. The wheels are in the air when the net contact force in the z-axis is 0. 
        """
        wheel_in_air = (self.net_cf_wheels[:, :, 2] == 0.0).float()
        return -torch.pow(torch.sum(wheel_in_air, dim=1), 3)

    def collision_reward(self):
        """
        A collision is detected when the two conditions meet: 
        1. net contact forces of the wheels have values in x and y axes; 
        2. the robot's base link's net contact force is non-zero in all directions. 
        """
        # any of the wheel has non-zero contact force in x and y axes
        wheel_in_contact = torch.any(torch.norm(self.net_cf_wheels[:, :, :2], dim=2) >= 1e-2, dim=-1) # contact force offset 
        # base has non-zero contact force in all directions
        base_in_contact = torch.norm(self.net_cf_base, dim=1) >= 1e-2
        in_collision = torch.logical_or(wheel_in_contact, base_in_contact).float()
        return -in_collision
    
    def jerk_reward(self):
        """Jerk is the third derivative of position.
        """
        linear_jerk_mag = torch.norm(self.base_net_force - self.prev_base_net_force, dim=1)
        angular_jerk_mag = torch.norm(self.base_net_torque - self.prev_base_net_torque, dim=1)
        # penalize the jerk magnitude
        return -linear_jerk_mag - angular_jerk_mag

    def soft_collision_reward(self):
        """This reward function provides a soft penalty for collision, i.e., the collision penalty is proportional to the contact forces of the wheels and the base.
        """
        wheel_contact_force_mag = torch.norm(self.net_cf_wheels[:, :, :2], dim=-1)
        base_contact_force_mag = torch.norm(self.net_cf_base, dim=-1)
        # collision penalty is proportional to the contact forces
        return -torch.sum(wheel_contact_force_mag, dim=1) - base_contact_force_mag
    
    def survival_reward(self):
        """Provide a small value for each time step to encourage the agent to survive.
        """
        return torch.ones(self.num_agents, device=self.device)
    
    def goal_orientation_reward(self):
        """Encourage the agent to meet the goal orientation.
           Instead of strictly set as terminal function
        """
        goal_near = torch.norm(self.goal_dist, dim = 1) < 0.7
        ang_reward = torch.squeeze(torch.t(torch.abs(torch.pi - self.goal_yaw_dist)))
        return torch.mul(goal_near.float(), ang_reward)

    ### terminal conditions ###
    def goal_terminal(self):
        goal_arrived = torch.norm(self.goal_dist, dim=1) < 0.1
        lin_vel = torch.norm(self.projected_lin_vel, dim=1)
        ang_vel = torch.norm(self.projected_ang_vel, dim=1)
        # tolerance on linear and angular velocities
        zero_vel = torch.logical_and(lin_vel < 1.0, ang_vel < 0.1)

        ang_dist = self.goal_yaw_dist < np.pi / 3.

        return goal_arrived
        return torch.logical_and(goal_arrived, zero_vel).float()

    def z_terminal(self):
        penetration = self.base_position[:, 2] < -0.1
        return penetration.float() 

    def wheel_in_air_terminal(self):
        """Instead of penalizing the wheels not in contact with the ground, we directly terminates the episode.
        """
        # we only check he agents that have previously contacted ground 
        wheel_in_air = (self.net_cf_wheels[:, :, 2] <= 1e-5).float()
        wheel_in_air = torch.any(wheel_in_air, dim=-1)
        wheel_in_air = torch.logical_and(wheel_in_air, self.has_contact)
        return wheel_in_air.float()

    def orientation_terminal(self):
        roll, pitch = self.euler_angles[:, 0], self.euler_angles[:, 1]
        # roll and pitch should be within [-pi/9, pi/9]
        roll_in_range = angle_in_between(roll, self.min_angle_terminal, self.max_angle_terminal)
        pitch_in_range = angle_in_between(pitch, self.min_angle_terminal, self.max_angle_terminal)
        return (~torch.logical_and(roll_in_range, pitch_in_range)).float()

    def collision_terminal(self):
        wheel_contact_force_mag = torch.norm(self.net_cf_wheels[..., :2], dim=-1)
        base_contact_force_mag = torch.norm(self.net_cf_base, dim=-1)
        # contact force in x and y directions is a surrogate for collision, although this is not a perfect indicator 
        wheel_in_collision = torch.any(wheel_contact_force_mag >= 1., dim=-1)
        # contact force on the base
        base_in_collision = base_contact_force_mag >= 1.
        return base_in_collision.float()
        return torch.logical_or(wheel_in_collision, base_in_collision).float() 
        
    def close(self):
        # This is harmful since some memory would stuck in GPU
        self.gym.destroy_sim(self.sim)