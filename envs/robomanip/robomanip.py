"""Pick and place task."""

from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from dm_control import mjcf
from gymnasium.spaces import Box

from envs.robomanip import controllers, env, lie, mjcf_utils

# XML files.
_HERE = Path(__file__).resolve().parent
ARENA_XML = _HERE / 'common' / 'floor_wall.xml'
UR5E_XML = _HERE / 'common' / 'universal_robots_ur5e' / 'ur5e.xml'
ROBOTIQ_XML = _HERE / 'common' / 'robotiq_2f85' / '2f85.xml'

# Default joint configuration for the arm (used by the IK controller).
_HOME_QPOS = np.asarray([-np.pi / 2, -np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0])

# Box bounds (xy) for the end-effector, in meters.
_WORKSPACE_BOUNDS = np.asarray([[0.25, -0.45, 0.02], [0.6, 0.45, 0.35]])

# Box bounds for sampling the initial object position, in meters.
_OBJECT_SAMPLING_BOUNDS = np.asarray([[0.3, -0.4], [0.55, 0.4]])

# Box bounds for sampling the target position, in meters.
_TARGET_SAMPLING_BOUNDS = np.asarray([[0.3, -0.4], [0.55, 0.4]])

# Actuator PD gains.
_ACTUATOR_KP = np.asarray([4500, 4500, 4500, 2000, 2000, 500])
_ACTUATOR_KD = np.asarray([-450, -450, -450, -200, -200, -50])

_EFFECTOR_DOWN_ROTATION = lie.SO3(np.asarray([0.0, 1.0, 0.0, 0.0]))

_ROBOTIQ_CONSTANT = 255.0

_OBJECT_RGBAS = np.asarray(
    [
        [0.96, 0.26, 0.33, 1.0],
        [1.0, 0.69, 0.21, 1.0],
        [0.06, 0.74, 0.21, 1.0],
        [0.35, 0.55, 0.91, 1.0],
    ]
)
_OBJECT_THICKNESS = 0.02
_OBJECT_SYMMETRY = np.pi / 2

_CAMERAS = {
    'overhead_left': {
        'pos': (-0.3, -0.263, 1),
        'xyaxes': (0, -1, 0, 0.821, 0, 0.571),
    },
    'overhead_right': {
        'pos': (-0.3, 0.263, 1),
        'xyaxes': (0, -1, 0, 0.821, 0, 0.571),
    },
    'front': {
        'pos': (1.171, 0.002, 0.753),
        'xyaxes': (-0.002, 1.000, -0.000, -0.569, -0.001, 0.822),
    },
    'topdown': {
        'pos': (0.5, 0.0, 1),
        'euler': (0.0, 0.0, 1.57),
    },
}


class RoboManipEnv(env.CustomMuJoCoEnv):
    def __init__(
        self,
        env_type: str = 'cubes',
        pixel_observation: bool = False,
        absolute_action_space: bool = False,
        physics_timestep: float = 0.002,
        control_timestep: float = 0.05,
        terminate_at_goal: bool = True,
        mode: str = 'evaluation',  # 'evaluation' or 'data_collection'
        visualize_info: bool = True,  # Whether to visualize the targets and successes.
        **kwargs,
    ):
        super().__init__(
            physics_timestep=physics_timestep,
            control_timestep=control_timestep,
            **kwargs,
        )

        self._workspace_bounds = _WORKSPACE_BOUNDS
        self._object_sampling_bounds = _OBJECT_SAMPLING_BOUNDS
        self._target_sampling_bounds = _TARGET_SAMPLING_BOUNDS
        self._env_type = env_type
        self._pixel_observation = pixel_observation
        self._absolute_action_space = absolute_action_space
        self._terminate_at_goal = terminate_at_goal
        self._mode = mode
        self._visualize_info = visualize_info

        if self._env_type == 'cube':
            self._object_xml = _HERE / 'common' / 'cube.xml'
            self._num_objects = 1
        else:
            self._object_xml = _HERE / 'common' / 'cubes.xml'
            self._num_objects = 4

        self._symmetry = _OBJECT_SYMMETRY
        self._object_z = _OBJECT_THICKNESS

        ik_mjcf = mjcf.from_path(UR5E_XML, escape_separators=True)
        xml_str = mjcf_utils.to_string(ik_mjcf)
        assets = mjcf_utils.get_assets(ik_mjcf)
        ik_model = mujoco.MjModel.from_xml_string(xml_str, assets)

        self._ik = controllers.DiffIKController(model=ik_model, sites=['attachment_site'])

        if self._absolute_action_space:
            self.action_low = np.array([*self._workspace_bounds[0], -np.pi, 0.0])
            self.action_high = np.array([*self._workspace_bounds[1], np.pi, 1.0])
        else:
            action_range = np.array([0.05] * 3 + [0.3, 1.0])
            self.action_low = -action_range
            self.action_high = action_range

        if self._mode == 'evaluation':
            self.task_infos = []
            self.num_tasks = None
            self.cur_task_idx = None
            self.cur_task_info = None
            self.set_tasks()

            self._cur_goal_ob = None

    def set_tasks(self):
        if self._env_type == 'cube':
            self.task_infos = [
                dict(
                    task_name='task1',
                    init_xyzs=np.array([[0.425, 0.1, 0.02]]),
                    goal_xyzs=np.array([[0.425, -0.1, 0.02]]),
                ),
                dict(
                    task_name='task2',
                    init_xyzs=np.array([[0.32, -0.38, 0.02]]),
                    goal_xyzs=np.array([[0.53, 0.38, 0.02]]),
                ),
            ]
        else:
            self.task_infos = [
                dict(
                    task_name='task1',
                    init_xyzs=np.array(
                        [
                            [0.32, -0.38, 0.02],
                            [0.53, -0.38, 0.02],
                            [0.53, 0.38, 0.02],
                            [0.32, 0.38, 0.02],
                        ]
                    ),
                    goal_xyzs=np.array(
                        [
                            [0.3, 0, 0.02],
                            [0.425, -0.125, 0.02],
                            [0.55, 0, 0.02],
                            [0.425, 0.125, 0.02],
                        ]
                    ),
                ),
                dict(
                    task_name='task2',
                    init_xyzs=np.array(
                        [
                            [0.35, -0.2, 0.02],
                            [0.35, 0.2, 0.02],
                            [0.5, 0.2, 0.02],
                            [0.5, -0.2, 0.02],
                        ]
                    ),
                    goal_xyzs=np.array(
                        [
                            [0.5, -0.2, 0.02],
                            [0.35, -0.2, 0.02],
                            [0.35, 0.2, 0.02],
                            [0.5, 0.2, 0.02],
                        ]
                    ),
                ),
                dict(
                    task_name='task3',
                    init_xyzs=np.array(
                        [
                            [0.55, 0, 0.02],
                            [0.425, -0.3, 0.02],
                            [0.3, 0, 0.02],
                            [0.425, 0.3, 0.02],
                        ]
                    ),
                    goal_xyzs=np.array(
                        [
                            [0.425, 0, 0.02],
                            [0.425, 0, 0.06],
                            [0.425, 0, 0.10],
                            [0.425, 0, 0.14],
                        ]
                    ),
                ),
                dict(
                    task_name='task4',
                    init_xyzs=np.array(
                        [
                            [0.425, -0.2, 0.02],
                            [0.425, -0.2, 0.06],
                            [0.425, 0.2, 0.02],
                            [0.425, 0.2, 0.06],
                        ]
                    ),
                    goal_xyzs=np.array(
                        [
                            [0.425, 0.2, 0.02],
                            [0.425, 0.2, 0.06],
                            [0.425, -0.2, 0.02],
                            [0.425, -0.2, 0.06],
                        ]
                    ),
                ),
                dict(
                    task_name='task5',
                    init_xyzs=np.array(
                        [
                            [0.425, 0.2, 0.02],
                            [0.425, 0.2, 0.06],
                            [0.425, 0.2, 0.10],
                            [0.425, 0.2, 0.14],
                        ]
                    ),
                    goal_xyzs=np.array(
                        [
                            [0.425, -0.22, 0.02],
                            [0.425, -0.22, 0.06],
                            [0.425, -0.18, 0.02],
                            [0.425, -0.18, 0.06],
                        ]
                    ),
                ),
                dict(
                    task_name='task6',
                    init_xyzs=np.array(
                        [
                            [0.35, -0.2, 0.02],
                            [0.5, -0.2, 0.02],
                            [0.5, 0.2, 0.02],
                            [0.35, 0.2, 0.02],
                        ]
                    ),
                    goal_xyzs=np.array(
                        [
                            [0.425, 0.0, 0.02],
                            [0.5, -0.2, 0.02],
                            [0.5, 0.2, 0.02],
                            [0.35, 0.2, 0.02],
                        ]
                    ),
                ),
            ]
        self.num_tasks = len(self.task_infos)

    def build_mjcf_model(self) -> mjcf.RootElement:
        # Scene.
        arena_mjcf = mjcf.from_path(ARENA_XML.as_posix())
        arena_mjcf.model = 'ur5e_arena'

        arena_mjcf.statistic.center = (0.3, 0, 0.15)
        arena_mjcf.statistic.extent = 0.7
        getattr(arena_mjcf.visual, 'global').elevation = -20
        getattr(arena_mjcf.visual, 'global').azimuth = 180
        arena_mjcf.statistic.meansize = 0.04
        arena_mjcf.visual.map.znear = 0.1
        arena_mjcf.visual.map.zfar = 10.0

        # UR5e.
        ur5e_mjcf = mjcf.from_path(UR5E_XML, escape_separators=True)
        ur5e_mjcf.model = 'ur5e'

        for light in ur5e_mjcf.find_all('light'):
            light.remove()

        # Attach the robotiq gripper to the ur5e flange.
        gripper_mjcf = mjcf.from_path(ROBOTIQ_XML, escape_separators=True)
        gripper_mjcf.model = 'robotiq'
        mjcf_utils.attach(ur5e_mjcf, gripper_mjcf, 'attachment_site')

        # Attach ur5e to scene.
        mjcf_utils.attach(arena_mjcf, ur5e_mjcf)

        # Add object to scene.
        object_mjcf = mjcf.from_path(self._object_xml.as_posix())
        arena_mjcf.include_copy(object_mjcf)

        self._object_geoms_list = []
        for i in range(self._num_objects):
            self._object_geoms_list.append(object_mjcf.find('body', f'object_{i}').find_all('geom'))

        self._object_target_geoms_list = []
        for i in range(self._num_objects):
            self._object_target_geoms_list.append(object_mjcf.find('body', f'object_target_{i}').find_all('geom'))

        # Add cameras.
        for camera_name, camera_kwargs in _CAMERAS.items():
            arena_mjcf.worldbody.add('camera', name=camera_name, **camera_kwargs)

        # Cache joint and actuator elements.
        self._arm_jnts = mjcf_utils.safe_find_all(
            ur5e_mjcf,
            'joint',
            exclude_attachments=True,
        )
        self._arm_acts = mjcf_utils.safe_find_all(
            ur5e_mjcf,
            'actuator',
            exclude_attachments=True,
        )
        self._gripper_jnts = mjcf_utils.safe_find_all(gripper_mjcf, 'joint', exclude_attachments=True)
        self._gripper_acts = mjcf_utils.safe_find_all(gripper_mjcf, 'actuator', exclude_attachments=True)

        # ================================ #
        # Visualization.
        # ================================ #
        mjcf_utils.add_bounding_box_site(
            arena_mjcf.worldbody,
            lower=self._workspace_bounds[0],
            upper=self._workspace_bounds[1],
            rgba=(0.2, 0.2, 0.6, 0.1),
            group=4,
            name='workspace_bounds',
        )
        mjcf_utils.add_bounding_box_site(
            arena_mjcf.worldbody,
            lower=np.asarray((*self._target_sampling_bounds[0], 0.02)),
            upper=np.asarray((*self._target_sampling_bounds[1], 0.02)),
            rgba=(0.6, 0.3, 0.3, 0.2),
            group=4,
            name='object_bounds',
        )
        # ================================ #

        # TODO: Remove this
        ################### FOR DEBUGGING ###################
        # with open('/Users/seohongpark/Downloads/manip/manip_cur.xml', 'w') as file:
        #     file.write(mjcf_utils.to_string(arena_mjcf))
        ################### FOR DEBUGGING ###################

        return arena_mjcf

    def post_compilation(self):
        # Arm joint and actuator IDs.
        arm_joint_names = [j.full_identifier for j in self._arm_jnts]
        self._arm_joint_ids = np.asarray([self._model.joint(name).id for name in arm_joint_names])
        actuator_names = [a.full_identifier for a in self._arm_acts]
        self._arm_actuator_ids = np.asarray([self._model.actuator(name).id for name in actuator_names])
        gripper_actuator_names = [a.full_identifier for a in self._gripper_acts]
        self._gripper_actuator_ids = np.asarray([self._model.actuator(name).id for name in gripper_actuator_names])
        self._gripper_opening_joint_id = self._model.joint('ur5e/robotiq/right_driver_joint').id

        # Modify PD gains.
        self._model.actuator_gainprm[self._arm_actuator_ids, 0] = _ACTUATOR_KP
        self._model.actuator_gainprm[self._arm_actuator_ids, 2] = _ACTUATOR_KD
        self._model.actuator_biasprm[self._arm_actuator_ids, 1] = -_ACTUATOR_KP

        # Object joint id.
        self._object_geom_ids_list = [
            [self._model.geom(geom.full_identifier).id for geom in object_geoms]
            for object_geoms in self._object_geoms_list
        ]

        # Mocap IDs.
        self._object_target_mocap_ids = [
            self._model.body(f'object_target_{i}').mocapid[0] for i in range(self._num_objects)
        ]
        self._object_target_geom_ids_list = [
            [self._model.geom(geom.full_identifier).id for geom in object_target_geoms]
            for object_target_geoms in self._object_target_geoms_list
        ]

        # Site IDs.
        self._pinch_site_id = self._model.site('ur5e/robotiq/pinch').id
        self._attach_site_id = self._model.site('ur5e/attachment_site').id

        pinch_pose = lie.SE3.from_rotation_and_translation(
            rotation=lie.SO3.from_matrix(self._data.site_xmat[self._pinch_site_id].reshape(3, 3)),
            translation=self._data.site_xpos[self._pinch_site_id],
        )
        attach_pose = lie.SE3.from_rotation_and_translation(
            rotation=lie.SO3.from_matrix(self._data.site_xmat[self._attach_site_id].reshape(3, 3)),
            translation=self._data.site_xpos[self._attach_site_id],
        )
        self._T_pa = pinch_pose.inverse() @ attach_pose

    def reset(self, options=None, *args, **kwargs):
        if self._mode == 'evaluation':
            if options is not None:
                if 'task_idx' in options:
                    self.cur_task_idx = options['task_idx']
                    self.cur_task_info = self.task_infos[self.cur_task_idx]
                elif 'task_info' in options:
                    self.cur_task_idx = None
                    self.cur_task_info = options['task_info']
                else:
                    raise ValueError('`options` must contain either `task_idx` or `task_info`')
            else:
                # Randomly sample task
                self.cur_task_idx = np.random.randint(self.num_tasks)
                self.cur_task_info = self.task_infos[self.cur_task_idx]

        return super().reset(*args, **kwargs)

    def initialize_episode(self):
        # TODO: Refactor

        for i in range(self._num_objects):
            for gid in self._object_geom_ids_list[i]:
                self._model.geom(gid).rgba = _OBJECT_RGBAS[i]
            for gid in self._object_target_geom_ids_list[i]:
                self._model.geom(gid).rgba[:3] = _OBJECT_RGBAS[i, :3]

        self._data.qpos[self._arm_joint_ids] = _HOME_QPOS
        mujoco.mj_kinematics(self._model, self._data)

        # Sample initial effector position and orientation.
        eff_pos = self.np_random.uniform(*self._workspace_bounds)
        cur_ori = _EFFECTOR_DOWN_ROTATION
        yaw = self.np_random.uniform(-np.pi, np.pi)
        rotz = lie.SO3.from_z_radians(yaw)
        eff_ori = rotz @ cur_ori

        # Solve for initial joint positions using IK.
        T_wp = lie.SE3.from_rotation_and_translation(eff_ori, eff_pos)
        T_wa = T_wp @ self._T_pa
        qpos_init = self._ik.solve(
            pos=T_wa.translation(),
            quat=T_wa.rotation().wxyz,
            curr_qpos=_HOME_QPOS,
        )

        self._data.qpos[self._arm_joint_ids] = qpos_init
        mujoco.mj_forward(self._model, self._data)

        if self._mode == 'data_collection':
            # Randomize object positions and orientations.
            for i in range(self._num_objects):
                xy = self.np_random.uniform(*self._object_sampling_bounds)
                obj_pos = (*xy, self._object_z)
                yaw = self.np_random.uniform(0, 2 * np.pi)
                obj_ori = lie.SO3.from_z_radians(yaw).wxyz.tolist()
                self._data.joint(f'object_joint_{i}').qpos[:3] = obj_pos
                self._data.joint(f'object_joint_{i}').qpos[3:] = obj_ori

            # Set a new target.
            self.set_new_target(return_info=False)
        else:
            # Set object positions and orientations based on the current task.
            init_xyzs = self.cur_task_info['init_xyzs'].copy()
            goal_xyzs = self.cur_task_info['goal_xyzs'].copy()
            permutation = self.np_random.permutation(self._num_objects)
            init_xyzs = init_xyzs[permutation]
            goal_xyzs = goal_xyzs[permutation]

            # First set the current scene to the goal state to get the goal observation.
            saved_qpos = self._data.qpos.copy()
            saved_qvel = self._data.qvel.copy()
            eff_pos = np.random.uniform([0.25, -0.45, 0.20], [0.6, 0.45, 0.35])
            cur_ori = _EFFECTOR_DOWN_ROTATION
            yaw = self.np_random.uniform(-np.pi, np.pi)
            rotz = lie.SO3.from_z_radians(yaw)
            eff_ori = rotz @ cur_ori

            T_wp = lie.SE3.from_rotation_and_translation(eff_ori, eff_pos)
            T_wa = T_wp @ self._T_pa
            qpos_init = self._ik.solve(
                pos=T_wa.translation(),
                quat=T_wa.rotation().wxyz,
                curr_qpos=_HOME_QPOS,
            )

            self._data.qpos[self._arm_joint_ids] = qpos_init
            for i in range(self._num_objects):
                self._data.joint(f'object_joint_{i}').qpos[:3] = goal_xyzs[i]
                self._data.joint(f'object_joint_{i}').qpos[3:] = lie.SO3.identity().wxyz.tolist()
                self._data.mocap_pos[self._object_target_mocap_ids[i]] = goal_xyzs[i]
                self._data.mocap_quat[self._object_target_mocap_ids[i]] = lie.SO3.identity().wxyz.tolist()
            mujoco.mj_forward(self._model, self._data)

            for _ in range(2):
                self.step(self.action_space.sample())

            self._cur_goal_ob = self.compute_observation()

            # Now do the actual reset.
            self._data.qpos[:] = saved_qpos
            self._data.qvel[:] = saved_qvel
            for i in range(self._num_objects):
                obj_pos = init_xyzs[i].copy()
                obj_pos[:2] += self.np_random.uniform(-0.01, 0.01, size=2)
                self._data.joint(f'object_joint_{i}').qpos[:3] = obj_pos
                yaw = self.np_random.uniform(0, 2 * np.pi)
                obj_ori = lie.SO3.from_z_radians(yaw).wxyz.tolist()
                self._data.joint(f'object_joint_{i}').qpos[3:] = obj_ori
                self._data.mocap_pos[self._object_target_mocap_ids[i]] = goal_xyzs[i]
                self._data.mocap_quat[self._object_target_mocap_ids[i]] = lie.SO3.identity().wxyz.tolist()

        # Forward kinematics to update site positions.
        mujoco.mj_forward(self._model, self._data)

        self._success = False

    def set_new_target(self, return_info=True, p_stack=0.5):
        assert self._mode == 'data_collection'

        block_xyzs = np.array([self._data.joint(f'object_joint_{i}').qpos[:3] for i in range(self._num_objects)])
        top_blocks = []
        for i in range(self._num_objects):
            for j in range(self._num_objects):
                if i == j:
                    continue
                if (
                        block_xyzs[j][2] > block_xyzs[i][2]
                        and np.linalg.norm(block_xyzs[i][:2] - block_xyzs[j][:2]) < 0.02
                ):
                    break
            else:
                top_blocks.append(i)

        # Pick one of the top objects as the target.
        self._target_block = self.np_random.choice(top_blocks)

        stack = len(top_blocks) >= 2 and self.np_random.uniform() < p_stack
        if stack:
            # Stack the target block on top of another block.
            block_idx = self.np_random.choice(list(set(top_blocks) - {self._target_block}))
            block_pos = self._data.joint(f'object_joint_{block_idx}').qpos[:3]
            tar_pos = np.array([block_pos[0], block_pos[1], block_pos[2] + 2 * self._object_z])
        else:
            # Randomize target position and orientation.
            xy = self.np_random.uniform(*self._target_sampling_bounds)
            tar_pos = (*xy, self._object_z)
        yaw = self.np_random.uniform(0, 2 * np.pi)
        tar_ori = lie.SO3.from_z_radians(yaw).wxyz.tolist()
        for i in range(self._num_objects):
            if i == self._target_block:
                self._data.mocap_pos[self._object_target_mocap_ids[i]] = tar_pos
                self._data.mocap_quat[self._object_target_mocap_ids[i]] = tar_ori
            else:
                self._data.mocap_pos[self._object_target_mocap_ids[i]] = (0, 0, -0.3)
                self._data.mocap_quat[self._object_target_mocap_ids[i]] = lie.SO3.identity().wxyz.tolist()
        for i in range(self._num_objects):
            if self._visualize_info and i == self._target_block:
                for gid in self._object_target_geom_ids_list[i]:
                    self._model.geom(gid).rgba[3] = 0.2
            else:
                for gid in self._object_target_geom_ids_list[i]:
                    self._model.geom(gid).rgba[3] = 0.0

        if return_info:
            return self.compute_observation(), self.get_reset_info()

    @property
    def action_space(self):
        return gym.spaces.Box(
            low=-np.ones(5),
            high=np.ones(5),
            shape=(5,),
            dtype=np.float32,
        )

    def normalize_action(self, action: np.ndarray) -> np.ndarray:
        # Normalize the action to the range [-1, 1].
        action = 2 * (action - self.action_low) / (self.action_high - self.action_low) - 1
        return np.clip(action, -1, 1)

    def unnormalize_action(self, action: np.ndarray) -> np.ndarray:
        # Unnormalize the action to the range [action_low, action_high].
        return 0.5 * (action + 1) * (self.action_high - self.action_low) + self.action_low

    @property
    def observation_space(self):
        if self._model is None:
            self.reset()

        ex_ob = self.compute_observation()

        return Box(low=-np.inf, high=np.inf, shape=ex_ob.shape, dtype=ex_ob.dtype)

    def set_control(self, action):
        action = self.unnormalize_action(action)
        a_pos, a_ori, a_gripper = action[:3], action[3], action[4]

        if self._absolute_action_space:
            target_effector_translation = a_pos
            target_effector_orientation = lie.SO3.from_z_radians(a_ori) @ _EFFECTOR_DOWN_ROTATION
            target_gripper_opening = a_gripper
        else:
            effector_pos = self._data.site_xpos[self._pinch_site_id].copy()
            effector_yaw = lie.SO3.from_matrix(
                self._data.site_xmat[self._pinch_site_id].copy().reshape(3, 3)
            ).compute_yaw_radians()
            gripper_opening = np.array(np.clip([self._data.qpos[self._gripper_opening_joint_id] / 0.8], 0, 1))
            target_effector_translation = effector_pos + a_pos
            target_effector_orientation = (
                lie.SO3.from_z_radians(a_ori) @ lie.SO3.from_z_radians(effector_yaw) @ _EFFECTOR_DOWN_ROTATION.inverse()
            )
            target_gripper_opening = gripper_opening + a_gripper

        # Make sure the target pose respects the action limits.
        np.clip(
            target_effector_translation,
            *self._workspace_bounds,
            out=target_effector_translation,
        )
        yaw = np.clip(
            target_effector_orientation.compute_yaw_radians(),
            -np.pi,
            +np.pi,
        )
        target_effector_orientation = lie.SO3.from_z_radians(yaw) @ _EFFECTOR_DOWN_ROTATION
        target_gripper_opening = np.clip(target_gripper_opening, 0.0, 1.0)

        # Pinch pose in the world frame -> attach pose in the world frame.
        self._target_effector_pose = lie.SE3.from_rotation_and_translation(
            rotation=target_effector_orientation,
            translation=target_effector_translation,
        )
        T_wa = self._target_effector_pose @ self._T_pa

        # Solve for the desired joint positions.
        qpos_target = self._ik.solve(
            pos=T_wa.translation(),
            quat=T_wa.rotation().wxyz,
            curr_qpos=self._data.qpos[self._arm_joint_ids],
        )

        # Set the desired joint positions for the underlying PD controller.
        self._data.ctrl[self._arm_actuator_ids] = qpos_target
        self._data.ctrl[self._gripper_actuator_ids] = _ROBOTIQ_CONSTANT * target_gripper_opening

    def post_step(self) -> None:
        object_successes = []
        for i in range(self._num_objects):
            obj_pos = self._data.joint(f'object_joint_{i}').qpos[:3]
            tar_pos = self._data.mocap_pos[self._object_target_mocap_ids[i]]
            if np.linalg.norm(obj_pos - tar_pos) <= 0.03:
                object_successes.append(True)
            else:
                object_successes.append(False)

        if self._mode == 'data_collection':
            self._success = object_successes[self._target_block]
        else:
            self._success = all(object_successes)

        for i in range(self._num_objects):
            if self._visualize_info and (self._mode == 'evaluation' or i == self._target_block):
                for gid in self._object_target_geom_ids_list[i]:
                    self._model.geom(gid).rgba[3] = 0.2
            else:
                for gid in self._object_target_geom_ids_list[i]:
                    self._model.geom(gid).rgba[3] = 0.0

            if self._visualize_info and object_successes[i]:
                for gid in self._object_geom_ids_list[i]:
                    self._model.geom(gid).rgba[:3] = (0, 1, 1)
            else:
                for gid in self._object_geom_ids_list[i]:
                    self._model.geom(gid).rgba[:3] = _OBJECT_RGBAS[i, :3]

    def compute_ob_info(self) -> dict[str, np.ndarray]:
        ob_info = {}

        # Proprioceptive observations.
        ob_info['proprio/joint_pos'] = self._data.qpos[self._arm_joint_ids].copy()
        ob_info['proprio/joint_vel'] = self._data.qvel[self._arm_joint_ids].copy()
        ob_info['proprio/effector_pos'] = self._data.site_xpos[self._pinch_site_id].copy()
        ob_info['proprio/effector_yaw'] = np.array(
            [lie.SO3.from_matrix(self._data.site_xmat[self._pinch_site_id].copy().reshape(3, 3)).compute_yaw_radians()]
        )
        ob_info['proprio/gripper_opening'] = np.array(
            np.clip([self._data.qpos[self._gripper_opening_joint_id] / 0.8], 0, 1)
        )
        ob_info['proprio/gripper_vel'] = self._data.qvel[[self._gripper_opening_joint_id]].copy()
        ob_info['proprio/gripper_contact'] = np.array(
            [np.clip(np.linalg.norm(self._data.body('ur5e/robotiq/right_pad').cfrc_ext) / 50, 0, 1)]
        )

        ob_info['proprio_aux/gripper_closed'] = (ob_info['proprio/gripper_contact'] > 0.5).astype(np.float64)
        ob_info['proprio_aux/gripper_open'] = (ob_info['proprio/gripper_contact'] < 0.1).astype(np.float64)
        ob_info['proprio_aux/above'] = (ob_info['proprio/effector_pos'][[2]] > 0.16).astype(np.float64)

        # Privileged observations.
        for i in range(self._num_objects):
            ob_info[f'privileged/block_{i}_pos'] = self._data.joint(f'object_joint_{i}').qpos[:3].copy()
            ob_info[f'privileged/block_{i}_quat'] = self._data.joint(f'object_joint_{i}').qpos[3:].copy()
            ob_info[f'privileged/block_{i}_yaw'] = np.array(
                [lie.SO3(wxyz=self._data.joint(f'object_joint_{i}').qpos[3:]).compute_yaw_radians()]
            )

            effector_pos = ob_info['proprio/effector_pos']
            block_pos = ob_info[f'privileged/block_{i}_pos']
            ob_info[f'privileged_aux/block_{i}_xy_aligned'] = np.array(
                [np.linalg.norm(block_pos[:2] - effector_pos[:2]) <= 0.04]
            ).astype(np.float64)
            ob_info[f'privileged_aux/block_{i}_pos_aligned'] = np.array(
                [np.linalg.norm(block_pos - effector_pos) <= 0.02]
            ).astype(np.float64)

        if self._mode == 'data_collection':
            target_mocap_id = self._object_target_mocap_ids[self._target_block]
            ob_info['target_block'] = self._target_block
            ob_info['privileged/target_pos'] = self._data.mocap_pos[target_mocap_id].copy()
            ob_info['privileged/target_yaw'] = np.array(
                [lie.SO3(wxyz=self._data.mocap_quat[target_mocap_id]).compute_yaw_radians()]
            )

        if self._pixel_observation:
            for cam_name in ['front', 'overhead_left', 'overhead_right', 'topdown']:
                ob_info[f'rgb/{cam_name}'] = self.render(camera=cam_name)
                ob_info[f'depth/{cam_name}'] = self.render(camera=cam_name, depth=True)

        ob_info['control'] = self._data.ctrl.copy()
        ob_info['contact'] = self._data.cfrc_ext.copy()
        ob_info['time'] = np.array([self._data.time])

        return ob_info

    def compute_observation(self) -> Any:
        xyz_center = np.array([0.425, 0.0, 0.0])
        xyz_scaler = 10
        gripper_scaler = 3

        ob_info = self.compute_ob_info()
        obs = [
            ob_info['proprio/joint_pos'],
            ob_info['proprio/joint_vel'],
            (ob_info['proprio/effector_pos'] - xyz_center) * xyz_scaler,
            np.cos(ob_info['proprio/effector_yaw']),
            np.sin(ob_info['proprio/effector_yaw']),
            ob_info['proprio/gripper_opening'] * gripper_scaler,
            ob_info['proprio/gripper_contact'],
            ob_info['proprio_aux/gripper_closed'],
            ob_info['proprio_aux/gripper_open'],
            ob_info['proprio_aux/above'],
        ]
        for i in range(self._num_objects):
            obs.extend(
                [
                    (ob_info[f'privileged/block_{i}_pos'] - xyz_center) * xyz_scaler,
                    ob_info[f'privileged/block_{i}_quat'],
                    np.cos(ob_info[f'privileged/block_{i}_yaw']),
                    np.sin(ob_info[f'privileged/block_{i}_yaw']),
                    ob_info[f'privileged_aux/block_{i}_xy_aligned'],
                    ob_info[f'privileged_aux/block_{i}_pos_aligned'],
                ]
            )

        return np.concatenate(obs)

    def compute_reward(self, obs: dict[str, np.ndarray], action: np.ndarray) -> float:
        return 0.0

    def get_reset_info(self) -> dict:
        reset_info = self.compute_ob_info()
        if self._mode == 'evaluation':
            reset_info['goal'] = self._cur_goal_ob
        return reset_info

    def get_step_info(self) -> dict:
        ob_info = self.compute_ob_info()
        ob_info['success'] = self._success
        return ob_info

    def terminate_episode(self) -> bool:
        if self._terminate_at_goal:
            return self._success
        else:
            return False
