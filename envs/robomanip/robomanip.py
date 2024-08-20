from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from dm_control import mjcf
from gymnasium.spaces import Box

from envs.robomanip import controllers, env, lie, mjcf_utils

# XML files
_HERE = Path(__file__).resolve().parent
ARENA_XML = _HERE / 'common' / 'floor_wall.xml'
UR5E_XML = _HERE / 'common' / 'universal_robots_ur5e' / 'ur5e.xml'
ROBOTIQ_XML = _HERE / 'common' / 'robotiq_2f85' / '2f85.xml'

# Default joint configuration for the arm (used by the IK controller)
_HOME_QPOS = np.asarray([-np.pi / 2, -np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0])

# Box bounds (xy) for the end-effector, in meters
_WORKSPACE_BOUNDS = np.asarray([[0.25, -0.35, 0.02], [0.6, 0.35, 0.35]])

# Box bounds for sampling the initial object position, in meters
_OBJECT_SAMPLING_BOUNDS = np.asarray([[0.3, -0.3], [0.55, 0.3]])

# Box bounds for sampling the target position, in meters
_TARGET_SAMPLING_BOUNDS = np.asarray([[0.3, -0.3], [0.55, 0.3]])

# Actuator PD gains
_ACTUATOR_KP = np.asarray([4500, 4500, 4500, 2000, 2000, 500])
_ACTUATOR_KD = np.asarray([-450, -450, -450, -200, -200, -50])

_EFFECTOR_DOWN_ROTATION = lie.SO3(np.asarray([0.0, 1.0, 0.0, 0.0]))

_ROBOTIQ_CONSTANT = 255.0

_OBJECT_RGBAS = np.asarray(
    [
        [0.96, 0.26, 0.33, 1.0],
        [0.35, 0.55, 0.91, 1.0],
        [1.0, 0.69, 0.21, 1.0],
        [0.06, 0.74, 0.21, 1.0],
    ]
)
_OBJECT_THICKNESS = 0.02

_CAMERAS = {
    'front': {
        'pos': (0.905, 0.000, 0.762),
        'xyaxes': (0.000, 1.000, 0.000, -0.771, 0.000, 0.637),
    },
}


class RoboManipEnv(env.CustomMuJoCoEnv):
    def __init__(
        self,
        ob_type='states',
        physics_timestep=0.002,
        control_timestep=0.05,
        terminate_at_goal=True,
        mode='evaluation',  # 'evaluation' or 'data_collection'
        visualize_info=True,
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
        self._ob_type = ob_type
        self._terminate_at_goal = terminate_at_goal
        self._mode = mode
        self._visualize_info = visualize_info

        assert ob_type in ['states', 'pixels']

        self._object_z = _OBJECT_THICKNESS

        ik_mjcf = mjcf.from_path(UR5E_XML, escape_separators=True)
        xml_str = mjcf_utils.to_string(ik_mjcf)
        assets = mjcf_utils.get_assets(ik_mjcf)
        ik_model = mujoco.MjModel.from_xml_string(xml_str, assets)

        self._ik = controllers.DiffIKController(model=ik_model, sites=['attachment_site'])

        action_range = np.array([0.05] * 3 + [0.3, 1.0])
        self.action_low = -action_range
        self.action_high = action_range

        if self._mode == 'evaluation':
            self.task_infos = []
            self.cur_task_idx = None
            self.cur_task_info = None
            self.set_tasks()
            self.num_tasks = len(self.task_infos)

            self._cur_goal_ob = None

    @property
    def observation_space(self):
        if self._model is None:
            self.reset()

        ex_ob = self.compute_observation()

        if self._ob_type == 'pixels':
            return Box(low=0, high=255, shape=ex_ob.shape, dtype=ex_ob.dtype)
        else:
            return Box(low=-np.inf, high=np.inf, shape=ex_ob.shape, dtype=ex_ob.dtype)

    @property
    def action_space(self):
        return gym.spaces.Box(
            low=-np.ones(5),
            high=np.ones(5),
            shape=(5,),
            dtype=np.float32,
        )

    def normalize_action(self, action):
        # Normalize the action to the range [-1, 1]
        action = 2 * (action - self.action_low) / (self.action_high - self.action_low) - 1
        return np.clip(action, -1, 1)

    def unnormalize_action(self, action):
        # Unnormalize the action to the range [action_low, action_high]
        return 0.5 * (action + 1) * (self.action_high - self.action_low) + self.action_low

    def set_tasks(self):
        pass

    def build_mjcf_model(self):
        # Scene
        arena_mjcf = mjcf.from_path(ARENA_XML.as_posix())
        arena_mjcf.model = 'ur5e_arena'

        arena_mjcf.statistic.center = (0.3, 0, 0.15)
        arena_mjcf.statistic.extent = 0.7
        getattr(arena_mjcf.visual, 'global').elevation = -20
        getattr(arena_mjcf.visual, 'global').azimuth = 180
        arena_mjcf.statistic.meansize = 0.04
        arena_mjcf.visual.map.znear = 0.1
        arena_mjcf.visual.map.zfar = 10.0

        # UR5e
        ur5e_mjcf = mjcf.from_path(UR5E_XML, escape_separators=True)
        ur5e_mjcf.model = 'ur5e'

        for light in ur5e_mjcf.find_all('light'):
            light.remove()

        # Attach the robotiq gripper to the ur5e flange
        gripper_mjcf = mjcf.from_path(ROBOTIQ_XML, escape_separators=True)
        gripper_mjcf.model = 'robotiq'
        mjcf_utils.attach(ur5e_mjcf, gripper_mjcf, 'attachment_site')

        # Attach ur5e to scene
        mjcf_utils.attach(arena_mjcf, ur5e_mjcf)

        self.add_objects(arena_mjcf)

        # TODO: Move
        # drawer_mjcf = mjcf.from_path((_HERE / 'common' / 'drawer.xml').as_posix())
        # arena_mjcf.include_copy(drawer_mjcf)
        # window_mjcf = mjcf.from_path((_HERE / 'common' / 'window.xml').as_posix())
        # arena_mjcf.include_copy(window_mjcf)
        # buttonbox_mjcf = mjcf.from_path((_HERE / 'common' / 'buttonboxes.xml').as_posix())
        # arena_mjcf.include_copy(buttonbox_mjcf)

        # Add cameras
        for camera_name, camera_kwargs in _CAMERAS.items():
            arena_mjcf.worldbody.add('camera', name=camera_name, **camera_kwargs)

        # Cache joint and actuator elements
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

        if self._ob_type == 'pixels':
            # Change the colors
            arena_mjcf.find('material', 'ur5e/robotiq/black').rgba = np.array([1.0, 1.0, 1.0, 1.0])
            arena_mjcf.find('material', 'ur5e/robotiq/gray').rgba = np.array([1.0, 1.0, 1.0, 1.0])
            grid = arena_mjcf.find('texture', 'grid')
            grid.builtin = 'gradient'
            grid.mark = 'edge'

        # TODO: Remove this
        ################## FOR DEBUGGING ###################
        with open('/Users/seohongpark/Downloads/manip/manip_cur.xml', 'w') as file:
            file.write(mjcf_utils.to_string(arena_mjcf))
        ################## FOR DEBUGGING ###################

        return arena_mjcf

    def add_objects(self, arena_mjcf):
        pass

    def post_compilation(self):
        # Arm joint and actuator IDs
        arm_joint_names = [j.full_identifier for j in self._arm_jnts]
        self._arm_joint_ids = np.asarray([self._model.joint(name).id for name in arm_joint_names])
        actuator_names = [a.full_identifier for a in self._arm_acts]
        self._arm_actuator_ids = np.asarray([self._model.actuator(name).id for name in actuator_names])
        gripper_actuator_names = [a.full_identifier for a in self._gripper_acts]
        self._gripper_actuator_ids = np.asarray([self._model.actuator(name).id for name in gripper_actuator_names])
        self._gripper_opening_joint_id = self._model.joint('ur5e/robotiq/right_driver_joint').id

        # Modify PD gains
        self._model.actuator_gainprm[self._arm_actuator_ids, 0] = _ACTUATOR_KP
        self._model.actuator_gainprm[self._arm_actuator_ids, 2] = _ACTUATOR_KD
        self._model.actuator_biasprm[self._arm_actuator_ids, 1] = -_ACTUATOR_KP

        # Site IDs
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

        self.post_compilation_objects()

    def post_compilation_objects(self):
        pass

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

    def initialize_arm(self):
        # Sample initial effector position and orientation
        eff_pos = self.np_random.uniform(*self._workspace_bounds)
        cur_ori = _EFFECTOR_DOWN_ROTATION
        yaw = self.np_random.uniform(-np.pi, np.pi)
        rotz = lie.SO3.from_z_radians(yaw)
        eff_ori = rotz @ cur_ori

        # Solve for initial joint positions using IK
        T_wp = lie.SE3.from_rotation_and_translation(eff_ori, eff_pos)
        T_wa = T_wp @ self._T_pa
        qpos_init = self._ik.solve(
            pos=T_wa.translation(),
            quat=T_wa.rotation().wxyz,
            curr_qpos=_HOME_QPOS,
        )

        self._data.qpos[self._arm_joint_ids] = qpos_init
        mujoco.mj_forward(self._model, self._data)

    def initialize_episode(self):
        pass

    def set_new_target(self, return_info=True):
        pass

    def set_control(self, action):
        action = self.unnormalize_action(action)
        a_pos, a_ori, a_gripper = action[:3], action[3], action[4]

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

        # Make sure the target pose respects the action limits
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

        # Pinch pose in the world frame -> attach pose in the world frame
        self._target_effector_pose = lie.SE3.from_rotation_and_translation(
            rotation=target_effector_orientation,
            translation=target_effector_translation,
        )
        T_wa = self._target_effector_pose @ self._T_pa

        # Solve for the desired joint positions
        qpos_target = self._ik.solve(
            pos=T_wa.translation(),
            quat=T_wa.rotation().wxyz,
            curr_qpos=self._data.qpos[self._arm_joint_ids],
        )

        # Set the desired joint positions for the underlying PD controller
        self._data.ctrl[self._arm_actuator_ids] = qpos_target
        self._data.ctrl[self._gripper_actuator_ids] = _ROBOTIQ_CONSTANT * target_gripper_opening

    def pre_step(self):
        self._prev_qpos = self._data.qpos.copy()
        self._prev_qvel = self._data.qvel.copy()
        self._prev_ob_info = self.compute_ob_info()

    def compute_ob_info(self):
        ob_info = {}

        # Proprioceptive observations
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

        self.add_object_info(ob_info)

        ob_info['prev_qpos'] = self._prev_qpos
        ob_info['prev_qvel'] = self._prev_qvel
        ob_info['qpos'] = self._data.qpos.copy()
        ob_info['qvel'] = self._data.qvel.copy()
        ob_info['control'] = self._data.ctrl.copy()
        ob_info['contact'] = self._data.cfrc_ext.copy()
        ob_info['time'] = np.array([self._data.time])

        return ob_info

    def compute_observation(self):
        if self._ob_type == 'pixels':
            frame = self.render(camera='front')
            return frame
        else:
            xyz_center = np.array([0.425, 0.0, 0.0])
            xyz_scaler = 10
            gripper_scaler = 3

            ob_info = self.compute_ob_info()
            ob = [
                ob_info['proprio/joint_pos'],
                ob_info['proprio/joint_vel'],
                (ob_info['proprio/effector_pos'] - xyz_center) * xyz_scaler,
                np.cos(ob_info['proprio/effector_yaw']),
                np.sin(ob_info['proprio/effector_yaw']),
                ob_info['proprio/gripper_opening'] * gripper_scaler,
                ob_info['proprio/gripper_contact'],
                ]

            return np.concatenate(ob)

    def add_object_info(self, ob_info):
        pass

    def compute_reward(self, ob, action):
        return 0.0

    def get_reset_info(self):
        reset_info = self.compute_ob_info()
        if self._mode == 'evaluation':
            reset_info['goal'] = self._cur_goal_ob
        return reset_info

    def get_step_info(self):
        ob_info = self.compute_ob_info()
        ob_info['success'] = self._success
        return ob_info

    def terminate_episode(self):
        if self._terminate_at_goal:
            return self._success
        else:
            return False
