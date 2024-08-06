"""Pick and place task."""

from pathlib import Path
import numpy as np
import mujoco
from dm_control import mjcf
import gymnasium as gym

from robot_descriptions import ur5e_mj_description
from robot_descriptions import robotiq_2f85_mj_description

from envs.robomanip import env
from envs.robomanip import mjcf_utils
from envs.robomanip import controllers
from envs.robomanip import lie

# XML files.
_HERE = Path(__file__).resolve().parent
ARENA_XML = _HERE / "common" / "floor.xml"

# Default joint configuration for the arm (used by the IK controller).
_HOME_QPOS = np.asarray([-np.pi / 2, -np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0])

# Box bounds (xy) for the end-effector, in meters.
_WORKSPACE_BOUNDS = np.asarray([[0.25, -0.45, 0.03], [0.6, 0.45, 0.35]])

# Box bounds for sampling the initial object position, in meters.
_OBJECT_SAMPLING_BOUNDS = np.asarray([[0.3, 0.0], [0.55, 0.4]])

# Box bounds for sampling the target position, in meters.
_TARGET_SAMPLING_BOUNDS = np.asarray([[0.3, -0.4], [0.55, 0.0]])

# Actuator PD gains.
_ACTUATOR_KP = np.asarray([4500, 4500, 4500, 2000, 2000, 500])
_ACTUATOR_KD = np.asarray([-450, -450, -450, -200, -200, -50])

_EFFECTOR_DOWN_ROTATION = lie.SO3(np.asarray([0.0, 1.0, 0.0, 0.0]))

_ROBOTIQ_CONSTANT = 255.0

_OBJECT_RGBA = np.asarray([0.2, 0.2, 0.6, 1.0])
_OBJECT_XML = _HERE / "common" / "cube.xml"
_OBJECT_THICKNESS = 0.03
_OBJECT_SYMMETRY = np.pi / 2

_CAMERAS = {
    "overhead_left": {
        "pos": (-0.3, -0.263, 1),
        "xyaxes": (0, -1, 0, 0.821, 0, 0.571),
    },
    "overhead_right": {
        "pos": (-0.3, 0.263, 1),
        "xyaxes": (0, -1, 0, 0.821, 0, 0.571),
    },
    "front": {
        "pos": (1.171, 0.002, 0.753),
        "xyaxes": (-0.002, 1.000, -0.000, -0.569, -0.001, 0.822),
    },
    "topdown": {
        "pos": (0.5, 0.0, 1),
        "euler": (0.0, 0.0, 1.57),
    },
}


class RoboManipEnv(env.MujocoEnv):
    """Pick and place task.

    The agent controls the end-effector of a UR5e robot to pick up the object and place
    it at the target location. The episode is considered successful if the object is
    successfully placed in the target area.

    Observations:
        - proprioceptive: joint positions, joint velocities, end-effector position, and
            target end-effector position.
        - privileged: target object position and orientation, object position and
            orientation.
        - pixel: RGB images from the front, overhead left, overhead right, and wrist
            cameras if `pixel_observation` is set to `True`.

    Actions (5D):
        - The agent controls the desired end-effector velocity in the x-y-z plane, the
            wrist yaw angle and the gripper open/close.

    Info:
        - success: whether the episode was successful.

    Notes:
        - At the moment, no reward is implemented for this task.
    """

    def __init__(
        self,
        randomize_object_position: bool = False,
        randomize_object_orientation: bool = False,
        randomize_target_position: bool = False,
        randomize_target_orientation: bool = False,
        randomize_effector_position: bool = False,
        randomize_effector_orientation: bool = False,
        pixel_observation: bool = False,
        absolute_action_space: bool = False,
        target_effector_mocap: bool = False,
        pos_threshold: float = 5e-3,
        ori_threshold: float = np.deg2rad(5),
        **kwargs,
    ):
        """Initializes the pick and place task.

        Args:
            randomize_object_position: Whether to sample a random block position at the
                beginning of each episode.
            randomize_object_orientation: Whether to sample a random block orientation
                at the beginning of each episode.
            randomize_target_position: Whether to sample a random target position at the
                beginning of each episode.
            randomize_target_orientation: Whether to sample a random target orientation
                at the beginning of each episode.
            randomize_effector_position: Whether to sample a random effector position
                at the beginning of each episode.
            randomize_effector_orientation: Whether to sample a random effector
                orientation at the beginning of each episode.
            pixel_observation: Whether to include RGB images from the front, overhead
                left, overhead right, and wrist cameras in the observations.
            absolute_action_space: Whether the action space is absolute (i.e., the
                desired end-effector position) or relative (i.e., the desired
                end-effector velocity).
            target_effector_mocap: When True, the target effector position is set from
                the mocap body rather than the `action` input. This is useful when
                collecting demonstrations with a mouse where the user can drag the
                mocap body around to set the target position.
        """
        super().__init__(**kwargs)

        self._workspace_bounds = _WORKSPACE_BOUNDS
        self._object_sampling_bounds = _OBJECT_SAMPLING_BOUNDS
        self._target_sampling_bounds = _TARGET_SAMPLING_BOUNDS
        self._pixel_observation = pixel_observation
        self._absolute_action_space = absolute_action_space
        self._target_effector_mocap = target_effector_mocap
        self._randomize_object_position = randomize_object_position
        self._randomize_object_orientation = randomize_object_orientation
        self._randomize_target_position = randomize_target_position
        self._randomize_target_orientation = randomize_target_orientation
        self._randomize_effector_position = randomize_effector_position
        self._randomize_effector_orientation = randomize_effector_orientation
        self._pos_threshold = pos_threshold
        self._ori_treshold = ori_threshold

        self._symmetry = _OBJECT_SYMMETRY
        self._object_z = _OBJECT_THICKNESS

        ik_mjcf = mjcf.from_path(ur5e_mj_description.MJCF_PATH, escape_separators=True)
        xml_str = mjcf_utils.to_string(ik_mjcf)
        assets = mjcf_utils.get_assets(ik_mjcf)
        ik_model = mujoco.MjModel.from_xml_string(xml_str, assets)

        self._ik = controllers.DiffIKController(
            model=ik_model, sites=["attachment_site"]
        )

        # self._scene_option.sitegroup[5] = 1.0
        # self._scene_option.frame = mujoco.mjtFrame.mjFRAME_SITE

    def build_mjcf_model(self) -> mjcf.RootElement:
        # Scene.
        arena_mjcf = mjcf.from_path(ARENA_XML.as_posix())
        arena_mjcf.model = "ur5e_pick_place_task"

        arena_mjcf.statistic.center = (0.3, 0, 0.15)
        arena_mjcf.statistic.extent = 0.7
        getattr(arena_mjcf.visual, "global").elevation = -20
        getattr(arena_mjcf.visual, "global").azimuth = 180
        arena_mjcf.statistic.meansize = 0.04
        arena_mjcf.visual.map.znear = 0.1
        arena_mjcf.visual.map.zfar = 10.0

        # UR5e.
        ur5e_mjcf = mjcf.from_path(
            ur5e_mj_description.MJCF_PATH, escape_separators=True
        )
        ur5e_mjcf.model = "ur5e"

        for light in ur5e_mjcf.find_all("light"):
            light.remove()

        # Attach the robotiq gripper to the ur5e flange.
        gripper_mjcf = mjcf.from_path(
            robotiq_2f85_mj_description.MJCF_PATH, escape_separators=True
        )
        gripper_mjcf.model = "robotiq"
        mjcf_utils.attach(ur5e_mjcf, gripper_mjcf, "attachment_site")

        # Attach ur5e to scene.
        mjcf_utils.attach(arena_mjcf, ur5e_mjcf)

        # Add object to scene.
        object_mjcf = mjcf.from_path(_OBJECT_XML.as_posix())
        arena_mjcf.include_copy(object_mjcf)
        self._object_geoms = object_mjcf.find("body", "object").find_all("geom")

        if self._target_effector_mocap:
            self._target_effector_mocap_body = arena_mjcf.worldbody.add(
                "body",
                name="target_effector_mocap",
                pos=(0.5, 0, 0.5),
                mocap=True,
            )
            self._target_effector_mocap_body.add(
                "geom",
                name="target_effector_mocap",
                type="box",
                size=(0.05,) * 3,
                contype=0,
                conaffinity=0,
                rgba=(0.3, 0.3, 0.6, 0.1),
                group=1,
            )
            self._target_effector_mocap_body.add(
                "site",
                name="target_effector_mocap",
                type="sphere",
                size=(0.01,),
                rgba=(1, 0, 0, 1),
                group=5,
            )
        else:
            self._target_effector_mocap_body = None

        # Add cameras.
        for camera_name, camera_kwargs in _CAMERAS.items():
            arena_mjcf.worldbody.add("camera", name=camera_name, **camera_kwargs)

        # Cache joint and actuator elements.
        self._arm_jnts = mjcf_utils.safe_find_all(
            ur5e_mjcf,
            "joint",
            exclude_attachments=True,
        )
        self._arm_acts = mjcf_utils.safe_find_all(
            ur5e_mjcf,
            "actuator",
            exclude_attachments=True,
        )
        self._gripper_jnts = mjcf_utils.safe_find_all(
            gripper_mjcf, "joint", exclude_attachments=True
        )
        self._gripper_acts = mjcf_utils.safe_find_all(
            gripper_mjcf, "actuator", exclude_attachments=True
        )

        # ================================ #
        # Visualization.
        # ================================ #
        mjcf_utils.add_bounding_box_site(
            arena_mjcf.worldbody,
            lower=self._workspace_bounds[0],
            upper=self._workspace_bounds[1],
            rgba=(0.2, 0.2, 0.6, 0.1),
            group=4,
            name="workspace_bounds",
        )
        mjcf_utils.add_bounding_box_site(
            arena_mjcf.worldbody,
            lower=np.asarray((*self._target_sampling_bounds[0], 0.03)),
            upper=np.asarray((*self._target_sampling_bounds[1], 0.03)),
            rgba=(0.6, 0.3, 0.3, 0.2),
            group=4,
            name="object_bounds",
        )
        mjcf_utils.add_bounding_box_site(
            arena_mjcf.worldbody,
            lower=np.asarray((*self._object_sampling_bounds[0], 0.03)),
            upper=np.asarray((*self._object_sampling_bounds[1], 0.03)),
            rgba=(0.3, 0.6, 0.3, 0.2),
            group=4,
            name="target_bounds",
        )
        # ================================ #

        return arena_mjcf

    def post_compilation(self):
        # Arm joint and actuator IDs.
        arm_joint_names = [j.full_identifier for j in self._arm_jnts]
        self._arm_joint_ids = np.asarray(
            [self._model.joint(name).id for name in arm_joint_names]
        )
        actuator_names = [a.full_identifier for a in self._arm_acts]
        self._arm_actuator_ids = np.asarray(
            [self._model.actuator(name).id for name in actuator_names]
        )
        gripper_actuator_names = [a.full_identifier for a in self._gripper_acts]
        self._gripper_actuator_ids = np.asarray(
            [self._model.actuator(name).id for name in gripper_actuator_names]
        )
        self._gripper_opening_joint_id = self._model.joint(
            "ur5e/robotiq/right_driver_joint"
        ).id

        # Modify PD gains.
        self._model.actuator_gainprm[self._arm_actuator_ids, 0] = _ACTUATOR_KP
        self._model.actuator_gainprm[self._arm_actuator_ids, 2] = _ACTUATOR_KD
        self._model.actuator_biasprm[self._arm_actuator_ids, 1] = -_ACTUATOR_KP

        # Object joint id.
        self._object_joint_id = self._model.joint("object_joint").id
        self._object_geom_ids = [
            self._model.geom(geom.full_identifier).id for geom in self._object_geoms
        ]

        # Mocap IDs.
        self._object_target_mocap_id = self._model.body("object_target").mocapid[0]
        if self._target_effector_mocap_body is not None:
            self._effector_target_mocap_id = self._model.body(
                self._target_effector_mocap_body.full_identifier
            ).mocapid[0]
        else:
            self._effector_target_mocap_id = None

        # Site IDs.
        self._pinch_site_id = self._model.site("ur5e/robotiq/pinch").id
        self._attach_site_id = self._model.site("ur5e/attachment_site").id

        pinch_pose = lie.SE3.from_rotation_and_translation(
            rotation=lie.SO3.from_matrix(
                self._data.site_xmat[self._pinch_site_id].reshape(3, 3)
            ),
            translation=self._data.site_xpos[self._pinch_site_id],
        )
        attach_pose = lie.SE3.from_rotation_and_translation(
            rotation=lie.SO3.from_matrix(
                self._data.site_xmat[self._attach_site_id].reshape(3, 3)
            ),
            translation=self._data.site_xpos[self._attach_site_id],
        )
        self._T_pa = pinch_pose.inverse() @ attach_pose

    def initialize_episode(self):
        for gid in self._object_geom_ids:
            self._model.geom(gid).rgba = _OBJECT_RGBA

        self._data.qpos[self._arm_joint_ids] = _HOME_QPOS
        mujoco.mj_kinematics(self._model, self._data)

        if (
            not self._randomize_effector_position
            and not self._randomize_effector_orientation
        ):
            qpos_init = _HOME_QPOS
        else:
            # Sample initial effector position and orientation.
            cur_pos = self._data.site_xpos[self._pinch_site_id].copy()
            if self._randomize_effector_position:
                eff_pos = self.np_random.uniform(*self._workspace_bounds)
            else:
                eff_pos = cur_pos
            cur_ori = _EFFECTOR_DOWN_ROTATION
            if self._randomize_effector_orientation:
                yaw = self.np_random.uniform(-np.pi, np.pi)
                rotz = lie.SO3.from_z_radians(yaw)
                eff_ori = rotz @ cur_ori
            else:
                eff_ori = cur_ori

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

        # Randomize object position and orientation.
        obj_pos = (0.425, 0.2, self._object_z)
        obj_ori = (1.0, 0.0, 0.0, 0.0)
        if self._randomize_object_position:
            xy = self.np_random.uniform(*self._object_sampling_bounds)
            obj_pos = (*xy, self._object_z)
        if self._randomize_object_orientation:
            yaw = self.np_random.uniform(0, 2 * np.pi)
            obj_ori = lie.SO3.from_z_radians(yaw).wxyz.tolist()
        self._data.qpos[self._object_joint_id : self._object_joint_id + 3] = obj_pos
        self._data.qpos[self._object_joint_id + 3 : self._object_joint_id + 7] = obj_ori

        # Randomize target position and orientation.
        tar_pos = (0.425, -0.2, self._object_z)
        tar_ori = (1.0, 0.0, 0.0, 0.0)
        if self._randomize_target_position:
            xy = self.np_random.uniform(*self._target_sampling_bounds)
            tar_pos = (*xy, self._object_z)
        if self._randomize_target_orientation:
            yaw = self.np_random.uniform(0, 2 * np.pi)
            tar_ori = lie.SO3.from_z_radians(yaw).wxyz.tolist()
        self._data.mocap_pos[self._object_target_mocap_id] = tar_pos
        self._data.mocap_quat[self._object_target_mocap_id] = tar_ori

        # Forward kinematics to update site positions.
        mujoco.mj_kinematics(self._model, self._data)

        # Initialize the target effector mocap body at the pinch site.
        pinch_xpos = self._data.site_xpos[self._pinch_site_id]
        pinch_xmat = self._data.site_xmat[self._pinch_site_id]
        pinch_quat = lie.mat2quat(pinch_xmat)
        if self._effector_target_mocap_id is not None:
            self._data.mocap_pos[self._effector_target_mocap_id] = pinch_xpos
            self._data.mocap_quat[self._effector_target_mocap_id] = pinch_quat

        self._target_effector_pose = lie.SE3.from_rotation_and_translation(
            rotation=lie.SO3(wxyz=pinch_quat),
            translation=pinch_xpos,
        )
        self._target_gripper_opening: float = 0.0

        # Reset metrics for the current episode.
        self._success: bool = False

    @property
    def action_space(self):
        if self._absolute_action_space:
            return gym.spaces.Box(
                low=np.asarray([*self._workspace_bounds[0], -np.pi, 0.0]),
                high=np.asarray([*self._workspace_bounds[1], np.pi, 1.0]),
                dtype=np.float64,
            )
        return gym.spaces.Box(
            low=np.asarray([-0.1, -0.1, -0.1, -0.1, 0.0]),
            high=np.asarray([0.1, 0.1, 0.1, 0.1, 0.1]),
            shape=(5,),
            dtype=np.float64,
        )

    @property
    def observation_space(self):
        if self._model is None:
            self.reset()
        spaces = {}

        # Proprioceptive observations.
        spaces["proprio/joint_pos"] = gym.spaces.Box(
            low=self._model.jnt_range[self._arm_joint_ids, 0],
            high=self._model.jnt_range[self._arm_joint_ids, 1],
            dtype=np.float64,
        )
        spaces["proprio/joint_vel"] = gym.spaces.Box(
            shape=(6,), low=-np.inf, high=np.inf, dtype=np.float64
        )
        spaces["proprio/effector_pos"] = gym.spaces.Box(
            low=self._workspace_bounds[0],
            high=self._workspace_bounds[1],
            dtype=np.float64,
        )
        spaces["proprio/effector_yaw"] = gym.spaces.Box(
            low=-np.pi,
            high=np.pi,
            dtype=np.float64,
        )
        spaces["proprio/effector_target_pos"] = gym.spaces.Box(
            low=self._workspace_bounds[0],
            high=self._workspace_bounds[1],
            dtype=np.float64,
        )
        spaces["proprio/effector_target_yaw"] = gym.spaces.Box(
            low=-np.pi,
            high=np.pi,
            dtype=np.float64,
        )
        spaces["proprio/gripper_opening"] = gym.spaces.Box(
            low=0.0,
            high=1.0,
            dtype=np.float64,
        )

        # Privileged observations.
        spaces["privileged/target_pos"] = gym.spaces.Box(
            low=self._workspace_bounds[0],
            high=self._workspace_bounds[1],
            dtype=np.float64,
        )
        spaces["privileged/target_yaw"] = gym.spaces.Box(
            low=0, high=2 * np.pi, shape=(1,), dtype=np.float64
        )
        spaces["privileged/block_pos"] = gym.spaces.Box(
            low=self._workspace_bounds[0],
            high=self._workspace_bounds[1],
            dtype=np.float64,
        )
        spaces["privileged/block_yaw"] = gym.spaces.Box(
            low=0, high=2 * np.pi, shape=(1,), dtype=np.float64
        )

        if self._pixel_observation:
            keys = [
                "front",
                "overhead_left",
                "overhead_right",
                "topdown",
            ]

            rgb_render_kwargs = dict(
                shape=(self._render_height, self._render_width, 3),
                low=0,
                high=255,
                dtype=np.uint8,
            )
            for key in keys:
                spaces[f"rgb/{key}"] = gym.spaces.Box(**rgb_render_kwargs)

            depth_render_kwargs = dict(
                shape=(self._render_height, self._render_width),
                low=0.0,
                high=np.inf,
                dtype=np.float32,
            )
            for key in keys:
                spaces[f"depth/{key}"] = gym.spaces.Box(**depth_render_kwargs)

        spaces["time"] = gym.spaces.Box(
            low=0, high=np.inf, shape=(1,), dtype=np.float64
        )

        return gym.spaces.Dict(spaces)

    @property
    def target_effector_pose(self) -> lie.SE3:
        return self._target_effector_pose

    @property
    def target_gripper_opening(self) -> float:
        return self._target_gripper_opening

    def set_control(self, action):
        if self._target_effector_mocap:
            np.clip(
                self._data.mocap_pos[self._effector_target_mocap_id],
                self._workspace_bounds[0],
                self._workspace_bounds[1],
                out=self._data.mocap_pos[self._effector_target_mocap_id],
            )
            mocap_quat = self._data.mocap_quat[self._effector_target_mocap_id]
            yaw = lie.SO3(mocap_quat).compute_yaw_radians()
            yaw = np.clip(yaw, -np.pi, np.pi)
            self._data.mocap_quat[self._effector_target_mocap_id] = (
                lie.SO3.from_z_radians(yaw) @ _EFFECTOR_DOWN_ROTATION
            ).wxyz

            # When target_effector_mocap=True, we read the action command from the
            # mocap body rather than from action.
            del action
            target_effector_translation = self._data.mocap_pos[
                self._effector_target_mocap_id
            ].copy()
            target_effector_orientation = lie.SO3(
                self._data.mocap_quat[self._effector_target_mocap_id].copy()
            )

            a_gripper = None
        else:
            a_pos, a_ori, a_gripper = action[:3], action[3], action[4]

            if self._absolute_action_space:
                target_effector_translation = a_pos
                target_effector_orientation = (
                    lie.SO3.from_z_radians(a_ori) @ _EFFECTOR_DOWN_ROTATION
                )
                self._target_gripper_opening = a_gripper
            else:
                target_effector_translation = (
                    self._target_effector_pose.translation() + a_pos
                )
                target_effector_orientation = (
                    lie.SO3.from_z_radians(a_ori)
                    @ self._target_effector_pose.rotation()
                )
                self._target_gripper_opening += a_gripper

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
        target_effector_orientation = (
            lie.SO3.from_z_radians(yaw) @ _EFFECTOR_DOWN_ROTATION
        )
        self._target_gripper_opening = np.clip(self._target_gripper_opening, 0.0, 1.0)

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
        self._data.ctrl[self._gripper_actuator_ids] = (
            _ROBOTIQ_CONSTANT * self._target_gripper_opening
        )

    def post_step(self) -> None:
        obj_pos = self._data.qpos[self._object_joint_id : self._object_joint_id + 3]
        tar_pos = self._data.mocap_pos[self._object_target_mocap_id]
        err_pos = tar_pos - obj_pos
        dist_pos = np.linalg.norm(err_pos)
        pos_success = dist_pos <= self._pos_threshold

        obj_yaw = lie.SO3(
            self._data.qpos[self._object_joint_id + 3 : self._object_joint_id + 7]
        ).compute_yaw_radians()
        tar_yaw = lie.SO3(
            self._data.mocap_quat[self._object_target_mocap_id]
        ).compute_yaw_radians()
        err_rot = 0.0
        if self._symmetry > 0.0:
            err_rot = abs(obj_yaw - tar_yaw) % self._symmetry
            if err_rot > (self._symmetry / 2):
                err_rot = self._symmetry - err_rot
        ori_success = err_rot <= self._ori_treshold

        self._success = pos_success and ori_success
        if self._success:
            for gid in self._object_geom_ids:
                self._model.geom(gid).rgba[:3] = (0, 1, 0)
        else:
            for gid in self._object_geom_ids:
                self._model.geom(gid).rgba = _OBJECT_RGBA

    def compute_observation(self) -> dict[str, np.ndarray]:
        obs = {}

        # Proprioceptive observations.
        obs["proprio/joint_pos"] = self._data.qpos[self._arm_joint_ids].copy()
        obs["proprio/joint_vel"] = self._data.qvel[self._arm_joint_ids].copy()
        obs["proprio/effector_pos"] = self._data.site_xpos[self._pinch_site_id].copy()
        obs["proprio/effector_yaw"] = np.array(
            [
                lie.SO3.from_matrix(
                    self._data.site_xmat[self._pinch_site_id].copy().reshape(3, 3)
                ).compute_yaw_radians()
            ]
        )
        obs["proprio/effector_target_pos"] = (
            self._target_effector_pose.translation().copy()
        )
        obs["proprio/effector_target_yaw"] = np.array(
            [self._target_effector_pose.rotation().compute_yaw_radians()]
        )
        obs["proprio/gripper_opening"] = np.array(
            np.clip([self._data.qpos[self._gripper_opening_joint_id] / 0.8], 0, 1)
        )

        # Privileged observations.
        obs["privileged/target_pos"] = self._data.mocap_pos[
            self._object_target_mocap_id
        ].copy()
        obs["privileged/target_yaw"] = np.array(
            [
                lie.SO3(
                    wxyz=self._data.mocap_quat[self._object_target_mocap_id]
                ).compute_yaw_radians()
            ]
        )
        obs["privileged/block_pos"] = self._data.qpos[
            self._object_joint_id : self._object_joint_id + 3
        ].copy()
        obs["privileged/block_yaw"] = np.array(
            [
                lie.SO3(
                    wxyz=self._data.qpos[
                        self._object_joint_id + 3 : self._object_joint_id + 7
                    ]
                ).compute_yaw_radians()
            ]
        )

        if self._pixel_observation:
            for cam_name in ["front", "overhead_left", "overhead_right", "topdown"]:
                obs[f"rgb/{cam_name}"] = self.render(camera=cam_name)
                obs[f"depth/{cam_name}"] = self.render(camera=cam_name, depth=True)

        obs["time"] = np.array([self._data.time])

        return obs

    def compute_reward(self, obs: dict[str, np.ndarray], action: np.ndarray) -> float:
        # TODO(kevin): Implement this.
        del obs, action  # Unused.
        return 0.0

    def get_step_info(self) -> dict:
        return dict(success=self._success)

    def terminate_episode(self) -> bool:
        return self._success
