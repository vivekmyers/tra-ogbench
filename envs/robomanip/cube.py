import mujoco
import numpy as np
from dm_control import mjcf

from envs.robomanip import lie
from envs.robomanip.robomanip import RoboManipEnv, _HERE, _OBJECT_RGBAS, _HOME_QPOS


class CubeEnv(RoboManipEnv):
    def __init__(self, env_type, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._env_type = env_type
        if self._env_type == 'cube_single':
            self._object_xml = _HERE / 'common' / 'cube_1.xml'
            self._num_objects = 1
        elif self._env_type == 'cube_double':
            self._object_xml = _HERE / 'common' / 'cube_2.xml'
            self._num_objects = 2
        elif self._env_type == 'cube_triple':
            self._object_xml = _HERE / 'common' / 'cube_3.xml'
            self._num_objects = 3
        elif self._env_type == 'cube_quadruple':
            self._object_xml = _HERE / 'common' / 'cube_4.xml'
            self._num_objects = 4

    def set_tasks(self):
        if self._env_type == 'cube_single':
            self.task_infos = [
                dict(
                    task_name='task1',
                    init_xyzs=np.array([[0.425, 0.1, 0.02]]),
                    goal_xyzs=np.array([[0.425, -0.1, 0.02]]),
                ),
                dict(
                    task_name='task2',
                    init_xyzs=np.array([[0.35, 0.0, 0.02]]),
                    goal_xyzs=np.array([[0.50, 0.0, 0.02]]),
                ),
                dict(
                    task_name='task3',
                    init_xyzs=np.array([[0.50, 0.0, 0.02]]),
                    goal_xyzs=np.array([[0.35, 0.0, 0.02]]),
                ),
                dict(
                    task_name='task4',
                    init_xyzs=np.array([[0.35, -0.2, 0.02]]),
                    goal_xyzs=np.array([[0.50, 0.2, 0.02]]),
                ),
                dict(
                    task_name='task5',
                    init_xyzs=np.array([[0.35, 0.2, 0.02]]),
                    goal_xyzs=np.array([[0.50, -0.2, 0.02]]),
                ),
            ]
        elif self._env_type == 'cube_double':
            self.task_infos = [
                dict(
                    task_name='task1',
                    init_xyzs=np.array(
                        [
                            [0.425, 0.0, 0.02],
                            [0.425, -0.1, 0.02],
                        ]
                    ),
                    goal_xyzs=np.array(
                        [
                            [0.425, 0.0, 0.02],
                            [0.425, 0.1, 0.02],
                        ]
                    ),
                ),
                dict(
                    task_name='task2',
                    init_xyzs=np.array(
                        [
                            [0.35, -0.1, 0.02],
                            [0.50, -0.1, 0.02],
                        ]
                    ),
                    goal_xyzs=np.array(
                        [
                            [0.35, 0.1, 0.02],
                            [0.50, 0.1, 0.02],
                        ]
                    ),
                ),
                dict(
                    task_name='task3',
                    init_xyzs=np.array(
                        [
                            [0.35, 0.0, 0.02],
                            [0.50, 0.0, 0.02],
                        ]
                    ),
                    goal_xyzs=np.array(
                        [
                            [0.425, -0.2, 0.02],
                            [0.425, 0.2, 0.02],
                        ]
                    ),
                ),
                dict(
                    task_name='task4',
                    init_xyzs=np.array(
                        [
                            [0.425, -0.1, 0.02],
                            [0.425, 0.1, 0.02],
                        ]
                    ),
                    goal_xyzs=np.array(
                        [
                            [0.425, 0.1, 0.02],
                            [0.425, -0.1, 0.02],
                        ]
                    ),
                ),
                dict(
                    task_name='task5',
                    init_xyzs=np.array(
                        [
                            [0.425, -0.2, 0.02],
                            [0.425, 0.2, 0.02],
                        ]
                    ),
                    goal_xyzs=np.array(
                        [
                            [0.425, 0.0, 0.02],
                            [0.425, 0.0, 0.06],
                        ]
                    ),
                ),
            ]
        elif self._env_type == 'cube_triple':
            self.task_infos = [
                dict(
                    task_name='task1',
                    init_xyzs=np.array(
                        [
                            [0.35, -0.1, 0.02],
                            [0.35, 0.1, 0.02],
                            [0.50, -0.1, 0.02],
                        ]
                    ),
                    goal_xyzs=np.array(
                        [
                            [0.35, -0.1, 0.02],
                            [0.35, 0.1, 0.02],
                            [0.50, 0.1, 0.02],
                        ]
                    ),
                ),
                dict(
                    task_name='task2',
                    init_xyzs=np.array(
                        [
                            [0.35, -0.2, 0.02],
                            [0.35, 0.0, 0.02],
                            [0.35, 0.2, 0.02],
                        ]
                    ),
                    goal_xyzs=np.array(
                        [
                            [0.50, 0.0, 0.02],
                            [0.50, 0.2, 0.02],
                            [0.50, -0.2, 0.02],
                        ]
                    ),
                ),
                dict(
                    task_name='task3',
                    init_xyzs=np.array(
                        [
                            [0.35, 0.0, 0.02],
                            [0.50, -0.1, 0.02],
                            [0.50, 0.1, 0.02],
                        ]
                    ),
                    goal_xyzs=np.array(
                        [
                            [0.50, -0.1, 0.02],
                            [0.50, 0.1, 0.02],
                            [0.35, 0.0, 0.02],
                        ]
                    ),
                ),
                dict(
                    task_name='task4',
                    init_xyzs=np.array(
                        [
                            [0.425, 0.2, 0.02],
                            [0.425, 0.2, 0.06],
                            [0.425, 0.2, 0.10],
                        ]
                    ),
                    goal_xyzs=np.array(
                        [
                            [0.35, -0.1, 0.02],
                            [0.50, -0.2, 0.02],
                            [0.50, 0.0, 0.02],
                        ]
                    ),
                ),
                dict(
                    task_name='task5',
                    init_xyzs=np.array(
                        [
                            [0.35, -0.1, 0.02],
                            [0.50, -0.2, 0.02],
                            [0.50, 0.0, 0.02],
                        ]
                    ),
                    goal_xyzs=np.array(
                        [
                            [0.425, 0.2, 0.02],
                            [0.425, 0.2, 0.06],
                            [0.425, 0.2, 0.10],
                        ]
                    ),
                ),
            ]
        else:
            raise NotImplementedError

    def add_objects(self, arena_mjcf):
        # Add object to scene
        object_mjcf = mjcf.from_path(self._object_xml.as_posix())
        arena_mjcf.include_copy(object_mjcf)

        self._object_geoms_list = []
        for i in range(self._num_objects):
            self._object_geoms_list.append(object_mjcf.find('body', f'object_{i}').find_all('geom'))

        self._object_target_geoms_list = []
        for i in range(self._num_objects):
            self._object_target_geoms_list.append(object_mjcf.find('body', f'object_target_{i}').find_all('geom'))

    def post_compilation_objects(self):
        # Object joint IDs
        self._object_geom_ids_list = [
            [self._model.geom(geom.full_identifier).id for geom in object_geoms]
            for object_geoms in self._object_geoms_list
        ]

        # Mocap IDs
        self._object_target_mocap_ids = [
            self._model.body(f'object_target_{i}').mocapid[0] for i in range(self._num_objects)
        ]
        self._object_target_geom_ids_list = [
            [self._model.geom(geom.full_identifier).id for geom in object_target_geoms]
            for object_target_geoms in self._object_target_geoms_list
        ]

    def initialize_episode(self):
        for i in range(self._num_objects):
            for gid in self._object_geom_ids_list[i]:
                self._model.geom(gid).rgba = _OBJECT_RGBAS[i]
            for gid in self._object_target_geom_ids_list[i]:
                self._model.geom(gid).rgba[:3] = _OBJECT_RGBAS[i, :3]

        self._data.qpos[self._arm_joint_ids] = _HOME_QPOS
        mujoco.mj_kinematics(self._model, self._data)

        if self._mode == 'data_collection':
            self.initialize_arm()

            # Randomize object positions and orientations
            for i in range(self._num_objects):
                xy = self.np_random.uniform(*self._object_sampling_bounds)
                obj_pos = (*xy, self._object_z)
                yaw = self.np_random.uniform(0, 2 * np.pi)
                obj_ori = lie.SO3.from_z_radians(yaw).wxyz.tolist()
                self._data.joint(f'object_joint_{i}').qpos[:3] = obj_pos
                self._data.joint(f'object_joint_{i}').qpos[3:] = obj_ori

            # Set a new target
            self.set_new_target(return_info=False)
        else:
            # Set object positions and orientations based on the current task
            init_xyzs = self.cur_task_info['init_xyzs'].copy()
            goal_xyzs = self.cur_task_info['goal_xyzs'].copy()
            permutation = self.np_random.permutation(self._num_objects)
            init_xyzs = init_xyzs[permutation]
            goal_xyzs = goal_xyzs[permutation]

            # First set the current scene to the goal state to get the goal observation
            saved_qpos = self._data.qpos.copy()
            saved_qvel = self._data.qvel.copy()
            self.initialize_arm()
            for i in range(self._num_objects):
                self._data.joint(f'object_joint_{i}').qpos[:3] = goal_xyzs[i]
                self._data.joint(f'object_joint_{i}').qpos[3:] = lie.SO3.identity().wxyz.tolist()
                self._data.mocap_pos[self._object_target_mocap_ids[i]] = goal_xyzs[i]
                self._data.mocap_quat[self._object_target_mocap_ids[i]] = lie.SO3.identity().wxyz.tolist()
            mujoco.mj_forward(self._model, self._data)
            for _ in range(2):
                self.step(self.action_space.sample())
            self._cur_goal_ob = self.compute_observation()

            # Now do the actual reset
            self._data.qpos[:] = saved_qpos
            self._data.qvel[:] = saved_qvel
            self.initialize_arm()
            for i in range(self._num_objects):
                obj_pos = init_xyzs[i].copy()
                obj_pos[:2] += self.np_random.uniform(-0.01, 0.01, size=2)
                self._data.joint(f'object_joint_{i}').qpos[:3] = obj_pos
                yaw = self.np_random.uniform(0, 2 * np.pi)
                obj_ori = lie.SO3.from_z_radians(yaw).wxyz.tolist()
                self._data.joint(f'object_joint_{i}').qpos[3:] = obj_ori
                self._data.mocap_pos[self._object_target_mocap_ids[i]] = goal_xyzs[i]
                self._data.mocap_quat[self._object_target_mocap_ids[i]] = lie.SO3.identity().wxyz.tolist()

        # Forward kinematics to update site positions
        mujoco.mj_forward(self._model, self._data)
        self.post_step()

        self._prev_qpos = self._data.qpos.copy()
        self._prev_qvel = self._data.qvel.copy()
        self._success = False

    def set_new_target(self, return_info=True, p_stack=0.5):
        assert self._mode  == 'data_collection'

        block_xyzs = np.array([self._data.joint(f'object_joint_{i}').qpos[:3] for i in range(self._num_objects)])
        top_blocks = []
        for i in range(self._num_objects):
            for j in range(self._num_objects):
                if i == j:
                    continue
                if block_xyzs[j][2] > block_xyzs[i][2] and np.linalg.norm(block_xyzs[i][:2] - block_xyzs[j][:2]) < 0.02:
                    break
            else:
                top_blocks.append(i)

        # Pick one of the top objects as the target
        self._target_block = self.np_random.choice(top_blocks)

        stack = len(top_blocks) >= 2 and self.np_random.uniform() < p_stack
        if stack:
            # Stack the target block on top of another block
            block_idx = self.np_random.choice(list(set(top_blocks) - {self._target_block}))
            block_pos = self._data.joint(f'object_joint_{block_idx}').qpos[:3]
            tar_pos = np.array([block_pos[0], block_pos[1], block_pos[2] + 2 * self._object_z])
        else:
            # Randomize target position and orientation
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

    def post_step(self):
        object_successes = []
        for i in range(self._num_objects):
            obj_pos = self._data.joint(f'object_joint_{i}').qpos[:3]
            tar_pos = self._data.mocap_pos[self._object_target_mocap_ids[i]]
            if np.linalg.norm(obj_pos - tar_pos) <= 0.04:
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

    def add_object_info(self, ob_info):
        for i in range(self._num_objects):
            ob_info[f'privileged/block_{i}_pos'] = self._data.joint(f'object_joint_{i}').qpos[:3].copy()
            ob_info[f'privileged/block_{i}_quat'] = self._data.joint(f'object_joint_{i}').qpos[3:].copy()
            ob_info[f'privileged/block_{i}_yaw'] = np.array(
                [lie.SO3(wxyz=self._data.joint(f'object_joint_{i}').qpos[3:]).compute_yaw_radians()]
            )

        if self._mode == 'data_collection':
            target_mocap_id = self._object_target_mocap_ids[self._target_block]
            ob_info['target_block'] = self._target_block
            ob_info['privileged/target_pos'] = self._data.mocap_pos[target_mocap_id].copy()
            ob_info['privileged/target_yaw'] = np.array(
                [lie.SO3(wxyz=self._data.mocap_quat[target_mocap_id]).compute_yaw_radians()]
            )

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
            for i in range(self._num_objects):
                ob.extend(
                    [
                        (ob_info[f'privileged/block_{i}_pos'] - xyz_center) * xyz_scaler,
                        ob_info[f'privileged/block_{i}_quat'],
                        np.cos(ob_info[f'privileged/block_{i}_yaw']),
                        np.sin(ob_info[f'privileged/block_{i}_yaw']),
                    ]
                )

            return np.concatenate(ob)
