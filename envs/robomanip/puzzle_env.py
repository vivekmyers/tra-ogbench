import mujoco
import numpy as np
from dm_control import mjcf

from envs.robomanip.robomanip import _COLORS, _HERE, _HOME_QPOS, RoboManipEnv


class PuzzleEnv(RoboManipEnv):
    def __init__(self, env_type, *args, **kwargs):
        self._env_type = env_type

        self._num_button_states = 2
        self._effect_type = 'plus'

        if '3x3' in env_type:
            self._num_rows = 3
            self._num_cols = 3
        elif '4x4' in env_type:
            self._num_rows = 4
            self._num_cols = 4
        elif '4x6' in env_type:
            self._num_rows = 4
            self._num_cols = 6
        else:
            raise ValueError(f'Unknown env_type: {env_type}')

        self._num_buttons = self._num_rows * self._num_cols

        self._cur_button_states = np.array([0] * self._num_buttons)

        super().__init__(*args, **kwargs)

        self._arm_sampling_bounds = np.asarray([[0.25, -0.2, 0.20], [0.6, 0.2, 0.25]])

        self._target_task = 'button'
        self._target_button = 0
        self._target_button_states = np.array([0] * self._num_buttons)

    def set_state(self, qpos, qvel, button_states):
        self._cur_button_states = button_states
        self._apply_button_states()
        super().set_state(qpos, qvel)

    def set_tasks(self):
        if self._num_rows == 3 and self._num_cols == 3:
            self.task_infos = [
                dict(
                    task_name='task1',
                    init_button_states=np.array(
                        [
                            [0, 0, 0],
                            [0, 0, 0],
                            [0, 0, 0],
                        ]
                    ).flatten(),
                    goal_button_states=np.array(
                        [
                            [1, 1, 0],
                            [1, 0, 1],
                            [0, 1, 1],
                        ]
                    ).flatten(),
                ),
                dict(
                    task_name='task2',
                    init_button_states=np.array(
                        [
                            [1, 1, 1],
                            [1, 1, 1],
                            [1, 1, 1],
                        ]
                    ).flatten(),
                    goal_button_states=np.array(
                        [
                            [0, 1, 1],
                            [1, 1, 1],
                            [1, 1, 1],
                        ]
                    ).flatten(),
                ),
                dict(
                    task_name='task3',
                    init_button_states=np.array(
                        [
                            [0, 1, 0],
                            [1, 1, 1],
                            [0, 1, 0],
                        ]
                    ).flatten(),
                    goal_button_states=np.array(
                        [
                            [1, 0, 1],
                            [0, 1, 0],
                            [1, 0, 1],
                        ]
                    ).flatten(),
                ),
                dict(
                    task_name='task4',
                    init_button_states=np.array(
                        [
                            [0, 1, 0],
                            [1, 0, 1],
                            [0, 1, 0],
                        ]
                    ).flatten(),
                    goal_button_states=np.array(
                        [
                            [1, 1, 1],
                            [1, 1, 1],
                            [1, 1, 1],
                        ]
                    ).flatten(),
                ),
                dict(
                    task_name='task5',
                    init_button_states=np.array(
                        [
                            [1, 1, 1],
                            [1, 1, 1],
                            [1, 1, 1],
                        ]
                    ).flatten(),
                    goal_button_states=np.array(
                        [
                            [1, 0, 1],
                            [1, 0, 1],
                            [1, 0, 1],
                        ]
                    ).flatten(),
                ),
            ]
        elif self._num_rows == 4 and self._num_cols == 4:
            self.task_infos = [
                dict(
                    task_name='task1',
                    init_button_states=np.array(
                        [
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                        ]
                    ).flatten(),
                    goal_button_states=np.array(
                        [
                            [1, 1, 1, 1],
                            [1, 1, 1, 1],
                            [1, 1, 1, 1],
                            [1, 1, 1, 1],
                        ]
                    ).flatten(),
                ),
                dict(
                    init_button_states=np.array(
                        [
                            [1, 1, 1, 1],
                            [1, 1, 1, 1],
                            [1, 1, 1, 1],
                            [1, 1, 1, 1],
                        ]
                    ).flatten(),
                    goal_button_states=np.array(
                        [
                            [1, 1, 1, 1],
                            [1, 0, 0, 1],
                            [1, 0, 0, 1],
                            [1, 1, 1, 1],
                        ]
                    ).flatten(),
                    task_name='task2',
                ),
                dict(
                    task_name='task3',
                    init_button_states=np.array(
                        [
                            [1, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 1],
                        ]
                    ).flatten(),
                    goal_button_states=np.array(
                        [
                            [0, 1, 0, 1],
                            [1, 0, 1, 0],
                            [0, 1, 0, 1],
                            [1, 0, 1, 0],
                        ]
                    ).flatten(),
                ),
                dict(
                    task_name='task4',
                    init_button_states=np.array(
                        [
                            [1, 0, 0, 1],
                            [1, 0, 0, 1],
                            [1, 0, 0, 1],
                            [1, 0, 0, 1],
                        ]
                    ).flatten(),
                    goal_button_states=np.array(
                        [
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                        ]
                    ).flatten(),
                ),
                dict(
                    task_name='task5',
                    init_button_states=np.array(
                        [
                            [0, 1, 0, 1],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1],
                            [1, 0, 0, 0],
                        ]
                    ).flatten(),
                    goal_button_states=np.array(
                        [
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                        ]
                    ).flatten(),
                ),
            ]
        elif self._num_rows == 4 and self._num_cols == 6:
            self.task_infos = [
                dict(
                    task_name='task1',
                    init_button_states=np.array(
                        [
                            [1, 1, 0, 1, 1, 1],
                            [0, 0, 1, 0, 1, 0],
                            [0, 1, 0, 1, 0, 0],
                            [1, 1, 1, 0, 1, 1],
                        ]
                    ).flatten(),
                    goal_button_states=np.array(
                        [
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                        ]
                    ).flatten(),
                ),
                dict(
                    task_name='task2',
                    init_button_states=np.array(
                        [
                            [1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1],
                        ]
                    ).flatten(),
                    goal_button_states=np.array(
                        [
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                        ]
                    ).flatten(),
                ),
                dict(
                    task_name='task3',
                    init_button_states=np.array(
                        [
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                        ]
                    ).flatten(),
                    goal_button_states=np.array(
                        [
                            [1, 1, 1, 1, 1, 0],
                            [1, 1, 0, 1, 0, 1],
                            [1, 0, 1, 0, 1, 1],
                            [0, 1, 1, 1, 1, 1],
                        ]
                    ).flatten(),
                ),
                dict(
                    task_name='task4',
                    init_button_states=np.array(
                        [
                            [0, 1, 0, 1, 0, 1],
                            [1, 0, 1, 0, 1, 0],
                            [0, 1, 0, 1, 0, 1],
                            [1, 0, 1, 0, 1, 0],
                        ]
                    ).flatten(),
                    goal_button_states=np.array(
                        [
                            [1, 0, 0, 0, 0, 1],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [1, 0, 0, 0, 0, 1],
                        ]
                    ).flatten(),
                ),
                dict(
                    task_name='task5',
                    init_button_states=np.array(
                        [
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                        ]
                    ).flatten(),
                    goal_button_states=np.array(
                        [
                            [1, 0, 0, 0, 0, 1],
                            [0, 1, 1, 1, 1, 0],
                            [0, 1, 1, 1, 1, 0],
                            [1, 0, 0, 0, 0, 1],
                        ]
                    ).flatten(),
                ),
            ]
        else:
            raise NotImplementedError

    def add_objects(self, arena_mjcf):
        # Add objects to scene
        button_outer_mjcf = mjcf.from_path((_HERE / 'common' / 'button_outer.xml').as_posix())
        arena_mjcf.include_copy(button_outer_mjcf)

        r = 0.05
        for i in range(self._num_rows):
            for j in range(self._num_cols):
                button_mjcf = mjcf.from_path((_HERE / 'common' / 'button_inner.xml').as_posix())
                pos_x = 0.425 - r * (self._num_rows - 1) + 2 * r * i
                pos_y = 0.0 - r * (self._num_cols - 1) + 2 * r * j
                button_mjcf.find('body', 'buttonbox_0').pos[:2] = np.array([pos_x, pos_y])
                for tag in ['body', 'joint', 'geom', 'site']:
                    for item in button_mjcf.find_all(tag):
                        if hasattr(item, 'name') and item.name is not None and item.name.endswith('_0'):
                            item.name = item.name[:-2] + f'_{i * self._num_cols + j}'
                arena_mjcf.include_copy(button_mjcf)

        self._button_geoms_list = []
        for i in range(self._num_buttons):
            self._button_geoms_list.append([arena_mjcf.find('geom', f'btngeom_{i}')])

    def post_compilation_objects(self):
        self._button_geom_ids_list = [
            [self._model.geom(geom.full_identifier).id for geom in button_geoms]
            for button_geoms in self._button_geoms_list
        ]
        self._button_site_ids = [self._model.site(f'btntop_{i}').id for i in range(self._num_buttons)]

    def _apply_button_states(self):
        # Change colors
        if self._num_button_states > 2:
            raise NotImplementedError
        for i in range(self._num_buttons):
            for gid in self._button_geom_ids_list[i]:
                self._model.geom(gid).rgba = _COLORS['black' if self._cur_button_states[i] == 0 else 'white']

        mujoco.mj_forward(self._model, self._data)

    def initialize_episode(self):
        self._data.qpos[self._arm_joint_ids] = _HOME_QPOS
        mujoco.mj_kinematics(self._model, self._data)

        if self._mode == 'data_collection':
            self.initialize_arm()

            # Randomize button states
            for i in range(self._num_buttons):
                self._cur_button_states[i] = self.np_random.choice(self._num_button_states)
            self._apply_button_states()

            # Set a new target
            self.set_new_target(return_info=False)
        else:
            # Set object positions and orientations based on the current task
            init_button_states = self.cur_task_info['init_button_states'].copy()
            goal_button_states = self.cur_task_info['goal_button_states'].copy()

            # First set the current scene to the goal state to get the goal observation
            saved_qpos = self._data.qpos.copy()
            saved_qvel = self._data.qvel.copy()
            self.initialize_arm()
            self._cur_button_states = goal_button_states.copy()
            self._apply_button_states()
            mujoco.mj_forward(self._model, self._data)
            for _ in range(2):
                self.step(self.action_space.sample())
            self._cur_goal_ob = self.compute_observation()

            # Now do the actual reset
            self._data.qpos[:] = saved_qpos
            self._data.qvel[:] = saved_qvel
            self.initialize_arm()
            self._cur_button_states = init_button_states.copy()
            self._target_button_states = goal_button_states.copy()
            self._apply_button_states()

        # Forward kinematics to update site positions
        self.pre_step()
        mujoco.mj_forward(self._model, self._data)
        self.post_step()

        self._success = False

    def set_new_target(self, return_info=True, p_stack=0.5):
        assert self._mode == 'data_collection'

        self._target_button = self.np_random.choice(self._num_buttons)
        self._target_button_states[self._target_button] = (
            self._cur_button_states[self._target_button] + 1
        ) % self._num_button_states

        mujoco.mj_kinematics(self._model, self._data)

        if return_info:
            return self.compute_observation(), self.get_reset_info()

    def pre_step(self):
        self._prev_button_states = self._cur_button_states.copy()
        super().pre_step()

    def post_step(self):
        # Change button states if pressed
        for i in range(self._num_buttons):
            prev_joint_pos = self._prev_ob_info[f'privileged/button_{i}_pos'][0]
            cur_joint_pos = self._data.joint(f'buttonbox_joint_{i}').qpos.copy()[0]
            if prev_joint_pos > -0.02 and cur_joint_pos <= -0.02:
                if self._effect_type == 'point':
                    self._cur_button_states[i] = (self._cur_button_states[i] + 1) % self._num_button_states
                elif self._effect_type == 'plus':
                    x, y = i // self._num_cols, i % self._num_cols
                    for dx, dy in [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < self._num_rows and 0 <= ny < self._num_cols:
                            self._cur_button_states[nx * self._num_cols + ny] = (
                                self._cur_button_states[nx * self._num_cols + ny] + 1
                            ) % self._num_button_states
        self._apply_button_states()

        # Evaluate successes
        button_successes = [
            (self._cur_button_states[i] == self._target_button_states[i]) for i in range(self._num_buttons)
        ]

        if self._mode == 'data_collection':
            self._success = button_successes[self._target_button]
        else:
            self._success = all(button_successes)

    def add_object_info(self, ob_info):
        for i in range(self._num_buttons):
            ob_info[f'privileged/button_{i}_state'] = self._cur_button_states[i]
            ob_info[f'privileged/button_{i}_pos'] = self._data.joint(f'buttonbox_joint_{i}').qpos.copy()
            ob_info[f'privileged/button_{i}_vel'] = self._data.joint(f'buttonbox_joint_{i}').qvel.copy()

        if self._mode == 'data_collection':
            ob_info['privileged/target_task'] = self._target_task

            ob_info['privileged/target_button'] = self._target_button
            ob_info['privileged/target_button_state'] = self._target_button_states[self._target_button]
            ob_info['privileged/target_button_top_pos'] = self._data.site_xpos[
                self._button_site_ids[self._target_button]
            ].copy()

        ob_info['prev_button_states'] = self._prev_button_states.copy()
        ob_info['button_states'] = self._cur_button_states.copy()

    def compute_observation(self):
        if self._ob_type == 'pixels':
            frame = self.render(camera='front')
            return frame
        else:
            xyz_center = np.array([0.425, 0.0, 0.0])
            xyz_scaler = 10

            ob_info = self.compute_ob_info()
            ob = [
                ob_info['proprio/joint_pos'],
                ob_info['proprio/joint_vel'],
                (ob_info['proprio/effector_pos'] - xyz_center) * xyz_scaler,
                np.cos(ob_info['proprio/effector_yaw']),
                np.sin(ob_info['proprio/effector_yaw']),
                ob_info['proprio/gripper_opening'] * 3,
                ob_info['proprio/gripper_contact'],
            ]
            for i in range(self._num_buttons):
                ob.extend(
                    [
                        np.eye(self._num_button_states)[self._cur_button_states[i]],
                        ob_info[f'privileged/button_{i}_pos'] * 120,
                        ob_info[f'privileged/button_{i}_vel'],
                    ]
                )

            return np.concatenate(ob)
