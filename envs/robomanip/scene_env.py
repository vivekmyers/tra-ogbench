import mujoco
import numpy as np
from dm_control import mjcf

from envs.robomanip import lie
from envs.robomanip.robomanip import RoboManipEnv, _HERE, _OBJECT_RGBAS, _HOME_QPOS


class SceneEnv(RoboManipEnv):
    def __init__(self, env_type, *args, **kwargs):
        self._env_type = env_type

        super().__init__(*args, **kwargs)

        self._arm_sampling_bounds = np.asarray([[0.25, -0.2, 0.20], [0.6, 0.2, 0.35]])
        self._object_sampling_bounds = np.asarray([[0.3, -0.07], [0.45, 0.18]])
        self._target_sampling_bounds = self._object_sampling_bounds
        self._drawer_center = np.array([0.33, -0.24, 0.066])
        self._num_cubes = 1
        self._num_buttons = 2
        self._num_button_states = 3
        self._cur_button_states = np.array([0] * self._num_buttons)

        self._target_task = 'cube'
        self._target_block = 0
        self._target_button = 0
        self._target_button_states = np.array([0] * self._num_buttons)
        self._target_drawer_pos = 0.0
        self._target_window_pos = 0.0

    def set_state(self, qpos, qvel, button_states):
        self._cur_button_states = button_states
        self._update_button_colors()
        super().set_state(qpos, qvel)

    def set_tasks(self):
        self.task_infos = [
            dict(
                task_name='task1_drawer_window',
                init=dict(
                    block_xyzs=np.array([[0.35, 0.05, 0.02]]),
                    button_states=np.array([0, 1]),
                    drawer_pos=0.0,
                    window_pos=0.0,
                ),
                goal=dict(
                    block_xyzs=np.array([[0.35, 0.05, 0.02]]),
                    button_states=np.array([0, 1]),
                    drawer_pos=-0.16,
                    window_pos=0.2,
                ),
            ),
            dict(
                task_name='task2_block_window',
                init=dict(
                    block_xyzs=np.array([[0.4, -0.05, 0.02]]),
                    button_states=np.array([1, 2]),
                    drawer_pos=0.0,
                    window_pos=0.2,
                ),
                goal=dict(
                    block_xyzs=np.array([[0.4, 0.15, 0.02]]),
                    button_states=np.array([1, 2]),
                    drawer_pos=0.0,
                    window_pos=0.0,
                ),
            ),
            dict(
                task_name='task3_block_in_drawer',
                init=dict(
                    block_xyzs=np.array([[0.35, 0.05, 0.02]]),
                    button_states=np.array([2, 0]),
                    drawer_pos=0.0,
                    window_pos=0.0,
                ),
                goal=dict(
                    block_xyzs=np.array([[0.33, -0.36, 0.066]]),
                    button_states=np.array([2, 0]),
                    drawer_pos=0.0,
                    window_pos=0.0,
                ),
            ),
            dict(
                task_name='task4_button_drawer_window',
                init=dict(
                    block_xyzs=np.array([[0.35, 0.05, 0.02]]),
                    button_states=np.array([1, 2]),
                    drawer_pos=0.0,
                    window_pos=0.2,
                ),
                goal=dict(
                    block_xyzs=np.array([[0.35, 0.05, 0.02]]),
                    button_states=np.array([0, 0]),
                    drawer_pos=-0.16,
                    window_pos=0.0,
                ),
            ),
            dict(
                task_name='task5_rearrange_all',
                init=dict(
                    block_xyzs=np.array([[0.35, 0.15, 0.02]]),
                    button_states=np.array([0, 1]),
                    drawer_pos=0.0,
                    window_pos=0.0,
                ),
                goal=dict(
                    block_xyzs=np.array([[0.4, -0.05, 0.02]]),
                    button_states=np.array([2, 0]),
                    drawer_pos=-0.16,
                    window_pos=0.2,
                ),
            ),
        ]

    def add_objects(self, arena_mjcf):
        # Add objects to scene
        cube_mjcf = mjcf.from_path((_HERE / 'common' / 'cube.xml').as_posix())
        arena_mjcf.include_copy(cube_mjcf)
        button_mjcf = mjcf.from_path((_HERE / 'common' / 'buttons.xml').as_posix())
        arena_mjcf.include_copy(button_mjcf)
        drawer_mjcf = mjcf.from_path((_HERE / 'common' / 'drawer.xml').as_posix())
        arena_mjcf.include_copy(drawer_mjcf)
        window_mjcf = mjcf.from_path((_HERE / 'common' / 'window.xml').as_posix())
        arena_mjcf.include_copy(window_mjcf)

        self._cube_geoms_list = []
        for i in range(self._num_cubes):
            self._cube_geoms_list.append(cube_mjcf.find('body', f'object_{i}').find_all('geom'))
        self._cube_target_geoms_list = []
        for i in range(self._num_cubes):
            self._cube_target_geoms_list.append(cube_mjcf.find('body', f'object_target_{i}').find_all('geom'))

        self._button_geoms_list = []
        for i in range(self._num_buttons):
            self._button_geoms_list.append([button_mjcf.find('geom', f'btngeom_{i}')])

    def post_compilation_objects(self):
        self._cube_geom_ids_list = [
            [self._model.geom(geom.full_identifier).id for geom in cube_geoms] for cube_geoms in self._cube_geoms_list
        ]
        self._cube_target_mocap_ids = [
            self._model.body(f'object_target_{i}').mocapid[0] for i in range(self._num_cubes)
        ]
        self._cube_target_geom_ids_list = [
            [self._model.geom(geom.full_identifier).id for geom in cube_target_geoms]
            for cube_target_geoms in self._cube_target_geoms_list
        ]

        self._button_geom_ids_list = [
            [self._model.geom(geom.full_identifier).id for geom in button_geoms]
            for button_geoms in self._button_geoms_list
        ]
        self._button_site_ids = [self._model.site(f'btntop_{i}').id for i in range(self._num_buttons)]

        self._drawer_site_id = self._model.site('drawer_handle_center').id
        self._drawer_target_site_id = self._model.site('drawer_handle_center_target').id

        self._window_site_id = self._model.site('window_handle_center').id
        self._window_target_site_id = self._model.site('window_handle_center_target').id

    def _update_button_colors(self):
        for i in range(self._num_buttons):
            for gid in self._button_geom_ids_list[i]:
                self._model.geom(gid).rgba = _OBJECT_RGBAS[self._cur_button_states[i] + 1]

    def initialize_episode(self):
        for i in range(self._num_cubes):
            for gid in self._cube_geom_ids_list[i]:
                self._model.geom(gid).rgba = _OBJECT_RGBAS[i]
            for gid in self._cube_target_geom_ids_list[i]:
                self._model.geom(gid).rgba[:3] = _OBJECT_RGBAS[i, :3]

        self._data.qpos[self._arm_joint_ids] = _HOME_QPOS
        mujoco.mj_kinematics(self._model, self._data)

        if self._mode == 'data_collection':
            self.initialize_arm()

            # Randomize block positions and orientations
            for i in range(self._num_cubes):
                xy = self.np_random.uniform(*self._object_sampling_bounds)
                obj_pos = (*xy, 0.02)
                yaw = self.np_random.uniform(0, 2 * np.pi)
                obj_ori = lie.SO3.from_z_radians(yaw).wxyz.tolist()
                self._data.joint(f'object_joint_{i}').qpos[:3] = obj_pos
                self._data.joint(f'object_joint_{i}').qpos[3:] = obj_ori

            # Randomize button states
            for i in range(self._num_buttons):
                self._cur_button_states[i] = self.np_random.choice(self._num_button_states)
            self._update_button_colors()

            # Randomize drawer and window positions
            self._data.joint('drawer_slide').qpos[0] = self.np_random.uniform(-0.16, 0)
            self._data.joint('window_slide').qpos[0] = self.np_random.uniform(0, 0.2)

            # Set a new target
            self.set_new_target(return_info=False)
        else:
            # Set object positions and orientations based on the current task
            block_permutation = self.np_random.permutation(self._num_cubes)
            init_block_xyzs = self.cur_task_info['init']['block_xyzs'].copy()[block_permutation]
            goal_block_xyzs = self.cur_task_info['goal']['block_xyzs'].copy()[block_permutation]
            init_button_states = self.cur_task_info['init']['button_states'].copy()
            goal_button_states = self.cur_task_info['goal']['button_states'].copy()
            init_drawer_pos = self.cur_task_info['init']['drawer_pos']
            goal_drawer_pos = self.cur_task_info['goal']['drawer_pos']
            init_window_pos = self.cur_task_info['init']['window_pos']
            goal_window_pos = self.cur_task_info['goal']['window_pos']

            # First set the current scene to the goal state to get the goal observation
            saved_qpos = self._data.qpos.copy()
            saved_qvel = self._data.qvel.copy()
            self.initialize_arm()
            for i in range(self._num_cubes):
                self._data.joint(f'object_joint_{i}').qpos[:3] = goal_block_xyzs[i]
                self._data.joint(f'object_joint_{i}').qpos[3:] = lie.SO3.identity().wxyz.tolist()
            self._cur_button_states = goal_button_states.copy()
            self._update_button_colors()
            self._data.joint('drawer_slide').qpos[0] = goal_drawer_pos
            self._data.joint('window_slide').qpos[0] = goal_window_pos
            mujoco.mj_forward(self._model, self._data)
            for _ in range(2):
                self.step(self.action_space.sample())
            self._cur_goal_ob = self.compute_observation()

            # Now do the actual reset
            self._data.qpos[:] = saved_qpos
            self._data.qvel[:] = saved_qvel
            self.initialize_arm()
            for i in range(self._num_cubes):
                obj_pos = init_block_xyzs[i].copy()
                obj_pos[:2] += self.np_random.uniform(-0.01, 0.01, size=2)
                self._data.joint(f'object_joint_{i}').qpos[:3] = obj_pos
                yaw = self.np_random.uniform(0, 2 * np.pi)
                obj_ori = lie.SO3.from_z_radians(yaw).wxyz.tolist()
                self._data.joint(f'object_joint_{i}').qpos[3:] = obj_ori
                self._data.mocap_pos[self._cube_target_mocap_ids[i]] = goal_block_xyzs[i]
                self._data.mocap_quat[self._cube_target_mocap_ids[i]] = lie.SO3.identity().wxyz.tolist()
            self._cur_button_states = init_button_states.copy()
            self._target_button_states = goal_button_states.copy()
            self._update_button_colors()
            self._data.joint('drawer_slide').qpos[0] = init_drawer_pos + self.np_random.uniform(-0.005, 0.005)
            self._model.site('drawer_handle_center_target').pos[1] = goal_drawer_pos
            self._target_drawer_pos = goal_drawer_pos
            self._data.joint('window_slide').qpos[0] = init_window_pos + self.np_random.uniform(-0.005, 0.005)
            self._model.site('window_handle_center_target').pos[0] = goal_window_pos
            self._target_window_pos = goal_window_pos

        # Forward kinematics to update site positions
        self.pre_step()
        mujoco.mj_forward(self._model, self._data)
        self.post_step()

        self._success = False

    def set_new_target(self, return_info=True, p_drawer=0.1, p_stack=0.5):
        assert self._mode == 'data_collection'

        # Only consider blocks not in the drawer
        available_blocks = []
        for i in range(self._num_cubes):
            if self._data.joint(f'object_joint_{self._target_block}').qpos[1] > -0.16:
                available_blocks.append(i)

        if len(available_blocks) == 0:
            self._target_task = self.np_random.choice(['button', 'drawer', 'window'])
        else:
            self._target_task = self.np_random.choice(['cube', 'button', 'drawer', 'window'])

        if self._target_task == 'cube':
            block_xyzs = np.array([self._data.joint(f'object_joint_{i}').qpos[:3] for i in range(self._num_cubes)])
            top_blocks = []
            for i in range(self._num_cubes):
                if i not in available_blocks:
                    continue
                for j in range(self._num_cubes):
                    if i == j:
                        continue
                    if (
                        block_xyzs[j][2] > block_xyzs[i][2]
                        and np.linalg.norm(block_xyzs[i][:2] - block_xyzs[j][:2]) < 0.02
                    ):
                        break
                else:
                    top_blocks.append(i)

            # Pick one of the top cubes as the target
            self._target_block = self.np_random.choice(top_blocks)

            put_in_drawer = (
                self._data.joint('drawer_slide').qpos[0] < -0.12
                and self.np_random.uniform() < p_drawer
            )
            stack = len(top_blocks) >= 2 and self.np_random.uniform() < p_stack
            if put_in_drawer:
                tar_pos = self._drawer_center.copy()
                tar_pos[:2] = tar_pos[:2] + self.np_random.uniform(-0.005, 0.005, size=2)
            elif stack:
                # Stack the target block on top of another block
                block_idx = self.np_random.choice(list(set(top_blocks) - {self._target_block}))
                block_pos = self._data.joint(f'object_joint_{block_idx}').qpos[:3]
                tar_pos = np.array([block_pos[0], block_pos[1], block_pos[2] + 0.04])
            else:
                # Randomize target position and orientation
                xy = self.np_random.uniform(*self._target_sampling_bounds)
                tar_pos = (*xy, 0.02)
            yaw = self.np_random.uniform(0, 2 * np.pi)
            tar_ori = lie.SO3.from_z_radians(yaw).wxyz.tolist()
            for i in range(self._num_cubes):
                if i == self._target_block:
                    self._data.mocap_pos[self._cube_target_mocap_ids[i]] = tar_pos
                    self._data.mocap_quat[self._cube_target_mocap_ids[i]] = tar_ori
                else:
                    self._data.mocap_pos[self._cube_target_mocap_ids[i]] = (0, 0, -0.3)
                    self._data.mocap_quat[self._cube_target_mocap_ids[i]] = lie.SO3.identity().wxyz.tolist()
            for i in range(self._num_cubes):
                if self._visualize_info and i == self._target_block:
                    for gid in self._cube_target_geom_ids_list[i]:
                        self._model.geom(gid).rgba[3] = 0.2
                else:
                    for gid in self._cube_target_geom_ids_list[i]:
                        self._model.geom(gid).rgba[3] = 0.0
        elif self._target_task == 'button':
            self._target_button = self.np_random.choice(self._num_buttons)
            self._target_button_states[self._target_button] = (
                self._cur_button_states[self._target_button] + 1
            ) % self._num_button_states
        elif self._target_task == 'drawer':
            if self._data.joint('drawer_slide').qpos[0] >= -0.08:  # Drawer closed
                self._target_drawer_pos = -0.16
            else:  # Drawer open
                self._target_drawer_pos = 0.0
            self._model.site('drawer_handle_center_target').pos[1] = self._target_drawer_pos
        elif self._target_task == 'window':
            if self._data.joint('window_slide').qpos[0] <= 0.1:  # Window closed
                self._target_window_pos = 0.2
            else:  # Window open
                self._target_window_pos = 0.0
            self._model.site('window_handle_center_target').pos[0] = self._target_window_pos

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
                self._cur_button_states[i] = (self._cur_button_states[i] + 1) % self._num_button_states
        self._update_button_colors()

        # Evaluate successes
        cube_successes = []
        for i in range(self._num_cubes):
            obj_pos = self._data.joint(f'object_joint_{i}').qpos[:3]
            tar_pos = self._data.mocap_pos[self._cube_target_mocap_ids[i]]
            if np.linalg.norm(obj_pos - tar_pos) <= 0.04:
                cube_successes.append(True)
            else:
                cube_successes.append(False)
        button_successes = [
            (self._cur_button_states[i] == self._target_button_states[i]) for i in range(self._num_buttons)
        ]
        drawer_success = np.abs(self._data.joint('drawer_slide').qpos[0] - self._target_drawer_pos) <= 0.04
        window_success = np.abs(self._data.joint('window_slide').qpos[0] - self._target_window_pos) <= 0.04

        if self._mode == 'data_collection':
            self._success = {
                'cube': cube_successes[self._target_block],
                'button': button_successes[self._target_button],
                'drawer': drawer_success,
                'window': window_success,
            }[self._target_task]
        else:
            self._success = all(cube_successes) and all(button_successes) and drawer_success and window_success

        for i in range(self._num_cubes):
            if self._visualize_info and (self._mode == 'evaluation' or i == self._target_block):
                for gid in self._cube_target_geom_ids_list[i]:
                    self._model.geom(gid).rgba[3] = 0.2
            else:
                for gid in self._cube_target_geom_ids_list[i]:
                    self._model.geom(gid).rgba[3] = 0.0

            if self._visualize_info and cube_successes[i]:
                for gid in self._cube_geom_ids_list[i]:
                    self._model.geom(gid).rgba[:3] = (0, 1, 1)
            else:
                for gid in self._cube_geom_ids_list[i]:
                    self._model.geom(gid).rgba[:3] = _OBJECT_RGBAS[i, :3]

    def add_object_info(self, ob_info):
        for i in range(self._num_cubes):
            ob_info[f'privileged/block_{i}_pos'] = self._data.joint(f'object_joint_{i}').qpos[:3].copy()
            ob_info[f'privileged/block_{i}_quat'] = self._data.joint(f'object_joint_{i}').qpos[3:].copy()
            ob_info[f'privileged/block_{i}_yaw'] = np.array(
                [lie.SO3(wxyz=self._data.joint(f'object_joint_{i}').qpos[3:]).compute_yaw_radians()]
            )

        for i in range(self._num_buttons):
            ob_info[f'privileged/button_{i}_state'] = self._cur_button_states[i]
            ob_info[f'privileged/button_{i}_pos'] = self._data.joint(f'buttonbox_joint_{i}').qpos.copy()
            ob_info[f'privileged/button_{i}_vel'] = self._data.joint(f'buttonbox_joint_{i}').qvel.copy()

        ob_info['privileged/drawer_pos'] = self._data.joint('drawer_slide').qpos.copy()
        ob_info['privileged/drawer_vel'] = self._data.joint('drawer_slide').qvel.copy()
        ob_info['privileged/drawer_handle_pos'] = self._data.site_xpos[self._drawer_site_id].copy()
        ob_info['privileged/drawer_handle_yaw'] = np.array(
            [lie.SO3.from_matrix(self._data.site_xmat[self._drawer_site_id].reshape(3, 3)).compute_yaw_radians()]
        )

        ob_info['privileged/window_pos'] = self._data.joint('window_slide').qpos.copy()
        ob_info['privileged/window_vel'] = self._data.joint('window_slide').qvel.copy()
        ob_info['privileged/window_handle_pos'] = self._data.site_xpos[self._window_site_id].copy()
        ob_info['privileged/window_handle_yaw'] = np.array(
            [lie.SO3.from_matrix(self._data.site_xmat[self._window_site_id].reshape(3, 3)).compute_yaw_radians()]
        )

        if self._mode == 'data_collection':
            ob_info['privileged/target_task'] = self._target_task

            target_mocap_id = self._cube_target_mocap_ids[self._target_block]
            ob_info['privileged/target_block'] = self._target_block
            ob_info['privileged/target_block_pos'] = self._data.mocap_pos[target_mocap_id].copy()
            ob_info['privileged/target_block_yaw'] = np.array(
                [lie.SO3(wxyz=self._data.mocap_quat[target_mocap_id]).compute_yaw_radians()]
            )

            ob_info['privileged/target_button'] = self._target_button
            ob_info['privileged/target_button_state'] = self._target_button_states[self._target_button]
            ob_info['privileged/target_button_top_pos'] = self._data.site_xpos[
                self._button_site_ids[self._target_button]
            ].copy()

            ob_info['privileged/target_drawer_pos'] = np.array([self._target_drawer_pos])
            ob_info['privileged/target_drawer_handle_pos'] = self._data.site_xpos[self._drawer_target_site_id].copy()

            ob_info['privileged/target_window_pos'] = np.array([self._target_window_pos])
            ob_info['privileged/target_window_handle_pos'] = self._data.site_xpos[self._window_target_site_id].copy()

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
            for i in range(self._num_cubes):
                ob.extend(
                    [
                        (ob_info[f'privileged/block_{i}_pos'] - xyz_center) * xyz_scaler,
                        ob_info[f'privileged/block_{i}_quat'],
                        np.cos(ob_info[f'privileged/block_{i}_yaw']),
                        np.sin(ob_info[f'privileged/block_{i}_yaw']),
                    ]
                )
            for i in range(self._num_buttons):
                ob.extend(
                    [
                        np.eye(self._num_button_states)[self._cur_button_states[i]],
                        ob_info[f'privileged/button_{i}_pos'] * 120,
                        ob_info[f'privileged/button_{i}_vel'],
                    ]
                )
            ob.extend(
                [
                    ob_info['privileged/drawer_pos'] * 18,
                    ob_info['privileged/drawer_vel'],
                    ob_info['privileged/window_pos'] * 15,
                    ob_info['privileged/window_vel'],
                ]
            )

            return np.concatenate(ob)
