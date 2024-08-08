import tempfile
import xml.etree.ElementTree as ET

import numpy as np
from gymnasium.spaces import Box

from envs.locomaze.quad import QuadEnv
from envs.locomaze.humanoid import HumanoidEnv


def make_maze_env(loco_env_type, maze_env_type, *args, **kwargs):
    if loco_env_type == 'quad':
        loco_env_class = QuadEnv
    elif loco_env_type == 'humanoid':
        loco_env_class = HumanoidEnv
    else:
        raise ValueError(f'Unknown locomotion environment type: {loco_env_type}')

    class MazeEnv(loco_env_class):
        def __init__(
            self,
            maze_type='large',
            maze_unit=4.0,
            maze_height=0.5,
            terminate_at_goal=True,
            ob_type='states',
            *args,
            **kwargs,
        ):
            self._maze_type = maze_type
            self._maze_unit = maze_unit
            self._maze_height = maze_height
            self._terminate_at_goal = terminate_at_goal
            self._ob_type = ob_type
            assert ob_type in ['states', 'pixels']

            if self._maze_type == 'arena':
                maze_map = [
                    [1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 0, 0, 0, 0, 0, 0, 1],
                    [1, 0, 0, 0, 0, 0, 0, 1],
                    [1, 0, 0, 0, 0, 0, 0, 1],
                    [1, 0, 0, 0, 0, 0, 0, 1],
                    [1, 0, 0, 0, 0, 0, 0, 1],
                    [1, 0, 0, 0, 0, 0, 0, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1],
                ]
            elif self._maze_type == 'medium':
                maze_map = [
                    [1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 0, 0, 1, 1, 0, 0, 1],
                    [1, 0, 0, 1, 0, 0, 0, 1],
                    [1, 1, 0, 0, 0, 1, 1, 1],
                    [1, 0, 0, 1, 0, 0, 0, 1],
                    [1, 0, 1, 0, 0, 1, 0, 1],
                    [1, 0, 0, 0, 1, 0, 0, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1],
                ]
            elif self._maze_type == 'large':
                maze_map = [
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
                    [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                    [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
                    [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
                    [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1],
                    [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
                    [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                ]
            elif self._maze_type == 'giant':
                maze_map = [
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
                    [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1],
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1],
                    [1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1],
                    [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1],
                    [1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1],
                    [1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1],
                    [1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1],
                    [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1],
                    [1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                ]
            else:
                raise ValueError(f'Unknown maze type: {self._maze_type}')

            self.maze_map = np.array(maze_map)

            self._offset_x = 4
            self._offset_y = 4
            self._noise = 1

            xml_file = self.xml_file
            tree = ET.parse(xml_file)
            self.update_tree(tree)
            _, maze_xml_file = tempfile.mkstemp(text=True, suffix='.xml')
            tree.write(maze_xml_file)

            super().__init__(xml_file=maze_xml_file, *args, **kwargs)

            self.task_infos = None
            self.num_tasks = None
            self.cur_task_idx = None
            self.cur_task_info = None
            self.set_tasks()
            self.cur_goal_xy = np.zeros(2)

            if self._ob_type == 'pixels':
                self.observation_space = Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)

                # Manually color the floor
                tex_grid = self.model.tex('grid')
                tex_height = tex_grid.height[0]
                tex_width = tex_grid.width[0]
                tex_rgb = self.model.tex_rgb[tex_grid.adr[0] : tex_grid.adr[0] + 3 * tex_height * tex_width]
                tex_rgb = tex_rgb.reshape(tex_height, tex_width, 3)
                for x in range(tex_height):
                    for y in range(tex_width):
                        min_value = 0
                        max_value = 192
                        r = int(x / tex_height * (max_value - min_value) + min_value)
                        g = int(y / tex_width * (max_value - min_value) + min_value)
                        tex_rgb[x, y, :] = [r, g, 128]
            else:
                ex_ob = self.get_ob()
                self.observation_space = Box(low=-np.inf, high=np.inf, shape=ex_ob.shape, dtype=ex_ob.dtype)

            # Set camera
            self.reset()
            self.render()
            self.mujoco_renderer.viewer.cam.lookat[0] = 2 * (self.maze_map.shape[1] - 3)
            self.mujoco_renderer.viewer.cam.lookat[1] = 2 * (self.maze_map.shape[0] - 3)
            self.mujoco_renderer.viewer.cam.distance = 5 * (self.maze_map.shape[1] - 2)
            self.mujoco_renderer.viewer.cam.elevation = -90

        def update_tree(self, tree):
            worldbody = tree.find('.//worldbody')

            for i in range(self.maze_map.shape[0]):
                for j in range(self.maze_map.shape[1]):
                    struct = self.maze_map[i, j]
                    if struct == 1:
                        ET.SubElement(
                            worldbody,
                            'geom',
                            name=f'block_{i}_{j}',
                            pos=f'{j * self._maze_unit - self._offset_x} {i * self._maze_unit - self._offset_y} {self._maze_height / 2 * self._maze_unit}',
                            size=f'{self._maze_unit / 2} {self._maze_unit / 2} {self._maze_height / 2 * self._maze_unit}',
                            type='box',
                            contype='1',
                            conaffinity='1',
                            material='wall',
                        )
            center_x, center_y = 2 * (self.maze_map.shape[1] - 3), 2 * (self.maze_map.shape[0] - 3)
            size_x, size_y = 2 * self.maze_map.shape[1], 2 * self.maze_map.shape[0]
            floor = tree.find('.//geom[@name="floor"]')
            floor.set('pos', f'{center_x} {center_y} 0')
            floor.set('size', f'{size_x} {size_y} 0.2')

            if self._ob_type == 'pixels':
                # Remove torso light
                torso_light = tree.find('.//light[@name="torso_light"]')
                torso_light_parent = tree.find('.//light[@name="torso_light"]/..')
                torso_light_parent.remove(torso_light)
                # Remove texture repeat
                grid = tree.find('.//material[@name="grid"]')
                grid.set('texuniform', 'false')
                if loco_env_type == 'quad':
                    # Color one leg to break symmetry
                    tree.find('.//geom[@name="aux_1_geom"]').set('material', 'self_white')
                    tree.find('.//geom[@name="left_leg_geom"]').set('material', 'self_white')
                    tree.find('.//geom[@name="left_ankle_geom"]').set('material', 'self_white')
            else:
                # Show target only when ob_type == states
                ET.SubElement(
                    worldbody,
                    'geom',
                    name='target',
                    type='cylinder',
                    size='.4 .05',
                    pos='0 0 .05',
                    material='target',
                    contype='0',
                    conaffinity='0',
                )

        def set_tasks(self):
            if self._maze_type == 'arena':
                tasks = [
                    [(1, 1), (6, 6)],
                ]
            elif self._maze_type == 'medium':
                tasks = [
                    [(1, 1), (6, 6)],
                    [(6, 1), (1, 6)],
                    [(5, 3), (4, 2)],
                    [(6, 5), (6, 1)],
                    [(2, 6), (1, 1)],
                ]
            elif self._maze_type == 'large':
                tasks = [
                    [(1, 1), (7, 10)],
                    [(5, 4), (7, 1)],
                    [(7, 4), (1, 10)],
                    [(3, 8), (5, 4)],
                    [(1, 1), (5, 4)],
                ]
            elif self._maze_type == 'giant':
                tasks = [
                    [(7, 1), (1, 14)],
                    [(2, 10), (1, 9)],
                    [(9, 1), (7, 1)],
                    [(1, 1), (10, 14)],
                    [(3, 13), (10, 10)],
                ]
            else:
                raise ValueError(f'Unknown maze type: {self._maze_type}')

            self.task_infos = []
            for i, task in enumerate(tasks):
                self.task_infos.append(
                    dict(
                        task_name=f'task{i + 1}',
                        init_ij=task[0],
                        init_xy=self.ij_to_xy(task[0]),
                        goal_ij=task[1],
                        goal_xy=self.ij_to_xy(task[1]),
                    )
                )
            self.num_tasks = len(self.task_infos)

        def reset(self, options=None, *args, **kwargs):
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

            init_xy = self.add_noise(self.ij_to_xy(self.cur_task_info['init_ij']))
            goal_xy = self.add_noise(self.ij_to_xy(self.cur_task_info['goal_ij']))

            # Get goal observation
            super().reset(*args, **kwargs)
            for _ in range(5):
                super().step(self.action_space.sample())
            self.set_xy(goal_xy)
            goal_ob = self.get_ob()

            ob, info = super().reset(*args, **kwargs)
            self.set_xy(init_xy)
            ob = self.get_ob()
            self.set_goal(goal_xy=goal_xy)
            info['goal'] = goal_ob

            return ob, info

        def step(self, action):
            ob, reward, terminated, truncated, info = super().step(action)

            if np.linalg.norm(self.get_xy() - self.cur_goal_xy) <= 0.5:
                if self._terminate_at_goal:
                    terminated = True
                info['success'] = 1.0
                reward = 1.0
            else:
                info['success'] = 0.0
                reward = 0.0

            return ob, reward, terminated, truncated, info

        def get_ob(self, ob_type=None):
            ob_type = self._ob_type if ob_type is None else ob_type
            if ob_type == 'states':
                return super().get_ob()
            else:
                frame = self.render()
                return frame

        def set_goal(self, goal_ij=None, goal_xy=None):
            if goal_xy is None:
                self.cur_goal_xy = self.add_noise(self.ij_to_xy(goal_ij))
            else:
                self.cur_goal_xy = goal_xy
            if self._ob_type == 'states':
                self.model.geom('target').pos[:2] = goal_xy

        def get_oracle_subgoal(self, start_xy, goal_xy):
            # Run BFS to find the next subgoal
            start_ij = self.xy_to_ij(start_xy)
            goal_ij = self.xy_to_ij(goal_xy)
            bfs_map = self.maze_map.copy()
            for i in range(self.maze_map.shape[0]):
                for j in range(self.maze_map.shape[1]):
                    bfs_map[i][j] = -1
            bfs_map[goal_ij[0], goal_ij[1]] = 0
            queue = [goal_ij]
            while len(queue) > 0:
                i, j = queue.pop(0)
                for di, dj in [(-1, 0), (0, -1), (1, 0), (0, 1)]:
                    ni, nj = i + di, j + dj
                    if (
                        0 <= ni < self.maze_map.shape[0]
                        and 0 <= nj < self.maze_map.shape[1]
                        and self.maze_map[ni, nj] == 0
                        and bfs_map[ni, nj] == -1
                    ):
                        bfs_map[ni][nj] = bfs_map[i][j] + 1
                        queue.append((ni, nj))
            subgoal_ij = start_ij
            for di, dj in [(-1, 0), (0, -1), (1, 0), (0, 1)]:
                ni, nj = start_ij[0] + di, start_ij[1] + dj
                if (
                    0 <= ni < self.maze_map.shape[0]
                    and 0 <= nj < self.maze_map.shape[1]
                    and self.maze_map[ni, nj] == 0
                    and bfs_map[ni, nj] < bfs_map[subgoal_ij[0], subgoal_ij[1]]
                ):
                    subgoal_ij = (ni, nj)
            subgoal_xy = self.ij_to_xy(subgoal_ij)
            return np.array(subgoal_xy), bfs_map

        def xy_to_ij(self, xy):
            maze_unit = self._maze_unit
            i = int((xy[1] + self._offset_y + 0.5 * maze_unit) / maze_unit)
            j = int((xy[0] + self._offset_x + 0.5 * maze_unit) / maze_unit)
            return i, j

        def ij_to_xy(self, ij):
            i, j = ij
            x = j * self._maze_unit - self._offset_x
            y = i * self._maze_unit - self._offset_y
            return x, y

        def add_noise(self, xy):
            random_x = np.random.uniform(low=-self._noise, high=self._noise) * self._maze_unit / 4
            random_y = np.random.uniform(low=-self._noise, high=self._noise) * self._maze_unit / 4
            return xy[0] + random_x, xy[1] + random_y

    class BallEnv(MazeEnv):
        def update_tree(self, tree):
            super().update_tree(tree)

            worldbody = tree.find('.//worldbody')
            ball = ET.SubElement(worldbody, 'body', name='ball', pos='0 0 3')
            ET.SubElement(ball, 'freejoint', name='ball_root')
            ET.SubElement(
                ball, 'geom', name='ball', size='.25', material='ball', priority='1', conaffinity='1', condim='6'
            )
            ET.SubElement(ball, 'light', name='ball_light', pos='0 0 4', mode='trackcom')

        def set_tasks(self):
            if self._maze_type == 'arena':
                tasks = [
                    [(1, 6), (2, 3), (5, 2)],
                    [(2, 2), (5, 5), (2, 2)],
                    [(6, 1), (2, 3), (6, 6)],
                    [(6, 6), (1, 1), (6, 1)],
                    [(4, 6), (6, 2), (1, 6)],
                ]
            elif self._maze_type == 'medium':
                tasks = [
                    [(1, 1), (3, 4), (6, 6)],
                    [(6, 1), (6, 5), (1, 1)],
                    [(5, 3), (4, 2), (6, 5)],
                    [(6, 5), (1, 1), (5, 3)],
                    [(1, 6), (6, 1), (1, 6)],
                ]
            else:
                raise ValueError(f'Unknown maze type: {self._maze_type}')

            self.task_infos = []
            for i, task in enumerate(tasks):
                self.task_infos.append(
                    dict(
                        task_name=f'task{i + 1}',
                        agent_init_ij=task[0],
                        agent_init_xy=self.ij_to_xy(task[0]),
                        ball_init_ij=task[1],
                        ball_init_xy=self.ij_to_xy(task[1]),
                        goal_ij=task[2],
                        goal_xy=self.ij_to_xy(task[2]),
                    )
                )
            self.num_tasks = len(self.task_infos)

        def reset(self, options=None, *args, **kwargs):
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

            agent_init_xy = self.add_noise(self.ij_to_xy(self.cur_task_info['agent_init_ij']))
            ball_init_xy = self.add_noise(self.ij_to_xy(self.cur_task_info['ball_init_ij']))
            goal_xy = self.add_noise(self.ij_to_xy(self.cur_task_info['goal_ij']))

            # Get goal observation
            super(MazeEnv, self).reset(*args, **kwargs)
            for _ in range(5):
                super(MazeEnv, self).step(self.action_space.sample())
            self.set_xy(goal_xy)
            self.set_agent_ball_xy(goal_xy, goal_xy)
            goal_ob = self.get_ob()

            ob, info = super(MazeEnv, self).reset(*args, **kwargs)
            self.set_agent_ball_xy(agent_init_xy, ball_init_xy)
            ob = self.get_ob()
            self.set_goal(goal_xy=goal_xy)
            info['goal'] = goal_ob

            return ob, info

        def step(self, action):
            ob, reward, terminated, truncated, info = super(MazeEnv, self).step(action)

            if np.linalg.norm(self.get_agent_ball_xy()[1] - self.cur_goal_xy) <= 0.5:
                if self._terminate_at_goal:
                    terminated = True
                info['success'] = 1.0
                reward = 1.0
            else:
                info['success'] = 0.0
                reward = 0.0

            return ob, reward, terminated, truncated, info

        def get_agent_ball_xy(self):
            agent_xy = self.data.qpos[:2].copy()
            ball_xy = self.data.qpos[-7:-5].copy()

            return agent_xy, ball_xy

        def set_agent_ball_xy(self, agent_xy, ball_xy):
            qpos = self.data.qpos.copy()
            qvel = self.data.qvel.copy()
            qpos[:2] = agent_xy
            qpos[-7:-5] = ball_xy
            self.set_state(qpos, qvel)

    if maze_env_type == 'maze':
        return MazeEnv(*args, **kwargs)
    elif maze_env_type == 'ball':
        return BallEnv(*args, **kwargs)
    else:
        raise ValueError(f'Unknown maze environment type: {maze_env_type}')
