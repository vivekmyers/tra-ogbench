import tempfile
import xml.etree.ElementTree as ET

import numpy as np

from envs.locomaze.quad import QuadEnv
from envs.locomaze.humanoid import HumanoidEnv


def make_maze_env(loco_env_type, *args, **kwargs):
    if loco_env_type == 'quad':
        loco_env_class = QuadEnv
    elif loco_env_type == 'humanoid':
        loco_env_class = HumanoidEnv
    else:
        raise ValueError(f'Unknown locomotion environment type: {loco_env_type}')

    class MazeEnv(loco_env_class):
        def __init__(
                self,
                maze_type,
                maze_unit=4.0,
                maze_height=0.5,
                terminate_at_goal=True,
                *args,
                **kwargs,
        ):
            self._maze_type = maze_type
            self._maze_unit = maze_unit
            self._maze_height = maze_height
            self._terminate_at_goal = terminate_at_goal

            xml_file = self.xml_file
            tree = ET.parse(xml_file)
            worldbody = tree.find('.//worldbody')

            if self._maze_type == 'large':
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
                tasks = [
                    # [(1, 1), (7, 9)],
                    # [(1, 1), (3, 10)],
                    # [(7, 4), (5, 2)],
                    # [(6, 4), (3, 10)],

                    [(1, 1), (7, 10)],
                    [(5, 4), (7, 1)],
                    [(1, 1), (3, 10)],
                    [(7, 4), (1, 10)],
                    [(7, 1), (3, 10)],
                    [(3, 8), (5, 4)],
                    [(7, 10), (5, 1)],
                    [(3, 8), (1, 1)],
                    [(1, 1), (5, 4)],
                ]
            else:
                raise ValueError(f'Unknown maze type: {self._maze_type}')

            self.maze_map = np.array(maze_map)

            ET.SubElement(
                tree.find('.//asset'),
                'material',
                name='wall',
                rgba='0.6 0.6 0.6 1',
            )
            self._offset_x = 4
            self._offset_y = 4
            self._noise = 1
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

            _, maze_xml_file = tempfile.mkstemp(text=True, suffix='.xml')
            tree.write(maze_xml_file)

            QuadEnv.__init__(self, xml_file=maze_xml_file, *args, **kwargs)

            self.task_infos = []
            for i, task in enumerate(tasks):
                self.task_infos.append(dict(
                    task_name=f'task{i + 1}',
                    init_ij=task[0],
                    init_xy=self._ij_to_xy(task[0]),
                    goal_ij=task[1],
                    goal_xy=self._ij_to_xy(task[1]),
                ))
            self.num_tasks = len(self.task_infos)
            self.cur_task_idx = None
            self.cur_task_info = None
            self.cur_goal_xy = None

            # Set up camera
            if self._maze_type == 'large' and self.render_mode == 'rgb_array':
                self.reset()
                self.render()
                self.mujoco_renderer.viewer.cam.lookat[0] = 18
                self.mujoco_renderer.viewer.cam.lookat[1] = 12
                self.mujoco_renderer.viewer.cam.distance = 50
                self.mujoco_renderer.viewer.cam.elevation = -90

        def reset(self, options=None, *args, **kwargs):
            goal_ob, _ = super().reset(*args, **kwargs)
            ob, info = super().reset(*args, **kwargs)

            if options is not None:
                # Set either task_idx or (init_ij and goal_ij)
                task_idx = options.pop('task_idx', None)
                init_ij = options.pop('init_ij', None)
                goal_ij = options.pop('goal_ij', None)
            else:
                task_idx = None
                init_ij = None
                goal_ij = None

            if init_ij is not None and goal_ij is not None:
                init_xy = self._add_noise(self._ij_to_xy(init_ij))
                goal_xy = self._add_noise(self._ij_to_xy(goal_ij))
            else:
                if task_idx is None:
                    task_idx = np.random.randint(self.num_tasks)
                self.cur_task_idx = task_idx
                self.cur_task_info = self.task_infos[task_idx]

                init_xy = self._add_noise(self.cur_task_info['init_xy'])
                goal_xy = self._add_noise(self.cur_task_info['goal_xy'])

            self.set_xy(init_xy)
            ob = self._get_obs()
            self.cur_goal_xy = goal_xy
            goal_ob = np.concatenate([goal_xy, goal_ob[2:]])
            info['goal'] = goal_ob

            return ob, info

        def step(self, action):
            ob, reward, terminated, truncated, info = super().step(action)

            if np.linalg.norm(self.get_xy() - self.cur_goal_xy) <= 0.5:
                if self._terminate_at_goal:
                    terminated = True
                info['success'] = True
            else:
                info['success'] = False

            return ob, reward, terminated, truncated, info

        def set_goal(self, goal_ij):
            self.cur_goal_xy = self._add_noise(self._ij_to_xy(goal_ij))

        def get_oracle_subgoal(self):
            # Run BFS to find the next subgoal
            start_ij = self._xy_to_ij(self.get_xy())
            goal_ij = self._xy_to_ij(self.cur_goal_xy)
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
                    if 0 <= ni < self.maze_map.shape[0] and 0 <= nj < self.maze_map.shape[1] and self.maze_map[ni, nj] == 0 and bfs_map[ni, nj] == -1:
                        bfs_map[ni][nj] = bfs_map[i][j] + 1
                        queue.append((ni, nj))
            subgoal_ij = start_ij
            for di, dj in [(-1, 0), (0, -1), (1, 0), (0, 1)]:
                ni, nj = start_ij[0] + di, start_ij[1] + dj
                if 0 <= ni < self.maze_map.shape[0] and 0 <= nj < self.maze_map.shape[1] and self.maze_map[ni, nj] == 0 and bfs_map[ni, nj] < bfs_map[subgoal_ij[0], subgoal_ij[1]]:
                    subgoal_ij = (ni, nj)
            subgoal_xy = self._ij_to_xy(subgoal_ij)
            return np.array(subgoal_xy)

        def _xy_to_ij(self, xy):
            maze_unit = self._maze_unit
            i = int((xy[1] + self._offset_y + 0.5 * maze_unit) / maze_unit)
            j = int((xy[0] + self._offset_x + 0.5 * maze_unit) / maze_unit)
            return i, j

        def _ij_to_xy(self, ij):
            i, j = ij
            x = j * self._maze_unit - self._offset_x
            y = i * self._maze_unit - self._offset_y
            return x, y

        def _add_noise(self, xy):
            random_x = np.random.uniform(low=-self._noise, high=self._noise) * self._maze_unit / 4
            random_y = np.random.uniform(low=-self._noise, high=self._noise) * self._maze_unit / 4
            return xy[0] + random_x, xy[1] + random_y

    return MazeEnv(*args, **kwargs)
