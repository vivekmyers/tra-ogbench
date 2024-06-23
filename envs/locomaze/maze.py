import tempfile
import xml.etree.ElementTree as ET
from copy import deepcopy

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
                *args,
                **kwargs,
        ):
            self._maze_type = maze_type
            self._maze_unit = maze_unit
            self._maze_height = maze_height

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
                    [(1, 1), (7, 9)],
                    [(1, 1), (3, 10)],
                    [(7, 4), (5, 2)],
                    [(6, 4), (3, 10)],
                ]
            else:
                raise ValueError(f'Unknown maze type: {self._maze_type}')

            self._maze_map = maze_map

            ET.SubElement(
                tree.find('.//asset'),
                'material',
                name='wall',
                rgba='0.6 0.6 0.6 1',
            )
            self._offset_x = 4
            self._offset_y = 4
            for i in range(len(self._maze_map)):
                for j in range(len(self._maze_map[0])):
                    struct = self._maze_map[i][j]
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
            if options is not None:
                task_idx = options.pop('task_idx', None)
            else:
                task_idx = None
            goal_ob, _ = super().reset(*args, **kwargs)
            ob, info = super().reset(*args, **kwargs)

            if task_idx is None:
                task_idx = np.random.randint(self.num_tasks)
            self.cur_task_idx = task_idx
            self.cur_task_info = self.task_infos[task_idx]

            init_xy = self._add_noise(self.cur_task_info['init_xy'])
            goal_xy = self._add_noise(self.cur_task_info['goal_xy'])

            self.set_xy(init_xy)
            self.cur_goal_xy = goal_xy
            goal_ob = np.concatenate([goal_xy, goal_ob[2:]])
            info['goal'] = goal_ob

            return ob, info

        def step(self, action):
            ob, reward, terminated, truncated, info = super().step(action)

            info = dict()
            if np.linalg.norm(self.get_xy() - self.cur_goal_xy) <= 0.5:
                terminated = True
                info['success'] = True
            else:
                info['success'] = False

            return ob, reward, terminated, truncated, info

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
            random_x = np.random.uniform(low=-1, high=1) * self._maze_unit / 4
            random_y = np.random.uniform(low=-1, high=1) * self._maze_unit / 4
            return xy[0] + random_x, xy[1] + random_y

    return MazeEnv(*args, **kwargs)
