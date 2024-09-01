import gymnasium
import numpy as np
from gymnasium.spaces import Box, Discrete

from envs.powderworld.sim import PWSim, PWRenderer, interp, pw_element_names


class PowderworldEnv(gymnasium.Env):
    metadata = {
        'render_modes': ['rgb_array'],
        'render_fps': 15,
    }

    def __init__(
        self,
        device='cpu',
        use_jit=True,
        world_size=32,
        grid_size=8,
        num_elems=5,
        mode='evaluation',  # 'evaluation' or 'data_collection'
    ):
        self.pw = PWSim(device=device, use_jit=use_jit)
        self.pwr = PWRenderer(device)

        self._world_size = world_size
        self._grid_size = grid_size
        self._brush_size = self._world_size // self._grid_size
        self._mode = mode
        self._num_elems = num_elems
        if num_elems == 2:
            self._elem_names = ['plant', 'stone']
        elif num_elems == 5:
            self._elem_names = ['sand', 'water', 'fire', 'plant', 'stone']
        else:
            raise NotImplementedError
        self._elems = [pw_element_names.index(elem_name) for elem_name in self._elem_names]

        self.observation_space = Box(low=0, high=255, shape=(self._world_size, self._world_size, 3), dtype=np.uint8)
        self.action_space = Discrete(len(self._elems) * 8 * 8)  # TODO: Refactor

        self._world = None

        if self._mode == 'evaluation':
            self.task_infos = []
            self.cur_task_idx = None
            self.cur_task_info = None
            self.set_tasks()
            self.num_tasks = len(self.task_infos)

    def compute_goal(self, action_seq):
        assert self._mode == 'evaluation'
        self._mode = 'internal'
        self.reset()
        for action in action_seq:
            self.step(self.get_action_from_semantics(*action))
        self._mode = 'evaluation'
        return self.render()

    def set_tasks(self):
        if self._num_elems == 2:
            # Task 1
            action_seq = []
            for y in reversed(range(8)):
                for x in range(8):
                    action_seq.append(('plant', x, y))
            goal = self.compute_goal(action_seq)
            self.task_infos.append(dict(task_name='task1_plant', action_seq=action_seq, goal=goal))

            # Task 2
            action_seq = []
            for y in reversed(range(8)):
                for x in range(8):
                    action_seq.append(('plant', x, y))
            for y in reversed(range(8)):
                for x in range(8):
                    action_seq.append(('stone', x, y))
            goal = self.compute_goal(action_seq)
            self.task_infos.append(dict(task_name='task2_stone', action_seq=action_seq, goal=goal))

            # Task 3
            action_seq = []
            for y in reversed(range(8)):
                for x in range(8):
                    action_seq.append(('plant', x, y))
            for i in range(1, 6):
                action_seq.append(('stone', 1, i))
            for i in range(1, 6):
                action_seq.append(('stone', i, 6))
            for i in range(6, 1, -1):
                action_seq.append(('stone', 6, i))
            for i in range(6, 1, -1):
                action_seq.append(('stone', i, 1))
            goal = self.compute_goal(action_seq)
            self.task_infos.append(dict(task_name='task3_plant_stone', action_seq=action_seq, goal=goal))

            # Task 4
            action_seq = []
            for y in reversed(range(8)):
                for x in range(8):
                    action_seq.append(('plant', x, y))
            for y in reversed(range(8)):
                for x in range(8):
                    action_seq.append(('stone', x, y))
            for sx, sy in [(0, 0), (0, 5), (5, 0), (5, 5)]:
                for i in range(0, 2):
                    action_seq.append(('plant', sx, sy + i))
                for i in range(0, 2):
                    action_seq.append(('plant', sx + i, sy + 2))
                for i in range(2, 0, -1):
                    action_seq.append(('plant', sx + 2, sy + i))
                for i in range(2, 0, -1):
                    action_seq.append(('plant', sx + i, sy))
            goal = self.compute_goal(action_seq)
            self.task_infos.append(dict(task_name='task4_stone_plant', action_seq=action_seq, goal=goal))

            # Task 5
            action_seq = []
            for y in reversed(range(8)):
                for x in range(8):
                    action_seq.append(('plant', x, y))
            for y in reversed(range(8)):
                for x in range(8):
                    if (x + y) % 2 == 0:
                        action_seq.append(('stone', x, y))
            goal = self.compute_goal(action_seq)
            self.task_infos.append(dict(task_name='task5_mosaic', action_seq=action_seq, goal=goal))
        elif self._num_elems == 5:
            # Task 1
            action_seq = []
            for i in range(1, 6):
                action_seq.append(('plant', 1, i))
            for i in range(1, 6):
                action_seq.append(('plant', i, 6))
            for i in range(6, 1, -1):
                action_seq.append(('plant', 6, i))
            for i in range(6, 1, -1):
                action_seq.append(('plant', i, 1))
            goal = self.compute_goal(action_seq)
            self.task_infos.append(dict(task_name='task1_plant', action_seq=action_seq, goal=goal))

            # Task 2
            action_seq = []
            for y in reversed(range(8)):
                for x in range(8):
                    action_seq.append(('water', x, y))
            for _ in range(16):
                action_seq.extend([
                    ('plant', 3, 3),
                    ('plant', 3, 4),
                    ('plant', 4, 4),
                    ('plant', 4, 3),
                ])
            goal = self.compute_goal(action_seq)
            self.task_infos.append(dict(task_name='task2_water_plant', action_seq=action_seq, goal=goal))

            # Task 3
            action_seq = []
            for i in range(8):
                action_seq.append(('stone', i, 7))
            for i in range(6, -1, -1):
                action_seq.append(('stone', 0, i))
            for i in range(6, -1, -1):
                action_seq.append(('stone', 7, i))
            for i in range(1, 7):
                action_seq.append(('stone', i, 0))
            for _ in range(32):
                action_seq.extend([
                    ('sand', 3, 1),
                    ('sand', 4, 1),
                ])
            goal = self.compute_goal(action_seq)
            self.task_infos.append(dict(task_name='task3_stone_sand', action_seq=action_seq, goal=goal))

            # Task 4
            action_seq = []
            for y in reversed(range(8)):
                for x in range(8):
                    action_seq.append(('sand', x, y))
            for _ in range(4):
                for x in range(8):
                    action_seq.append(('water', x, 7))
            goal = self.compute_goal(action_seq)
            self.task_infos.append(dict(task_name='task4_sand_water', action_seq=action_seq, goal=goal))

            # Task 5
            action_seq = []
            for y in reversed(range(8)):
                for x in range(8):
                    action_seq.append(('plant', x, y))
            for _ in range(4):
                for x in range(8):
                    action_seq.append(('fire', x, 0))
            goal = self.compute_goal(action_seq)
            self.task_infos.append(dict(task_name='task5_plant_fire', action_seq=action_seq, goal=goal))
        else:
            raise NotImplementedError

    def reset(self, *, seed=None, options=None):
        if self._mode == 'evaluation':
            render_goal = False
            if options is not None:
                if 'task_idx' in options:
                    self.cur_task_idx = options['task_idx']
                    self.cur_task_info = self.task_infos[self.cur_task_idx]
                elif 'task_info' in options:
                    self.cur_task_idx = None
                    self.cur_task_info = options['task_info']
                else:
                    raise ValueError('`options` must contain either `task_idx` or `task_info`')

                if 'render_goal' in options:
                    render_goal = options['render_goal']
            else:
                # Randomly sample task
                self.cur_task_idx = np.random.randint(self.num_tasks)
                self.cur_task_info = self.task_infos[self.cur_task_idx]

        np_world = np.zeros((1, self._world_size, self._world_size), dtype=np.uint8)
        np_world[:, 0, :] = 1
        np_world[:, -1, :] = 1
        np_world[:, :, 0] = 1
        np_world[:, :, -1] = 1

        self._world = self.pw.np_to_pw(np_world).clone()

        if self._mode == 'evaluation':
            # Add random action
            self._mode = 'internal'
            self.step(self.action_space.sample())
            self._mode = 'evaluation'

        ob = self._get_ob()
        info = dict()

        if self._mode == 'evaluation':
            info['goal'] = self.cur_task_info['goal']
            if render_goal:
                info['goal_frame'] = info['goal']

        return ob, info

    def step(self, action):
        # Step world
        self._world = self.pw(self._world)

        np_action_world = np.zeros((1, self._world_size, self._world_size), dtype=np.uint8)
        elem_id, y, x = np.unravel_index(action, (len(self._elems), self._grid_size, self._grid_size))
        elem = self._elems[elem_id]
        brush_size = self._brush_size
        real_x = x * brush_size
        real_y = y * brush_size
        np_action_world[:, real_y : real_y + brush_size, real_x : real_x + brush_size] = elem
        world_delta = self.pw.np_to_pw(np_action_world)
        self._world = interp(
            ~self.pw.get_bool(world_delta, 'empty') & ~self.pw.get_bool(self._world, 'wall'), self._world, world_delta
        )

        ob = self._get_ob()
        reward = 0.0
        done = False
        info = dict()

        if self._mode == 'evaluation':
            goal = self.cur_task_info['goal']
            ob_shifts = []
            for dx, dy in [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]:
                ob_shifts.append(np.roll(ob, (dy, dx), axis=(0, 1)))
            ob_shifts = np.stack(ob_shifts, axis=0)
            match = (ob_shifts == goal).all(axis=3).any(axis=0)
            error = ~match

            success = error.sum() < 64
            if success:
                done = True
                info['success'] = 1.0
                reward = 1.0
            else:
                info['success'] = 0.0
                reward = 0.0

        return ob, reward, done, done, info

    def get_action_from_semantics(self, elem_name, x, y):
        elem_id = self._elem_names.index(elem_name)
        return np.ravel_multi_index((elem_id, y, x), (len(self._elems), self._grid_size, self._grid_size))

    def render(self):
        im = self.pwr.render(self._world).copy()
        return im

    def _get_ob(self):
        return self.render()
