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
        grid_size=4,
        brush_size=4,
        num_elems=5,
        mode='evaluation',  # 'evaluation' or 'data_collection'
    ):
        self.pw = PWSim(device=device, use_jit=use_jit)
        self.pwr = PWRenderer(device)

        self._world_size = world_size
        self._grid_size = grid_size
        self._brush_size = brush_size
        self._mode = mode
        self._num_elems = num_elems
        if num_elems == 2:
            self._elem_names = ['plant', 'stone']
        elif num_elems == 5:
            self._elem_names = ['sand', 'water', 'fire', 'plant', 'stone']
        elif num_elems == 8:
            self._elem_names = ['sand', 'water', 'fire', 'plant', 'stone', 'gas', 'wood', 'ice']
        else:
            raise NotImplementedError
        self._elems = [pw_element_names.index(elem_name) for elem_name in self._elem_names]
        self._elem_colors = self.pwr.elem_vecs_array.weight.detach().cpu().numpy()[self._elems]

        self.observation_space = Box(low=0, high=255, shape=(self._world_size, self._world_size, 6), dtype=np.uint8)
        self._xy_action_size = (world_size - brush_size) // grid_size + 1
        self.action_space = Discrete(max(len(self._elems), self._xy_action_size))

        self._world = None

        self._action_step = 0  # 0: set element id, 1: set x, 2: set y
        self._action_elem_id = None
        self._action_x = None
        self._action_y = None

        if self._mode == 'evaluation':
            self.task_infos = []
            self.cur_task_idx = None
            self.cur_task_info = None
            self.set_tasks()
            self.num_tasks = len(self.task_infos)
            self.cur_goal_world = None

    def set_tasks(self):
        def add_square(action_seq, elem_name, x, y, size):
            for i in range(0, size):
                action_seq.append((elem_name, x + i, y + size - 1))
            for i in range(size - 2, -1, -1):
                action_seq.append((elem_name, x, y + i))
            for i in range(size - 2, -1, -1):
                action_seq.append((elem_name, x + size - 1, y + i))
            for i in range(1, size - 1):
                action_seq.append((elem_name, x + i, y))

        if self._num_elems == 2:
            # Task 1
            action_seq = []
            for y in reversed(range(8)):
                for x in range(8):
                    action_seq.append(('plant', x, y))
            self.task_infos.append(dict(task_name='task1_plant', action_seq=action_seq, tol=32))

            # Task 2
            action_seq = []
            for y in reversed(range(8)):
                for x in range(8):
                    action_seq.append(('plant', x, y))
            for y in reversed(range(8)):
                for x in range(8):
                    action_seq.append(('stone', x, y))
            self.task_infos.append(dict(task_name='task2_stone', action_seq=action_seq, tol=32))

            # Task 3
            action_seq = []
            for y in reversed(range(8)):
                for x in range(8):
                    action_seq.append(('plant', x, y))
            add_square(action_seq, 'stone', 1, 1, 6)
            self.task_infos.append(dict(task_name='task3_plant_stone', action_seq=action_seq, tol=32))

            # Task 4
            action_seq = []
            for y in reversed(range(8)):
                for x in range(8):
                    action_seq.append(('plant', x, y))
            for y in reversed(range(8)):
                for x in range(8):
                    action_seq.append(('stone', x, y))
            for sx, sy in [(0, 0), (0, 5), (5, 0), (5, 5)]:
                add_square(action_seq, 'plant', sx, sy, 3)
            self.task_infos.append(dict(task_name='task4_stone_plant', action_seq=action_seq, tol=32))

            # Task 5
            action_seq = []
            for y in reversed(range(8)):
                for x in range(8):
                    action_seq.append(('plant', x, y))
            for y in reversed(range(8)):
                for x in range(8):
                    if (x + y) % 2 == 0:
                        action_seq.append(('stone', x, y))
            self.task_infos.append(dict(task_name='task5_mosaic', action_seq=action_seq, tol=32))
        elif self._num_elems == 5:
            # Task 1
            action_seq = []
            add_square(action_seq, 'plant', 1, 1, 6)
            add_square(action_seq, 'sand', 0, 0, 8)
            add_square(action_seq, 'stone', 2, 2, 4)
            add_square(action_seq, 'water', 3, 3, 2)
            self.task_infos.append(dict(task_name='task1_square', action_seq=action_seq, tol=64))

            # Task 2
            action_seq = []
            for y in reversed(range(8)):
                for x in range(8):
                    action_seq.append(('water', x, y))
            add_square(action_seq, 'plant', 0, 0, 8)
            self.task_infos.append(dict(task_name='task2_water_plant', action_seq=action_seq, tol=96))

            # Task 3
            action_seq = []
            add_square(action_seq, 'stone', 0, 0, 8)
            for _ in range(32):
                action_seq.extend(
                    [
                        ('sand', 3, 1),
                        ('sand', 4, 1),
                    ]
                )
            self.task_infos.append(dict(task_name='task3_stone_sand', action_seq=action_seq, tol=64))

            # Task 4
            action_seq = []
            for x in range(0, 8):
                action_seq.append(('plant', x, 6))
            for x in range(0, 8):
                action_seq.append(('plant', x, 7))
            for y in range(7, -1, -1):
                action_seq.append(('stone', 0, y))
            for y in range(7, -1, -1):
                action_seq.append(('stone', 7, y))
            for x in range(1, 7):
                action_seq.append(('stone', x, 3))
            add_square(action_seq, 'water', 2, 0, 3)
            add_square(action_seq, 'water', 3, 0, 3)
            for _ in range(4):
                action_seq.append(('fire', 3, 7))
                action_seq.append(('fire', 4, 7))
            self.task_infos.append(dict(task_name='task4_stone_bridge', action_seq=action_seq, tol=96))

            # Task 5
            action_seq = []
            for y in reversed(range(8)):
                for x in range(8):
                    action_seq.append(('plant', x, y))
            for y in [4, 7]:
                for x in range(8):
                    action_seq.append(('water', x, y))
            for _ in range(2):
                for x in range(8):
                    action_seq.append(('fire', x, 0))
            self.task_infos.append(dict(task_name='task5_plant_water_fire', action_seq=action_seq, tol=96))
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
        self._action_step = 0
        self._action_elem_id = None
        self._action_x = None
        self._action_y = None

        if self._mode == 'evaluation':
            # First get a goal observation
            self._mode = 'internal'
            self.reset()
            for semantic_action in self.cur_task_info['action_seq']:
                for _ in range(3):
                    self.step(self.semantic_action_to_action(*semantic_action))
            self._mode = 'evaluation'
            goal = self._get_ob()
            self.cur_goal_world = self._world[0, 0].numpy().copy()
            if render_goal:
                goal_frame = self.render()

            # Do actual reset and perform one random action
            self._mode = 'internal'
            self.reset()
            semantic_action = self.sample_semantic_action()
            for _ in range(3):
                self.step(self.semantic_action_to_action(*semantic_action))
            self._mode = 'evaluation'

        ob = self._get_ob()
        info = dict()

        if self._mode == 'evaluation':
            info['goal'] = goal
            if render_goal:
                info['goal_frame'] = goal_frame

        return ob, info

    def step(self, action):
        if self._action_step == 0:
            if action < len(self._elems):
                self._action_elem_id = action
            else:
                self._action_elem_id = np.random.randint(len(self._elems))
        elif self._action_step == 1:
            if action < self._xy_action_size:
                self._action_x = action
            else:
                self._action_x = np.random.randint(self._xy_action_size)
        else:
            if action < self._xy_action_size:
                self._action_y = action
            else:
                self._action_y = np.random.randint(self._xy_action_size)

            # Step world
            self._world = self.pw(self._world)

            np_action_world = np.zeros((1, self._world_size, self._world_size), dtype=np.uint8)
            elem = self._elems[self._action_elem_id]
            real_x = self._action_x * self._grid_size
            real_y = self._action_y * self._grid_size
            np_action_world[:, real_y : real_y + self._brush_size, real_x : real_x + self._brush_size] = elem
            world_delta = self.pw.np_to_pw(np_action_world)
            self._world = interp(
                ~self.pw.get_bool(world_delta, 'empty') & ~self.pw.get_bool(self._world, 'wall'),
                self._world,
                world_delta,
            )

            self._action_elem_id = None
            self._action_x = None
            self._action_y = None

        self._action_step = (self._action_step + 1) % 3

        ob = self._get_ob()
        reward = 0.0
        done = False
        info = dict()

        if self._mode == 'evaluation':
            cur_world = self._world[0, 0].numpy().copy()
            world_shifts = []
            for dx, dy in [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]:
                world_shifts.append(np.roll(cur_world, (dy, dx), axis=(0, 1)))
            world_shifts = np.stack(world_shifts, axis=0)
            match = (self.cur_goal_world == world_shifts).any(axis=0)
            error = ~match

            success = error.sum() < self.cur_task_info['tol']
            if success:
                done = True
                info['success'] = 1.0
                reward = 1.0
            else:
                info['success'] = 0.0
                reward = 0.0

        return ob, reward, done, done, info

    def semantic_action_to_action(self, elem_name, x, y):
        elem_id = self._elem_names.index(elem_name)
        if self._action_step == 0:
            return elem_id
        elif self._action_step == 1:
            return x
        else:
            return y

    def sample_action(self):
        if self._action_step == 0:
            return np.random.randint(len(self._elems))
        else:
            return np.random.randint(self._xy_action_size)

    def sample_semantic_action(self):
        elem_name = np.random.choice(self._elem_names)
        x = np.random.randint(self._xy_action_size)
        y = np.random.randint(self._xy_action_size)
        return elem_name, x, y

    def render(self):
        ob = self._get_ob()
        world_frame, action_frame = np.split(ob, 2, axis=2)
        return world_frame

    def _get_ob(self):
        world_frame = self.pwr.render(self._world).copy()
        action_frame = np.zeros_like(world_frame)
        if self._action_step == 1:
            color = self._elem_colors[self._action_elem_id]
            action_frame[..., :] = (color * 255.0).astype(np.uint8)
        elif self._action_step == 2:
            color = self._elem_colors[self._action_elem_id]
            real_x = self._action_x * self._grid_size
            action_frame[:, real_x : real_x + self._brush_size, :] = (color * 255.0).astype(np.uint8)
        return np.concatenate([world_frame, action_frame], axis=2)
