import gymnasium
import numpy as np
from crafter import Env, constants
from gymnasium.spaces import Box, Discrete


class CrafterEnv(gymnasium.Env):
    metadata = {
        'render_modes': ['rgb_array'],
        'render_fps': 15,
    }

    def __init__(
        self,
        mode='task',  # ['task', 'data_collection', 'online']
        *args,
        **kwargs,
    ):
        self.env = Env(*args, **kwargs)
        self.observation_space = Box(0, 255, tuple(self.env._size) + (3,), np.uint8)
        self.action_space = Discrete(len(constants.actions))

        self._mode = mode
        self._internal_done = False
        self._num_last_steps = 3
        self._cur_last_count = 0

        if self._mode == 'task':
            self.task_infos = []
            self.cur_task_id = None
            self.cur_task_info = None
            self.set_tasks()
            self.num_tasks = len(self.task_infos)

            self.cur_goal_items = None

    def set_tasks(self):
        self.task_infos = [
            dict(
                task_name='task1_wood',
                goal_items={'wood', 'wood_pickaxe', 'wood_sword'},
            ),
            dict(
                task_name='task2_stone',
                goal_items={'wood', 'stone', 'sapling', 'wood_pickaxe', 'wood_sword', 'stone_pickaxe', 'stone_sword'},
            ),
            dict(
                task_name='task3_stone_exact',
                goal_items={'sapling', 'wood_pickaxe', 'wood_sword', 'stone_pickaxe', 'stone_sword'},
            ),
            dict(
                task_name='task4_stone_coal_iron',
                goal_items={
                    'wood',
                    'stone',
                    'coal',
                    'iron',
                    'sapling',
                    'wood_pickaxe',
                    'wood_sword',
                    'stone_pickaxe',
                    'stone_sword',
                },
            ),
            dict(
                task_name='task5_iron',
                goal_items={
                    'wood',
                    'stone',
                    'coal',
                    'iron',
                    'sapling',
                    'wood_pickaxe',
                    'wood_sword',
                    'stone_pickaxe',
                    'stone_sword',
                    'iron_pickaxe',
                    'iron_sword',
                },
            ),
        ]

    def reset(self, *, seed=None, options=None):
        if self._mode == 'task':
            render_goal = False
            if options is not None:
                if 'task_id' in options:
                    self.cur_task_id = options['task_id']
                    self.cur_task_info = self.task_infos[self.cur_task_id - 1]
                elif 'task_info' in options:
                    self.cur_task_id = None
                    self.cur_task_info = options['task_info']
                else:
                    raise ValueError('`options` must contain either `task_id` or `task_info`')

                if 'render_goal' in options:
                    render_goal = options['render_goal']
            else:
                # Randomly sample task
                self.cur_task_id = np.random.randint(1, self.num_tasks + 1)
                self.cur_task_info = self.task_infos[self.cur_task_id - 1]

        ob = self.env.reset()
        info = dict()
        self._internal_done = False
        self._cur_last_count = 0

        if self._mode == 'task':
            goal_inventory = dict(self.env._player.inventory)
            for item, amount in goal_inventory.items():
                if item in self.cur_task_info['goal_items']:
                    goal_inventory[item] = 1
                else:
                    goal_inventory[item] = 0
            goal_ob = self.goal_render(goal_inventory)
            self.cur_goal_items = self.cur_task_info['goal_items']
            info['goal'] = goal_ob
            if render_goal:
                info['goal_rendered'] = goal_ob

        return ob, info

    def step(self, action):
        if self._mode == 'online':
            ob, reward, done, info = self.env.step(action)
            return ob, reward, done, done, info
        else:
            ob, reward, done, info = self.env.step(action)

            if done:
                self._internal_done = True

            if self._mode == 'data_collection':
                if self._internal_done:
                    ob = self.goal_render(self.env._player.inventory)
                    self._cur_last_count += 1
                    if self._cur_last_count >= self._num_last_steps:
                        done = True
                    else:
                        done = False

                    return ob, 0.0, done, done, info
                else:
                    return ob, reward, done, done, info
            elif self._mode == 'task':
                success = True
                for item, amount in self.env._player.inventory.items():
                    if item in ['health', 'food', 'drink', 'energy']:
                        continue
                    if amount > 0 and item not in self.cur_goal_items:
                        success = False
                        break
                    if amount == 0 and item in self.cur_goal_items:
                        success = False
                        break
                if success:
                    done = True
                    info['success'] = 1.0
                    reward = 1.0
                else:
                    info['success'] = 0.0
                    reward = 0.0

                return ob, reward, done, done, info

    def goal_render(self, inventory):
        size = self.env._size
        unit = size // self.env._view
        canvas = np.zeros(tuple(size) + (3,), np.uint8)
        local_view = self.env._local_view(self.env._player, unit)
        local_view = np.zeros_like(local_view)

        unit = np.array(unit)
        item_canvas = np.zeros(tuple(self.env._item_view._grid * unit) + (3,), np.uint8)
        for index, (item, amount) in enumerate(inventory.items()):
            if item in ['health', 'food', 'drink', 'energy']:
                continue
            if amount < 1:
                continue
            self.env._item_view._item(item_canvas, index, item, unit)
        item_view = item_canvas

        view = np.concatenate([local_view, item_view], 1)
        border = (size - (size // self.env._view) * self.env._view) // 2
        (x, y), (w, h) = border, view.shape[:2]
        canvas[x : x + w, y : y + h] = view

        return canvas.transpose((1, 0, 2))

    def render(self):
        return self.env.render()
