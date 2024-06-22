import numpy as np
from gymnasium import Wrapper


class AntMazeGoalWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)

        # Set camera position
        env.reset()
        self.render()
        env.viewer.cam.lookat[0] = 18
        env.viewer.cam.lookat[1] = 12
        env.viewer.cam.distance = 50
        env.viewer.cam.elevation = -90

        self.task_infos = [
            dict(
                task_name='task1',
                init_pos=np.array([0., 0.]),
                goal_pos=np.array([32., 24.]),
            ),
            dict(
                task_name='task2',
                init_pos=np.array([0., 0.]),
                goal_pos=np.array([36., 8.]),
            ),
            dict(
                task_name='task3',
                init_pos=np.array([12., 24.]),
                goal_pos=np.array([4., 16.]),
            ),
            dict(
                task_name='task4',
                init_pos=np.array([12., 20.]),
                goal_pos=np.array([36., 8.]),
            ),
        ]
        self.num_tasks = len(self.task_infos)

        self.cur_task_idx = None
        self.cur_task_info = None

    def reset(self, options=None, *args, **kwargs):
        if options is not None:
            task_idx = options.pop('task_idx', None)
        else:
            task_idx = None
        goal_ob, _ = self.env.reset(*args, **kwargs)
        ob, _ = self.env.reset(*args, **kwargs)

        if task_idx is None:
            task_idx = np.random.randint(self.num_tasks)
        self.cur_task_idx = task_idx
        self.cur_task_info = self.task_infos[task_idx]

        self.env.set_xy(self.cur_task_info['init_pos'])

        # Mimic the original goal sampling
        goal_x, goal_y = self.cur_task_info['goal_pos']
        goal_x = goal_x + np.random.uniform(low=0, high=4 * 0.25)
        goal_y = goal_y + np.random.uniform(low=0, high=4 * 0.25)
        goal_x = goal_x + np.random.uniform(low=0, high=0.5) * 0.25 * 4
        goal_y = goal_y + np.random.uniform(low=0, high=0.5) * 0.25 * 4
        goal_xy = (max(goal_x, 0), max(goal_y, 0))
        self.env.set_target_goal(goal_xy)

        goal_ob = np.concatenate([goal_xy, goal_ob[2:]])

        return ob, dict(
            goal=goal_ob,
        )

    def step(self, action):
        ob, reward, terminated, truncated, info = self.env.step(action)

        info = dict()
        if np.linalg.norm(self.get_xy() - self.target_goal) <= 0.5:
            terminated = True
            info['success'] = True
        else:
            info['success'] = False

        return ob, reward, terminated, truncated, info

    def render(self):
        frame = self.gym_env.render(mode='rgb_array', width=200, height=200)
        return frame
