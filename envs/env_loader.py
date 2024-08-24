import collections
import os
import platform
import time

import gymnasium
import numpy as np
from gymnasium.spaces import Box

from utils.dataset import Dataset


class EpisodeMonitor(gymnasium.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._reset_stats()
        self.total_timesteps = 0

    def _reset_stats(self):
        self.reward_sum = 0.0
        self.episode_length = 0
        self.start_time = time.time()

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)

        self.reward_sum += reward
        self.episode_length += 1
        self.total_timesteps += 1
        info['total'] = {'timesteps': self.total_timesteps}

        if terminated or truncated:
            info['episode'] = {}
            info['episode']['return'] = self.reward_sum
            info['episode']['length'] = self.episode_length
            info['episode']['duration'] = time.time() - self.start_time

            if hasattr(self.unwrapped, 'get_normalized_score'):
                info['episode']['normalized_return'] = (
                    self.unwrapped.get_normalized_score(info['episode']['return']) * 100.0
                )

        return observation, reward, terminated, truncated, info

    def reset(self, *args, **kwargs):
        self._reset_stats()
        return self.env.reset(*args, **kwargs)


class FrameStackWrapper(gymnasium.Wrapper):
    def __init__(self, env, num_stack):
        super().__init__(env)

        self.num_stack = num_stack
        self.frames = collections.deque(maxlen=num_stack)

        low = np.concatenate([self.observation_space.low] * num_stack, axis=-1)
        high = np.concatenate([self.observation_space.high] * num_stack, axis=-1)
        self.observation_space = Box(low=low, high=high, dtype=self.observation_space.dtype)

    def get_observation(self):
        assert len(self.frames) == self.num_stack
        return np.concatenate(list(self.frames), axis=-1)

    def reset(self, **kwargs):
        ob, info = self.env.reset(**kwargs)
        for _ in range(self.num_stack):
            self.frames.append(ob)
        if 'goal' in info:
            info['goal'] = np.concatenate([info['goal']] * self.num_stack, axis=-1)
        return self.get_observation(), info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(observation)
        return self.get_observation(), reward, terminated, truncated, info


def setup_egl():
    if 'mac' in platform.platform():
        pass
    else:
        os.environ['MUJOCO_GL'] = 'egl'
        if 'SLURM_STEP_GPUS' in os.environ:
            os.environ['EGL_DEVICE_ID'] = os.environ['SLURM_STEP_GPUS']


def truncate_dataset(dataset, ratio, return_both=False):
    size = dataset.size
    traj_idxs = []
    traj_start = 0
    for i in range(len(dataset['observations'])):
        if dataset['terminals'][i] == 1.0:
            traj_idxs.append(np.arange(traj_start, i + 1))
            traj_start = i + 1
    np.random.seed(0)
    traj_idxs = np.random.permutation(traj_idxs)
    new_idxs = []
    num_states = 0
    for idxs in traj_idxs:
        new_idxs.extend(idxs)
        num_states += len(idxs)
        if num_states >= size * ratio:
            break
    trunc_dataset = Dataset(dataset.get_subset(new_idxs))
    if return_both:
        codataset = Dataset(dataset.get_subset(np.setdiff1d(np.arange(size), new_idxs)))
        return trunc_dataset, codataset
    else:
        return trunc_dataset


def get_dataset(dataset_path, ob_dtype=np.float32):
    train_path = dataset_path
    val_path = dataset_path.replace('.npz', '-val.npz')
    train_dataset = dict()
    val_dataset = dict()
    for path, dataset in [(train_path, train_dataset), (val_path, val_dataset)]:
        file = np.load(path)

        for k in ['observations', 'actions', 'terminals']:
            dtype = ob_dtype if 'observation' in k else np.float32
            dataset[k] = file[k][...].astype(dtype)

        # Since the dataset doesn't contain next_observations, we need to invalidate the last state of each trajectory
        # so that we can safely get next_observations[t] by using observations[t + 1].
        dataset['valids'] = 1.0 - dataset['terminals']
        new_terminals = np.concatenate([dataset['terminals'][1:], [1.0]])
        dataset['terminals'] = np.minimum(dataset['terminals'] + new_terminals, 1.0)
        # Now, we have the following:
        # terminals: [0, 0, ..., 0, 1, 1, 0, 0, 0, ..., 0, 1, 1, 0, 0, ..., 1, 1]
        # valids   : [1, 1, ..., 1, 1, 0, 1, 1, 1, ..., 1, 0, 1, 1, 1, ..., 1, 0]

    return Dataset.create(**train_dataset), Dataset.create(**val_dataset)


def make_env_and_dataset(env_name, dataset_path=None):
    setup_egl()

    if 'antmaze' in env_name:
        from envs.antmaze.wrappers import AntMazeGoalWrapper
        from envs.d4rl import d4rl_utils

        env = d4rl_utils.make_env(env_name)
        env = AntMazeGoalWrapper(env)
        dataset = d4rl_utils.get_dataset(env, env_name)
        train_dataset, val_dataset = truncate_dataset(dataset, 0.95, return_both=True)
    elif 'kitchen' in env_name:
        # HACK: Monkey patching to make it compatible with Python 3.10.
        from envs.d4rl import d4rl_utils

        if not hasattr(collections, 'Mapping'):
            collections.Mapping = collections.abc.Mapping

        env = d4rl_utils.make_env(env_name)
        dataset = d4rl_utils.get_dataset(env, env_name, filter_terminals=True)
        dataset = dataset.copy(
            {
                'observations': dataset['observations'][:, :30],
                'next_observations': dataset['next_observations'][:, :30],
            }
        )
        train_dataset, val_dataset = truncate_dataset(dataset, 0.95, return_both=True)
    elif 'quadmaze' in env_name or 'quadball' in env_name or 'humanoidmaze' in env_name:
        import envs.locomaze  # noqa

        env = gymnasium.make(env_name)
        train_dataset, val_dataset = get_dataset(
            dataset_path, ob_dtype=np.uint8 if 'visual' in env_name else np.float32
        )
    elif 'cube' in env_name or 'button' in env_name or 'scene' in env_name:
        import envs.manipspace  # noqa

        env = gymnasium.make(env_name)
        train_dataset, val_dataset = get_dataset(
            dataset_path, ob_dtype=np.uint8 if 'visual' in env_name else np.float32
        )
    else:
        raise ValueError(f'Unknown environment: {env_name}')

    if val_dataset is not None and val_dataset.size == 0:
        val_dataset = None

    env.reset()

    return env, train_dataset, val_dataset


def make_online_env(env_name):
    setup_egl()

    if 'Ant' in env_name or 'Humanoid' in env_name:
        import envs.locomotion  # noqa

        if '-xy' in env_name:
            env_name = env_name.replace('-xy', '')
            apply_xy_wrapper = True
        else:
            apply_xy_wrapper = False

        if 'Ant' in env_name:
            xml_file = os.path.join(os.path.dirname(__file__), 'locomotion/assets/ant.xml')
            env = gymnasium.make(env_name, render_mode='rgb_array', height=200, width=200, xml_file=xml_file)
        elif 'HumanoidCustom' in env_name:
            env = gymnasium.make(env_name, render_mode='rgb_array', height=200, width=200, camera_id=0)
        else:
            env = gymnasium.make(env_name, render_mode='rgb_array', height=200, width=200)

        if apply_xy_wrapper:
            from envs.locomotion.wrappers import DMCHumanoidXYWrapper, GymXYWrapper

            if 'HumanoidCustom' in env_name:
                env = DMCHumanoidXYWrapper(env, resample_interval=200)
            else:
                env = GymXYWrapper(env, resample_interval=100 if 'Ant' in env_name else 200)

        env = EpisodeMonitor(env)
    else:
        raise ValueError(f'Unknown environment: {env_name}')

    return env


def make_vec_env(env_name, num_envs, **kwargs):
    from gymnasium.vector import SyncVectorEnv

    envs = [lambda: make_online_env(env_name, **kwargs) for _ in range(num_envs)]
    env = SyncVectorEnv(envs)
    return env
