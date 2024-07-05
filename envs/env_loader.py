import collections
import os
import platform
import time

import gymnasium
import h5py
import numpy as np

from utils.dataset import Dataset


class EpisodeMonitor(gymnasium.ActionWrapper):
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
                info['episode']['normalized_return'] = self.unwrapped.get_normalized_score(info['episode']['return']) * 100.0

        return observation, reward, terminated, truncated, info

    def reset(self, *args, **kwargs):
        self._reset_stats()
        return self.env.reset(*args, **kwargs)


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


def get_dataset(dataset_path, obs_dtype=np.float32):
    train_path = dataset_path
    val_path = dataset_path.replace('.hdf5', '-val.hdf5')
    train_dataset = dict()
    val_dataset = dict()
    for path, dataset in [(train_path, train_dataset), (val_path, val_dataset)]:
        file = h5py.File(path, 'r')

        for k in ['observations', 'actions', 'terminals']:
            dtype = obs_dtype if 'observation' in k else np.float32
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
    if 'antmaze' in env_name:
        from envs.d4rl import d4rl_utils
        from envs.antmaze.wrappers import AntMazeGoalWrapper
        env = d4rl_utils.make_env(env_name)
        env = AntMazeGoalWrapper(env)
        dataset = d4rl_utils.get_dataset(env, env_name)
        train_dataset, val_dataset = truncate_dataset(dataset, 0.95, return_both=True)
    elif 'kitchen' in env_name:
        # HACK: Monkey patching to make it compatible with Python 3.10.
        from envs.d4rl import d4rl_utils
        if not hasattr(collections, 'Mapping'):
            collections.Mapping = collections.abc.Mapping

        setup_egl()

        env = d4rl_utils.make_env(env_name)
        dataset = d4rl_utils.get_dataset(env, env_name, filter_terminals=True)
        dataset = dataset.copy({
            'observations': dataset['observations'][:, :30],
            'next_observations': dataset['next_observations'][:, :30],
        })
        train_dataset, val_dataset = truncate_dataset(dataset, 0.95, return_both=True)
    elif 'quadmaze' in env_name or 'quadball' in env_name or 'humanoidmaze' in env_name:
        import gymnasium
        import envs.locomaze  # noqa

        env = gymnasium.make(env_name, render_mode='rgb_array', width=200, height=200)
        train_dataset, val_dataset = get_dataset(dataset_path)
    else:
        raise ValueError(f'Unknown environment: {env_name}')

    env.reset()

    return env, train_dataset, val_dataset


def make_online_env(env_name, eval=False):
    if 'ant' in env_name or 'gymhum' in env_name or 'humanoid' in env_name:
        import envs.locomotion  # noqa
        import gymnasium

        if 'ant' in env_name:
            xml_file = os.path.join(os.path.dirname(__file__), 'locomotion/assets/ant.xml')
            env = gymnasium.make('AntCustom-v0', render_mode='rgb_array', height=200, width=200, xml_file=xml_file)
        elif 'gymhum' in env_name:
            env = gymnasium.make('Humanoid-v4', render_mode='rgb_array', height=200, width=200)
        elif 'humanoid' in env_name:
            env = gymnasium.make('HumanoidCustom-v0', render_mode='rgb_array', height=200, width=200, camera_id=0)
        else:
            raise ValueError(f'Unknown environment: {env_name}')

        if env_name.endswith('-xy'):
            from envs.locomotion.wrappers import GymXYWrapper, DMCHumanoidXYWrapper
            if 'ant' in env_name or 'gymhum' in env_name:
                env = GymXYWrapper(env, resample_interval=100 if 'ant' in env_name else 200)
            else:
                env = DMCHumanoidXYWrapper(env, resample_interval=200)

        env = EpisodeMonitor(env)
    elif 'quadball' in env_name:
        import gymnasium
        import envs.locomaze  # noqa

        env = gymnasium.make(env_name, render_mode='rgb_array', width=400, height=400, max_episode_steps=200, terminate_at_goal=False)
        env = EpisodeMonitor(env)
    else:
        raise ValueError(f'Unknown environment: {env_name}')

    return env


def make_vec_env(env_name, num_envs, **kwargs):
    from gymnasium.vector import SyncVectorEnv
    envs = [lambda: make_online_env(env_name, **kwargs) for _ in range(num_envs)]
    env = SyncVectorEnv(envs)
    return env
