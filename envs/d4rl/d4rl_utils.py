import d4rl
import gymnasium
import numpy as np

from envs.env_loader import EpisodeMonitor
from utils.dataset import Dataset


def make_env(env_name):
    env = gymnasium.make('GymV21Environment-v0', env_id=env_name)
    env = EpisodeMonitor(env)
    return env


def get_dataset(
        env,
        env_name,
        dataset=None,
        filter_terminals=False,
        obs_dtype=np.float32,
):
    if dataset is None:
        dataset = d4rl.qlearning_dataset(env)

    dataset['terminals'][-1] = 1

    if filter_terminals:
        # drop terminal transitions
        non_last_idxs, _ = np.nonzero(~dataset['terminals'])
        last_idxs, _ = np.nonzero(dataset['terminals'])
        penult_idxs = last_idxs - 1
        for k, v in dataset.items():
            if k == 'terminals':
                v[penult_idxs] = 1
            dataset[k] = v[non_last_idxs]

    terminals = np.zeros_like(dataset['rewards'])
    if 'antmaze' in env_name:
        for i in range(len(terminals) - 1):
            terminals[i] = float(np.linalg.norm(dataset['observations'][i + 1] - dataset['next_observations'][i]) > 1e-6)
    else:
        for i in range(len(terminals) - 1):
            if np.linalg.norm(dataset['observations'][i + 1] - dataset['next_observations'][i]) > 1e-6 or dataset['terminals'][i] == 1.0:
                terminals[i] = 1
            else:
                terminals[i] = 0
    terminals[-1] = 1

    return Dataset.create(
        observations=dataset['observations'].astype(obs_dtype),
        actions=dataset['actions'].astype(np.float32),
        next_observations=dataset['next_observations'].astype(obs_dtype),
        terminals=terminals.astype(np.float32),
    )
