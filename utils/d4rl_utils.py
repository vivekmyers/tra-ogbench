import d4rl
import gym
import numpy as np

from utils.dataset import Dataset
from utils.evaluation import EpisodeMonitor


def make_env(env_name: str):
    env = gym.make(env_name)
    env = EpisodeMonitor(env)
    return env


def get_dataset(env: gym.Env,
                env_name: str,
                clip_to_eps: bool = True,
                eps: float = 1e-5,
                dataset=None,
                filter_terminals=False,
                obs_dtype=np.float32,
                ):
    # traj_ends: trajectory boundaries
    # dones_float: either trajectory boundaries or "dead"
    # masks: 1 - trajectory boundaries (GCRL)
    # In GCRL, traj_ends and dones_float are the same. In RL, they can be different.

    if dataset is None:
        dataset = d4rl.qlearning_dataset(env)

    if clip_to_eps:
        lim = 1 - eps
        dataset['actions'] = np.clip(dataset['actions'], -lim, lim)

    dataset['terminals'][-1] = 1

    if filter_terminals:
        # drop terminal transitions
        non_last_idx = np.nonzero(~dataset['terminals'])[0]
        last_idx = np.nonzero(dataset['terminals'])[0]
        penult_idx = last_idx - 1
        for k, v in dataset.items():
            if k == 'terminals':
                v[penult_idx] = 1
            dataset[k] = v[non_last_idx]

    if 'antmaze' in env_name:
        dones_float = np.zeros_like(dataset['rewards'])
        traj_ends = np.zeros_like(dataset['rewards'])

        for i in range(len(dones_float) - 1):
            traj_end = (np.linalg.norm(dataset['observations'][i + 1] - dataset['next_observations'][i]) > 1e-6)
            traj_ends[i] = traj_end
            dones_float[i] = int(traj_end)
    else:
        dones_float = np.zeros_like(dataset['rewards'])
        traj_ends = np.zeros_like(dataset['rewards'])

        for i in range(len(dones_float) - 1):
            if np.linalg.norm(dataset['observations'][i + 1] - dataset['next_observations'][i]) > 1e-6 or \
                    dataset['terminals'][i] == 1.0:
                dones_float[i] = traj_ends[i] = 1
            else:
                dones_float[i] = traj_ends[i] = 0
    dones_float[-1] = 1
    traj_ends[-1] = 1

    observations = dataset['observations'].astype(obs_dtype)
    next_observations = dataset['next_observations'].astype(obs_dtype)

    masks = 1.0 - dones_float

    return Dataset.create(
        observations=observations,
        actions=dataset['actions'].astype(np.float32),
        rewards=dataset['rewards'].astype(np.float32),
        masks=masks,
        dones_float=dones_float.astype(np.float32),
        next_observations=next_observations,
        traj_ends=traj_ends
    )
