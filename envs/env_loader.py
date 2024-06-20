import os
import platform

import numpy as np

from envs.antmaze.wrapper import AntMazeGoalWrapper
from envs.d4rl import d4rl_utils
from utils.dataset import Dataset


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


def make_env_and_dataset(env_name):
    if 'antmaze' in env_name:
        env = d4rl_utils.make_env(env_name)
        env = AntMazeGoalWrapper(env)
        dataset = d4rl_utils.get_dataset(env, env_name)
    elif 'kitchen' in env_name:
        # HACK: Monkey patching to make it compatible with Python 3.10.
        import collections
        if not hasattr(collections, 'Mapping'):
            collections.Mapping = collections.abc.Mapping

        setup_egl()

        env = d4rl_utils.make_env(env_name)
        dataset = d4rl_utils.get_dataset(env, env_name, filter_terminals=True)
        dataset = dataset.copy({
            'observations': dataset['observations'][:, :30],
            'next_observations': dataset['next_observations'][:, :30],
        })
    else:
        raise NotImplementedError

    env.reset()
    train_dataset, val_dataset = truncate_dataset(dataset, 0.95, return_both=True)

    return env, train_dataset, val_dataset


def make_online_env(env_name, eval=False):
    if 'ant' in env_name:
        from envs.locomotion.ant import AntEnv
        from d4rl.locomotion import wrappers
        from gym.wrappers.order_enforcing import OrderEnforcing
        from gym.wrappers.time_limit import TimeLimit
        from envs.d4rl.d4rl_utils import EpisodeMonitor

        env = AntEnv()
        env = wrappers.NormalizedBoxEnv(env)
        env = OrderEnforcing(env)
        env = TimeLimit(env, max_episode_steps=1000)
        env = EpisodeMonitor(env)

        if env_name == 'ant-xy':
            from envs.locomotion.xy_wrapper import XYWrapper
            env = XYWrapper(env, resample_interval=500 if eval else 100)
    else:
        raise NotImplementedError

    return env


