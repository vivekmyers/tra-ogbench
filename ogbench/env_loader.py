import collections
import os
import platform
import time

import gymnasium
import numpy as np
from gymnasium.spaces import Box

from utils.datasets import Dataset


class EpisodeMonitor(gymnasium.Wrapper):
    """Environment wrapper to monitor episode statistics."""

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
    """Environment wrapper to stack observations."""

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
    """Set up EGL for rendering."""
    if 'mac' in platform.platform():
        # macOS doesn't support EGL.
        pass
    else:
        os.environ['MUJOCO_GL'] = 'egl'
        if 'SLURM_STEP_GPUS' in os.environ:
            os.environ['EGL_DEVICE_ID'] = os.environ['SLURM_STEP_GPUS']


def truncate_dataset(dataset, ratio, return_both=False):
    """Truncate dataset to a certain ratio of states.

    This can be useful for splitting the dataset into training and validation sets.

    Args:
        dataset: Dataset to truncate.
        ratio: Ratio of states to keep.
        return_both: Whether to return both truncated and complementary datasets.
    """
    size = dataset.size
    traj_idxs = []
    traj_start = 0
    for i in range(len(dataset['observations'])):
        if dataset['terminals'][i] == 1.0:
            traj_idxs.append(np.arange(traj_start, i + 1))
            traj_start = i + 1
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


def get_dataset(dataset_path, ob_dtype=np.float32, action_dtype=np.float32):
    """Load OGBench dataset.

    Args:
        dataset_path: Path to the dataset file.
        ob_dtype: dtype for observations.
        action_dtype: dtype for actions.

    Returns:
        Tuple of training and validation datasets.
    """
    train_path = dataset_path
    val_path = dataset_path.replace('.npz', '-val.npz')
    train_dataset = dict()
    val_dataset = dict()
    for path, dataset in [(train_path, train_dataset), (val_path, val_dataset)]:
        file = np.load(path)

        for k in ['observations', 'actions', 'terminals']:
            if k == 'observations':
                dtype = ob_dtype
            elif k == 'actions':
                dtype = action_dtype
            else:
                dtype = np.float32
            dataset[k] = file[k][...].astype(dtype)

        # At this point, we have:
        # terminals: [0, 0, ..., 0, 0, 1, 0, 0, ..., 0, 0, 1, 0, 0, ..., 0, 0, 1]
        #             |<--trajectory-->|  |<--trajectory-->|  |<- trajectory-->|

        # Since the dataset doesn't contain next_observations, we need to invalidate the last state of each trajectory
        # so that we can safely get next_observations[t] by using observations[t + 1].
        dataset['valids'] = 1.0 - dataset['terminals']
        new_terminals = np.concatenate([dataset['terminals'][1:], [1.0]])
        dataset['terminals'] = np.minimum(dataset['terminals'] + new_terminals, 1.0)

        # Now, we have the following:
        # terminals: [0, 0, ..., 0, 1, 1, 0, 0, ..., 0, 1, 1, 0, 0, ..., 0, 1, 1]
        # valids   : [1, 1, ..., 1, 1, 0, 1, 1, ..., 1, 0, 1, 1, 1, ..., 1, 1, 0]
        #             |<--trajectory-->|  |<--trajectory-->|  |<--trajectory-->|

    return Dataset.create(**train_dataset), Dataset.create(**val_dataset)


def make_env_and_dataset(env_name, dataset_path=None, frame_stack=None):
    """Make environment and dataset.

    Args:
        env_name: Name of the environment.
        dataset_path: Path to the dataset file.
        frame_stack: Number of frames to stack.
    """
    setup_egl()

    if 'antmaze' in env_name and ('diverse' in env_name or 'play' in env_name):
        from ogbench.d4rl import d4rl_utils
        from ogbench.d4rl.wrappers import AntMazeGoalWrapper

        env = d4rl_utils.make_env(env_name)
        env = AntMazeGoalWrapper(env)
        dataset = d4rl_utils.get_dataset(env, env_name)
        train_dataset, val_dataset = truncate_dataset(dataset, 0.95, return_both=True)
    elif 'kitchen' in env_name:
        from ogbench.d4rl import d4rl_utils
        from ogbench.d4rl.wrappers import KitchenGoalWrapper

        # HACK: Monkey patching to make it compatible with Python 3.10.
        if not hasattr(collections, 'Mapping'):
            collections.Mapping = collections.abc.Mapping

        env = d4rl_utils.make_env(env_name)
        env = KitchenGoalWrapper(env)
        dataset = d4rl_utils.get_dataset(env, env_name, filter_terminals=True)
        dataset = dataset.copy(
            {
                'observations': dataset['observations'][:, :30],
                'next_observations': dataset['next_observations'][:, :30],
            }
        )
        train_dataset, val_dataset = truncate_dataset(dataset, 0.95, return_both=True)
    elif 'pointmaze' in env_name or 'antmaze' in env_name or 'antsoccer' in env_name or 'humanoidmaze' in env_name:
        # OGBench Locomotion environments.
        import ogbench.locomaze  # noqa

        env = gymnasium.make(env_name)
        train_dataset, val_dataset = get_dataset(
            dataset_path, ob_dtype=np.uint8 if 'visual' in env_name else np.float32
        )
    elif 'cube' in env_name or 'scene' in env_name or 'puzzle' in env_name:
        # OGBench Manipulation environments.
        import ogbench.manipspace  # noqa

        env = gymnasium.make(env_name)
        train_dataset, val_dataset = get_dataset(
            dataset_path, ob_dtype=np.uint8 if 'visual' in env_name else np.float32
        )
    elif 'powderworld' in env_name:
        # OGBench Drawing environments.
        import ogbench.powderworld  # noqa

        env = gymnasium.make(env_name)
        train_dataset, val_dataset = get_dataset(dataset_path, ob_dtype=np.uint8, action_dtype=np.int32)
    elif 'crafter' in env_name:
        import ogbench.crafter  # noqa

        env = gymnasium.make(env_name)
        train_dataset, val_dataset = get_dataset(dataset_path, ob_dtype=np.uint8, action_dtype=np.int32)
    else:
        raise ValueError(f'Unknown environment: {env_name}')

    if frame_stack is not None:
        env = FrameStackWrapper(env, frame_stack)

    if val_dataset is not None and val_dataset.size == 0:
        val_dataset = None

    env.reset()

    return env, train_dataset, val_dataset


def make_online_env(env_name):
    """Make online environment.

    Args:
        env_name: Name of the environment.
    """
    setup_egl()

    if 'online-ant' in env_name or 'online-humanoid' in env_name:
        import ogbench.online_locomotion  # noqa

        # Manually recognize the '-xy' suffix, which indicates that the environment should be wrapped with a
        # directional locomotion wrapper.
        if '-xy' in env_name:
            env_name = env_name.replace('-xy', '')
            apply_xy_wrapper = True
        else:
            apply_xy_wrapper = False

        if 'humanoid' in env_name:
            env = gymnasium.make(env_name, render_mode='rgb_array', height=200, width=200, camera_id=0)
        else:
            env = gymnasium.make(env_name, render_mode='rgb_array', height=200, width=200)

        if apply_xy_wrapper:
            # Apply the directional locomotion wrapper.
            from ogbench.online_locomotion.wrappers import DMCHumanoidXYWrapper, GymXYWrapper

            if 'humanoid' in env_name:
                env = DMCHumanoidXYWrapper(env, resample_interval=200)
            else:
                env = GymXYWrapper(env, resample_interval=100)

        env = EpisodeMonitor(env)
    elif 'crafter' in env_name:
        import ogbench.crafter  # noqa

        env = gymnasium.make(env_name, mode='online')
        env = EpisodeMonitor(env)
    else:
        raise ValueError(f'Unknown environment: {env_name}')

    return env


def make_vec_env(env_name, num_envs, **kwargs):
    from gymnasium.vector import SyncVectorEnv

    envs = [lambda: make_online_env(env_name, **kwargs) for _ in range(num_envs)]
    env = SyncVectorEnv(envs)
    return env
