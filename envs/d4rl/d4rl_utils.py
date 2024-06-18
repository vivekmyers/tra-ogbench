import d4rl
import gym
import time
import numpy as np

from utils.dataset import Dataset


class EpisodeMonitor(gym.ActionWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._reset_stats()
        self.total_timesteps = 0

    def _reset_stats(self):
        self.reward_sum = 0.0
        self.episode_length = 0
        self.start_time = time.time()

    def step(self, action: np.ndarray):
        observation, reward, done, info = self.env.step(action)

        self.reward_sum += reward
        self.episode_length += 1
        self.total_timesteps += 1
        info['total'] = {'timesteps': self.total_timesteps}

        if done:
            info['episode'] = {}
            info['episode']['return'] = self.reward_sum
            info['episode']['length'] = self.episode_length
            info['episode']['duration'] = time.time() - self.start_time

            if hasattr(self, 'get_normalized_score'):
                info['episode']['normalized_return'] = self.get_normalized_score(info['episode']['return']) * 100.0

        return observation, reward, done, info

    def reset(self) -> np.ndarray:
        self._reset_stats()
        return self.env.reset()


def make_env(env_name: str):
    env = gym.make(env_name)
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
        non_last_idx = np.nonzero(~dataset['terminals'])[0]
        last_idx = np.nonzero(dataset['terminals'])[0]
        penult_idx = last_idx - 1
        for k, v in dataset.items():
            if k == 'terminals':
                v[penult_idx] = 1
            dataset[k] = v[non_last_idx]

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
