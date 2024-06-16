import time
from collections import defaultdict

import gym
import jax
import numpy as np
from tqdm import trange


def supply_rng(f, rng=jax.random.PRNGKey(0)):
    def wrapped(*args, **kwargs):
        nonlocal rng
        rng, key = jax.random.split(rng)
        return f(*args, seed=key, **kwargs)

    return wrapped


def flatten(d, parent_key="", sep="."):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if hasattr(v, "items"):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def add_to(dict_of_lists, single_dict):
    for k, v in single_dict.items():
        dict_of_lists[k].append(v)


def kitchen_render(kitchen_env, wh=64):
    from dm_control.mujoco import engine
    camera = engine.MovableCamera(kitchen_env.sim, wh, wh)
    camera.set_pose(distance=1.86, lookat=[-0.3, .5, 2.], azimuth=90, elevation=-60)
    img = camera.render()
    return img


def adroit_render(adroit_env, wh=64):
    adroit_env.viewer.render(width=wh, height=wh)
    frame = np.asarray(
        adroit_env.viewer.read_pixels(wh, wh, depth=False)[::-1, :, :], dtype=np.uint8,
    )
    return frame


def evaluate_with_trajectories(
        agent, env: gym.Env, env_name, goal_info=None,
        num_episodes=10, base_observation=None, num_video_episodes=0,
        eval_temperature=0, eval_gaussian=None,
        config=None,
):
    actor_fn = supply_rng(agent.sample_actions)
    trajectories = []
    stats = defaultdict(list)

    renders = []
    for i in trange(num_episodes + num_video_episodes):
        trajectory = defaultdict(list)

        # Reset
        if 'roboverse' in env_name:
            env.unwrapped.reset_counter = env.unwrapped.reset_interval - 1
        observation, done = env.reset(), False

        # Set goal
        if 'antmaze' in env_name:
            goal = env.wrapped_env.target_goal
            obs_goal = np.concatenate([goal, base_observation[-27:]])
        elif 'kitchen' in env_name:
            observation, obs_goal = observation[:30], observation[30:]
            obs_goal[:9] = base_observation[:9]
        else:
            raise NotImplementedError

        render = []
        step = 0
        while not done:
            actor_obs = observation
            actor_goal = obs_goal

            action = actor_fn(observations=actor_obs, goals=actor_goal, temperature=eval_temperature)
            action = np.array(action)
            if eval_gaussian is not None:
                action = np.random.normal(action, eval_gaussian)
                action = np.clip(action, -1, 1)

            # Step
            if 'antmaze' in env_name:
                next_observation, reward, done, info = env.step(action)
            elif 'kitchen' in env_name:
                next_observation, reward, done, info = env.step(action)
                next_observation = next_observation[:30]
            else:
                next_observation, reward, done, info = env.step(action)

            step += 1

            # Render
            if i >= num_episodes and step % 3 == 0:
                if 'antmaze' in env_name:
                    size = 200
                    cur_frame = env.render(mode='rgb_array', width=size, height=size).transpose(2, 0, 1).copy()
                elif 'kitchen' in env_name:
                    cur_frame = kitchen_render(env, wh=200).transpose(2, 0, 1)
                else:
                    size = 200
                    cur_frame = env.render(mode='rgb_array', width=size, height=size).transpose(2, 0, 1).copy()
                render.append(cur_frame)
            transition = dict(
                observation=observation,
                next_observation=next_observation,
                action=action,
                reward=reward,
                done=done,
                info=info,
            )
            add_to(trajectory, transition)
            add_to(stats, flatten(info))
            observation = next_observation
        if i < num_episodes:
            add_to(stats, flatten(info, parent_key="final"))
            trajectories.append(trajectory)
        else:
            renders.append(np.array(render))

    for k, v in stats.items():
        stats[k] = np.mean(v)
    return stats, trajectories, renders


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
        info["total"] = {"timesteps": self.total_timesteps}

        if done:
            info["episode"] = {}
            info["episode"]["return"] = self.reward_sum
            info["episode"]["length"] = self.episode_length
            info["episode"]["duration"] = time.time() - self.start_time

            if hasattr(self, "get_normalized_score"):
                info["episode"]["normalized_return"] = (
                        self.get_normalized_score(info["episode"]["return"]) * 100.0
                )

        return observation, reward, done, info

    def reset(self) -> np.ndarray:
        self._reset_stats()
        return self.env.reset()
