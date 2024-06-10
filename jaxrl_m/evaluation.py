import time
from collections import defaultdict
from typing import Dict

import gym
import jax
import numpy as np
from tqdm import trange


def supply_rng(f, rng=jax.random.PRNGKey(0)):
    """
    Wrapper that supplies a jax random key to a function (using keyword `seed`).
    Useful for stochastic policies that require randomness.

    Similar to functools.partial(f, seed=seed), but makes sure to use a different
    key for each new call (to avoid stale rng keys).

    """

    def wrapped(*args, **kwargs):
        nonlocal rng
        rng, key = jax.random.split(rng)
        return f(*args, seed=key, **kwargs)

    return wrapped


def flatten(d, parent_key="", sep="."):
    """
    Helper function that flattens a dictionary of dictionaries into a single dictionary.
    E.g: flatten({'a': {'b': 1}}) -> {'a.b': 1}
    """
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
        eval_temperature=0, eval_gaussian=None, sfbc_samples=None,
        actor_temperature=1, action_grad_coef=None, action_grad_normalize=0,
        config=None,
):
    policy_fn = supply_rng(agent.sample_actions)
    action_grad_fn = jax.jit(jax.grad(agent.min_critic, argnums=-1))
    sfbc_policy_fn = supply_rng(agent.sample_sfbc_actions)
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
        if config['goal_conditioned']:
            if 'antmaze' in env_name:
                goal = env.wrapped_env.target_goal
                obs_goal = np.concatenate([goal, base_observation[-27:]])
            elif 'kitchen' in env_name:
                observation, obs_goal = observation[:30], observation[30:]
                obs_goal[:9] = base_observation[:9]
            elif 'roboverse' in env_name:
                observation = (observation['image_observation'].reshape(3, 48, 48).transpose() * 255).astype(np.uint8)
                goal_idx = np.random.randint(len(goal_info['goals']))
                obs_goal = goal_info['goals'][goal_idx]
                goal_state = goal_info['state_desired_goals'][goal_idx]
                state_obs = []
                state_goals = []
            elif 'calvin' in env_name:
                import xxhash
                from calvin_agent.evaluation.utils import get_env_state_for_initial_condition

                start_goal_pairs = goal_info['start_goal_pairs']
                goal_idx = np.random.randint(len(start_goal_pairs))
                pair = start_goal_pairs[goal_idx]
                start_state, goal_state = pair['start'], pair['goal']

                # Render the goal image
                seed = xxhash.xxh32_intdigest(str(start_state.values()))
                robot_obs, scene_obs = get_env_state_for_initial_condition(goal_state, seed=seed)
                env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
                obs = env.get_obs()
                if 'state' in env_name:
                    obs_goal = np.concatenate([obs['robot_obs'], obs['scene_obs']])
                else:
                    obs_goal = obs['rgb_obs']['rgb_static']
                pixel_goal = obs['rgb_obs']['rgb_static']

                robot_obs, scene_obs = get_env_state_for_initial_condition(start_state, seed=seed)
                observation = env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
                start_info = env.get_info()
                final_success = False

                if 'state' in env_name:
                    observation = np.concatenate([observation['robot_obs'], observation['scene_obs']])
                else:
                    observation = observation['rgb_obs']['rgb_static']
            else:
                raise NotImplementedError
        else:
            if 'kitchen' in env_name:
                observation = observation[:30]
            obs_goal = None

        render = []
        action_grad_norms = []
        step = 0
        while not done:
            policy_obs = observation
            policy_goal = obs_goal

            if sfbc_samples is not None:
                action = sfbc_policy_fn(observations=policy_obs, goals=policy_goal, actor_temperature=actor_temperature, temperature=eval_temperature, num_samples=sfbc_samples)
            else:
                action = policy_fn(observations=policy_obs, goals=policy_goal, actor_temperature=actor_temperature, temperature=eval_temperature)
            if action_grad_coef is not None:
                action_grad = action_grad_fn(policy_obs, policy_goal, action)
                action_grad_norms.append(np.linalg.norm(action_grad))
                if action_grad_normalize:
                    action_grad = action_grad / (np.linalg.norm(action_grad) + 1e-9)

                action = action + action_grad * action_grad_coef
                action = np.array(action)
                action = np.clip(action, -1, 1)
            action = np.array(action)
            if eval_gaussian is not None:
                action = np.random.normal(action, eval_gaussian)
                action = np.clip(action, -1, 1)

            # Check if nan in action
            if np.isnan(action).any():
                print('NAN in action!!')
                print(step, policy_obs, action)
                print(trajectory)
                exit(1)

            # Step
            if 'antmaze' in env_name:
                next_observation, reward, done, info = env.step(action)
            elif 'kitchen' in env_name:
                next_observation, reward, done, info = env.step(action)
                next_observation = next_observation[:30]
            elif 'roboverse' in env_name:
                next_dict_observation, reward, done, info = env.step(np.array(action).copy())
                info = dict()
                next_observation = (next_dict_observation['image_observation'].reshape(3, 48, 48).transpose() * 255).astype(np.uint8)
                state_obs.append(next_dict_observation['state_observation'])
                state_goals.append(goal_state)
                if step >= 199:
                    done = True
                cur_render = next_observation.copy()
                cur_success = env.get_success_metric(next_dict_observation['state_observation'][None, ...], goal_state[None, ...], key='overall')[0][0]
                if cur_success:
                    cur_render[:5, -5:, :] = [0, 255, 0]
                else:
                    cur_render[:5, -5:, :] = [255, 0, 0]
                cur_render = np.concatenate([cur_render, obs_goal], axis=1)
                render_info = dict(
                    cur_render=cur_render,
                )
            elif 'calvin' in env_name:
                action = np.array(action).copy()
                if action[-1] < 0:
                    action[-1] = -1
                else:
                    action[-1] = 1
                next_observation, reward, done, info = env.step(action)
                render_info = dict(
                    cur_render=np.concatenate([next_observation['rgb_obs']['rgb_static'], pixel_goal], axis=1),
                )
                if 'state' in env_name:
                    next_observation = np.concatenate([next_observation['robot_obs'], next_observation['scene_obs']])
                else:
                    next_observation = next_observation['rgb_obs']['rgb_static']
                current_task_info = goal_info['task_oracle'].get_task_info_for_set(start_info, info, {goal_info['goal_name']})
                if len(current_task_info) > 0:
                    done = True
                    final_success = True
                if step >= 144:
                    done = True
                info = dict()
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
                elif 'pen' in env_name or 'hammer' in env_name or 'door' in env_name or 'relocate' in env_name:
                    cur_frame = adroit_render(env, wh=200).transpose(2, 0, 1)
                elif 'roboverse' in env_name:
                    cur_frame = render_info['cur_render'].transpose(2, 0, 1).copy()
                elif 'calvin' in env_name:
                    cur_frame = render_info['cur_render'].transpose(2, 0, 1).copy()
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
            # Add info
            if 'roboverse' in env_name:
                success = env.get_success_metric(np.array(state_obs), np.array(state_goals), key='overall')
                final_success = success[-1][0]
                info['return'] = final_success
                info['once_success'] = (success.sum() > 0)
            elif 'calvin' in env_name:
                info['return'] = final_success
            elif 'dmc' in env_name:
                info['return'] = sum(trajectory['reward'])

            if action_grad_coef is not None:
                info['action_grad_norm'] = np.mean(action_grad_norms)

            add_to(stats, flatten(info, parent_key="final"))
            trajectories.append(trajectory)
        else:
            renders.append(np.array(render))

    for k, v in stats.items():
        stats[k] = np.mean(v)
    return stats, trajectories, renders


class EpisodeMonitor(gym.ActionWrapper):
    """A class that computes episode returns and lengths."""

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
