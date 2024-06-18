from collections import defaultdict

import jax
import numpy as np
from tqdm import trange


def supply_rng(f, rng=jax.random.PRNGKey(0)):
    def wrapped(*args, **kwargs):
        nonlocal rng
        rng, key = jax.random.split(rng)
        return f(*args, seed=key, **kwargs)

    return wrapped


def flatten(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if hasattr(v, 'items'):
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


def evaluate(
        agent,
        env,
        task_idx,
        config=None,
        num_eval_episodes=50,
        num_video_episodes=0,
        video_frame_skip=3,
        eval_temperature=0,
        eval_gaussian=None,
):
    actor_fn = supply_rng(agent.sample_actions, rng=agent.rng)
    trajs = []
    stats = defaultdict(list)

    renders = []
    for i in trange(num_eval_episodes + num_video_episodes):
        traj = defaultdict(list)

        observation, goal = env.reset(task_idx=task_idx)
        done = False
        step = 0
        render = []
        while not done:
            action = actor_fn(observations=observation, goals=goal, temperature=eval_temperature)
            action = np.array(action)
            if eval_gaussian is not None:
                action = np.random.normal(action, eval_gaussian)
                action = np.clip(action, -1, 1)

            next_observation, reward, done, info = env.step(action)
            step += 1

            if i >= num_eval_episodes and step % video_frame_skip == 0:
                frame = env.render().copy()
                render.append(frame)

            transition = dict(
                observation=observation,
                next_observation=next_observation,
                action=action,
                reward=reward,
                done=done,
                info=info,
            )
            add_to(traj, transition)
            observation = next_observation
        if i < num_eval_episodes:
            add_to(stats, flatten(info))
            trajs.append(traj)
        else:
            renders.append(np.array(render))

    for k, v in stats.items():
        stats[k] = np.mean(v)
    return stats, trajs, renders
