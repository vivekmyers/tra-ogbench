from collections import defaultdict

import gymnasium
import numpy as np
from absl import app, flags
from tqdm import trange

import envs.manipspace  # noqa
from envs.manipspace.oracles.button_oracle import ButtonOracle
from envs.manipspace.oracles.cube_oracle import CubeOracle
from envs.manipspace.oracles.drawer_oracle import DrawerOracle
from envs.manipspace.oracles.window_oracle import WindowOracle

FLAGS = flags.FLAGS

flags.DEFINE_integer('seed', 0, 'Random seed')
flags.DEFINE_string('env_name', 'cube-single-v0', 'Environment name')
flags.DEFINE_string('dataset_type', 'play', 'Dataset type')
flags.DEFINE_string('save_path', None, 'Save path')
flags.DEFINE_float('noise', 0.1, 'Action noise')
flags.DEFINE_float('min_norm', 0.4, 'Action min norm')
flags.DEFINE_float('p_random_action', 0.1, 'Random action probability')
flags.DEFINE_integer('num_episodes', 1000, 'Number of episodes')
flags.DEFINE_integer('max_episode_steps', 1001, 'Number of episodes')


def main(_):
    env = gymnasium.make(
        FLAGS.env_name,
        terminate_at_goal=False,
        mode='data_collection',
        max_episode_steps=FLAGS.max_episode_steps,
    )

    has_button_states = hasattr(env.unwrapped, '_cur_button_states')
    if 'cube' in FLAGS.env_name:
        agents = {
            'cube': CubeOracle(env=env, min_norm=FLAGS.min_norm),
        }
    elif 'puzzle' in FLAGS.env_name:
        agents = {
            'button': ButtonOracle(env=env, min_norm=FLAGS.min_norm, gripper_always_closed=True),
        }
    elif 'scene' in FLAGS.env_name:
        agents = {
            'cube': CubeOracle(env=env, min_norm=FLAGS.min_norm, max_step=100),
            'button': ButtonOracle(env=env, min_norm=FLAGS.min_norm),
            'drawer': DrawerOracle(env=env, min_norm=FLAGS.min_norm),
            'window': WindowOracle(env=env, min_norm=FLAGS.min_norm),
        }

    dataset = defaultdict(list)

    total_steps = 0
    total_train_steps = 0
    num_train_episodes = FLAGS.num_episodes
    num_val_episodes = FLAGS.num_episodes // 10
    for ep_idx in trange(num_train_episodes + num_val_episodes):
        if FLAGS.dataset_type in ['play']:
            ob, info = env.reset()
            if 'single' in FLAGS.env_name:
                p_stack = 0.0
            elif 'double' in FLAGS.env_name:
                p_stack = np.random.uniform(0.0, 0.25)
            elif 'triple' in FLAGS.env_name:
                p_stack = np.random.uniform(0.05, 0.35)
            else:
                p_stack = np.random.uniform(0.1, 0.5)
        else:
            ob, info = env.reset()
        xi = np.random.uniform(0, FLAGS.noise)
        agent = agents[info['privileged/target_task']]
        agent.reset(ob, info)

        done = False
        step = 0

        while not done:
            if np.random.rand() < FLAGS.p_random_action:
                action = env.action_space.sample()
            else:
                action = agent.select_action(ob, info)
                action = np.array(action)
                action = action + np.random.normal(0, [xi, xi, xi, xi * 3, xi * 10], action.shape)
            action = np.clip(action, -1, 1)
            next_ob, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            if agent.done and FLAGS.dataset_type == 'play':
                agent_ob, agent_info = env.unwrapped.set_new_target(p_stack=p_stack)
                agent = agents[agent_info['privileged/target_task']]
                agent.reset(agent_ob, agent_info)

            dataset['observations'].append(ob)
            dataset['actions'].append(action)
            dataset['terminals'].append(done)
            dataset['qpos'].append(info['prev_qpos'])
            dataset['qvel'].append(info['prev_qvel'])
            if has_button_states:
                dataset['button_states'].append(info['prev_button_states'])

            ob = next_ob
            step += 1

        total_steps += step
        if ep_idx < num_train_episodes:
            total_train_steps += step

    print('Total steps:', total_steps)

    train_path = FLAGS.save_path
    val_path = FLAGS.save_path.replace('.npz', '-val.npz')

    train_dataset = {}
    val_dataset = {}
    for k, v in dataset.items():
        if 'observations' in k and v[0].dtype == np.uint8:
            dtype = np.uint8
        elif k == 'terminals':
            dtype = bool
        elif k == 'button_states':
            dtype = np.int64
        else:
            dtype = np.float32
        train_dataset[k] = np.array(v[:total_train_steps], dtype=dtype)
        val_dataset[k] = np.array(v[total_train_steps:], dtype=dtype)

    for path, dataset in [(train_path, train_dataset), (val_path, val_dataset)]:
        np.savez_compressed(path, **dataset)


if __name__ == '__main__':
    app.run(main)
