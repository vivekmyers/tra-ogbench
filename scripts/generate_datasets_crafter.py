import glob
import json
import pickle
from collections import defaultdict

import flax
import gymnasium
import numpy as np
from absl import app, flags
from tqdm import trange

import envs.crafter  # noqa
from algos import PPOAgent
from utils.evaluation import supply_rng

FLAGS = flags.FLAGS

flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'cube-single-v0', 'Environment name.')
flags.DEFINE_string('dataset_type', 'play', 'Dataset type.')
flags.DEFINE_string('restore_path', None, 'Expert agent restore path.')
flags.DEFINE_integer('restore_epoch', None, 'Expert agent restore epoch.')
flags.DEFINE_string('save_path', None, 'Save path.')
flags.DEFINE_integer('num_episodes', 1000, 'Number of episodes.')
flags.DEFINE_integer('add_demos', 0, 'Whether to add human demonstrations.')


def add_demo_to_dataset(env, dataset, demo_file):
    step = 0
    demo_data = dict(np.load(demo_file))
    demo_len = len(demo_data['image'])
    for j in range(demo_len - 1):
        dataset['observations'].append(demo_data['image'][j])
        dataset['actions'].append(demo_data['action'][j + 1])  # Warning: dataset actions are shifted by 1
        dataset['terminals'].append(False)
        step += 1

    # Add goal-view images
    inventory = {}
    for item, _ in env.unwrapped.env._player.inventory.items():
        inventory[item] = demo_data[f'inventory_{item}'][demo_len - 2]  # Get the penultimate inventory

    num_last_steps = env.unwrapped._num_last_steps
    for j in range(num_last_steps - 1):
        dataset['observations'].append(env.unwrapped.goal_render(inventory))
        dataset['actions'].append(np.random.randint(17))
        dataset['terminals'].append(False if j < num_last_steps - 2 else True)
        step += 1

    return step


def main(_):
    env = gymnasium.make(
        FLAGS.env_name,
        mode='data_collection',
    )
    env.reset()
    action_dim = 17  # Excluding goal-view action

    restore_path = FLAGS.restore_path
    candidates = glob.glob(restore_path)
    assert len(candidates) == 1
    restore_path = candidates[0]

    with open(restore_path + '/flags.json', 'r') as f:
        agent_config = json.load(f)['agent']

    agent = PPOAgent.create(
        FLAGS.seed,
        env.observation_space.sample(),
        np.array(action_dim - 1),
        agent_config,
    )

    if FLAGS.restore_epoch is None:
        param_path = restore_path + '/params.pkl'
    else:
        param_path = restore_path + f'/params_{FLAGS.restore_epoch}.pkl'
    with open(param_path, 'rb') as f:
        load_dict = pickle.load(f)
    agent = flax.serialization.from_state_dict(agent, load_dict['agent'])
    actor_fn = supply_rng(agent.sample_actions, rng=agent.rng)

    dataset = defaultdict(list)

    total_steps = 0
    total_train_steps = 0
    num_train_episodes = FLAGS.num_episodes
    num_val_episodes = FLAGS.num_episodes // 10

    if FLAGS.add_demos:
        # Add human demonstrations
        demo_files = glob.glob('data/crafter/demos/*.npz')
        np.random.shuffle(demo_files)
        num_demo_episodes = len(demo_files)
        num_val_demo_episodes = num_demo_episodes // 11
        num_train_demo_episodes = num_demo_episodes - num_val_demo_episodes
        num_train_episodes += num_train_demo_episodes
        num_val_episodes += num_val_demo_episodes
        demo_episode_idxs = set(
            np.random.choice(num_train_episodes + num_val_episodes, num_demo_episodes, replace=False)
        )
    else:
        demo_episode_idxs = set()

    for ep_idx in trange(num_train_episodes + num_val_episodes):
        if ep_idx in demo_episode_idxs:
            demo_file = demo_files.pop()
            step = add_demo_to_dataset(env, dataset, demo_file)
        else:
            ob, info = env.reset()

            done = False
            step = 0

            while not done:
                action = actor_fn(ob, temperature=1)
                next_ob, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                dataset['observations'].append(ob)
                dataset['actions'].append(action)
                dataset['terminals'].append(done)

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
        elif 'actions' in k and v[0].dtype == np.int32:
            dtype = np.int32
        elif k == 'terminals':
            dtype = bool
        else:
            dtype = np.float32
        train_dataset[k] = np.array(v[:total_train_steps], dtype=dtype)
        val_dataset[k] = np.array(v[total_train_steps:], dtype=dtype)

    for path, dataset in [(train_path, train_dataset), (val_path, val_dataset)]:
        np.savez_compressed(path, **dataset)


if __name__ == '__main__':
    app.run(main)
