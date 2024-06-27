import glob
import json
import pickle
from collections import defaultdict

import flax
import gymnasium
import h5py
import numpy as np
from absl import app, flags
from tqdm import trange

import envs.locomaze  # noqa
from algos import SACAgent
from utils.evaluation import supply_rng

FLAGS = flags.FLAGS

flags.DEFINE_string('run_group', 'Debug', 'Run group')
flags.DEFINE_integer('seed', 0, 'Random seed')
flags.DEFINE_string('env_name', 'quadmaze-large-v0', 'Environment name')
flags.DEFINE_string('dataset_type', 'standard', 'Dataset type')
flags.DEFINE_string('restore_path', None, 'Expert agent paht')
flags.DEFINE_integer('restore_epoch', None, 'Expert agent paht')
flags.DEFINE_string('save_path', None, 'Save path')
flags.DEFINE_float('noise', 0.2, 'Action noise')
flags.DEFINE_integer('num_episodes', 1000, 'Number of episodes')


def main(_):
    env = gymnasium.make(FLAGS.env_name, render_mode='rgb_array', width=200, height=200, terminate_at_goal=False)
    ob_dim = 27

    restore_path = FLAGS.restore_path
    candidates = glob.glob(restore_path)
    assert len(candidates) == 1
    restore_path = candidates[0]

    with open(restore_path + '/flags.json', 'r') as f:
        agent_config = json.load(f)['agent']

    agent = SACAgent.create(
        FLAGS.seed,
        np.zeros(ob_dim),
        env.action_space.sample(),
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
    for _ in trange(FLAGS.num_episodes):
        if FLAGS.dataset_type == 'standard':
            valid_init_cells = []
            valid_goal_cells = []
            maze_map = env.unwrapped.maze_map

            for i in range(maze_map.shape[0]):
                for j in range(maze_map.shape[1]):
                    if maze_map[i, j] == 0:
                        valid_init_cells.append((i, j))

                        # Exclude hallway cells
                        if maze_map[i - 1, j] == 0 and maze_map[i + 1, j] == 0 and maze_map[i, j - 1] == 1 and maze_map[i, j + 1] == 1:
                            continue
                        if maze_map[i, j - 1] == 0 and maze_map[i, j + 1] == 0 and maze_map[i - 1, j] == 1 and maze_map[i + 1, j] == 1:
                            continue

                        valid_goal_cells.append((i, j))

            init_ij = valid_init_cells[np.random.randint(len(valid_init_cells))]
            goal_ij = valid_goal_cells[np.random.randint(len(valid_goal_cells))]
            ob, _ = env.reset(options=dict(init_ij=init_ij, goal_ij=goal_ij))
        else:
            ob, _ = env.reset()

        done = False
        step = 0
        success = False

        while not done:
            subgoal_xy = env.unwrapped.get_oracle_subgoal()
            subgoal_dir = subgoal_xy - ob[:2]
            subgoal_dir = subgoal_dir / (np.linalg.norm(subgoal_dir) + 1e-6)

            agent_ob = np.concatenate([ob[2:], subgoal_dir])
            action = actor_fn(agent_ob, temperature=0)
            action = action + np.random.normal(0, FLAGS.noise, action.shape)
            action = np.clip(action, -1, 1)
            next_ob, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            success = success or info['success']

            dataset['observations'].append(ob)
            dataset['actions'].append(action)
            dataset['next_observations'].append(next_ob)
            dataset['terminals'].append(done)
            dataset['infos/qpos'].append(info['qpos'])
            dataset['infos/qvel'].append(info['qvel'])

            ob = next_ob
            step += 1

        total_steps += step

    print('Total steps:', total_steps)
    file = h5py.File(FLAGS.save_path, 'w')
    for k in dataset:
        dataset[k] = np.array(dataset[k], dtype=np.float32 if k != 'terminals' else bool)
    for k in dataset:
        file.create_dataset(k, data=dataset[k], compression='gzip')


if __name__ == '__main__':
    app.run(main)
