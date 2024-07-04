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
flags.DEFINE_string('env_name', 'quadball-arena-v0', 'Environment name')
flags.DEFINE_string('dataset_type', 'path', 'Dataset type')
flags.DEFINE_string('loco_restore_path', None, 'Locomotion agent restore path')
flags.DEFINE_integer('loco_restore_epoch', None, 'Locomotion agent restore epoch')
flags.DEFINE_string('ball_restore_path', None, 'Ball agent restore path')
flags.DEFINE_integer('ball_restore_epoch', None, 'Ball agent restore epoch')
flags.DEFINE_string('save_path', None, 'Save path')
flags.DEFINE_float('noise', 0.2, 'Action noise')
flags.DEFINE_integer('num_episodes', 1000, 'Number of episodes')
flags.DEFINE_integer('max_episode_steps', 1001, 'Number of episodes')


def load_agent(restore_path, restore_epoch, ob_dim, action_dim):
    candidates = glob.glob(restore_path)
    assert len(candidates) == 1
    restore_path = candidates[0]

    with open(restore_path + '/flags.json', 'r') as f:
        agent_config = json.load(f)['agent']

    agent = SACAgent.create(
        FLAGS.seed,
        np.zeros(ob_dim),
        np.zeros(action_dim),
        agent_config,
    )

    if restore_epoch is None:
        param_path = restore_path + '/params.pkl'
    else:
        param_path = restore_path + f'/params_{restore_epoch}.pkl'
    with open(param_path, 'rb') as f:
        load_dict = pickle.load(f)
    agent = flax.serialization.from_state_dict(agent, load_dict['agent'])

    return agent


def main(_):
    env = gymnasium.make(
        FLAGS.env_name,
        render_mode='rgb_array',
        width=200,
        height=200,
        terminate_at_goal=False,
        max_episode_steps=FLAGS.max_episode_steps,
    )
    ob_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    loco_agent = load_agent(FLAGS.loco_restore_path, FLAGS.loco_restore_epoch, ob_dim, action_dim)
    ball_agent = load_agent(FLAGS.ball_restore_path, FLAGS.ball_restore_epoch, ob_dim, action_dim)

    loco_actor_fn = supply_rng(loco_agent.sample_actions, rng=loco_agent.rng)
    ball_actor_fn = supply_rng(ball_agent.sample_actions, rng=ball_agent.rng)

    all_cells = []
    maze_map = env.unwrapped.maze_map

    for i in range(maze_map.shape[0]):
        for j in range(maze_map.shape[1]):
            if maze_map[i, j] == 0:
                all_cells.append((i, j))

    dataset = defaultdict(list)

    total_steps = 0
    total_train_steps = 0
    num_train_episodes = FLAGS.num_episodes
    num_val_episodes = FLAGS.num_episodes // 10
    for ep_idx in trange(num_train_episodes + num_val_episodes):
        if FLAGS.dataset_type in ['path', 'play']:
            agent_init_idx, ball_init_idx, goal_idx = np.random.choice(len(all_cells), 3, replace=False)
            agent_init_ij = all_cells[agent_init_idx]
            ball_init_ij = all_cells[ball_init_idx]
            goal_ij = all_cells[goal_idx]
            ob, _ = env.reset(options=dict(agent_init_ij=agent_init_ij, ball_init_ij=ball_init_ij, goal_ij=goal_ij))
        elif FLAGS.dataset_type == 'stitch':
            raise NotImplementedError
            init_ij = all_cells[np.random.randint(len(all_cells))]

            adj_cells = []
            adj_steps = 4
            bfs_map = maze_map.copy()
            for i in range(bfs_map.shape[0]):
                for j in range(bfs_map.shape[1]):
                    bfs_map[i][j] = -1
            bfs_map[init_ij[0], init_ij[1]] = 0
            queue = [init_ij]
            while len(queue) > 0:
                i, j = queue.pop(0)
                for di, dj in [(-1, 0), (0, -1), (1, 0), (0, 1)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < bfs_map.shape[0] and 0 <= nj < bfs_map.shape[1] and maze_map[ni, nj] == 0 and bfs_map[ni, nj] == -1:
                        bfs_map[ni][nj] = bfs_map[i][j] + 1
                        queue.append((ni, nj))
                        if bfs_map[ni][nj] == adj_steps:
                            adj_cells.append((ni, nj))

            goal_ij = adj_cells[np.random.randint(len(adj_cells))]
            ob, _ = env.reset(options=dict(init_ij=init_ij, goal_ij=goal_ij))
        else:
            ob, _ = env.reset()

        done = False
        step = 0

        virtual_agent_goal_xy = None

        while not done:
            # subgoal_xy, _ = env.unwrapped.get_oracle_subgoal(env.unwrapped.get_xy(), env.unwrapped.cur_goal_xy)
            # subgoal_dir = subgoal_xy - ob[:2]
            # subgoal_dir = subgoal_dir / (np.linalg.norm(subgoal_dir) + 1e-6)

            agent_xy, ball_xy = env.unwrapped.get_agent_ball_xy()
            agent_xy, ball_xy = np.array(agent_xy), np.array(ball_xy)
            goal_xy = np.array(env.unwrapped.cur_goal_xy)
            if virtual_agent_goal_xy is None:
                if np.linalg.norm(agent_xy - ball_xy) > 2:
                    # Move the agent to the ball
                    ball_dir = ball_xy - agent_xy
                    ball_dir = ball_dir / (np.linalg.norm(ball_dir) + 1e-6)
                    agent_ob = np.concatenate([ob[2:15], ob[22:36], ball_dir])
                    action = loco_actor_fn(agent_ob, temperature=0)
                else:
                    # Move the ball to the goal
                    if np.linalg.norm(goal_xy - ball_xy) > 10:
                        goal_xy = ball_xy + 10 * (goal_xy - ball_xy) / np.linalg.norm(goal_xy - ball_xy)
                    agent_ob = np.concatenate([ob[2:15], ob[17:], ball_xy - agent_xy, goal_xy - ball_xy])
                    action = ball_actor_fn(agent_ob, temperature=0)
            else:
                # Move the agent to the virtual goal
                agent_goal_dir = virtual_agent_goal_xy - ob[:2]
                agent_goal_dir = agent_goal_dir / (np.linalg.norm(agent_goal_dir) + 1e-6)
                agent_ob = np.concatenate([ob[2:15], ob[22:36], agent_goal_dir])
                action = loco_actor_fn(agent_ob, temperature=0)
            action = action + np.random.normal(0, FLAGS.noise, action.shape)
            action = np.clip(action, -1, 1)
            next_ob, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            success = info['success']

            if virtual_agent_goal_xy is not None and np.linalg.norm(virtual_agent_goal_xy - next_ob[:2]) <= 0.5:
                # Clear the virtual goal
                virtual_agent_goal_xy = None
            if FLAGS.dataset_type == 'play':
                if success:
                    # Resample a new goal
                    goal_ij = all_cells[np.random.randint(len(all_cells))]
                    env.unwrapped.set_goal(goal_ij)
                if step > 150 and virtual_agent_goal_xy is None and np.linalg.norm(np.array(dataset['observations'][-150:])[:, :2] - next_ob[:2], axis=1).max() <= 2:
                    # If stuck, move the agent to a random position
                    virtual_agent_goal_xy = np.array(all_cells[np.random.randint(len(all_cells))])

            dataset['observations'].append(ob)
            dataset['actions'].append(action)
            dataset['terminals'].append(done)
            dataset['infos/qpos'].append(info['prev_qpos'])
            dataset['infos/qvel'].append(info['prev_qvel'])

            ob = next_ob
            step += 1

        total_steps += step
        if ep_idx < num_train_episodes:
            total_train_steps += step

    print('Total steps:', total_steps)

    train_path = FLAGS.save_path.replace('.hdf5', '-train.hdf5')
    val_path = FLAGS.save_path.replace('.hdf5', '-val.hdf5')

    train_dataset = {k: np.array(v[:total_train_steps], dtype=np.float32 if k != 'terminals' else bool) for k, v in dataset.items()}
    val_dataset = {k: np.array(v[total_train_steps:], dtype=np.float32 if k != 'terminals' else bool) for k, v in dataset.items()}

    for path, dataset in [(train_path, train_dataset), (val_path, val_dataset)]:
        file = h5py.File(path, 'w')
        for k in dataset:
            file.create_dataset(k, data=dataset[k], compression='gzip')


if __name__ == '__main__':
    app.run(main)
