import json
import os
import random
import time

import jax
import numpy as np
import tqdm
import wandb
from absl import app, flags
from ml_collections import config_flags

from agents import agents
from ogbench.env_loader import make_online_env, make_vec_env
from ogbench.viz_utils import visualize_trajs
from utils.datasets import ReplayBuffer
from utils.evaluation import evaluate, flatten
from utils.flax_utils import restore_agent, save_agent
from utils.log_utils import CsvLogger, get_exp_name, get_flag_dict, get_wandb_video, setup_wandb

FLAGS = flags.FLAGS

flags.DEFINE_string('run_group', 'Debug', 'Run group.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'antmaze-large-diverse-v2', 'Environment name.')
flags.DEFINE_string('dataset_path', None, 'Dataset path.')
flags.DEFINE_string('save_dir', 'exp/', 'Save directory.')
flags.DEFINE_string('restore_path', None, 'Restore path.')
flags.DEFINE_integer('restore_epoch', None, 'Restore epoch.')

flags.DEFINE_integer('num_envs', 64, 'Number of environments.')
flags.DEFINE_integer('train_steps', 1000000, 'Number of training steps.')
flags.DEFINE_integer('train_interval', 128, 'Training interval.')
flags.DEFINE_integer('log_interval', 5000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 100000, 'Evaluation interval.')
flags.DEFINE_integer('save_interval', 1000000, 'Saving interval.')

flags.DEFINE_integer('eval_tasks', None, 'Number of tasks to evaluate (None for all).')
flags.DEFINE_integer('eval_episodes', 50, 'Number of episodes for each task.')
flags.DEFINE_float('eval_temperature', 0, 'Actor temperature for evaluation.')
flags.DEFINE_float('eval_gaussian', None, 'Action Gaussian noise for evaluation.')
flags.DEFINE_integer('video_episodes', 1, 'Number of video episodes for each task.')
flags.DEFINE_integer('video_frame_skip', 3, 'Frame skip for videos.')

config_flags.DEFINE_config_file('agent', 'agents/ppo.py', lock_config=False)


def main(_):
    # Set up logger.
    exp_name = get_exp_name(FLAGS.seed)
    setup_wandb(project='ogcrl', group=FLAGS.run_group, name=exp_name)

    FLAGS.save_dir = os.path.join(FLAGS.save_dir, wandb.run.project, FLAGS.run_group, exp_name)
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    flag_dict = get_flag_dict()
    with open(os.path.join(FLAGS.save_dir, 'flags.json'), 'w') as f:
        json.dump(flag_dict, f)

    # Set up environments and replay buffer.
    config = FLAGS.agent
    env = make_vec_env(FLAGS.env_name, FLAGS.num_envs)
    eval_env = make_online_env(FLAGS.env_name)

    zeros = np.zeros((FLAGS.num_envs,))
    example_transition = dict(
        observations=env.observation_space.sample(),
        actions=env.action_space.sample(),
        rewards=zeros,
        masks=zeros,
        next_observations=env.observation_space.sample(),
        log_probs=zeros,
    )
    if config['discrete']:
        # Fill with the maximum action to let the agent know the action space size.
        example_transition['actions'] = np.full_like(example_transition['actions'], env.single_action_space.n - 1)

    replay_buffer = ReplayBuffer.create(example_transition, size=FLAGS.train_interval)

    # Initialize agent.
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    agent_class = agents[config['agent_name']]
    agent = agent_class.create(
        FLAGS.seed,
        example_transition['observations'],
        example_transition['actions'],
        config,
    )

    # Restore agent.
    if FLAGS.restore_path is not None:
        agent = restore_agent(agent, FLAGS.restore_path, FLAGS.restore_epoch)

    # Train agent.
    expl_metrics = dict()
    expl_rng = jax.random.PRNGKey(FLAGS.seed)
    obs, _ = env.reset()

    train_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'train.csv'))
    eval_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'eval.csv'))
    first_time = time.time()
    last_time = time.time()
    update_info = None
    for i in tqdm.tqdm(range(1, FLAGS.train_steps + 1), smoothing=0.1, dynamic_ncols=True):
        # Sample transitions.
        expl_rng, key = jax.random.split(expl_rng)
        actions, log_probs = agent.sample_actions(observations=obs, seed=key, info=True)
        if not agent.config['discrete']:
            actions = np.clip(actions, -1.0, 1.0)

        next_obs, rewards, terminateds, truncateds, infos = env.step(actions)

        replay_buffer.add_transition(
            dict(
                observations=obs,
                actions=actions,
                rewards=rewards,
                masks=(~terminateds).astype(np.float64),
                next_observations=next_obs,
                log_probs=log_probs,
            )
        )
        obs = next_obs

        if terminateds[0] or truncateds[0]:
            info = {k: v[0] for k, v in infos.items() if not k.startswith('_')}
            expl_metrics = {f'exploration/{k}': v for k, v in flatten(info).items()}

        # Update agent.
        if i % FLAGS.train_interval == 0:
            traj_batch = replay_buffer.get_subset(slice(0, FLAGS.train_interval))
            agent, update_info = agent.train(traj_batch)
            replay_buffer.clear()

        # Log metrics.
        if i % FLAGS.log_interval == 0 and update_info is not None:
            train_metrics = {f'training/{k}': v for k, v in update_info.items()}
            train_metrics['time/epoch_time'] = (time.time() - last_time) / FLAGS.log_interval
            train_metrics['time/total_time'] = time.time() - first_time
            train_metrics.update(expl_metrics)
            last_time = time.time()
            wandb.log(train_metrics, step=i)
            train_logger.log(train_metrics, step=i)

        # Evaluate agent.
        if i % FLAGS.eval_interval == 0:
            eval_metrics = {}
            eval_info, trajs, renders = evaluate(
                agent=agent,
                env=eval_env,
                task_id=None,
                config=config,
                num_eval_episodes=FLAGS.eval_episodes,
                num_video_episodes=FLAGS.video_episodes,
                video_frame_skip=FLAGS.video_frame_skip,
                eval_temperature=FLAGS.eval_temperature,
                eval_gaussian=FLAGS.eval_gaussian,
            )
            eval_metrics.update({f'evaluation/{k}': v for k, v in eval_info.items()})

            if FLAGS.video_episodes > 0:
                video = get_wandb_video(renders=renders)
                eval_metrics['video'] = video

            traj_image = visualize_trajs(FLAGS.env_name, trajs)
            if traj_image is not None:
                eval_metrics['traj'] = wandb.Image(traj_image)

            wandb.log(eval_metrics, step=i)
            eval_logger.log(eval_metrics, step=i)

        # Save agent.
        if i % FLAGS.save_interval == 0:
            save_agent(agent, FLAGS.save_dir, i)

    train_logger.close()
    eval_logger.close()


if __name__ == '__main__':
    app.run(main)
