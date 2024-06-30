import glob
import json
import os
import pickle
import random
import time
from datetime import datetime

import jax
import flax
import numpy as np
import tqdm
import wandb
from absl import app, flags
from ml_collections import config_flags

from algos import algos
from envs.env_loader import make_online_env
from envs.viz_utils import visualize_trajs
from utils.dataset import ReplayBuffer
from utils.evaluation import evaluate, flatten
from utils.logger import setup_wandb, get_flag_dict, get_wandb_video, CsvLogger

FLAGS = flags.FLAGS

flags.DEFINE_string('run_group', 'Debug', 'Run group')
flags.DEFINE_integer('seed', 0, 'Random seed')
flags.DEFINE_string('env_name', 'antmaze-large-diverse-v2', 'Environment name')
flags.DEFINE_string('dataset_path', None, 'Dataset path')
flags.DEFINE_string('save_dir', 'exp/', 'Save directory')
flags.DEFINE_string('restore_path', None, 'Restore path')
flags.DEFINE_integer('restore_epoch', None, 'Restore epoch')

flags.DEFINE_integer('seed_steps', 10000, 'Number of seed steps')
flags.DEFINE_integer('train_steps', 1000000, 'Number of training steps')
flags.DEFINE_integer('train_interval', 1, 'Train interval')
flags.DEFINE_integer('log_interval', 1000, 'Log interval')
flags.DEFINE_integer('eval_interval', 100000, 'Evaluation interval')
flags.DEFINE_integer('save_interval', 100000, 'Save interval')
flags.DEFINE_integer('terminate_at_end', 0, 'Whether to set terminated=True when truncated=True')

flags.DEFINE_integer('eval_tasks', None, 'Number of tasks to evaluate (None for all)')
flags.DEFINE_integer('eval_episodes', 50, 'Number of episodes for each task')
flags.DEFINE_float('eval_temperature', 0, 'Evaluation temperature')
flags.DEFINE_float('eval_gaussian', None, 'Evaluation Gaussian noise')
flags.DEFINE_integer('video_episodes', 2, 'Number of video episodes for each task')
flags.DEFINE_integer('video_frame_skip', 3, 'Frame skip for video')
flags.DEFINE_integer('eval_on_cpu', 1, 'Whether to evaluate on CPU')

config_flags.DEFINE_config_file('agent', 'algos/sac.py', lock_config=False)


def get_exp_name():
    exp_name = ''
    exp_name += f'sd{FLAGS.seed:03d}_'
    if 'SLURM_JOB_ID' in os.environ:
        exp_name += f's_{os.environ["SLURM_JOB_ID"]}.'
    if 'SLURM_PROCID' in os.environ:
        exp_name += f'{os.environ["SLURM_PROCID"]}.'
    exp_name += f'{datetime.now().strftime("%Y%m%d_%H%M%S")}'

    return exp_name


def main(_):
    exp_name = get_exp_name()
    setup_wandb(project='ogcrl', group=FLAGS.run_group, name=exp_name)

    FLAGS.save_dir = os.path.join(FLAGS.save_dir, wandb.run.project, FLAGS.run_group, exp_name)
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    flag_dict = get_flag_dict()
    with open(os.path.join(FLAGS.save_dir, 'flags.json'), 'w') as f:
        json.dump(flag_dict, f)

    config = FLAGS.agent

    env = make_online_env(FLAGS.env_name, eval=False)
    eval_env = make_online_env(FLAGS.env_name, eval=True)

    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    example_transition = dict(
        observations=env.observation_space.sample(),
        actions=env.action_space.sample(),
        rewards=0.0,
        masks=1.0,
        next_observations=env.observation_space.sample(),
    )

    replay_buffer = ReplayBuffer.create(example_transition, size=int(1e6))

    agent_class = algos[config.agent_name]
    agent = agent_class.create(
        FLAGS.seed,
        example_transition['observations'],
        example_transition['actions'],
        config,
    )

    if FLAGS.restore_path is not None:
        restore_path = FLAGS.restore_path
        candidates = glob.glob(restore_path)
        if len(candidates) == 0:
            raise Exception(f'Path does not exist: {restore_path}')
        if len(candidates) > 1:
            raise Exception(f'Multiple matching paths exist for: {restore_path}')
        if FLAGS.restore_epoch is None:
            restore_path = candidates[0] + '/params.pkl'
        else:
            restore_path = candidates[0] + f'/params_{FLAGS.restore_epoch}.pkl'
        with open(restore_path, 'rb') as f:
            load_dict = pickle.load(f)
        agent = flax.serialization.from_state_dict(agent, load_dict['agent'])
        print(f'Restored from {restore_path}')

    expl_metrics = dict()
    expl_rng = jax.random.PRNGKey(FLAGS.seed)
    ob, _ = env.reset()

    train_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'train.csv'))
    eval_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'eval.csv'))
    first_time = time.time()
    last_time = time.time()
    update_info = None
    for i in tqdm.tqdm(range(1, FLAGS.train_steps + 1), smoothing=0.1, dynamic_ncols=True):
        if i < FLAGS.seed_steps:
            action = env.action_space.sample()
        else:
            expl_rng, key = jax.random.split(expl_rng)
            action = agent.sample_actions(observations=ob, seed=key)

        next_ob, reward, terminated, truncated, info = env.step(action)
        if FLAGS.terminate_at_end and truncated:
            terminated = True

        replay_buffer.add_transition(dict(
            observations=ob,
            actions=action,
            rewards=reward,
            masks=float(not terminated),
            next_observations=next_ob,
        ))
        ob = next_ob

        if terminated or truncated:
            expl_metrics = {f'exploration/{k}': v for k, v in flatten(info).items()}
            ob, _ = env.reset()

        if replay_buffer.size < FLAGS.seed_steps:
            continue

        if i % FLAGS.train_interval == 0:
            batch = replay_buffer.sample(config.batch_size)
            agent, update_info = agent.update(batch)

        if i % FLAGS.log_interval == 0 and update_info is not None:
            train_metrics = {f'training/{k}': v for k, v in update_info.items()}
            train_metrics['time/epoch_time'] = (time.time() - last_time) / FLAGS.log_interval
            train_metrics['time/total_time'] = (time.time() - first_time)
            train_metrics.update(expl_metrics)
            last_time = time.time()
            wandb.log(train_metrics, step=i)
            train_logger.log(train_metrics, step=i)

        if i % FLAGS.eval_interval == 0:
            if FLAGS.eval_on_cpu:
                eval_agent = jax.device_put(agent, device=jax.devices('cpu')[0])
            else:
                eval_agent = agent
            eval_metrics = {}
            eval_info, trajs, renders = evaluate(
                agent=eval_agent,
                env=eval_env,
                task_idx=None,
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

        if i % FLAGS.save_interval == 0:
            save_dict = dict(
                agent=flax.serialization.to_state_dict(agent),
            )

            fname = os.path.join(FLAGS.save_dir, f'params_{i}.pkl')
            print(f'Saving to {fname}')
            with open(fname, 'wb') as f:
                pickle.dump(save_dict, f)
    train_logger.close()
    eval_logger.close()


if __name__ == '__main__':
    app.run(main)
