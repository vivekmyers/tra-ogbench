import glob
import json
import os
import pickle
import random
import time
from collections import defaultdict
from datetime import datetime

import flax
import numpy as np
import tqdm
import wandb
from absl import app, flags
from ml_collections import config_flags

from algos import algos
from envs.env_loader import make_env_and_dataset
from utils.dataset import Dataset, GCDataset
from utils.evaluation import evaluate
from utils.logger import setup_wandb, get_flag_dict, get_wandb_video, CsvLogger

FLAGS = flags.FLAGS

flags.DEFINE_string('run_group', 'Debug', 'Run group')
flags.DEFINE_integer('seed', 0, 'Random seed')
flags.DEFINE_string('env_name', 'antmaze-large-diverse-v2', 'Environment name')
flags.DEFINE_string('dataset_path', None, 'Dataset path')
flags.DEFINE_string('save_dir', 'exp/', 'Save directory')
flags.DEFINE_string('restore_path', None, 'Restore path')
flags.DEFINE_integer('restore_epoch', None, 'Restore epoch')

flags.DEFINE_integer('train_steps', 1000000, 'Number of training steps')
flags.DEFINE_integer('log_interval', 1000, 'Log interval')
flags.DEFINE_integer('eval_interval', 100000, 'Evaluation interval')
flags.DEFINE_integer('save_interval', 100000, 'Save interval')

flags.DEFINE_integer('eval_tasks', None, 'Number of tasks to evaluate (None for all)')
flags.DEFINE_integer('eval_episodes', 50, 'Number of episodes for each task')
flags.DEFINE_float('eval_temperature', 0, 'Evaluation temperature')
flags.DEFINE_float('eval_gaussian', None, 'Evaluation Gaussian noise')
flags.DEFINE_integer('video_episodes', 2, 'Number of video episodes for each task')
flags.DEFINE_integer('video_frame_skip', 3, 'Frame skip for video')

config_flags.DEFINE_config_file('agent', 'algos/gciql.py', lock_config=False)


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

    env, train_dataset, val_dataset = make_env_and_dataset(FLAGS.env_name)

    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    train_dataset = GCDataset(Dataset.create(**train_dataset), config)
    val_dataset = GCDataset(Dataset.create(**val_dataset), config)

    total_steps = FLAGS.train_steps
    example_batch = train_dataset.sample(1)

    agent_class = algos[config.agent_name]
    agent = agent_class.create(
        FLAGS.seed,
        example_batch['observations'],
        example_batch['actions'],
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

    train_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'train.csv'))
    eval_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'eval.csv'))
    first_time = time.time()
    last_time = time.time()
    for i in tqdm.tqdm(range(1, total_steps + 1), smoothing=0.1, dynamic_ncols=True):
        batch = train_dataset.sample(config.batch_size)
        update_info = dict()
        agent, info = agent.update(batch)
        update_info.update(info)

        if i % FLAGS.log_interval == 0:
            train_metrics = {f'training/{k}': v for k, v in update_info.items()}
            if val_dataset is not None:
                val_batch = val_dataset.sample(config.batch_size)
                _, val_info = agent.total_loss(val_batch, grad_params=None)
                train_metrics.update({f'validation/{k}': v for k, v in val_info.items()})
            train_metrics['time/epoch_time'] = (time.time() - last_time) / FLAGS.log_interval
            train_metrics['time/total_time'] = (time.time() - first_time)
            last_time = time.time()
            wandb.log(train_metrics, step=i)
            train_logger.log(train_metrics, step=i)

        if i == 1 or i % FLAGS.eval_interval == 0:
            renders = []
            eval_metrics = {}
            overall_metrics = defaultdict(list)
            num_tasks = FLAGS.eval_tasks if FLAGS.eval_tasks is not None else len(env.task_infos)
            for task_idx in tqdm.trange(num_tasks):
                task_name = env.task_infos[task_idx]['task_name']
                eval_info, trajs, cur_renders = evaluate(
                    agent=agent,
                    env=env,
                    task_idx=task_idx,
                    config=config,
                    num_eval_episodes=FLAGS.eval_episodes,
                    num_video_episodes=FLAGS.video_episodes,
                    video_frame_skip=FLAGS.video_frame_skip,
                    eval_temperature=FLAGS.eval_temperature,
                    eval_gaussian=FLAGS.eval_gaussian,
                )
                renders.extend(cur_renders)
                eval_metrics.update({f'evaluation/{task_name}_{k}': v for k, v in eval_info.items()})
                for k, v in eval_info.items():
                    overall_metrics[k].append(v)
            for k, v in overall_metrics.items():
                eval_metrics[f'evaluation/overall_{k}'] = np.mean(v)

            if FLAGS.video_episodes > 0:
                video = get_wandb_video(renders=renders)
                eval_metrics['video'] = video

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
