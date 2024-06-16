import glob
import json
import os
import pickle
import platform
import time
from datetime import datetime

import flax
import jax
import numpy as np
import tqdm
import wandb
from absl import app, flags
from ml_collections import config_flags

from utils import d4rl_utils
from utils.utils import record_video, CsvLogger
from utils.dataset import Dataset, GCDataset
from utils.evaluation import evaluate
from utils.wandb_utils import setup_wandb, get_flag_dict

if 'mac' in platform.platform():
    pass
else:
    os.environ['MUJOCO_GL'] = 'egl'
    if 'SLURM_STEP_GPUS' in os.environ:
        os.environ['EGL_DEVICE_ID'] = os.environ['SLURM_STEP_GPUS']

FLAGS = flags.FLAGS
flags.DEFINE_string('env_name', 'antmaze-large-diverse-v2', '')
flags.DEFINE_string('dataset_path', None, '')
flags.DEFINE_string('save_dir', 'exp/', '')
flags.DEFINE_string('restore_path', None, '')
flags.DEFINE_integer('restore_epoch', None, '')
flags.DEFINE_string('run_group', 'Debug', '')
flags.DEFINE_integer('seed', 0, '')
flags.DEFINE_integer('eval_episodes', 50, '')
flags.DEFINE_integer('video_episodes', 2, '')
flags.DEFINE_integer('video_frame_skip', 3, '')
flags.DEFINE_integer('log_interval', 1000, '')
flags.DEFINE_integer('eval_interval', 100000, '')
flags.DEFINE_integer('save_interval', 100000, '')
flags.DEFINE_integer('train_steps', 1000000, '')
flags.DEFINE_float('eval_temperature', 0, '')
flags.DEFINE_float('eval_gaussian', None, '')

config_flags.DEFINE_config_file('agent', 'algos/gciql.py', lock_config=False)


def truncate_dataset(dataset, ratio, return_both=False):
    size = dataset.size
    traj_idxs = []
    traj_start = 0
    for i in range(len(dataset['observations'])):
        if dataset['traj_ends'][i] == 1.0:
            traj_idxs.append(np.arange(traj_start, i + 1))
            traj_start = i + 1
    np.random.seed(0)
    traj_idxs = np.random.permutation(traj_idxs)
    np.random.seed(FLAGS.seed)
    new_idxs = []
    num_states = 0
    for idxs in traj_idxs:
        new_idxs.extend(idxs)
        num_states += len(idxs)
        if num_states >= size * ratio:
            break
    trunc_dataset = Dataset(dataset.get_subset(new_idxs))
    if return_both:
        codataset = Dataset(dataset.get_subset(np.setdiff1d(np.arange(size), new_idxs)))
        return trunc_dataset, codataset
    else:
        return trunc_dataset


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
    # Set up logger
    exp_name = get_exp_name()
    setup_wandb(project='ogcrl', group=FLAGS.run_group, name=exp_name)

    FLAGS.save_dir = os.path.join(FLAGS.save_dir, wandb.run.project, FLAGS.run_group, exp_name)
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    flag_dict = get_flag_dict()
    with open(os.path.join(FLAGS.save_dir, 'flags.json'), 'w') as f:
        json.dump(flag_dict, f)

    config = FLAGS.agent

    goal_infos = [{}]
    if 'antmaze' in FLAGS.env_name:
        import d4rl
        env_name = FLAGS.env_name

        env = d4rl_utils.make_env(env_name)

        if FLAGS.dataset_path is not None:
            dataset = d4rl.qlearning_dataset(env, dataset=env.get_dataset(FLAGS.dataset_path))
            # Manually replace dense rewards with sparse rewards
            if 'large' in FLAGS.env_name:
                dataset['rewards'] = (np.linalg.norm(dataset['observations'][:, :2] - np.array([32.75, 24.75]), axis=1) <= 0.5).astype(np.float32)
                dataset['terminals'] = dataset['rewards']
        else:
            dataset = None
        dataset = d4rl_utils.get_dataset(env, FLAGS.env_name, dataset=dataset)
        dataset = dataset.copy({'rewards': dataset['rewards'] - 1.0})

        env.render(mode='rgb_array', width=200, height=200)
        if 'large' in FLAGS.env_name:
            env.viewer.cam.lookat[0] = 18
            env.viewer.cam.lookat[1] = 12
            env.viewer.cam.distance = 50
            env.viewer.cam.elevation = -90
        else:
            env.viewer.cam.lookat[0] = 18
            env.viewer.cam.lookat[1] = 12
            env.viewer.cam.distance = 50
            env.viewer.cam.elevation = -90
    elif 'kitchen' in FLAGS.env_name:
        # HACK: Monkey patching to make it compatible with Python 3.10.
        import collections
        if not hasattr(collections, 'Mapping'):
            collections.Mapping = collections.abc.Mapping

        env = d4rl_utils.make_env(FLAGS.env_name)
        if FLAGS.dataset_path is not None:
            dataset = d4rl_utils.get_dataset(
                env, FLAGS.env_name, dataset=dict(np.load(FLAGS.dataset_path)), filter_terminals=False,
            )
        else:
            dataset = d4rl_utils.get_dataset(env, FLAGS.env_name, filter_terminals=True)
        dataset = dataset.copy({'observations': dataset['observations'][:, :30],
                                'next_observations': dataset['next_observations'][:, :30]})

    train_dataset, val_dataset = truncate_dataset(dataset, 0.95, return_both=True)

    base_observation = jax.tree_util.tree_map(lambda arr: arr[0], dataset['observations'])

    env.reset()

    train_dataset = GCDataset(Dataset.create(**train_dataset), config)
    val_dataset = GCDataset(Dataset.create(**val_dataset), config)

    total_steps = FLAGS.train_steps
    example_batch = dataset.sample(1)
    if config.agent_name == 'gciql':
        from algos.gciql import GCIQLAgent
        agent_class = GCIQLAgent
    else:
        raise NotImplementedError

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
            for goal_info in goal_infos:
                eval_info, trajs, cur_renders = evaluate(
                    agent=agent,
                    env=env if 'env' not in goal_info else goal_info['env'],
                    env_name=FLAGS.env_name,
                    config=config,
                    base_observation=base_observation,
                    num_eval_episodes=FLAGS.eval_episodes,
                    num_video_episodes=FLAGS.video_episodes,
                    video_frame_skip=FLAGS.video_frame_skip,
                    eval_temperature=FLAGS.eval_temperature,
                    eval_gaussian=FLAGS.eval_gaussian,
                )
                renders.extend(cur_renders)
                eval_metrics.update({f'evaluation/{k}': v for k, v in eval_info.items()})

            if FLAGS.video_episodes > 0:
                video = record_video('Video', i, renders=renders)
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
