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
from utils.evaluation import evaluate_with_trajectories
from utils.wandb import setup_wandb, default_wandb_config, get_flag_dict

if 'mac' in platform.platform():
    pass
else:
    os.environ['MUJOCO_GL'] = 'egl'
    if 'SLURM_STEP_GPUS' in os.environ:
        os.environ['EGL_DEVICE_ID'] = os.environ['SLURM_STEP_GPUS']

FLAGS = flags.FLAGS
flags.DEFINE_string('env_name', 'antmaze-large-diverse-v2', '')
flags.DEFINE_string('dataset_path', None, '')
flags.DEFINE_string('agent_name', 'gciql', '')
flags.DEFINE_integer('dataset_size', 10000000, '')  # Only for DMC
flags.DEFINE_string('expl_agent', 'rnd', '')  # Only for DMC
flags.DEFINE_string('task_name', None, '')  # Only for DMC
flags.DEFINE_string('save_dir', 'exp/', '')
flags.DEFINE_string('restore_path', None, '')
flags.DEFINE_integer('restore_epoch', None, '')
flags.DEFINE_string('run_group', 'Debug', '')
flags.DEFINE_integer('seed', 0, '')
flags.DEFINE_integer('eval_episodes', 50, '')
flags.DEFINE_integer('num_video_episodes', 2, '')
flags.DEFINE_integer('log_interval', 1000, '')
flags.DEFINE_integer('eval_interval', 100000, '')
flags.DEFINE_integer('save_interval', 100000, '')
flags.DEFINE_integer('batch_size', 1024, '')
flags.DEFINE_integer('train_steps', 1000000, '')

flags.DEFINE_float('lr', 3e-4, '')
flags.DEFINE_integer('value_hidden_dim', 512, '')
flags.DEFINE_integer('value_num_layers', 3, '')
flags.DEFINE_integer('actor_hidden_dim', 256, '')
flags.DEFINE_integer('actor_num_layers', 2, '')
flags.DEFINE_float('discount', 0.99, '')
flags.DEFINE_float('tau', 0.005, '')
flags.DEFINE_float('expectile', 0.9, '')
flags.DEFINE_integer('layer_norm', 1, '')

flags.DEFINE_string('value_type', 'mono', '')  # ['mono', 'bilinear', 'critic_bilinear']
flags.DEFINE_string('actor_type', 'mono', '')  # ['mono', 'bilinear']
flags.DEFINE_integer('latent_dim', 512, '')

flags.DEFINE_float('value_p_curgoal', 0.2, '')
flags.DEFINE_float('value_p_trajgoal', 0.5, '')
flags.DEFINE_float('value_p_randomgoal', 0.3, '')
flags.DEFINE_integer('value_geom_sample', 1, '')
flags.DEFINE_float('policy_p_randomgoal', 0., '')
flags.DEFINE_integer('policy_geom_sample', 0, '')
flags.DEFINE_float('temperature', 1, '')
flags.DEFINE_float('eval_temperature', 0, '')
flags.DEFINE_float('eval_gaussian', None, '')
flags.DEFINE_integer('sfbc_samples', None, '')

flags.DEFINE_float('gc_negative', 1, '')
flags.DEFINE_float('value_only', 1, '')
flags.DEFINE_integer('const_std', 0, '')
flags.DEFINE_string('actor_loss_type', 'awr', '')  # ['awr', 'ddpg']
flags.DEFINE_integer('value_exp', 0, '')  # Exp parameterization
flags.DEFINE_integer('use_log_q', 0, '')  # log Q for advantage

flags.DEFINE_string('encoder', None, '')
flags.DEFINE_float('p_aug', None, '')

config_flags.DEFINE_config_dict('wandb', default_wandb_config(), lock_config=False)


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


def main(_):
    g_start_time = int(datetime.now().timestamp())

    exp_name = ''
    exp_name += f'sd{FLAGS.seed:03d}_'
    if 'SLURM_JOB_ID' in os.environ:
        exp_name += f's_{os.environ["SLURM_JOB_ID"]}.'
    if 'SLURM_PROCID' in os.environ:
        exp_name += f'{os.environ["SLURM_PROCID"]}.'
    exp_name += f'{g_start_time}'
    exp_name += f'_{FLAGS.wandb["name"]}'

    # Create wandb logger
    FLAGS.wandb['project'] = 'ogcrl'
    FLAGS.wandb['name'] = FLAGS.wandb['exp_descriptor'] = exp_name
    FLAGS.wandb['group'] = FLAGS.wandb['exp_prefix'] = FLAGS.run_group
    setup_wandb(dict(), **FLAGS.wandb)

    FLAGS.save_dir = os.path.join(FLAGS.save_dir, wandb.run.project, wandb.config.exp_prefix,
                                  wandb.config.experiment_id)
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    flag_dict = get_flag_dict()
    with open(os.path.join(FLAGS.save_dir, 'flags.json'), 'w') as f:
        json.dump(flag_dict, f)

    goal_infos = [{}]
    if 'antmaze' in FLAGS.env_name:
        import d4rl
        env_name = FLAGS.env_name

        env = d4rl_utils.make_env(env_name)

        if FLAGS.dataset_path is not None:
            dataset = d4rl.qlearning_dataset(env, dataset=env.get_dataset(FLAGS.dataset_path))
            # Manually replace dense rewards with sparse rewards
            if 'large' in FLAGS.env_name:
                dataset['rewards'] = (np.linalg.norm(dataset['observations'][:, :2] - np.array([32.75, 24.75]),
                                                     axis=1) <= 0.5).astype(np.float32)
                dataset['terminals'] = dataset['rewards']
        else:
            dataset = None
        dataset = d4rl_utils.get_dataset(env, FLAGS.env_name, goal_conditioned=True, dataset=dataset)
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
                env, FLAGS.env_name, dataset=dict(np.load(FLAGS.dataset_path)),
                goal_conditioned=True, filter_terminals=False,
            )
        else:
            dataset = d4rl_utils.get_dataset(env, FLAGS.env_name, goal_conditioned=True, filter_terminals=True)
        dataset = dataset.copy({'observations': dataset['observations'][:, :30],
                                'next_observations': dataset['next_observations'][:, :30]})

    train_dataset, val_dataset = truncate_dataset(dataset, 0.95, return_both=True)

    base_observation = jax.tree_util.tree_map(lambda arr: arr[0], dataset['observations'])

    env.reset()

    gcdataset_config = dict(
        value_p_curgoal=FLAGS.value_p_curgoal,
        value_p_trajgoal=FLAGS.value_p_trajgoal,
        value_p_randomgoal=FLAGS.value_p_randomgoal,
        value_geom_sample=FLAGS.value_geom_sample,
        policy_p_curgoal=0.,
        policy_p_trajgoal=1 - FLAGS.policy_p_randomgoal,
        policy_p_randomgoal=FLAGS.policy_p_randomgoal,
        policy_geom_sample=FLAGS.policy_geom_sample,
        discount=FLAGS.discount,
        p_aug=FLAGS.p_aug,
    )
    train_dataset = GCDataset(Dataset.create(**train_dataset), **gcdataset_config)
    val_dataset = GCDataset(Dataset.create(**val_dataset), **gcdataset_config)

    total_steps = FLAGS.train_steps
    example_batch = dataset.sample(1)
    if FLAGS.agent_name == 'gciql':
        from algos import gciql as learner
    else:
        raise NotImplementedError

    config = dict(
        lr=FLAGS.lr,
        value_hidden_dims=(FLAGS.value_hidden_dim,) * FLAGS.value_num_layers,
        actor_hidden_dims=(FLAGS.actor_hidden_dim,) * FLAGS.actor_num_layers,
        discount=FLAGS.discount,
        tau=FLAGS.tau,
        temperature=FLAGS.temperature,
        expectile=FLAGS.expectile,
        layer_norm=FLAGS.layer_norm,
        value_type=FLAGS.value_type,
        actor_type=FLAGS.actor_type,
        latent_dim=FLAGS.latent_dim,
        gc_negative=FLAGS.gc_negative,
        value_only=FLAGS.value_only,
        const_std=FLAGS.const_std,
        actor_loss_type=FLAGS.actor_loss_type,
        value_exp=FLAGS.value_exp,
        use_log_q=FLAGS.use_log_q,
        encoder=FLAGS.encoder,
    )
    agent = learner.create_learner(
        FLAGS.seed,
        example_batch['observations'],
        example_batch['actions'],
        **config,
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
        with open(restore_path, "rb") as f:
            load_dict = pickle.load(f)
        agent = flax.serialization.from_state_dict(agent, load_dict['agent'])
        print(f'Restored from {restore_path}')

    train_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'train.csv'))
    eval_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'eval.csv'))
    first_time = time.time()
    last_time = time.time()
    for i in tqdm.tqdm(range(1, total_steps + 1),
                       smoothing=0.1,
                       dynamic_ncols=True):
        batch = train_dataset.sample(FLAGS.batch_size)
        update_info = dict()
        agent, info = agent.update(batch)
        update_info.update(info)
        if not FLAGS.value_only and FLAGS.agent_name == 'gciql':
            agent, info = agent.update_q(batch)
            update_info.update(info)

        if i % FLAGS.log_interval == 0:
            train_metrics = {f'training/{k}': v for k, v in update_info.items()}
            if val_dataset is not None and hasattr(agent, 'get_loss_info'):
                val_batch = val_dataset.sample(FLAGS.batch_size)
                val_info = agent.get_loss_info(val_batch)
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
                eval_info, trajs, cur_renders = evaluate_with_trajectories(
                    agent, env if 'env' not in goal_info else goal_info['env'], goal_info=goal_info,
                    env_name=FLAGS.env_name, num_episodes=FLAGS.eval_episodes,
                    base_observation=base_observation, num_video_episodes=FLAGS.num_video_episodes,
                    eval_temperature=FLAGS.eval_temperature, eval_gaussian=FLAGS.eval_gaussian,
                    sfbc_samples=FLAGS.sfbc_samples,
                    config=config,
                )
                renders.extend(cur_renders)
                eval_metrics.update({f'evaluation/{k}': v for k, v in eval_info.items()})

            if FLAGS.num_video_episodes > 0:
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
            with open(fname, "wb") as f:
                pickle.dump(save_dict, f)
    train_logger.close()
    eval_logger.close()


if __name__ == '__main__':
    app.run(main)
