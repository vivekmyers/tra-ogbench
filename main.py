import glob
import os
import platform
import pickle
import time
from collections import defaultdict
from datetime import datetime
from functools import partial
from pathlib import Path

import flax
import jax
import jax.numpy as jnp
import numpy as np
import tqdm
import wandb
import json
from absl import app, flags
from ml_collections import config_flags

from jaxrl_m.dataset import Dataset
from jaxrl_m.evaluation import supply_rng, evaluate_with_trajectories, EpisodeMonitor
from jaxrl_m.wandb import setup_wandb, default_wandb_config, get_flag_dict
from src import d4rl_utils, d4rl_ant, ant_diagnostics, viz_utils
from src.d4rl_utils import normalize_dataset
from src.dataset_utils import GCDataset, merge_datasets
from src.utils import record_video, CsvLogger

if 'mac' in platform.platform():
    pass
else:
    os.environ['MUJOCO_GL'] = 'egl'
    if 'SLURM_STEP_GPUS' in os.environ:
        os.environ['EGL_DEVICE_ID'] = os.environ['SLURM_STEP_GPUS']

FLAGS = flags.FLAGS
flags.DEFINE_string('env_name', 'antmaze-large-diverse-v2', '')
flags.DEFINE_string('dataset_path', None, '')
flags.DEFINE_string('agent_name', 'trl', '')
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

flags.DEFINE_float('p_randomgoal', 0.3, '')
flags.DEFINE_float('p_trajgoal', 0.5, '')
flags.DEFINE_float('p_currgoal', 0.2, '')
flags.DEFINE_float('policy_p_randomgoal', 0., '')
flags.DEFINE_integer('geom_sample', 1, '')
flags.DEFINE_integer('policy_geom_sample', 0, '')
flags.DEFINE_float('geom_discount', None, '')
flags.DEFINE_string('temperature', '1', '')
flags.DEFINE_float('goal_conditioned', 1, '')
flags.DEFINE_float('eval_temperature', 0, '')
flags.DEFINE_float('eval_gaussian', None, '')
flags.DEFINE_string('sfbc_samples', None, '')

flags.DEFINE_string('value_algo', 'iql', '')  # ['iql', 'crl']
flags.DEFINE_float('gc_negative', 1, '')
flags.DEFINE_float('value_only', 1, '')
flags.DEFINE_integer('const_std', 0, '')
flags.DEFINE_string('actor_loss_type', 'awr', '')  # ['awr', 'ddpg']
flags.DEFINE_integer('ddqn_trick', 0, '')  # Only used when value_only is 0
flags.DEFINE_integer('use_target_v', 0, '')  # Only used when value_only is 0
flags.DEFINE_integer('value_exp', 0, '')  # Exp parameterization
flags.DEFINE_integer('use_log_q', 0, '')  # log Q for advantage
flags.DEFINE_string('dual_type', 'none', '')  # ['none', 'scalar']
flags.DEFINE_integer('ddpg_tanh', 0, '')
flags.DEFINE_integer('tanh_squash', 0, '')

flags.DEFINE_string('encoder', None, '')
flags.DEFINE_float('p_aug', None, '')
flags.DEFINE_integer('color_aug', 0, '')

flags.DEFINE_float('value_data_ratio', None, '')
flags.DEFINE_float('actor_data_ratio', None, '')

flags.DEFINE_float('action_grad_coef', None, '')
flags.DEFINE_integer('action_grad_normalize', 0, '')

flags.DEFINE_string('algo_name', None, '')  # Not used, only for logging

config_flags.DEFINE_config_dict('wandb', default_wandb_config(), lock_config=False)


@jax.jit
def get_traj_v(agent, trajectory):
    def get_v(s, g):
        v = agent.network(jax.tree_map(lambda x: x[None], s), jax.tree_map(lambda x: x[None], g), method='value')
        if FLAGS.value_only:
            v1, v2 = v
            return (v1 + v2) / 2
        else:
            return v

    observations = trajectory['observations']
    all_values = jax.vmap(jax.vmap(get_v, in_axes=(None, 0)), in_axes=(0, None))(observations, observations)
    return {
        'dist_to_beginning': all_values[:, 0],
        'dist_to_end': all_values[:, -1],
        'dist_to_middle': all_values[:, all_values.shape[1] // 2],
    }


@jax.jit
def get_v_goal(agent, goal, observations):
    goal = jnp.tile(goal, (observations.shape[0], 1))
    v = agent.network(observations, goal, method='value')
    if FLAGS.value_only:
        v1, v2 = v
        return (v1 + v2) / 2
    else:
        return v


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

    FLAGS.save_dir = os.path.join(FLAGS.save_dir, wandb.run.project, wandb.config.exp_prefix, wandb.config.experiment_id)
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    flag_dict = get_flag_dict()
    with open(os.path.join(FLAGS.save_dir, 'flags.json'), 'w') as f:
        json.dump(flag_dict, f)

    goal_infos = [{}]
    if 'antmaze' in FLAGS.env_name:
        import d4rl
        env_name = FLAGS.env_name

        if 'ultra' in FLAGS.env_name:
            import gym
            import d4rl_ext  # noqa
            env = gym.make(env_name)
            env = EpisodeMonitor(env)
        else:
            env = d4rl_utils.make_env(env_name)

        if FLAGS.dataset_path is not None:
            dataset = d4rl.qlearning_dataset(env, dataset=env.get_dataset(FLAGS.dataset_path))
            # Manually replace dense rewards with sparse rewards
            if 'large' in FLAGS.env_name:
                dataset['rewards'] = (np.linalg.norm(dataset['observations'][:, :2] - np.array([32.75, 24.75]), axis=1) <= 0.5).astype(np.float32)
                dataset['terminals'] = dataset['rewards']
        else:
            dataset = None
        dataset = d4rl_utils.get_dataset(env, FLAGS.env_name, goal_conditioned=FLAGS.goal_conditioned, dataset=dataset)
        dataset = dataset.copy({'rewards': dataset['rewards'] - 1.0})

        env.render(mode='rgb_array', width=200, height=200)
        if 'large' in FLAGS.env_name:
            env.viewer.cam.lookat[0] = 18
            env.viewer.cam.lookat[1] = 12
            env.viewer.cam.distance = 50
            env.viewer.cam.elevation = -90

            viz_env, viz_dataset = d4rl_ant.get_env_and_dataset(env_name)
            viz = ant_diagnostics.Visualizer(env_name, viz_env, viz_dataset, discount=FLAGS.discount)
            init_state = np.copy(viz_dataset['observations'][0])
            init_state[:2] = (12.5, 8)
        elif 'ultra' in FLAGS.env_name:
            env.viewer.cam.lookat[0] = 26
            env.viewer.cam.lookat[1] = 18
            env.viewer.cam.distance = 70
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
                goal_conditioned=FLAGS.goal_conditioned, filter_terminals=False,
            )
        else:
            dataset = d4rl_utils.get_dataset(env, FLAGS.env_name, goal_conditioned=FLAGS.goal_conditioned, filter_terminals=True)
        dataset = dataset.copy({'observations': dataset['observations'][:, :30],
                                'next_observations': dataset['next_observations'][:, :30]})
    elif 'halfcheetah' in FLAGS.env_name or 'hopper' in FLAGS.env_name or 'walker2d' in FLAGS.env_name:
        import d4rl.gym_mujoco
        env = d4rl_utils.make_env(FLAGS.env_name)
        dataset = d4rl_utils.get_dataset(env, FLAGS.env_name, goal_conditioned=FLAGS.goal_conditioned)
        dataset = normalize_dataset(FLAGS.env_name, dataset)
    elif 'pen' in FLAGS.env_name or 'hammer' in FLAGS.env_name or 'door' in FLAGS.env_name or 'relocate' in FLAGS.env_name:
        import d4rl.hand_manipulation_suite
        import mujoco_py
        env = d4rl_utils.make_env(FLAGS.env_name)
        if FLAGS.dataset_path is not None:
            dataset = d4rl.qlearning_dataset(env, dataset=env.get_dataset(FLAGS.dataset_path))
        else:
            dataset = None
        dataset = d4rl_utils.get_dataset(env, FLAGS.env_name, goal_conditioned=FLAGS.goal_conditioned, dataset=dataset)
        # HACK: For rendering
        env.viewer = mujoco_py.MjRenderContextOffscreen(env.sim, device_id=-1)
    elif 'roboverse' in FLAGS.env_name:
        import sys
        sys.path.append('../bullet-manipulation-cz')
        sys.path.append('../bullet-manipulation-cz/roboverse/envs/assets/bullet-objects')

        if '270k' in FLAGS.env_name:
            dataset = dict(np.load(f'data/roboverse/data270k.npz'))
        elif '1080k' in FLAGS.env_name:
            dataset = dict(np.load(f'data/roboverse/data1080k.npz'))
        elif '1125k' in FLAGS.env_name:
            dataset = dict(np.load(f'data/roboverse/data1125k.npz'))
        else:
            raise NotImplementedError
        dataset['terminals'][:] = 0
        dataset['terminals'][299::300] = 1
        dataset = d4rl_utils.get_dataset(None, FLAGS.env_name, goal_conditioned=FLAGS.goal_conditioned, dataset=dataset, obs_dtype=dataset['observations'].dtype, filter_terminals=False)

        goal_infos = []
        for i, seed_id in enumerate([50, 12, 37, 31, 14]):
            import roboverse
            from roboverse.utils.renderer import EnvRenderer, InsertImageEnv
            from eval_scripts.drawer_pnp_push_commands import drawer_pnp_push_commands
            from gym.wrappers import ClipAction
            goal_pkl = glob.glob(f'../bullet-manipulation-cz/goals/*seed{seed_id}.pkl')[0]
            with open(goal_pkl, 'rb') as f:
                goal_info = pickle.load(f)
            goal_name = {
                12: 'pick_and_place_table',
                14: 'push_block_close_drawer',
                31: 'push_block_open_drawer',
                37: 'pick_and_place_drawer',
                50: 'drawer',
            }[seed_id]
            raw_env = roboverse.make(
                "SawyerRigAffordances-v6",
                gui=False,
                expl=False,
                env_obs_img_dim=196,
                obs_img_dim=48,
                test_env=True,
                test_env_command=drawer_pnp_push_commands[seed_id],
                downsample=True,
            )
            state_env = ClipAction(raw_env)
            renderer = EnvRenderer(
                create_image_format='HWC',
                output_image_format='CWH',
                flatten_image=True,
                width=48,
                height=48,
            )
            env = InsertImageEnv(state_env, renderer=renderer)
            env.reset()
            goals = [(goal.reshape(3, 48, 48).transpose() * 255).astype(np.uint8) for goal in
                     goal_info['image_desired_goal']]
            goal_infos.append({
                'env': env,
                'seed_id': seed_id,
                'goal_name': goal_name,
                'goals': goals,
                'state_desired_goals': goal_info['state_desired_goal'],
            })
    elif 'calvin' in FLAGS.env_name:
        import hydra
        from calvin_env.envs.play_table_env import get_env
        from omegaconf import OmegaConf
        env = get_env(Path('../calvin-my/env_settings/small'), show_gui=False)
        dataset_path = 'data/calvin_visual/calvin_D_64.npz' if FLAGS.dataset_path is None else FLAGS.dataset_path
        if 'state' in FLAGS.env_name:
            dataset = np.load(dataset_path)
            dataset = {k: dataset[k] for k in dataset if 'image' not in k}

            dataset['next_observations'] = dataset['states'][1:]
            dataset['observations'] = dataset.pop('states')[:-1]
        else:
            dataset = dict(np.load(dataset_path))

            dataset['next_observations'] = dataset['images'][1:]
            dataset['next_states'] = dataset['states'][1:]
            dataset['observations'] = dataset.pop('images')[:-1]
            dataset['states'] = dataset.pop('states')[:-1]
        dataset['actions'] = dataset['actions'][:-1]
        dataset['rewards'] = dataset['rewards'][:-1]
        dataset['terminals'] = dataset['terminals'][:-1]
        dataset = d4rl_utils.get_dataset(None, FLAGS.env_name, goal_conditioned=FLAGS.goal_conditioned, dataset=dataset, obs_dtype=dataset['observations'].dtype, filter_terminals=True)

        goal_infos = []
        conf_dir = Path('../calvin-my/calvin_models/conf')
        task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
        task_oracle = hydra.utils.instantiate(task_cfg)

        task_state_assignments_f = open("data/calvin_visual/task_state_assignments.json")
        task_state_assignments = json.load(task_state_assignments_f)
        task_state_assignments_f.close()

        filtered_subtasks_f = open("data/calvin_visual/filtered_subtasks.json")
        filtered_subtasks = json.load(filtered_subtasks_f)
        filtered_subtasks_f.close()

        for task_id in range(8):
            task = filtered_subtasks[task_id]
            start_goal_pairs = task_state_assignments[task]

            goal_infos.append({
                'task_id': task_id,
                'goal_name': task,
                'start_goal_pairs': start_goal_pairs,
                'task_oracle': task_oracle,
            })
    elif 'dmc' in FLAGS.env_name:
        from src.envs import dmc
        from src.envs.dmc import DMCWrapper
        _, env_name, task_name = FLAGS.env_name.split('_')
        env = dmc.make(f'{env_name}_{task_name}', obs_type='states', frame_stack=1, action_repeat=1, seed=FLAGS.seed)
        env = DMCWrapper(env, FLAGS.seed)
        env.reset()

        path = f'data/exorl/datasets/{env_name}/{FLAGS.expl_agent}/buffer'
        npzs = sorted(glob.glob(f'{path}/*.npz'))
        dataset = defaultdict(list)
        num_steps = 0
        for i, npz in tqdm.tqdm(enumerate(npzs)):
            traj_data = dict(np.load(npz))
            dataset['observations'].append(traj_data['observation'][:-1, :])
            dataset['next_observations'].append(traj_data['observation'][1:, :])
            dataset['actions'].append(traj_data['action'][1:, :])
            if FLAGS.task_name is not None:
                rewards = []
                reward_spec = env.reward_spec()
                states = traj_data['physics']
                for k in range(states.shape[0]):
                    with env.physics.reset_context():
                        env.physics.set_state(states[k])
                    reward = env.task.get_reward(env.physics)
                    reward = np.full(reward_spec.shape, reward, reward_spec.dtype)
                    rewards.append(reward)
                traj_data['reward'] = np.array(rewards, dtype=reward_spec.dtype)
                dataset['rewards'].append(traj_data['reward'][1:])
            else:
                dataset['rewards'].append(traj_data['reward'][1:, 0])
            terminals = np.full((len(traj_data['observation']) - 1,), False)
            if FLAGS.goal_conditioned:
                terminals[-1] = True
            dataset['terminals'].append(terminals)
            num_steps += len(traj_data['observation']) - 1
            if num_steps >= FLAGS.dataset_size:
                break
        for k, v in dataset.items():
            dataset[k] = np.concatenate(v, axis=0)
        dataset = d4rl_utils.get_dataset(None, FLAGS.env_name, goal_conditioned=FLAGS.goal_conditioned, dataset=dataset)

    train_dataset, val_dataset = truncate_dataset(dataset, 0.95, return_both=True)
    if FLAGS.value_data_ratio is not None:
        value_dataset = truncate_dataset(train_dataset, FLAGS.value_data_ratio)
    else:
        value_dataset = train_dataset
    if FLAGS.actor_data_ratio is not None:
        actor_dataset = truncate_dataset(train_dataset, FLAGS.actor_data_ratio)
    else:
        actor_dataset = train_dataset

    base_observation = jax.tree_map(lambda arr: arr[0], dataset['observations'])

    env.reset()

    if FLAGS.goal_conditioned:
        gcdataset_config = dict(
            p_currgoal=FLAGS.p_currgoal,
            p_trajgoal=FLAGS.p_trajgoal,
            p_randomgoal=FLAGS.p_randomgoal,
            policy_p_randomgoal=FLAGS.policy_p_randomgoal,
            geom_sample=FLAGS.geom_sample,
            policy_geom_sample=FLAGS.policy_geom_sample,
            discount=FLAGS.geom_discount if FLAGS.geom_discount is not None else FLAGS.discount,
            p_aug=FLAGS.p_aug,
            color_aug=FLAGS.color_aug,
        )
        train_dataset = GCDataset(Dataset.create(**train_dataset), **gcdataset_config)
        val_dataset = GCDataset(Dataset.create(**val_dataset), **gcdataset_config)
        value_dataset = GCDataset(Dataset.create(**value_dataset), **gcdataset_config) if FLAGS.value_data_ratio is not None else train_dataset
        actor_dataset = GCDataset(Dataset.create(**actor_dataset), **gcdataset_config) if FLAGS.actor_data_ratio is not None else train_dataset
    else:
        train_dataset = Dataset.create(**train_dataset)
        val_dataset = Dataset.create(**val_dataset)
        value_dataset = Dataset.create(**value_dataset) if FLAGS.value_data_ratio is not None else train_dataset
        actor_dataset = Dataset.create(**actor_dataset) if FLAGS.actor_data_ratio is not None else train_dataset

    FLAGS.temperature = [float(x) for x in FLAGS.temperature.split(',')]
    if FLAGS.sfbc_samples is not None:
        FLAGS.sfbc_samples = [int(x) for x in FLAGS.sfbc_samples.split(',')]
        assert len(FLAGS.temperature) == 1

    total_steps = FLAGS.train_steps
    example_batch = dataset.sample(1)
    if FLAGS.agent_name == 'trl':
        from src.agents import trl as learner
    elif FLAGS.agent_name == 'gciql':
        from src.agents import gciql as learner
    elif FLAGS.agent_name == 'iql':
        from src.agents import iql as learner
    elif FLAGS.agent_name == 'viql':
        from src.agents import viql as learner
    else:
        raise NotImplementedError

    config = dict(
        lr=FLAGS.lr,
        value_hidden_dims=(FLAGS.value_hidden_dim,) * FLAGS.value_num_layers,
        actor_hidden_dims=(FLAGS.actor_hidden_dim,) * FLAGS.actor_num_layers,
        discount=FLAGS.discount,
        tau=FLAGS.tau,
        temperatures=FLAGS.temperature,
        dual_type=FLAGS.dual_type,
        ddpg_tanh=FLAGS.ddpg_tanh,
        tanh_squash=FLAGS.tanh_squash,
        expectile=FLAGS.expectile,
        layer_norm=FLAGS.layer_norm,
        value_type=FLAGS.value_type,
        actor_type=FLAGS.actor_type,
        latent_dim=FLAGS.latent_dim,
        goal_conditioned=FLAGS.goal_conditioned,
        value_algo=FLAGS.value_algo,
        gc_negative=FLAGS.gc_negative,
        value_only=FLAGS.value_only,
        const_std=FLAGS.const_std,
        actor_loss_type=FLAGS.actor_loss_type,
        ddqn_trick=FLAGS.ddqn_trick,
        use_target_v=FLAGS.use_target_v,
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

    if FLAGS.goal_conditioned:
        if 'antmaze' in FLAGS.env_name:
            example_trajectory = train_dataset.sample(50, indx=np.arange(1000, 1050), evaluation=True)
        else:
            example_trajectory = train_dataset.sample(50, indx=np.arange(0, 50), evaluation=True)

    train_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'train.csv'))
    eval_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'eval.csv'))
    first_time = time.time()
    last_time = time.time()
    for i in tqdm.tqdm(range(1, total_steps + 1),
                       smoothing=0.1,
                       dynamic_ncols=True):
        value_batch = value_dataset.sample(FLAGS.batch_size)
        actor_batch = actor_dataset.sample(FLAGS.batch_size)
        update_info = dict()
        agent, info = agent.update(value_batch, actor_batch)
        update_info.update(info)
        if not FLAGS.value_only and FLAGS.agent_name == 'trl':
            agent, info = agent.update_q(value_batch)
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
            overall_metrics = defaultdict(list)
            eval_params = FLAGS.temperature if FLAGS.sfbc_samples is None else FLAGS.sfbc_samples
            for temp in eval_params:  # Abuse the name "temp" for sfbc_samples as well
                temp_prefix = f'/temp{temp}' if len(eval_params) > 1 else ''
                for goal_info in goal_infos:
                    eval_info, trajs, cur_renders = evaluate_with_trajectories(
                        agent, env if 'env' not in goal_info else goal_info['env'], goal_info=goal_info, env_name=FLAGS.env_name, num_episodes=FLAGS.eval_episodes,
                        base_observation=base_observation, num_video_episodes=FLAGS.num_video_episodes,
                        eval_temperature=FLAGS.eval_temperature, eval_gaussian=FLAGS.eval_gaussian, sfbc_samples=temp if FLAGS.sfbc_samples is not None else None,
                        actor_temperature=temp if FLAGS.sfbc_samples is None else FLAGS.temperature[0],
                        action_grad_coef=FLAGS.action_grad_coef, action_grad_normalize=FLAGS.action_grad_normalize,
                        config=config,
                    )
                    renders.extend(cur_renders)
                    if 'goal_name' in goal_info:
                        eval_metrics.update({f'evaluation{temp_prefix}/{goal_info["goal_name"]}/{k}': v for k, v in eval_info.items()})
                        for k, v in eval_info.items():
                            overall_metrics[k].append(v)
                    else:
                        eval_metrics.update({f'evaluation{temp_prefix}/{k}': v for k, v in eval_info.items()})
                if 'goal_name' in goal_info:
                    for k, v in overall_metrics.items():
                        eval_metrics[f'evaluation{temp_prefix}/overall/{k}'] = np.mean(v)

            if FLAGS.num_video_episodes > 0:
                video = record_video('Video', i, renders=renders)
                eval_metrics['video'] = video

            if FLAGS.goal_conditioned:
                traj_metrics = get_traj_v(agent, example_trajectory)
                value_viz = viz_utils.make_visual_no_image(
                    traj_metrics,
                    [partial(viz_utils.visualize_metric, metric_name=k) for k in traj_metrics.keys()]
                )
                eval_metrics['value_traj_viz'] = wandb.Image(value_viz)

            if FLAGS.goal_conditioned and 'antmaze' in FLAGS.env_name and 'large' in FLAGS.env_name:
                traj_image = d4rl_ant.trajectory_image(viz_env, viz_dataset, trajs)
                eval_metrics['trajectories'] = wandb.Image(traj_image)

                new_metrics_dist = viz.get_distance_metrics(trajs)
                eval_metrics.update({f'debugging/{k}': v for k, v in new_metrics_dist.items()})

                image_goal = d4rl_ant.gcvalue_image(
                    viz_env,
                    viz_dataset,
                    partial(get_v_goal, agent),
                )
                eval_metrics['v_goal'] = wandb.Image(image_goal)

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
