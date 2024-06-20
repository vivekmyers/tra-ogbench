import copy
import glob
import json
import os
import pickle
import random
import time
from collections import defaultdict
from datetime import datetime
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import flax
from flax.core import freeze, unfreeze
import optax
import ml_collections
import numpy as np
import tqdm
import wandb
from absl import app, flags
from ml_collections import config_flags

from utils.dataset import ReplayBuffer
from utils.networks import GCValue, GCActor, LogParam
from utils.evaluation import evaluate, flatten
from utils.logger import setup_wandb, get_flag_dict, get_wandb_video, CsvLogger
from utils.train_state import TrainState, nonpytree_field, ModuleDict

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
flags.DEFINE_integer('log_interval', 1000, 'Log interval')
flags.DEFINE_integer('eval_interval', 100000, 'Evaluation interval')
flags.DEFINE_integer('save_interval', 100000, 'Save interval')

flags.DEFINE_integer('eval_tasks', None, 'Number of tasks to evaluate (None for all)')
flags.DEFINE_integer('eval_episodes', 50, 'Number of episodes for each task')
flags.DEFINE_float('eval_temperature', 0, 'Evaluation temperature')
flags.DEFINE_float('eval_gaussian', None, 'Evaluation Gaussian noise')
flags.DEFINE_integer('video_episodes', 2, 'Number of video episodes for each task')
flags.DEFINE_integer('video_frame_skip', 3, 'Frame skip for video')

agent_config = ml_collections.ConfigDict(dict(
    agent_name='sac',
    lr=3e-4,
    batch_size=1024,
    actor_hidden_dims=(512, 512, 512),
    value_hidden_dims=(512, 512, 512),
    layer_norm=True,
    discount=0.99,
    tau=0.005,  # Target network update rate
    target_entropy=ml_collections.config_dict.placeholder(float),
    target_entropy_multiplier=0.5,
    tanh_squash=True,
    state_dependent_std=True,
    actor_fc_scale=1.0,
))

config_flags.DEFINE_config_dict('agent', agent_config, lock_config=False)


def get_exp_name():
    exp_name = ''
    exp_name += f'sd{FLAGS.seed:03d}_'
    if 'SLURM_JOB_ID' in os.environ:
        exp_name += f's_{os.environ["SLURM_JOB_ID"]}.'
    if 'SLURM_PROCID' in os.environ:
        exp_name += f'{os.environ["SLURM_PROCID"]}.'
    exp_name += f'{datetime.now().strftime("%Y%m%d_%H%M%S")}'

    return exp_name


def make_env(env_name):
    if env_name == 'ant':
        from envs.locomotion.ant import AntEnv
        from d4rl.locomotion import wrappers
        from gym.wrappers.order_enforcing import OrderEnforcing
        from gym.wrappers.time_limit import TimeLimit

        env = AntEnv()
        env = wrappers.NormalizedBoxEnv(env)
        env = OrderEnforcing(env)
        env = TimeLimit(env, max_episode_steps=1000)
    else:
        raise NotImplementedError

    return env


class SACAgent(flax.struct.PyTreeNode):
    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def critic_loss(self, batch, grad_params, rng):
        next_dist = self.network.select('actor')(batch['next_observations'])
        next_actions, next_log_probs = next_dist.sample_and_log_prob(seed=rng)

        next_qs = self.network.select('target_critic')(batch['next_observations'], next_actions)
        next_q = jnp.min(next_qs, axis=0)

        target_q = batch['rewards'] + self.config['discount'] * batch['masks'] * next_q
        target_q = target_q - self.config['discount'] * batch['masks'] * next_log_probs * self.network.select('alpha')()

        q = self.network.select('critic')(batch['observations'], batch['actions'], params=grad_params)
        critic_loss = jnp.square(q - target_q).mean()

        return critic_loss, {
            'critic_loss': critic_loss,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
        }

    def actor_loss(self, batch, grad_params, rng):
        # Actor loss
        dist = self.network.select('actor')(batch['observations'], params=grad_params)
        actions, log_probs = dist.sample_and_log_prob(seed=rng)

        qs = self.network.select('critic')(batch['observations'], actions)
        q = jnp.min(qs, axis=0)

        actor_loss = (log_probs * self.network.select('alpha')() - q).mean()

        # Alpha loss
        alpha = self.network.select('alpha')(params=grad_params)
        entropy = -jax.lax.stop_gradient(log_probs).mean()
        alpha_loss = (alpha * (entropy - self.config['target_entropy'])).mean()

        total_loss = actor_loss + alpha_loss

        if self.config['tanh_squash']:
            action_std = dist._distribution.stddev()
        else:
            action_std = dist.stddev().mean()

        return actor_loss, {
            'total_loss': total_loss,
            'actor_loss': actor_loss,
            'alpha_loss': alpha_loss,
            'alpha': alpha,
            'entropy': -log_probs.mean(),
            'std': action_std,
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        info = {}
        rng = rng if rng is not None else self.rng

        rng, actor_rng, critic_rng = jax.random.split(rng, 3)

        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = critic_loss + actor_loss
        return loss, info

    def target_update(self, network, module_name):
        params = unfreeze(network.params)
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'], self.network.params[f'modules_target_{module_name}']
        )
        params[f'modules_target_{module_name}'] = new_target_params
        network = network.replace(params=freeze(params))
        return network

    @jax.jit
    def update(self, batch):
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        new_network = self.target_update(new_network, 'critic')

        return self.replace(network=new_network, rng=new_rng), info

    @partial(jax.jit, static_argnames=('discrete',))
    def sample_actions(
            self,
            observations,
            goals=None,
            seed=None,
            temperature=1.0,
            discrete=False,
    ):
        dist = self.network.select('actor')(observations, goals, temperature=temperature)
        actions = dist.sample(seed=seed)
        if not discrete:
            actions = jnp.clip(actions, -1, 1)
        return actions

    @classmethod
    def create(
            cls,
            seed,
            ex_observations,
            ex_actions,
            config,
    ):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        action_dim = ex_actions.shape[-1]

        if config['target_entropy'] is None:
            config['target_entropy'] = -config['target_entropy_multiplier'] * action_dim

        critic_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            ensemble=True,
        )

        actor_def = GCActor(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            log_std_min=-10,
            tanh_squash=config['tanh_squash'],
            state_dependent_std=config['state_dependent_std'],
            const_std=False,
            final_fc_init_scale=config['actor_fc_scale'],
        )

        alpha_def = LogParam()

        networks = dict(
            critic=critic_def,
            target_critic=copy.deepcopy(critic_def),
            actor=actor_def,
            alpha=alpha_def,
        )
        network_args = dict(
            critic=[ex_observations, None, ex_actions],
            target_critic=[ex_observations, None, ex_actions],
            actor=[ex_observations, None],
            alpha=[],
        )
        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = unfreeze(network.params)
        params['modules_target_critic'] = params['modules_critic']
        network = network.replace(params=freeze(params))

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def main(_):
    exp_name = get_exp_name()
    setup_wandb(project='ogcrl', group=FLAGS.run_group, name=exp_name)

    FLAGS.save_dir = os.path.join(FLAGS.save_dir, wandb.run.project, FLAGS.run_group, exp_name)
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    flag_dict = get_flag_dict()
    with open(os.path.join(FLAGS.save_dir, 'flags.json'), 'w') as f:
        json.dump(flag_dict, f)

    config = FLAGS.agent

    env = make_env(FLAGS.env_name)
    eval_env = make_env(FLAGS.env_name)

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

    agent = SACAgent.create(
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
    expl_rng = jax.random.PRNGKey(0)
    ob = env.reset()

    train_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'train.csv'))
    eval_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'eval.csv'))
    first_time = time.time()
    last_time = time.time()
    for i in tqdm.tqdm(range(1, FLAGS.train_steps + 1), smoothing=0.1, dynamic_ncols=True):
        if i < FLAGS.seed_steps:
            action = env.action_space.sample()
        else:
            expl_rng, key = jax.random.split(expl_rng)
            action = agent.sample_actions(ob, seed=key)

        next_ob, reward, done, info = env.step(action)
        mask = float(not done or 'TimeLimit.truncated' in info)

        replay_buffer.add_transition(dict(
            observations=ob,
            actions=action,
            rewards=reward,
            masks=mask,
            next_observations=next_ob,
        ))
        ob = next_ob

        if done:
            expl_metrics = {f'exploration/{k}': v for k, v in flatten(info).items()}
            ob = env.reset()

        if replay_buffer.size < FLAGS.seed_steps:
            continue

        batch = replay_buffer.sample(config.batch_size)
        agent, update_info = agent.update(batch)

        if i % FLAGS.log_interval == 0:
            train_metrics = {f'training/{k}': v for k, v in update_info.items()}
            train_metrics['time/epoch_time'] = (time.time() - last_time) / FLAGS.log_interval
            train_metrics['time/total_time'] = (time.time() - first_time)
            train_metrics.update(expl_metrics)
            last_time = time.time()
            wandb.log(train_metrics, step=i)
            train_logger.log(train_metrics, step=i)

        if i % FLAGS.eval_interval == 0:
            eval_metrics = {}
            eval_info, trajs, renders = evaluate(
                agent=agent,
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
