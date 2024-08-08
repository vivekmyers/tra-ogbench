from functools import partial
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.networks import GCValue, GCActor, RunningMeanStd
from utils.train_state import TrainState, nonpytree_field, ModuleDict


class PPOAgent(flax.struct.PyTreeNode):
    rng: Any
    network: Any
    rms_ob: RunningMeanStd
    rms_reward: RunningMeanStd
    config: Any = nonpytree_field()

    def compute_gae(self, traj_batch):
        # traj_batch: dict of (num_steps, num_envs, ob_dim)
        values = self.network.select('value')(traj_batch['observations'])
        next_values = self.network.select('value')(traj_batch['next_observations'])

        def scan_fn(lastgaelam, inputs):
            reward, mask, value, next_value = inputs
            delta = reward + mask * self.config['discount'] * next_value - value
            advantage = delta + mask * self.config['discount'] * self.config['lam'] * lastgaelam
            return advantage, advantage

        zeros = jnp.zeros(traj_batch['rewards'].shape[1])
        _, advs = jax.lax.scan(
            scan_fn, zeros, (traj_batch['rewards'], traj_batch['masks'], values, next_values), reverse=True
        )
        returns = values + advs
        normalized_advs = (advs - jnp.mean(advs)) / (jnp.std(advs) + 1e-6)

        return returns, normalized_advs

    def value_loss(self, batch, grad_params, rng):
        v = self.network.select('value')(batch['observations'], params=grad_params)
        value_loss = ((v - batch['returns']) ** 2).mean()

        return value_loss, {
            'value_loss': value_loss,
            'v_mean': v.mean(),
            'v_max': v.max(),
            'v_min': v.min(),
        }

    def actor_loss(self, batch, grad_params, rng):
        if self.config['tanh_squash']:
            actions = jnp.clip(batch['actions'], -1.0 + 1e-6, 1.0 - 1e-6)
        else:
            actions = batch['actions']

        dist = self.network.select('actor')(batch['observations'], params=grad_params)
        log_prob = dist.log_prob(actions)

        log_ratio = log_prob - batch['log_probs']
        ratio = jnp.exp(log_ratio)

        pg_loss1 = -batch['normalized_advs'] * ratio
        pg_loss2 = -batch['normalized_advs'] * jnp.clip(
            ratio, 1 - self.config['clip_ratio'], 1 + self.config['clip_ratio']
        )
        actor_loss = jnp.maximum(pg_loss1, pg_loss2).mean()

        entropy = dist.entropy()
        entropy_loss = -(self.config['ent_coef'] * entropy).mean()

        total_loss = actor_loss + entropy_loss

        approx_kl = ((ratio - 1) - log_ratio).mean()
        clip_frac = (jnp.abs(ratio - 1.0) > self.config['clip_ratio']).mean()
        if self.config['tanh_squash']:
            action_std = dist._distribution.stddev()
        else:
            action_std = dist.stddev().mean()

        return total_loss, {
            'total_loss': total_loss,
            'actor_loss': actor_loss,
            'entropy_loss': entropy_loss,
            'entropy': entropy.mean(),
            'ratio': ratio.mean(),
            'std': action_std.mean(),
            'approx_kl': approx_kl,
            'clip_frac': clip_frac,
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        info = {}
        rng = rng if rng is not None else self.rng

        rng, actor_rng, critic_rng = jax.random.split(rng, 3)

        value_loss, value_info = self.value_loss(batch, grad_params, critic_rng)
        for k, v in value_info.items():
            info[f'value/{k}'] = v

        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = value_loss + actor_loss
        return loss, info

    @jax.jit
    def update(self, batch):
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def train(self, traj_batch):
        assert len(traj_batch['observations'].shape) == 3  # (num_steps, num_envs, ob_dim)

        if self.config['normalize_ob']:
            ob_dim = traj_batch['observations'].shape[-1]
            self = self.replace(rms_ob=self.rms_ob.update(traj_batch['observations'].reshape(-1, ob_dim)))
            traj_batch['observations'] = self.rms_ob.normalize(traj_batch['observations'])
            traj_batch['next_observations'] = self.rms_ob.normalize(traj_batch['next_observations'])
        if self.config['normalize_reward']:
            self = self.replace(rms_reward=self.rms_reward.update(traj_batch['rewards'].reshape(-1)))
            traj_batch['rewards'] = self.rms_reward.normalize(traj_batch['rewards'])
        new_rng, rng = jax.random.split(self.rng)
        self = self.replace(rng=new_rng)

        returns, normalized_advs = self.compute_gae(traj_batch)
        traj_batch['returns'] = returns
        traj_batch['normalized_advs'] = normalized_advs

        traj_batch = jax.tree_util.tree_map(lambda x: x.reshape(x.shape[0] * x.shape[1], *x.shape[2:]), traj_batch)
        traj_batch_size = len(traj_batch['rewards'])

        idxs = jnp.tile(jnp.arange(traj_batch_size), (self.config['num_epochs'], 1))
        idxs = jax.random.permutation(rng, idxs, axis=1, independent=True)
        idxs = jnp.reshape(idxs, (-1, self.config['batch_size']))

        def train_step(agent, batch_idxs):
            batch = jax.tree_util.tree_map(lambda x: x[batch_idxs], traj_batch)
            agent, info = agent.update(batch)
            return agent, info

        self, infos = jax.lax.scan(f=train_step, init=self, xs=idxs)
        info = jax.tree_util.tree_map(lambda x: x.mean(), infos)
        return self, info

    @partial(jax.jit, static_argnames=('info'))
    def sample_actions(
        self,
        observations,
        goals=None,
        seed=None,
        temperature=1.0,
        info=False,
    ):
        if self.config['normalize_ob']:
            observations = self.rms_ob.normalize(observations)
        dist = self.network.select('actor')(observations, goals, temperature=temperature)
        actions = dist.sample(seed=seed)
        if info:
            logprobs = dist.log_prob(actions)
            return actions, logprobs
        else:
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

        value_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            ensemble=False,
        )

        actor_def = GCActor(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            log_std_min=-5,
            tanh_squash=config['tanh_squash'],
            state_dependent_std=config['state_dependent_std'],
            const_std=False,
            final_fc_init_scale=config['actor_fc_scale'],
        )

        network_info = dict(
            value=(value_def, (ex_observations, None)),
            actor=(actor_def, (ex_observations, None)),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.chain(
            optax.clip_by_global_norm(config['clip_grad_norm']), optax.adam(learning_rate=config['lr'])
        )
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        rms_ob = RunningMeanStd()
        rms_reward = RunningMeanStd()

        return cls(
            rng=rng, network=network, rms_ob=rms_ob, rms_reward=rms_reward, config=flax.core.FrozenDict(**config)
        )


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='ppo',
            lr=3e-4,
            batch_size=64,
            num_epochs=2,
            actor_hidden_dims=(512, 512, 512),
            value_hidden_dims=(512, 512, 512),
            layer_norm=False,
            discount=0.99,
            tanh_squash=False,
            state_dependent_std=False,
            actor_fc_scale=0.01,
            lam=0.95,
            ent_coef=0.0,
            clip_grad_norm=0.5,
            clip_ratio=0.2,
            normalize_ob=True,
            normalize_reward=True,
        )
    )
    return config
