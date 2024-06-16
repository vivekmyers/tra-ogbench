import copy
from functools import partial
from typing import Any

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import optax
from flax.core import freeze, unfreeze

from utils.networks import GoalConditionedValue, GoalConditionedCritic
from utils.networks import Actor
from utils.train_state import TrainState, nonpytree_field


class GCIQLNetwork(nn.Module):
    networks: Any
    v_only: bool = True

    def value(self, observations, goals, **kwargs):
        return self.networks['value'](observations, goals, **kwargs)

    def target_value(self, observations, goals, **kwargs):
        return self.networks['target_value'](observations, goals, **kwargs)

    def critic(self, observations, goals, actions, **kwargs):
        return self.networks['critic'](observations, goals, actions, **kwargs)

    def target_critic(self, observations, goals, actions, **kwargs):
        return self.networks['target_critic'](observations, goals, actions, **kwargs)

    def actor(self, observations, goals, **kwargs):
        return self.networks['actor'](observations, goals, **kwargs)

    def __call__(self, observations, goals=None, actions=None):
        # Only for initialization
        rets = {}
        rets.update({
            'actor': self.actor(observations, goals),
            'value': self.value(observations, goals),
            'target_value': self.target_value(observations, goals),
        })
        if not self.v_only:
            rets.update({
                'critic': self.critic(observations, goals, actions),
                'target_critic': self.target_critic(observations, goals, actions),
            })
        return rets


class GCIQLAgent(flax.struct.PyTreeNode):
    rng: Any
    network: Any
    config: Any = nonpytree_field()

    @staticmethod
    def expectile_loss(adv, diff, expectile):
        weight = jnp.where(adv >= 0, expectile, (1 - expectile))
        return weight * (diff ** 2)

    def value_loss(self, batch, grad_params):
        goals = batch['value_goals']

        if self.config['v_only']:
            (next_v1_t, next_v2_t) = self.network('target_value')(batch['next_observations'], goals)
            next_v_t = jnp.minimum(next_v1_t, next_v2_t)
            q = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v_t

            (v1_t, v2_t) = self.network('target_value')(batch['observations'], goals)
            v_t = (v1_t + v2_t) / 2
            adv = q - v_t

            q1 = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v1_t
            q2 = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v2_t
            (v1, v2) = self.network('value')(batch['observations'], goals, params=grad_params)
            v = (v1 + v2) / 2

            value_loss1 = self.expectile_loss(adv, q1 - v1, self.config['expectile']).mean()
            value_loss2 = self.expectile_loss(adv, q2 - v2, self.config['expectile']).mean()
            value_loss = value_loss1 + value_loss2
        else:
            q1, q2 = self.network('target_critic')(batch['observations'], goals, batch['actions'])
            q = jnp.minimum(q1, q2)
            v = self.network('value')(batch['observations'], goals, params=grad_params)
            value_loss = self.expectile_loss(q - v, q - v, self.config['expectile']).mean()

        return value_loss, {
            'value_loss': value_loss,
            'v_mean': v.mean(),
            'v_max': v.max(),
            'v_min': v.min(),
        }

    def critic_loss(self, batch, grad_params):
        goals = batch['value_goals']

        next_v = self.network('value')(batch['next_observations'], goals)
        q = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v

        q1, q2 = self.network('critic')(batch['observations'], goals, batch['actions'], params=grad_params)
        critic_loss = ((q1 - q) ** 2 + (q2 - q) ** 2).mean()

        return critic_loss, {
            'critic_loss': critic_loss,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
        }

    def actor_loss(self, batch, grad_params, rng=None):
        cur_goals = batch['actor_goals']

        if self.config['actor_loss'] == 'awr':
            if self.config['v_only']:
                v1, v2 = self.network('value')(batch['observations'], cur_goals)
                nv1, nv2 = self.network('value')(batch['next_observations'], cur_goals)
                v = (v1 + v2) / 2
                nv = (nv1 + nv2) / 2

                adv = nv - v
            else:
                v = self.network('value')(batch['observations'], cur_goals)
                q1, q2 = self.network('target_critic')(batch['observations'], cur_goals, batch['actions'])
                q = jnp.minimum(q1, q2)
                adv = q - v

            exp_a = jnp.exp(adv * self.config['alpha'])
            exp_a = jnp.minimum(exp_a, 100.)

            dist = self.network('actor')(batch['observations'], cur_goals, params=grad_params)
            log_prob = dist.log_prob(batch['actions'])

            actor_loss = -(exp_a * log_prob).mean()

            return actor_loss, {
                'actor_loss': actor_loss,
                'adv': adv.mean(),
                'bc_log_prob': log_prob.mean(),
                'mse': jnp.mean((dist.mode() - batch['actions']) ** 2),
                'std': jnp.mean(dist.scale_diag),
            }
        elif self.config['actor_loss'] == 'ddpg':
            assert not self.config['v_only']

            dist = self.network('actor')(batch['observations'], cur_goals, params=grad_params)
            if self.config['const_std']:
                q1, q2 = self.network('critic')(batch['observations'], cur_goals, jnp.clip(dist.mode(), -1, 1))
            else:
                q_actions = jnp.clip(dist.sample(seed=rng), -1, 1)
                q1, q2 = self.network('critic')(batch['observations'], cur_goals, q_actions)
            q = jnp.minimum(q1, q2)

            q_loss = -q.mean()
            log_prob = dist.log_prob(batch['actions'])

            bc_loss = -(self.config['alpha'] * log_prob).mean()

            actor_loss = q_loss + bc_loss

            return actor_loss, {
                'actor_loss': actor_loss,
                'q_loss': q_loss,
                'bc_loss': bc_loss,
                'bc_log_prob': log_prob.mean(),
                'mse': jnp.mean((dist.mode() - batch['actions']) ** 2),
                'std': jnp.mean(dist.scale_diag),
            }
        else:
            raise NotImplementedError

    def total_loss(self, batch, grad_params, rng=None):
        info = {}

        value_loss, value_info = self.value_loss(batch, grad_params)
        for k, v in value_info.items():
            info[f'value/{k}'] = v

        if not self.config['v_only']:
            critic_loss, critic_info = self.critic_loss(batch, grad_params)
            for k, v in critic_info.items():
                info[f'critic/{k}'] = v
        else:
            critic_loss = 0.

        actor_loss, actor_info = self.actor_loss(batch, grad_params, rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = value_loss + critic_loss + actor_loss
        return loss, info

    @jax.jit
    def update(self, batch):
        rng, new_rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_target_value_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params['networks_value'], self.network.params['networks_target_value']
        )
        if not self.config['v_only']:
            new_target_critic_params = jax.tree_util.tree_map(
                lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
                self.network.params['networks_critic'], self.network.params['networks_target_critic']
            )

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)

        params = unfreeze(new_network.params)
        params['networks_target_value'] = new_target_value_params
        if not self.config['v_only']:
            params['networks_target_critic'] = new_target_critic_params
        new_network = new_network.replace(params=freeze(params))

        return self.replace(network=new_network, rng=new_rng), info

    @partial(jax.jit, static_argnames=('discrete',))
    def sample_actions(
            self,
            observations,
            goals=None,
            *,
            seed=None,
            temperature=1.0,
            discrete=False
    ):
        dist = self.network('actor')(observations, goals, temperature=temperature)
        actions = dist.sample(seed=seed)
        if not discrete:
            actions = jnp.clip(actions, -1, 1)
        return actions

    @classmethod
    def create(
            cls,
            seed,
            example_observations,
            example_actions,
            config,
    ):
        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, value_key = jax.random.split(rng, 4)

        encoder_module = None

        action_dim = example_actions.shape[-1]

        if config['v_only']:
            value_def = GoalConditionedValue(hidden_dims=config['value_hidden_dims'], layer_norm=config['layer_norm'], ensemble=True, encoder=encoder_module)
            critic_def = None
        else:
            value_def = GoalConditionedValue(hidden_dims=config['value_hidden_dims'], layer_norm=config['layer_norm'], ensemble=False, encoder=encoder_module)
            critic_def = GoalConditionedCritic(hidden_dims=config['value_hidden_dims'], layer_norm=config['layer_norm'], ensemble=True, encoder=encoder_module)

        actor_def = Actor(config['actor_hidden_dims'], action_dim=action_dim, log_std_min=-5.0, state_dependent_std=False, const_std=config['const_std'], encoder=encoder_module)

        network_def = GCIQLNetwork(
            networks={
                'value': value_def,
                'target_value': copy.deepcopy(value_def),
                'critic': critic_def,
                'target_critic': copy.deepcopy(critic_def),
                'actor': copy.deepcopy(actor_def),
            },
            v_only=config['v_only'],
        )
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(value_key, example_observations, example_observations, example_actions)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)
        params = unfreeze(network.params)
        if config['v_only']:
            params['networks_target_value'] = params['networks_value']
        else:
            params['networks_target_value'] = params['networks_value']
            params['networks_target_critic'] = params['networks_critic']
        network = network.replace(params=freeze(params))

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict({
        'agent_name': 'gciql',
        'lr': 3e-4,
        'batch_size': 1024,
        'actor_hidden_dims': (512, 512, 512),
        'value_hidden_dims': (512, 512, 512),
        'layer_norm': True,
        'discount': 0.99,
        'tau': 0.005,  # Target network update rate
        'expectile': 0.9,
        'actor_loss': 'awr',  # 'awr' or 'ddpgbc'
        'alpha': 3.0,  # AWR temperature or DDPG+BC coefficient
        'v_only': True,  # True for GCIVL, False for GCIQL
        'const_std': True,
        'encoder': ml_collections.config_dict.placeholder(str),

        'value_p_curgoal': 0.2,
        'value_p_trajgoal': 0.5,
        'value_p_randomgoal': 0.3,
        'value_geom_sample': True,
        'actor_p_curgoal': 0.0,
        'actor_p_trajgoal': 0.7,
        'actor_p_randomgoal': 0.3,
        'actor_geom_sample': False,
        'gc_negative': True,  # True for (-1, 0) rewards, False for (0, 1) rewards
        'p_aug': 0.0,
    })
    return config
