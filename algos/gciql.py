import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.encoders import GCEncoder, encoder_modules
from utils.networks import GCActor, GCValue, GCDiscreteActor, GCDiscreteCritic
from utils.train_state import ModuleDict, TrainState, nonpytree_field


class GCIQLAgent(flax.struct.PyTreeNode):
    rng: Any
    network: Any
    config: Any = nonpytree_field()

    @staticmethod
    def expectile_loss(adv, diff, expectile):
        weight = jnp.where(adv >= 0, expectile, (1 - expectile))
        return weight * (diff**2)

    def value_loss(self, batch, grad_params):
        if self.config['use_q']:
            q1, q2 = self.network.select('target_critic')(batch['observations'], batch['value_goals'], batch['actions'])
            q = jnp.minimum(q1, q2)
            v = self.network.select('value')(batch['observations'], batch['value_goals'], params=grad_params)
            value_loss = self.expectile_loss(q - v, q - v, self.config['expectile']).mean()
        else:
            (next_v1_t, next_v2_t) = self.network.select('target_value')(
                batch['next_observations'], batch['value_goals']
            )
            next_v_t = jnp.minimum(next_v1_t, next_v2_t)
            q = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v_t

            (v1_t, v2_t) = self.network.select('target_value')(batch['observations'], batch['value_goals'])
            v_t = (v1_t + v2_t) / 2
            adv = q - v_t

            q1 = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v1_t
            q2 = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v2_t
            (v1, v2) = self.network.select('value')(batch['observations'], batch['value_goals'], params=grad_params)
            v = (v1 + v2) / 2

            value_loss1 = self.expectile_loss(adv, q1 - v1, self.config['expectile']).mean()
            value_loss2 = self.expectile_loss(adv, q2 - v2, self.config['expectile']).mean()
            value_loss = value_loss1 + value_loss2

        return value_loss, {
            'value_loss': value_loss,
            'v_mean': v.mean(),
            'v_max': v.max(),
            'v_min': v.min(),
        }

    def critic_loss(self, batch, grad_params):
        next_v = self.network.select('value')(batch['next_observations'], batch['value_goals'])
        q = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v

        q1, q2 = self.network.select('critic')(
            batch['observations'], batch['value_goals'], batch['actions'], params=grad_params
        )
        critic_loss = ((q1 - q) ** 2 + (q2 - q) ** 2).mean()

        return critic_loss, {
            'critic_loss': critic_loss,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
        }

    def actor_loss(self, batch, grad_params, rng=None):
        if self.config['actor_loss'] == 'awr':
            if self.config['use_q']:
                v = self.network.select('value')(batch['observations'], batch['actor_goals'])
                q1, q2 = self.network.select('critic')(batch['observations'], batch['actor_goals'], batch['actions'])
                q = jnp.minimum(q1, q2)
                adv = q - v
            else:
                v1, v2 = self.network.select('value')(batch['observations'], batch['actor_goals'])
                nv1, nv2 = self.network.select('value')(batch['next_observations'], batch['actor_goals'])
                v = (v1 + v2) / 2
                nv = (nv1 + nv2) / 2
                adv = nv - v

            exp_a = jnp.exp(adv * self.config['alpha'])
            exp_a = jnp.minimum(exp_a, 100.0)

            dist = self.network.select('actor')(batch['observations'], batch['actor_goals'], params=grad_params)
            log_prob = dist.log_prob(batch['actions'])

            actor_loss = -(exp_a * log_prob).mean()

            return actor_loss, {
                'actor_loss': actor_loss,
                'adv': adv.mean(),
                'bc_log_prob': log_prob.mean(),
                'mse': jnp.mean((dist.mode() - batch['actions']) ** 2),
                'std': jnp.mean(dist.scale_diag) if not self.config['discrete'] else 0.0,
            }
        elif self.config['actor_loss'] == 'ddpgbc':
            assert self.config['use_q']

            dist = self.network.select('actor')(batch['observations'], batch['actor_goals'], params=grad_params)
            if self.config['const_std']:
                q_actions = jnp.clip(dist.mode(), -1, 1)
            else:
                q_actions = jnp.clip(dist.sample(seed=rng), -1, 1)
            q1, q2 = self.network.select('critic')(batch['observations'], batch['actor_goals'], q_actions)
            q = jnp.minimum(q1, q2)

            q_loss = -q.mean() / jax.lax.stop_gradient(jnp.abs(q).mean() + 1e-6)
            log_prob = dist.log_prob(batch['actions'])

            bc_loss = -(self.config['alpha'] * log_prob).mean()

            actor_loss = q_loss + bc_loss

            return actor_loss, {
                'actor_loss': actor_loss,
                'q_loss': q_loss,
                'bc_loss': bc_loss,
                'q_mean': q.mean(),
                'q_abs_mean': jnp.abs(q).mean(),
                'bc_log_prob': log_prob.mean(),
                'mse': jnp.mean((dist.mode() - batch['actions']) ** 2),
                'std': jnp.mean(dist.scale_diag),
            }
        else:
            raise ValueError(f'Unsupported actor loss: {self.config["actor_loss"]}')

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        info = {}
        rng = rng if rng is not None else self.rng

        value_loss, value_info = self.value_loss(batch, grad_params)
        for k, v in value_info.items():
            info[f'value/{k}'] = v

        if self.config['use_q']:
            critic_loss, critic_info = self.critic_loss(batch, grad_params)
            for k, v in critic_info.items():
                info[f'critic/{k}'] = v
        else:
            critic_loss = 0.0

        rng, actor_rng = jax.random.split(rng)
        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = value_loss + critic_loss + actor_loss
        return loss, info

    def target_update(self, network, module_name):
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @jax.jit
    def update(self, batch):
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, 'value')
        if self.config['use_q']:
            self.target_update(new_network, 'critic')

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(
        self,
        observations,
        goals=None,
        seed=None,
        temperature=1.0,
    ):
        if self.config['qmax_actor']:
            q1, q2 = self.network.select('critic')(observations, goals)
            q = jnp.minimum(q1, q2)
            actions = jnp.argmax(q, axis=-1)
            return actions
        else:
            dist = self.network.select('actor')(observations, goals, temperature=temperature)
            actions = dist.sample(seed=seed)
            if not self.config['discrete']:
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

        ex_goals = ex_observations
        if config['discrete']:
            action_dim = ex_actions.max() + 1
        else:
            action_dim = ex_actions.shape[-1]

        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['value'] = GCEncoder(concat_encoder=encoder_module())
            encoders['actor'] = GCEncoder(concat_encoder=encoder_module())
            if config['use_q']:
                encoders['critic'] = GCEncoder(concat_encoder=encoder_module())

        if config['use_q']:
            value_def = GCValue(
                hidden_dims=config['value_hidden_dims'],
                layer_norm=config['layer_norm'],
                ensemble=False,
                gc_encoder=encoders.get('value'),
            )
            if config['discrete']:
                critic_def = GCDiscreteCritic(
                    hidden_dims=config['value_hidden_dims'],
                    layer_norm=config['layer_norm'],
                    ensemble=True,
                    gc_encoder=encoders.get('critic'),
                    action_dim=action_dim,
                )
            else:
                critic_def = GCValue(
                    hidden_dims=config['value_hidden_dims'],
                    layer_norm=config['layer_norm'],
                    ensemble=True,
                    gc_encoder=encoders.get('critic'),
                )
        else:
            value_def = GCValue(
                hidden_dims=config['value_hidden_dims'],
                layer_norm=config['layer_norm'],
                ensemble=True,
                gc_encoder=encoders.get('value'),
            )
            critic_def = None

        if config['discrete']:
            actor_def = GCDiscreteActor(
                hidden_dims=config['actor_hidden_dims'],
                action_dim=action_dim,
                gc_encoder=encoders.get('actor'),
            )
        else:
            actor_def = GCActor(
                hidden_dims=config['actor_hidden_dims'],
                action_dim=action_dim,
                state_dependent_std=False,
                const_std=config['const_std'],
                gc_encoder=encoders.get('actor'),
            )

        network_info = dict(
            value=(value_def, (ex_observations, ex_goals)),
            target_value=(copy.deepcopy(value_def), (ex_observations, ex_goals)),
            actor=(actor_def, (ex_observations, ex_goals)),
        )
        if config['use_q']:
            network_info.update(
                critic=(critic_def, (ex_observations, ex_goals, ex_actions)),
                target_critic=(copy.deepcopy(critic_def), (ex_observations, ex_goals, ex_actions)),
            )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network_params
        params['modules_target_value'] = params['modules_value']
        if config['use_q']:
            params['modules_target_critic'] = params['modules_critic']

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='gciql',
            lr=3e-4,
            batch_size=1024,
            actor_hidden_dims=(512, 512, 512),
            value_hidden_dims=(512, 512, 512),
            layer_norm=True,
            discount=0.99,
            tau=0.005,  # Target network update rate
            expectile=0.9,
            actor_loss='awr',  # ['awr', 'ddpgbc']
            alpha=3.0,  # AWR temperature or DDPG+BC coefficient
            use_q=True,  # True for GCIQL, False for GCIVL
            const_std=True,
            discrete=False,
            qmax_actor=False,  # Use argmax_a Q(s, a) as actor (only for discrete actions)
            encoder=ml_collections.config_dict.placeholder(str),
            dataset_class='GCDataset',
            value_p_curgoal=0.2,
            value_p_trajgoal=0.5,
            value_p_randomgoal=0.3,
            value_geom_sample=True,
            actor_p_curgoal=0.0,
            actor_p_trajgoal=1.0,
            actor_p_randomgoal=0.0,
            actor_geom_sample=False,
            gc_negative=True,  # True for (-1, 0) rewards, False for (0, 1) rewards
            p_aug=0.0,
            frame_stack=ml_collections.config_dict.placeholder(int),
        )
    )
    return config
