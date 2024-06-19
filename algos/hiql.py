from functools import partial
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax
from flax.core import freeze, unfreeze

from utils.networks import GCValue, GCActor, RelativeRepresentation
from utils.train_state import TrainState, nonpytree_field, ModuleDict


class HIQLAgent(flax.struct.PyTreeNode):
    rng: Any
    network: Any
    config: Any = nonpytree_field()

    @staticmethod
    def expectile_loss(adv, diff, expectile):
        weight = jnp.where(adv >= 0, expectile, (1 - expectile))
        return weight * (diff ** 2)

    def value_loss(self, batch, grad_params):
        (next_v1_t, next_v2_t) = self.network.select('target_value')(batch['next_observations'], batch['value_goals'])
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

    def low_actor_loss(self, batch, grad_params):
        v1, v2 = self.network.select('value')(batch['observations'], batch['low_actor_goals'])
        nv1, nv2 = self.network.select('value')(batch['next_observations'], batch['low_actor_goals'])
        v = (v1 + v2) / 2
        nv = (nv1 + nv2) / 2
        adv = nv - v

        exp_a = jnp.exp(adv * self.config['low_alpha'])
        exp_a = jnp.minimum(exp_a, 100.0)

        goal_reps = self.network.select('goal_rep')(targets=batch['low_actor_goals'], bases=batch['observations'], params=grad_params)
        if not self.config['low_actor_rep_grad']:
            goal_reps = jax.lax.stop_gradient(goal_reps)
        dist = self.network.select('low_actor')(batch['observations'], goal_reps, goal_encoded=True, params=grad_params)
        log_prob = dist.log_prob(batch['actions'])

        actor_loss = -(exp_a * log_prob).mean()

        return actor_loss, {
            'actor_loss': actor_loss,
            'adv': adv.mean(),
            'bc_log_prob': log_prob.mean(),
            'mse': jnp.mean((dist.mode() - batch['actions']) ** 2),
            'std': jnp.mean(dist.scale_diag),
        }

    def high_actor_loss(self, batch, grad_params):
        v1, v2 = self.network.select('value')(batch['observations'], batch['high_actor_goals'])
        nv1, nv2 = self.network.select('value')(batch['high_actor_targets'], batch['high_actor_goals'])
        v = (v1 + v2) / 2
        nv = (nv1 + nv2) / 2
        adv = nv - v

        exp_a = jnp.exp(adv * self.config['high_alpha'])
        exp_a = jnp.minimum(exp_a, 100.0)

        dist = self.network.select('high_actor')(batch['observations'], batch['high_actor_goals'], params=grad_params)
        target = self.network.select('goal_rep')(targets=batch['high_actor_targets'], bases=batch['observations'])
        log_prob = dist.log_prob(target)

        actor_loss = -(exp_a * log_prob).mean()

        return actor_loss, {
            'actor_loss': actor_loss,
            'adv': adv.mean(),
            'bc_log_prob': log_prob.mean(),
            'mse': jnp.mean((dist.mode() - target) ** 2),
            'std': jnp.mean(dist.scale_diag),
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        info = {}

        value_loss, value_info = self.value_loss(batch, grad_params)
        for k, v in value_info.items():
            info[f'value/{k}'] = v

        low_actor_loss, low_actor_info = self.low_actor_loss(batch, grad_params)
        for k, v in low_actor_info.items():
            info[f'low_actor/{k}'] = v

        high_actor_loss, high_actor_info = self.high_actor_loss(batch, grad_params)
        for k, v in high_actor_info.items():
            info[f'high_actor/{k}'] = v

        loss = value_loss + low_actor_loss + high_actor_loss
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
        new_network = self.target_update(new_network, 'value')

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
        high_seed, low_seed = jax.random.split(seed)

        high_dist = self.network.select('high_actor')(observations, goals, temperature=temperature)
        goal_reps = high_dist.sample(seed=high_seed)
        goal_reps = goal_reps / jnp.linalg.norm(goal_reps, axis=-1, keepdims=True) * jnp.sqrt(goal_reps.shape[-1])

        low_dist = self.network.select('low_actor')(observations, goal_reps, goal_encoded=True, temperature=temperature)
        actions = low_dist.sample(seed=low_seed)

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

        ex_goals = ex_observations
        action_dim = ex_actions.shape[-1]

        def make_rep():
            return RelativeRepresentation(
                rep_dim=config['rep_dim'],
                hidden_dims=config['value_hidden_dims'],
                layer_norm=config['layer_norm'],
                rep_type=config['rep_type'],
            )

        goal_rep_def = make_rep()

        value_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            ensemble=True,
            state_encoder=None,
            goal_encoder=goal_rep_def,
        )
        target_value_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            ensemble=True,
            state_encoder=None,
            goal_encoder=goal_rep_def,
        )

        low_actor_def = GCActor(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            state_dependent_std=False,
            const_std=config['const_std'],
            state_encoder=None,
            goal_encoder=goal_rep_def,
        )

        high_actor_def = GCActor(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=config['rep_dim'],
            state_dependent_std=False,
            const_std=config['const_std'],
            state_encoder=None,
            goal_encoder=None,
        )

        networks = dict(
            value=value_def,
            target_value=target_value_def,
            low_actor=low_actor_def,
            high_actor=high_actor_def,
            goal_rep=goal_rep_def,
        )
        network_args = dict(
            value=[ex_observations, ex_goals],
            target_value=[ex_observations, ex_goals],
            low_actor=[ex_observations, ex_goals],
            high_actor=[ex_observations, ex_goals],
            goal_rep=[ex_goals, ex_observations],
        )
        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = unfreeze(network.params)
        params['modules_target_value'] = params['modules_value']
        network = network.replace(params=freeze(params))

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(dict(
        agent_name='hiql',
        lr=3e-4,
        batch_size=1024,
        actor_hidden_dims=(512, 512, 512),
        value_hidden_dims=(512, 512, 512),
        layer_norm=True,
        discount=0.99,
        tau=0.005,  # Target network update rate
        expectile=0.7,
        low_alpha=1.0,
        high_alpha=1.0,
        subgoal_steps=25,
        rep_dim=10,
        rep_type='concat',  # ['state', 'diff', 'concat']
        low_actor_rep_grad=False,  # Whether the gradient flows from the low actor to the goal representation
        const_std=True,
        encoder=ml_collections.config_dict.placeholder(str),

        dataset_class='HGCDataset',
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
    ))
    return config
