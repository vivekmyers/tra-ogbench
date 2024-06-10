import copy

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.core import freeze, unfreeze

from jaxrl_m.common import TrainState
from jaxrl_m.networks import Policy
from jaxrl_m.types import *
from src.special_networks import ActorCritic, GoalConditionedValue, GoalConditionedCritic


def expectile_loss(adv, diff, expectile=0.7):
    weight = jnp.where(adv >= 0, expectile, (1 - expectile))
    return weight * (diff ** 2)


def compute_actor_loss(agent, batch, network_params):
    if agent.config['goal_conditioned']:
        cur_goals = batch['policy_goals']
    else:
        cur_goals = None

    if agent.config['value_only']:
        v1, v2 = agent.network(batch['observations'], cur_goals, method='value')
        nv1, nv2 = agent.network(batch['next_observations'], cur_goals, method='value')
        v = (v1 + v2) / 2
        nv = (nv1 + nv2) / 2

        if agent.config['goal_conditioned']:
            adv = nv - v
        else:
            adv = batch['rewards'] + agent.config['discount'] * batch['masks'] * nv - v
    else:
        v = agent.network(batch['observations'], cur_goals, method='value')
        q1, q2 = agent.network(batch['observations'], cur_goals, batch['actions'], method='target_critic')
        q = jnp.minimum(q1, q2)
        adv = q - v

    exp_a = jnp.exp(adv * agent.config['temperature'])
    exp_a = jnp.minimum(exp_a, 100.0)

    dist = agent.network(batch['observations'], cur_goals, method='actor', params=network_params)
    log_probs = dist.log_prob(batch['actions'])
    actor_loss = -(exp_a * log_probs).mean()

    return actor_loss, {
        'actor_loss': actor_loss,
        'adv': adv.mean(),
        'bc_log_probs': log_probs.mean(),
        'adv_median': jnp.median(adv),
        'mse': jnp.mean((dist.mode() - batch['actions']) ** 2),
    }


def compute_value_loss(agent, batch, network_params):
    if agent.config['goal_conditioned']:
        goals = batch['goals']
        # masks are 0 if terminal, 1 otherwise
        batch['masks'] = 1.0 - batch['rewards']
        # rewards are 0 if terminal, -1 otherwise
        batch['rewards'] = batch['rewards'] - 1.0
    else:
        goals = None

    if agent.config['value_only']:
        (next_v1, next_v2) = agent.network(batch['next_observations'], goals, method='target_value')
        next_v = jnp.minimum(next_v1, next_v2)
        q = batch['rewards'] + agent.config['discount'] * batch['masks'] * next_v

        (v1_t, v2_t) = agent.network(batch['observations'], goals, method='target_value')
        v_t = (v1_t + v2_t) / 2
        adv = q - v_t

        q1 = batch['rewards'] + agent.config['discount'] * batch['masks'] * next_v1
        q2 = batch['rewards'] + agent.config['discount'] * batch['masks'] * next_v2
        (v1, v2) = agent.network(batch['observations'], goals, method='value', params=network_params)
        v = (v1 + v2) / 2

        value_loss1 = expectile_loss(adv, q1 - v1, agent.config['expectile']).mean()
        value_loss2 = expectile_loss(adv, q2 - v2, agent.config['expectile']).mean()
        value_loss = value_loss1 + value_loss2
    else:
        q1, q2 = agent.network(batch['observations'], goals, batch['actions'], method='target_critic')
        q = jnp.minimum(q1, q2)
        v = agent.network(batch['observations'], goals, method='value', params=network_params)
        adv = q - v
        value_loss = expectile_loss(adv, q - v, agent.config['expectile']).mean()

    return value_loss, {
        'value_loss': value_loss,
        'v max': v.max(),
        'v min': v.min(),
        'v mean': v.mean(),
        'abs adv mean': jnp.abs(adv).mean(),
        'adv mean': adv.mean(),
        'adv max': adv.max(),
        'adv min': adv.min(),
        'accept prob': (adv >= 0).mean(),
    }


def compute_critic_loss(agent, batch, network_params):
    if agent.config['goal_conditioned']:
        goals = batch['goals']
        # masks are 0 if terminal, 1 otherwise
        batch['masks'] = 1.0 - batch['rewards']
        # rewards are 0 if terminal, -1 otherwise
        batch['rewards'] = batch['rewards'] - 1.0
    else:
        goals = None

    next_v = agent.network(batch['next_observations'], goals, method='value')
    q = batch['rewards'] + agent.config['discount'] * batch['masks'] * next_v

    q1, q2 = agent.network(batch['observations'], goals, batch['actions'], method='critic', params=network_params)
    critic_loss = ((q1 - q) ** 2 + (q2 - q) ** 2).mean()

    return critic_loss, {
        'critic_loss': critic_loss,
        'q max': q.max(),
        'q min': q.min(),
        'q mean': q.mean(),
    }


class JointTrainAgent(flax.struct.PyTreeNode):
    rng: PRNGKey
    network: TrainState
    config: dict = flax.struct.field(pytree_node=False)

    def update(agent, batch):
        def loss_fn(network_params):
            info = {}

            value_loss, value_info = compute_value_loss(agent, batch, network_params)
            for k, v in value_info.items():
                info[f'value/{k}'] = v

            if not agent.config['value_only']:
                critic_loss, critic_info = compute_critic_loss(agent, batch, network_params)
                for k, v in critic_info.items():
                    info[f'critic/{k}'] = v
            else:
                critic_loss = 0.

            actor_loss, actor_info = compute_actor_loss(agent, batch, network_params)
            for k, v in actor_info.items():
                info[f'actor/{k}'] = v

            loss = value_loss + critic_loss + actor_loss

            return loss, info

        if agent.config['value_only']:
            new_target_params = jax.tree_map(
                lambda p, tp: p * agent.config['target_update_rate'] + tp * (1 - agent.config['target_update_rate']),
                agent.network.params['networks_value'], agent.network.params['networks_target_value']
            )
        else:
            new_target_params = jax.tree_map(
                lambda p, tp: p * agent.config['target_update_rate'] + tp * (1 - agent.config['target_update_rate']),
                agent.network.params['networks_critic'], agent.network.params['networks_target_critic']
            )

        new_network, info = agent.network.apply_loss_fn(loss_fn=loss_fn, has_aux=True)

        params = unfreeze(new_network.params)
        if agent.config['value_only']:
            params['networks_target_value'] = new_target_params
        else:
            params['networks_target_critic'] = new_target_params
        new_network = new_network.replace(params=freeze(params))

        return agent.replace(network=new_network), info

    update = jax.jit(update)

    @jax.jit
    def get_loss_info(agent, batch):
        def loss_fn(network_params):
            info = {}

            value_loss, value_info = compute_value_loss(agent, batch, network_params)
            for k, v in value_info.items():
                info[f'value/{k}'] = v
            if not agent.config['value_only']:
                critic_loss, critic_info = compute_critic_loss(agent, batch, network_params)
                for k, v in critic_info.items():
                    info[f'critic/{k}'] = v
            actor_loss, actor_info = compute_actor_loss(agent, batch, network_params)
            for k, v in actor_info.items():
                info[f'actor/{k}'] = v

            return info

        info = loss_fn(agent.network.params)

        return info

    def sample_actions(agent,
                       observations: np.ndarray,
                       goals: np.ndarray = None,
                       *,
                       seed: PRNGKey = None,
                       temperature: float = 1.0,
                       discrete: int = 0,
                       num_samples: int = None) -> jnp.ndarray:
        dist = agent.network(observations, goals, temperature=temperature, method='actor')
        if num_samples is None:
            actions = dist.sample(seed=seed)
        else:
            actions = dist.sample(seed=seed, sample_shape=num_samples)
        if not discrete:
            actions = jnp.clip(actions, -1, 1)
        return actions

    sample_actions = jax.jit(sample_actions, static_argnames=('num_samples', 'discrete'))


def create_learner(
        seed: int,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        lr: float = 3e-4,
        actor_hidden_dims: Sequence[int] = (256, 256),
        value_hidden_dims: Sequence[int] = (256, 256),
        discount: float = 0.99,
        tau: float = 0.005,
        temperature: float = 1,
        expectile: float = 0.7,
        layer_norm: int = 0,
        goal_conditioned: int = 1,
        value_only: int = 1,
        **kwargs):
    print('Extra kwargs:', kwargs)

    rng = jax.random.PRNGKey(seed)
    rng, actor_key, critic_key, value_key = jax.random.split(rng, 4)

    if value_only:
        value_def = GoalConditionedValue(hidden_dims=value_hidden_dims, layer_norm=layer_norm, ensemble=True)
        critic_def = GoalConditionedCritic(hidden_dims=value_hidden_dims, layer_norm=layer_norm,
                                           ensemble=True)  # Not used!
    else:
        value_def = GoalConditionedValue(hidden_dims=value_hidden_dims, layer_norm=layer_norm, ensemble=False)
        critic_def = GoalConditionedCritic(hidden_dims=value_hidden_dims, layer_norm=layer_norm, ensemble=True)

    action_dim = actions.shape[-1]
    actor_def = Policy(actor_hidden_dims, action_dim=action_dim, log_std_min=-5.0, state_dependent_std=False,
                       tanh_squash_distribution=False)

    network_def = ActorCritic(
        networks={
            'value': value_def,
            'target_value': copy.deepcopy(value_def),
            'critic': critic_def,
            'target_critic': copy.deepcopy(critic_def),
            'actor': actor_def,
        },
        goal_conditioned=goal_conditioned,
    )
    network_tx = optax.chain(optax.zero_nans(), optax.adam(learning_rate=lr))
    if goal_conditioned:
        network_params = network_def.init(value_key, observations, observations, actions)['params']
    else:
        network_params = network_def.init(value_key, observations, None, actions)['params']
    network = TrainState.create(network_def, network_params, tx=network_tx)
    params = unfreeze(network.params)
    params['networks_target_value'] = params['networks_value']
    params['networks_target_critic'] = params['networks_critic']
    network = network.replace(params=freeze(params))

    config = flax.core.FrozenDict(dict(
        discount=discount, temperature=temperature, target_update_rate=tau, expectile=expectile,
        goal_conditioned=goal_conditioned, value_only=value_only,
    ))

    return JointTrainAgent(rng, network=network, config=config)
