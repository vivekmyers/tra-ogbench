import copy
from functools import partial
from typing import Any

import distrax
import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.core import freeze, unfreeze

from utils.networks import GoalConditionedValue, TRLNetwork, GoalConditionedCritic
from utils.networks import Policy
from utils.train_state import TrainState, nonpytree_field


def expectile_loss(adv, diff, expectile=0.7):
    weight = jnp.where(adv >= 0, expectile, (1 - expectile))
    return weight * (diff ** 2)


def compute_actor_loss(agent, batch, network_params, rng=None):
    cur_goals = batch['policy_goals']

    if agent.config['actor_loss_type'] == 'awr':
        if agent.config['value_only']:
            v1, v2 = agent.network(batch['observations'], cur_goals, method='value')
            nv1, nv2 = agent.network(batch['next_observations'], cur_goals, method='value')
            v = (v1 + v2) / 2
            nv = (nv1 + nv2) / 2

            adv = nv - v
        else:
            v = agent.network(batch['observations'], cur_goals, method='value')
            q1, q2 = agent.network(batch['observations'], cur_goals, batch['actions'], method='target_critic')
            q = jnp.minimum(q1, q2)
            adv = q - v

        total_actor_loss = 0.
        info = {}
        exp_a = jnp.exp(adv * agent.config['temperature'])
        exp_a = jnp.minimum(exp_a, 100.0)

        dist = agent.network(batch['observations'], cur_goals, method='actor', params=network_params)
        log_probs = dist.log_prob(batch['actions'])

        actor_loss = -(exp_a * log_probs).mean()

        total_actor_loss = total_actor_loss + actor_loss
        info.update({
            'actor_loss': actor_loss,
            'adv': adv.mean(),
            'bc_log_probs': log_probs.mean(),
            'mse': jnp.mean((dist.mode() - batch['actions']) ** 2),
            'std': jnp.mean(dist.scale_diag),
        })
        return total_actor_loss, info
    elif agent.config['actor_loss_type'] == 'ddpg':
        assert not agent.config['value_only']

        total_actor_loss = 0.
        info = {}

        dist = agent.network(batch['observations'], cur_goals, method='actor', params=network_params)
        if agent.config['const_std']:
            q1, q2 = agent.network(batch['observations'], cur_goals, jnp.clip(dist.mode(), -1, 1), method='critic')
        else:
            q_actions = jnp.clip(dist.sample(seed=rng), -1, 1)
            q1, q2 = agent.network(batch['observations'], cur_goals, q_actions, method='critic')
        q = jnp.minimum(q1, q2)

        q_loss = -q.mean()
        log_probs = dist.log_prob(batch['actions'])

        bc_loss = -(agent.config['temperature'] * log_probs).mean()  # Abuse the name 'temperature' for the BC coefficient

        actor_loss = q_loss + bc_loss

        total_actor_loss = total_actor_loss + actor_loss

        info.update({
            'actor_loss': actor_loss,
            'bc_log_probs': log_probs.mean(),
            'mse': jnp.mean((dist.mode() - batch['actions']) ** 2),
            'q_loss': q_loss,
            'std': jnp.mean(dist.scale_diag),
        })
        return total_actor_loss, info
    else:
        raise NotImplementedError


def compute_value_loss(agent, batch, network_params):
    goals = batch['value_goals']
    # masks are 0 if terminal, 1 otherwise
    batch['masks'] = 1.0 - batch['rewards']
    if agent.config['gc_negative']:
        # rewards are 0 if terminal, -1 otherwise
        batch['rewards'] = batch['rewards'] - 1.0

    if agent.config['value_only']:
        (next_v1_t, next_v2_t) = agent.network(batch['next_observations'], goals, method='target_value')
        next_v_t = jnp.minimum(next_v1_t, next_v2_t)
        q = batch['rewards'] + agent.config['discount'] * batch['masks'] * next_v_t

        (v1_t, v2_t) = agent.network(batch['observations'], goals, method='target_value')
        v_t = (v1_t + v2_t) / 2
        adv = q - v_t

        q1 = batch['rewards'] + agent.config['discount'] * batch['masks'] * next_v1_t
        q2 = batch['rewards'] + agent.config['discount'] * batch['masks'] * next_v2_t
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
    goals = batch['value_goals']
    # masks are 0 if terminal, 1 otherwise
    batch['masks'] = 1.0 - batch['rewards']
    if agent.config['gc_negative']:
        # rewards are 0 if terminal, -1 otherwise
        batch['rewards'] = batch['rewards'] - 1.0

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
    rng: Any
    network: Any
    config: Any = nonpytree_field()

    @jax.jit
    def update(agent, batch):
        rng, new_rng = jax.random.split(agent.rng)

        def loss_fn(network_params):
            info = {}

            value_loss, value_info = compute_value_loss(agent, batch, network_params)
            for k, v in value_info.items():
                info[f'value/{k}'] = v

            actor_loss, actor_info = compute_actor_loss(agent, batch, network_params, rng)
            for k, v in actor_info.items():
                info[f'actor/{k}'] = v

            loss = value_loss + actor_loss

            return loss, info

        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * agent.config['target_update_rate'] + tp * (1 - agent.config['target_update_rate']),
            agent.network.params['networks_value'], agent.network.params['networks_target_value']
        )

        new_network, info = agent.network.apply_loss_fn(loss_fn=loss_fn)

        params = unfreeze(new_network.params)
        params['networks_target_value'] = new_target_params
        new_network = new_network.replace(params=freeze(params))

        return agent.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def update_q(agent, batch):
        def loss_fn(network_params):
            info = {}

            critic_loss, critic_info = compute_critic_loss(agent, batch, network_params)
            for k, v in critic_info.items():
                info[f'critic/{k}'] = v

            loss = critic_loss

            return loss, info

        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * agent.config['target_update_rate'] + tp * (1 - agent.config['target_update_rate']),
            agent.network.params['networks_critic'], agent.network.params['networks_target_critic']
        )

        new_network, info = agent.network.apply_loss_fn(loss_fn=loss_fn)

        params = unfreeze(new_network.params)
        params['networks_target_critic'] = new_target_params
        new_network = new_network.replace(params=freeze(params))

        return agent.replace(network=new_network), info

    @jax.jit
    def get_loss_info(agent, batch):
        def loss_fn(network_params):
            info = {}

            value_loss, value_info = compute_value_loss(agent, batch, network_params)
            for k, v in value_info.items():
                info[f'value/{k}'] = v
            actor_loss, actor_info = compute_actor_loss(agent, batch, network_params, agent.rng)
            for k, v in actor_info.items():
                info[f'actor/{k}'] = v
            if not agent.config['value_only']:
                critic_loss, critic_info = compute_critic_loss(agent, batch, network_params)
                for k, v in critic_info.items():
                    info[f'critic/{k}'] = v

            return info

        info = loss_fn(agent.network.params)

        return info

    @partial(jax.jit, static_argnames=('num_samples', 'discrete'))
    def sample_actions(agent,
                       observations,
                       goals=None,
                       *,
                       seed=None,
                       temperature=1.0,
                       discrete=False,
                       num_samples=None):
        dist = agent.network(observations, goals, temperature=temperature, method='actor')
        if num_samples is None:
            actions = dist.sample(seed=seed)
        else:
            actions = dist.sample(seed=seed, sample_shape=num_samples)
        if not discrete:
            actions = jnp.clip(actions, -1, 1)
        return actions

    @partial(jax.jit, static_argnames=('num_samples', 'discrete'))
    def sample_sfbc_actions(agent,
                            observations,
                            goals=None,
                            *,
                            seed=None,
                            temperature=1.0,
                            discrete=False,
                            num_samples=None):
        action_dist = agent.network(observations, goals, temperature=temperature,
                                    method='actor')
        actions = action_dist.sample(seed=seed, sample_shape=(num_samples,))
        actions = jnp.clip(actions, -1, 1)
        rep_obs = np.repeat(observations[None, :], num_samples, axis=0)
        if goals is not None:
            rep_goals = np.repeat(goals[None, :], num_samples, axis=0)
        else:
            rep_goals = None
        qs = agent.network(observations=rep_obs, actions=actions, goals=rep_goals, method='critic').min(axis=0)
        action = actions[np.argmax(qs)]
        if not discrete:
            action = jnp.clip(action, -1, 1)
        return action


def create_learner(
        seed,
        observations,
        actions,
        lr=3e-4,
        actor_hidden_dims=(256, 256),
        value_hidden_dims=(512, 512, 512),
        discount=0.99,
        tau=0.005,
        temperature=1,
        expectile=0.7,
        layer_norm=0,
        goal_conditioned=1,
        value_algo='iql',
        gc_negative=1,
        value_only=1,
        const_std=0,
        actor_loss_type='awr',
        value_exp=0,
        use_log_q=0,
        encoder=None,
        **kwargs):
    print('Extra kwargs:', kwargs)

    rng = jax.random.PRNGKey(seed)
    rng, actor_key, critic_key, value_key = jax.random.split(rng, 4)

    encoder_module = None

    if value_only:
        value_def = GoalConditionedValue(hidden_dims=value_hidden_dims, layer_norm=layer_norm, ensemble=True, value_exp=value_exp, encoder=encoder_module)
        critic_def = None
    else:
        value_def = GoalConditionedValue(hidden_dims=value_hidden_dims, layer_norm=layer_norm, ensemble=False, value_exp=value_exp, encoder=encoder_module)
        critic_def = GoalConditionedCritic(hidden_dims=value_hidden_dims, layer_norm=layer_norm, ensemble=True, value_exp=value_exp, encoder=encoder_module)

    action_dim = actions.shape[-1]
    actor_def = Policy(actor_hidden_dims, action_dim=action_dim, log_std_min=-5.0, state_dependent_std=False, const_std=const_std, encoder=encoder_module)

    network_def = TRLNetwork(
        networks={
            'value': value_def,
            'target_value': copy.deepcopy(value_def),
            'critic': critic_def,
            'target_critic': copy.deepcopy(critic_def),
            'actor': copy.deepcopy(actor_def),
        },
        goal_conditioned=goal_conditioned,
        value_only=value_only,
    )
    network_tx = optax.chain(optax.zero_nans(), optax.adam(learning_rate=lr))
    network_params = network_def.init(value_key, observations, observations, actions)['params']
    network = TrainState.create(network_def, network_params, tx=network_tx)
    params = unfreeze(network.params)
    if value_only:
        params['networks_target_value'] = params['networks_value']
    else:
        params['networks_target_value'] = params['networks_value']
        params['networks_target_critic'] = params['networks_critic']
    network = network.replace(params=freeze(params))

    config = flax.core.FrozenDict(dict(
        discount=discount, temperature=temperature, target_update_rate=tau,
        expectile=expectile,
        goal_conditioned=goal_conditioned, value_algo=value_algo, gc_negative=gc_negative, value_only=value_only,
        const_std=const_std,
        actor_loss_type=actor_loss_type, use_log_q=use_log_q,
    ))

    return JointTrainAgent(rng, network=network, config=config)
