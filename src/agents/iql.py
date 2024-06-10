"""Implementations of algorithms for continuous control."""
import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from functools import partial

from jaxrl_m.common import TrainState, target_update
from jaxrl_m.networks import Policy, ValueCritic, Critic, ensemblize
from jaxrl_m.types import *


def expectile_loss(diff, expectile=0.8):
    weight = jnp.where(diff > 0, expectile, (1 - expectile))
    return weight * (diff ** 2)


class IQLAgent(flax.struct.PyTreeNode):
    rng: PRNGKey
    critic: TrainState
    target_critic: TrainState
    value: TrainState
    actor: TrainState
    config: dict = flax.struct.field(pytree_node=False)

    @partial(jax.jit, static_argnames=('dry_run',))
    def update(agent, value_batch, actor_batch, dry_run=False):
        def critic_loss_fn(critic_params):
            next_v = agent.value(value_batch['next_observations'])
            target_q = value_batch['rewards'] + agent.config['discount'] * value_batch['masks'] * next_v
            q1, q2 = agent.critic(value_batch['observations'], value_batch['actions'], params=critic_params)
            critic_loss = ((q1 - target_q) ** 2 + (q2 - target_q) ** 2).mean()
            return critic_loss, {
                'critic_loss': critic_loss,
                'q1': q1.mean(),
            }

        def value_loss_fn(value_params):
            q1, q2 = agent.target_critic(value_batch['observations'], value_batch['actions'])
            q = jnp.minimum(q1, q2)
            v = agent.value(value_batch['observations'], params=value_params)
            value_loss = expectile_loss(q - v, agent.config['expectile']).mean()
            return value_loss, {
                'value_loss': value_loss,
                'v': v.mean(),
            }

        def actor_loss_fn(actor_params):
            v = agent.value(actor_batch['observations'])
            q1, q2 = agent.critic(actor_batch['observations'], actor_batch['actions'])
            q = jnp.minimum(q1, q2)
            exp_a = jnp.exp((q - v) * agent.config['temperature'])
            exp_a = jnp.minimum(exp_a, 100.0)

            dist = agent.actor(actor_batch['observations'], params=actor_params)
            log_probs = dist.log_prob(actor_batch['actions'])
            actor_loss = -(exp_a * log_probs).mean()

            return actor_loss, {'actor_loss': actor_loss, 'adv': q - v}

        new_critic, critic_info = agent.critic.apply_loss_fn(loss_fn=critic_loss_fn, has_aux=True)
        new_target_critic = target_update(agent.critic, agent.target_critic, agent.config['target_update_rate'])
        new_value, value_info = agent.value.apply_loss_fn(loss_fn=value_loss_fn, has_aux=True)
        new_actor, actor_info = agent.actor.apply_loss_fn(loss_fn=actor_loss_fn, has_aux=True)

        if dry_run:
            return agent, {**critic_info, **value_info, **actor_info}
        else:
            return agent.replace(critic=new_critic, target_critic=new_target_critic, value=new_value, actor=new_actor), {
                **critic_info, **value_info, **actor_info
            }

    @jax.jit
    def get_loss_info(agent, batch):
        return agent.update(batch, batch, dry_run=True)[1]

    @jax.jit
    def sample_actions(agent,
                       observations: np.ndarray,
                       goals: np.ndarray = None,
                       *,
                       seed: PRNGKey = None,
                       actor_temperature: float = 1.0,
                       temperature: float = 1.0) -> jnp.ndarray:
        actions = agent.actor(observations, temperature=temperature).sample(seed=seed)
        actions = jnp.clip(actions, -1, 1)
        return actions

    @partial(jax.jit, static_argnames=('num_samples', 'discrete', 'actor_temperature'))
    def sample_sfbc_actions(agent,
                            observations: np.ndarray,
                            goals: np.ndarray = None,
                            *,
                            seed: PRNGKey = None,
                            actor_temperature: float = 1.0,
                            temperature: float = 1.0,
                            num_samples: int = None) -> jnp.ndarray:
        action_dist = agent.actor(observations, temperature=temperature)
        actions = action_dist.sample(seed=seed, sample_shape=(num_samples,))
        actions = jnp.clip(actions, -1, 1)
        rep_obs = np.repeat(observations[None, :], num_samples, axis=0)
        qs = agent.critic(rep_obs, actions).min(axis=0)
        action = actions[np.argmax(qs)]
        return action

    @jax.jit
    def min_critic(agent,
                   observations: np.ndarray,
                   goals: np.ndarray = None,
                   actions: np.ndarray = None) -> jnp.ndarray:
        q1, q2 = agent.critic(observations, actions)
        return jnp.minimum(q1, q2)


def create_learner(
        seed: int,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        lr: float = 3e-4,
        actor_hidden_dims: Sequence[int] = (256, 256),
        value_hidden_dims: Sequence[int] = (256, 256),
        discount: float = 0.99,
        temperatures: Sequence[float] = [1.],
        tau: float = 0.005,
        expectile: float = 0.8,
        layer_norm: int = 0,
        const_std: int = 0,
        **kwargs):
    print('Extra kwargs:', kwargs)

    assert len(temperatures) == 1
    temperature = temperatures[0]

    rng = jax.random.PRNGKey(seed)
    rng, actor_key, critic_key, value_key = jax.random.split(rng, 4)

    action_dim = actions.shape[-1]
    actor_def = Policy(actor_hidden_dims, action_dim=action_dim, log_std_min=-5.0, state_dependent_std=False, tanh_squash_distribution=False, const_std=const_std)

    actor_tx = optax.adam(learning_rate=lr)

    actor_params = actor_def.init(actor_key, observations)['params']
    actor = TrainState.create(actor_def, actor_params, tx=actor_tx)

    critic_def = ensemblize(Critic, num_qs=2)(value_hidden_dims, layer_norm=layer_norm)
    critic_params = critic_def.init(critic_key, observations, actions)['params']
    critic = TrainState.create(critic_def, critic_params, tx=optax.adam(learning_rate=lr))
    target_critic = TrainState.create(critic_def, critic_params)

    value_def = ValueCritic(value_hidden_dims, layer_norm=layer_norm)
    value_params = value_def.init(value_key, observations)['params']
    value = TrainState.create(value_def, value_params, tx=optax.adam(learning_rate=lr))

    config = flax.core.FrozenDict(dict(
        discount=discount, temperature=temperature, expectile=expectile, target_update_rate=tau,
    ))

    return IQLAgent(rng, critic=critic, target_critic=target_critic, value=value, actor=actor, config=config)
