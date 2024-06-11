import copy
from functools import partial

import distrax
import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.core import freeze, unfreeze

from jaxrl_m.common import TrainState
from jaxrl_m.networks import Policy, BilinearPolicy
from jaxrl_m.types import *
from jaxrl_m.vision import encoders
from src.special_networks import ActorCritic, GoalConditionedValue, GoalConditionedQuasiPhiValue, \
    GoalConditionedPhiValue, GoalConditionedBilinearValue, TRLNetwork, GoalConditionedCritic, \
    GoalConditionedBilinearCritic


def expectile_loss(adv, diff, expectile=0.7):
    weight = jnp.where(adv >= 0, expectile, (1 - expectile))
    return weight * (diff ** 2)


def compute_actor_loss(agent, batch, network_params, rng=None):
    if agent.config['goal_conditioned']:
        cur_goals = batch['policy_goals']
    else:
        cur_goals = None

    if agent.config['use_log_q']:
        if agent.config['gc_negative']:
            value_transform = lambda x: jnp.log(jnp.maximum(x * (1 - agent.config['discount']) + 1, 1e-6))
        else:
            value_transform = lambda x: jnp.log(jnp.maximum(x, 1e-6))
    else:
        value_transform = lambda x: x

    if agent.config['actor_loss_type'] == 'awr':
        if agent.config['value_only']:
            v1, v2 = value_transform(agent.network(batch['observations'], cur_goals, method='value'))
            nv1, nv2 = value_transform(agent.network(batch['next_observations'], cur_goals, method='value'))
            v = (v1 + v2) / 2
            nv = (nv1 + nv2) / 2

            if agent.config['goal_conditioned']:
                adv = nv - v
            else:
                adv = batch['rewards'] + agent.config['discount'] * batch['masks'] * nv - v
        else:
            v = value_transform(agent.network(batch['observations'], cur_goals, method='value'))
            q1, q2 = value_transform(agent.network(batch['observations'], cur_goals, batch['actions'], method='target_critic'))
            q = jnp.minimum(q1, q2)
            adv = q - v

        total_actor_loss = 0.
        info = {}
        for temp in agent.config['temperatures']:
            exp_a = jnp.exp(adv * temp)
            exp_a = jnp.minimum(exp_a, 100.0)

            dist = agent.network(batch['observations'], cur_goals, temp=temp, method='actor', params=network_params)
            log_probs = dist.log_prob(batch['actions'])

            actor_loss = -(exp_a * log_probs).mean()

            total_actor_loss = total_actor_loss + actor_loss
            postfix = f'_{temp}' if len(agent.config['temperatures']) > 1 else ''
            info.update({
                f'actor_loss{postfix}': actor_loss,
                f'adv{postfix}': adv.mean(),
                f'bc_log_probs{postfix}': log_probs.mean(),
                f'mse{postfix}': jnp.mean((dist.mode() - batch['actions']) ** 2),
                f'std{postfix}': jnp.mean(dist.scale_diag),
            })
        return total_actor_loss, info
    elif agent.config['actor_loss_type'] == 'ddpg':
        assert not agent.config['value_only']

        total_actor_loss = 0.
        info = {}

        bc_dist = agent.network(batch['observations'], cur_goals, method='bc_actor', params=network_params)
        if agent.config['ddpg_tanh']:
            bc_dist = distrax.MultivariateNormalDiag(
                loc=jnp.tanh(bc_dist.loc), scale_diag=bc_dist.scale_diag
            )
        bc_log_probs = bc_dist.log_prob(batch['actions'])
        bc_actor_loss = -bc_log_probs.mean()
        total_actor_loss = total_actor_loss + bc_actor_loss

        for temp in agent.config['temperatures']:
            postfix = f'_{temp}' if len(agent.config['temperatures']) > 1 else ''

            if agent.config['dual_type'] in ['none', 'avg']:
                dist = agent.network(batch['observations'], cur_goals, temp=temp, method='actor', params=network_params)
            else:
                dist, log_lam = agent.network(batch['observations'], cur_goals, temp=temp, info=True, method='actor', params=network_params)
            if agent.config['ddpg_tanh']:
                dist = distrax.MultivariateNormalDiag(
                    loc=jnp.tanh(dist.loc), scale_diag=dist.scale_diag
                )
                q1, q2 = value_transform(agent.network(batch['observations'], cur_goals, dist.loc, method='critic'))
            else:
                if agent.config['const_std']:
                    q1, q2 = value_transform(agent.network(batch['observations'], cur_goals, jnp.clip(dist.mode(), -1, 1), method='critic'))
                else:
                    q_actions = jnp.clip(dist.sample(seed=rng), -1, 1)
                    q1, q2 = value_transform(agent.network(batch['observations'], cur_goals, q_actions, method='critic'))
            q = jnp.minimum(q1, q2)

            q_loss = -q.mean()
            log_probs = dist.log_prob(batch['actions'])

            if agent.config['dual_type'] in ['none', 'avg']:
                if agent.config['dual_type'] == 'none':
                    bc_loss = -(temp * log_probs).mean()  # Abuse the name 'temperature' for the BC coefficient
                else:
                    avg_q = jax.lax.stop_gradient(jnp.abs(q).mean())
                    bc_loss = -(temp * avg_q * log_probs).mean()
                dual_loss = 0
                kl = dist.kl_divergence(bc_dist).mean()
                info.update({
                    f'kl{postfix}': kl,
                })
            else:
                log_lam = jnp.minimum(log_lam, 10)
                lam = jnp.exp(log_lam)
                bc_loss = -(jax.lax.stop_gradient(lam) * log_probs).mean()  # Abuse the name 'temperature' for the BC coefficient
                kl = dist.kl_divergence(bc_dist).mean()
                dual_loss = -log_lam * (jax.lax.stop_gradient(kl) - temp).mean()
                info.update({
                    f'dual_loss{postfix}': dual_loss,
                    f'lam{postfix}': lam,
                    f'kl{postfix}': kl,
                })

            actor_loss = q_loss + bc_loss + dual_loss

            total_actor_loss = total_actor_loss + actor_loss

            info.update({
                f'actor_loss{postfix}': actor_loss,
                f'bc_log_probs{postfix}': log_probs.mean(),
                f'mse{postfix}': jnp.mean((dist.mode() - batch['actions']) ** 2),
                f'q_loss{postfix}': q_loss,
                f'std{postfix}': jnp.mean(dist.scale_diag),
            })
        return total_actor_loss, info
    else:
        raise NotImplementedError


def compute_value_loss(agent, batch, network_params):
    if agent.config['goal_conditioned']:
        goals = batch['goals']
        # masks are 0 if terminal, 1 otherwise
        batch['masks'] = 1.0 - batch['rewards']
        if agent.config['gc_negative']:
            # rewards are 0 if terminal, -1 otherwise
            batch['rewards'] = batch['rewards'] - 1.0
    else:
        goals = None

    if agent.config['value_algo'] == 'iql':
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
            if agent.config['ddqn_trick']:
                v_t = agent.network(batch['observations'], goals, method='target_value')
                adv = q - v_t
            else:
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
    elif agent.config['value_algo'] == 'crl':
        batch_size = batch['observations'].shape[0]
        weights = np.ones((batch_size, batch_size)) / (batch_size - 1)
        weights[np.arange(batch_size), np.arange(batch_size)] = 1
        if agent.config['value_only']:
            v, phi, psi = agent.network(batch['observations'], goals, info=True, method='value', params=network_params)
        else:
            v, phi, psi = agent.network(batch['observations'], goals, batch['actions'], info=True, method='critic', params=network_params)
        logits = jnp.einsum('eik,ejk->ije', phi, psi) / jnp.sqrt(phi.shape[-1])
        # logits.shape = (B, B, 2) with 1 term for positive pair and (B - 1) terms for negative pairs in each row
        I = jnp.eye(batch_size)
        value_loss = jax.vmap(
            lambda _logits: optax.sigmoid_binary_cross_entropy(logits=_logits, labels=I),
            in_axes=-1,
            out_axes=-1,
        )(logits)
        value_loss = jnp.mean(value_loss)

        # Take the mean here so that we can compute the accuracy.
        logits = jnp.mean(logits, axis=-1)
        correct = jnp.argmax(logits, axis=1) == jnp.argmax(I, axis=1)
        logits_pos = jnp.sum(logits * I) / jnp.sum(I)
        logits_neg = jnp.sum(logits * (1 - I)) / jnp.sum(1 - I)

        if agent.config['actor_loss_type'] == 'awr' and agent.config['value_only'] == 0:
            # Fit v as well for advantages
            _, phi, psi = agent.network(batch['observations'], goals, info=True, method='value', params=network_params)
            phi = phi[None, ...]
            psi = psi[None, ...]
            v_logits = jnp.einsum('eik,ejk->ije', phi, psi) / jnp.sqrt(phi.shape[-1])
            v_value_loss = jax.vmap(
                lambda _logits: optax.sigmoid_binary_cross_entropy(logits=_logits, labels=I),
                in_axes=-1,
                out_axes=-1,
            )(v_logits)
            value_loss = value_loss + jnp.mean(v_value_loss)

        return value_loss, {
            'value_loss': value_loss,
            "binary_accuracy": jnp.mean((logits > 0) == I),
            "categorical_accuracy": jnp.mean(correct),
            "logits_pos": logits_pos,
            "logits_neg": logits_neg,
            "logits": logits.mean(),
            'v max': v.max(),
            'v min': v.min(),
            'v mean': v.mean(),
        }


def compute_critic_loss(agent, batch, network_params):
    if agent.config['value_algo'] == 'crl':
        return 0., dict()

    if agent.config['goal_conditioned']:
        goals = batch['goals']
        # masks are 0 if terminal, 1 otherwise
        batch['masks'] = 1.0 - batch['rewards']
        if agent.config['gc_negative']:
            # rewards are 0 if terminal, -1 otherwise
            batch['rewards'] = batch['rewards'] - 1.0
    else:
        goals = None

    if agent.config['use_target_v']:
        next_v = agent.network(batch['next_observations'], goals, method='target_value')
    else:
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

    @jax.jit
    def update(agent, value_batch, actor_batch):
        rng, new_rng = jax.random.split(agent.rng)

        def loss_fn(network_params):
            info = {}

            value_loss, value_info = compute_value_loss(agent, value_batch, network_params)
            for k, v in value_info.items():
                info[f'value/{k}'] = v

            actor_loss, actor_info = compute_actor_loss(agent, actor_batch, network_params, rng)
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
            lambda p, tp: p * agent.config['target_update_rate'] + tp * (1 - agent.config['target_update_rate']), agent.network.params['networks_critic'], agent.network.params['networks_target_critic']
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

    @partial(jax.jit, static_argnames=('num_samples', 'discrete', 'actor_temperature'))
    def sample_actions(agent,
                       observations: np.ndarray,
                       goals: np.ndarray = None,
                       *,
                       seed: PRNGKey = None,
                       actor_temperature: float = 1.0,
                       temperature: float = 1.0,
                       discrete: int = 0,
                       num_samples: int = None) -> jnp.ndarray:
        dist = agent.network(observations, goals, temp=actor_temperature, temperature=temperature, method='actor')
        if num_samples is None:
            actions = dist.sample(seed=seed)
        else:
            actions = dist.sample(seed=seed, sample_shape=num_samples)
        if not discrete:
            if agent.config['actor_loss_type'] == 'ddpg' and agent.config['ddpg_tanh']:
                actions = jnp.tanh(actions)
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
                            discrete: int = 0,
                            num_samples: int = None) -> jnp.ndarray:
        action_dist = agent.network(observations, goals, temp=actor_temperature, temperature=temperature, method='actor')
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

    @jax.jit
    def min_critic(agent,
                   observations: np.ndarray,
                   goals: np.ndarray,
                   actions: np.ndarray) -> jnp.ndarray:
        q1, q2 = agent.network(observations, goals, actions, method='critic')
        return jnp.minimum(q1, q2)


def create_learner(
        seed: int,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        lr: float = 3e-4,
        actor_hidden_dims: Sequence[int] = (256, 256),
        value_hidden_dims: Sequence[int] = (256, 256),
        discount: float = 0.99,
        tau: float = 0.005,
        temperatures: Sequence[float] = [1.],
        dual_type: str = 'none',
        ddpg_tanh: int = 1,
        tanh_squash: int = 0,
        expectile: float = 0.7,
        layer_norm: int = 0,
        value_type: str = 'mono',
        actor_type: str = 'mono',
        latent_dim: int = 512,
        goal_conditioned: int = 1,
        value_algo: str = 'iql',
        gc_negative: int = 1,
        value_only: int = 1,
        const_std: int = 0,
        actor_loss_type: str = 'awr',
        ddqn_trick: int = 0,
        use_target_v: int = 0,
        value_exp: int = 0,
        use_log_q: int = 0,
        encoder: str = None,
        **kwargs):
    print('Extra kwargs:', kwargs)

    rng = jax.random.PRNGKey(seed)
    rng, actor_key, critic_key, value_key = jax.random.split(rng, 4)

    if encoder is not None:
        encoder_module = encoders[encoder]
    else:
        encoder_module = None

    if value_algo == 'crl':
        assert value_type == 'bilinear'
        assert not gc_negative

    if value_only:
        if value_type == 'mono':
            value_def = GoalConditionedValue(hidden_dims=value_hidden_dims, layer_norm=layer_norm, ensemble=True, value_exp=value_exp, encoder=encoder_module)
        elif value_type == 'bilinear':
            value_def = GoalConditionedBilinearValue(hidden_dims=value_hidden_dims, layer_norm=layer_norm, ensemble=True, latent_dim=latent_dim, value_exp=value_exp, encoder=encoder_module)
        critic_def = None
    else:
        if value_type == 'mono':
            value_def = GoalConditionedValue(hidden_dims=value_hidden_dims, layer_norm=layer_norm, ensemble=False, value_exp=value_exp, encoder=encoder_module)
            critic_def = GoalConditionedCritic(hidden_dims=value_hidden_dims, layer_norm=layer_norm, ensemble=True, value_exp=value_exp, encoder=encoder_module)
        elif value_type == 'bilinear':
            value_def = GoalConditionedBilinearValue(hidden_dims=value_hidden_dims, layer_norm=layer_norm, ensemble=False, latent_dim=latent_dim, value_exp=value_exp, encoder=encoder_module)
            critic_def = GoalConditionedBilinearCritic(hidden_dims=value_hidden_dims, layer_norm=layer_norm, ensemble=True, latent_dim=latent_dim, value_exp=value_exp, encoder=encoder_module)
        elif value_type == 'critic_bilinear':
            value_def = GoalConditionedValue(hidden_dims=value_hidden_dims, layer_norm=layer_norm, ensemble=False, value_exp=value_exp, encoder=encoder_module)
            critic_def = GoalConditionedBilinearCritic(hidden_dims=value_hidden_dims, layer_norm=layer_norm, ensemble=True, latent_dim=latent_dim, value_exp=value_exp, encoder=encoder_module)

    action_dim = actions.shape[-1]
    if actor_type == 'mono':
        actor_def = Policy(actor_hidden_dims, action_dim=action_dim, log_std_min=-5.0, state_dependent_std=False, tanh_squash_distribution=tanh_squash, const_std=const_std, encoder=encoder_module, dual_type=dual_type)
    elif actor_type == 'bilinear':
        assert dual_type == 'none'
        actor_def = BilinearPolicy(actor_hidden_dims, latent_dim=latent_dim // 4, action_dim=action_dim, log_std_min=-5.0, tanh_squash_distribution=tanh_squash, const_std=const_std, encoder=encoder_module)

    network_def = TRLNetwork(
        networks={
            'value': value_def,
            'target_value': copy.deepcopy(value_def),
            'critic': critic_def,
            'target_critic': copy.deepcopy(critic_def),
            'bc_actor': copy.deepcopy(actor_def),
            **{f'actor_{temp}': copy.deepcopy(actor_def) for temp in temperatures},
        },
        goal_conditioned=goal_conditioned,
        value_only=value_only,
    )
    network_tx = optax.chain(optax.zero_nans(), optax.adam(learning_rate=lr))
    if goal_conditioned:
        network_params = network_def.init(value_key, observations, observations, actions, temperatures)['params']
    else:
        network_params = network_def.init(value_key, observations, None, actions, temperatures)['params']
    network = TrainState.create(network_def, network_params, tx=network_tx)
    params = unfreeze(network.params)
    if value_only:
        params['networks_target_value'] = params['networks_value']
    else:
        params['networks_target_value'] = params['networks_value']
        params['networks_target_critic'] = params['networks_critic']
    network = network.replace(params=freeze(params))

    config = flax.core.FrozenDict(dict(
        discount=discount, temperatures=temperatures, dual_type=dual_type, ddpg_tanh=ddpg_tanh, target_update_rate=tau, expectile=expectile,
        goal_conditioned=goal_conditioned, value_algo=value_algo, gc_negative=gc_negative, value_only=value_only, const_std=const_std,
        actor_loss_type=actor_loss_type, ddqn_trick=ddqn_trick, use_target_v=use_target_v, use_log_q=use_log_q,
    ))

    return JointTrainAgent(rng, network=network, config=config)
