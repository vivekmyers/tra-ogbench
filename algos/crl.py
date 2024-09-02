from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.encoders import GCEncoder, encoder_modules
from utils.networks import GCActor, GCBilinearValue, GCDiscreteActor, GCDiscreteBilinearCritic
from utils.train_state import ModuleDict, TrainState, nonpytree_field


class CRLAgent(flax.struct.PyTreeNode):
    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def contrastive_loss(self, batch, grad_params, module_name='critic'):
        batch_size = batch['observations'].shape[0]

        if module_name == 'critic':
            v, phi, psi = self.network.select(module_name)(
                batch['observations'], batch['value_goals'], batch['actions'], info=True, params=grad_params
            )
        else:
            v, phi, psi = self.network.select(module_name)(
                batch['observations'], batch['value_goals'], info=True, params=grad_params
            )
        if len(phi.shape) == 2:  # Non-ensemble
            phi = phi[None, ...]
            psi = psi[None, ...]
        logits = jnp.einsum('eik,ejk->ije', phi, psi) / jnp.sqrt(phi.shape[-1])
        # logits.shape = (B, B, e) with 1 term for positive pair and (B - 1) terms for negative pairs in each row
        I = jnp.eye(batch_size)
        contrastive_loss = jax.vmap(
            lambda _logits: optax.sigmoid_binary_cross_entropy(logits=_logits, labels=I),
            in_axes=-1,
            out_axes=-1,
        )(logits)
        contrastive_loss = jnp.mean(contrastive_loss)

        logits = jnp.mean(logits, axis=-1)
        correct = jnp.argmax(logits, axis=1) == jnp.argmax(I, axis=1)
        logits_pos = jnp.sum(logits * I) / jnp.sum(I)
        logits_neg = jnp.sum(logits * (1 - I)) / jnp.sum(1 - I)

        return contrastive_loss, {
            'contrastive_loss': contrastive_loss,
            'v_mean': v.mean(),
            'v_max': v.max(),
            'v_min': v.min(),
            'binary_accuracy': jnp.mean((logits > 0) == I),
            'categorical_accuracy': jnp.mean(correct),
            'logits_pos': logits_pos,
            'logits_neg': logits_neg,
            'logits': logits.mean(),
        }

    def actor_loss(self, batch, grad_params, rng=None):
        if self.config['actor_log_q']:

            def value_transform(x):
                return jnp.log(jnp.maximum(x, 1e-6))
        else:

            def value_transform(x):
                return x

        if self.config['actor_loss'] == 'awr':
            if self.config['use_q']:
                v = value_transform(self.network.select('value')(batch['observations'], batch['actor_goals']))
                q1, q2 = value_transform(
                    self.network.select('critic')(batch['observations'], batch['actor_goals'], batch['actions'])
                )
                q = jnp.minimum(q1, q2)
                adv = q - v
            else:
                v1, v2 = value_transform(self.network.select('value')(batch['observations'], batch['actor_goals']))
                nv1, nv2 = value_transform(
                    self.network.select('value')(batch['next_observations'], batch['actor_goals'])
                )
                v = (v1 + v2) / 2
                nv = (nv1 + nv2) / 2
                adv = nv - v

            exp_a = jnp.exp(adv * self.config['alpha'])
            exp_a = jnp.minimum(exp_a, 100.0)

            dist = self.network.select('actor')(batch['observations'], batch['actor_goals'], params=grad_params)
            log_prob = dist.log_prob(batch['actions'])

            actor_loss = -(exp_a * log_prob).mean()

            actor_info = {
                'actor_loss': actor_loss,
                'adv': adv.mean(),
                'bc_log_prob': log_prob.mean(),
            }
            if not self.config['discrete']:
                actor_info.update(
                    {
                        'mse': jnp.mean((dist.mode() - batch['actions']) ** 2),
                        'std': jnp.mean(dist.scale_diag),
                    }
                )

            return actor_loss, actor_info
        elif self.config['actor_loss'] == 'ddpgbc':
            assert self.config['use_q'] and not self.config['discrete']

            dist = self.network.select('actor')(batch['observations'], batch['actor_goals'], params=grad_params)
            if self.config['const_std']:
                q_actions = jnp.clip(dist.mode(), -1, 1)
            else:
                q_actions = jnp.clip(dist.sample(seed=rng), -1, 1)
            q1, q2 = value_transform(
                self.network.select('critic')(batch['observations'], batch['actor_goals'], q_actions)
            )
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

        if self.config['use_q']:
            critic_loss, critic_info = self.contrastive_loss(batch, grad_params, 'critic')
            for k, v in critic_info.items():
                info[f'critic/{k}'] = v
        else:
            critic_loss = 0.0

        if not self.config['use_q'] or (self.config['use_q'] and self.config['actor_loss'] == 'awr'):
            value_loss, value_info = self.contrastive_loss(batch, grad_params, 'value')
            for k, v in value_info.items():
                info[f'value/{k}'] = v
        else:
            value_loss = 0.0

        rng, actor_rng = jax.random.split(rng)
        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = value_loss + critic_loss + actor_loss
        return loss, info

    @jax.jit
    def update(self, batch):
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(
        self,
        observations,
        goals=None,
        seed=None,
        temperature=1.0,
    ):
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
            encoders['value_state'] = encoder_module()
            encoders['value_goal'] = encoder_module()
            encoders['actor'] = GCEncoder(concat_encoder=encoder_module())
            if config['use_q']:
                encoders['critic_state'] = encoder_module()
                encoders['critic_goal'] = encoder_module()

        if config['use_q']:
            value_def = GCBilinearValue(
                hidden_dims=config['value_hidden_dims'],
                latent_dim=config['latent_dim'],
                layer_norm=config['layer_norm'],
                ensemble=False,
                value_exp=True,
                state_encoder=encoders.get('value_state'),
                goal_encoder=encoders.get('value_goal'),
            )
            if config['discrete']:
                critic_def = GCDiscreteBilinearCritic(
                    hidden_dims=config['value_hidden_dims'],
                    latent_dim=config['latent_dim'],
                    layer_norm=config['layer_norm'],
                    ensemble=True,
                    value_exp=True,
                    state_encoder=encoders.get('critic_state'),
                    goal_encoder=encoders.get('critic_goal'),
                    action_dim=action_dim,
                )
            else:
                critic_def = GCBilinearValue(
                    hidden_dims=config['value_hidden_dims'],
                    latent_dim=config['latent_dim'],
                    layer_norm=config['layer_norm'],
                    ensemble=True,
                    value_exp=True,
                    state_encoder=encoders.get('critic_state'),
                    goal_encoder=encoders.get('critic_goal'),
                )
        else:
            value_def = GCBilinearValue(
                hidden_dims=config['value_hidden_dims'],
                latent_dim=config['latent_dim'],
                layer_norm=config['layer_norm'],
                ensemble=True,
                value_exp=True,
                state_encoder=encoders.get('value_state'),
                goal_encoder=encoders.get('value_goal'),
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
            actor=(actor_def, (ex_observations, ex_goals)),
        )
        if config['use_q']:
            network_info.update(
                critic=(critic_def, (ex_observations, ex_goals, ex_actions)),
            )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='crl',
            lr=3e-4,
            batch_size=1024,
            actor_hidden_dims=(512, 512, 512),
            value_hidden_dims=(512, 512, 512),
            latent_dim=512,
            layer_norm=True,
            discount=0.99,
            actor_loss='ddpgbc',  # ['awr', 'ddpgbc']
            alpha=1.0,  # AWR temperature or DDPG+BC coefficient
            use_q=True,  # Whether to fit Q or V with contrastive RL
            actor_log_q=True,  # Whether to use log Q in actor loss
            const_std=True,
            discrete=False,
            encoder=ml_collections.config_dict.placeholder(str),
            dataset_class='GCDataset',
            value_p_curgoal=0.0,
            value_p_trajgoal=1.0,
            value_p_randomgoal=0.0,
            value_geom_sample=True,
            actor_p_curgoal=0.0,
            actor_p_trajgoal=1.0,
            actor_p_randomgoal=0.0,
            actor_geom_sample=False,
            gc_negative=False,
            p_aug=0.0,
            frame_stack=ml_collections.config_dict.placeholder(int),
        )
    )
    return config
