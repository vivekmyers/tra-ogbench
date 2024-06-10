import jax

from jaxrl_m.networks import *


class LayerNormRepresentation(nn.Module):
    hidden_dims: tuple = (256, 256)
    activate_final: bool = True
    ensemble: bool = True

    @nn.compact
    def __call__(self, observations):
        module = LayerNormMLP
        if self.ensemble:
            module = ensemblize(module, 2)
        return module(self.hidden_dims, activate_final=self.activate_final)(observations)


class Representation(nn.Module):
    hidden_dims: tuple = (256, 256)
    activate_final: bool = True
    ensemble: bool = True

    @nn.compact
    def __call__(self, observations):
        module = MLP
        if self.ensemble:
            module = ensemblize(module, 2)
        return module(self.hidden_dims, activate_final=self.activate_final, activations=nn.gelu)(observations)


class RelativeRepresentation(nn.Module):
    rep_dim: int = 256
    hidden_dims: tuple = (256, 256)
    module: nn.Module = None
    visual: bool = False
    layer_norm: bool = False
    rep_type: str = 'state'
    bottleneck: bool = True  # Meaning that we're using this representation for high-level actions

    @nn.compact
    def __call__(self, targets, bases=None):
        if bases is None:
            inputs = targets
        else:
            if self.rep_type == 'state':
                inputs = targets
            elif self.rep_type == 'diff':
                inputs = jax.tree_map(lambda t, b: t - b + jnp.ones_like(t) * 1e-6, targets, bases)
            elif self.rep_type == 'concat':
                inputs = jax.tree_map(lambda t, b: jnp.concatenate([t, b], axis=-1), targets, bases)
            else:
                raise NotImplementedError

        if self.visual:
            inputs = self.module()(inputs)
        if self.layer_norm:
            rep = LayerNormMLP(self.hidden_dims, activate_final=not self.bottleneck, activations=nn.gelu)(inputs)
        else:
            rep = MLP(self.hidden_dims, activate_final=not self.bottleneck, activations=nn.gelu)(inputs)

        if self.bottleneck:
            rep = rep / jnp.linalg.norm(rep, axis=-1, keepdims=True) * jnp.sqrt(self.rep_dim)

        return rep


class GoalConditionedValue(nn.Module):
    hidden_dims: tuple = (256, 256)
    readout_size: tuple = (256,)
    layer_norm: bool = True
    ensemble: bool = True
    value_exp: bool = False
    encoder: nn.Module = None

    def setup(self) -> None:
        repr_class = LayerNormRepresentation if self.layer_norm else Representation
        value_net = repr_class((*self.hidden_dims, 1), activate_final=False, ensemble=self.ensemble)
        if self.encoder is not None:
            value_net = nn.Sequential([self.encoder(), value_net])
        self.value_net = value_net

    def __call__(self, observations, goals=None, return_log=False, info=False):
        if goals is None:
            v = self.value_net(observations).squeeze(-1)
        else:
            v = self.value_net(jnp.concatenate([observations, goals], axis=-1)).squeeze(-1)
        if self.value_exp and not return_log:
            v = jnp.exp(v)
        elif not self.value_exp and return_log:
            v = jnp.log(jnp.maximum(v, 1e-6))

        return v


class GoalConditionedBilinearValue(nn.Module):
    hidden_dims: tuple = (256, 256)
    readout_size: tuple = (256,)
    latent_dim: int = 2
    layer_norm: bool = True
    ensemble: bool = True
    value_exp: bool = False
    encoder: nn.Module = None

    def setup(self) -> None:
        repr_class = LayerNormRepresentation if self.layer_norm else Representation
        phi = repr_class((*self.hidden_dims, self.latent_dim), activate_final=False, ensemble=self.ensemble)
        psi = repr_class((*self.hidden_dims, self.latent_dim), activate_final=False, ensemble=self.ensemble)
        if self.encoder is not None:
            phi = nn.Sequential([self.encoder(), phi])
            psi = nn.Sequential([self.encoder(), psi])
        self.phi = phi
        self.psi = psi

    def __call__(self, observations, goals=None, return_log=False, info=False):
        phi = self.phi(observations)
        psi = self.psi(goals)
        v = (phi * psi / jnp.sqrt(self.latent_dim)).sum(axis=-1)
        if self.value_exp and not return_log:
            v = jnp.exp(v)
        elif not self.value_exp and return_log:
            v = jnp.log(jnp.maximum(v, 1e-6))

        if info:
            return v, phi, psi
        else:
            return v


class GoalConditionedPhiValue(nn.Module):
    hidden_dims: tuple = (256, 256)
    readout_size: tuple = (256,)
    latent_dim: int = 2
    layer_norm: bool = True
    ensemble: bool = True
    encoder: nn.Module = None

    def setup(self) -> None:
        repr_class = LayerNormRepresentation if self.layer_norm else Representation
        phi = repr_class((*self.hidden_dims, self.latent_dim), activate_final=False, ensemble=self.ensemble)
        if self.encoder is not None:
            phi = nn.Sequential([self.encoder(), phi])
        self.phi = phi

    def get_phi(self, observations):
        return self.phi(observations)[0]  # Use the first vf

    def __call__(self, observations, goals=None, info=False):
        phi_s = self.phi(observations)
        phi_g = self.phi(goals)
        squared_dist = ((phi_s - phi_g) ** 2).sum(axis=-1)
        v = -jnp.sqrt(jnp.maximum(squared_dist, 1e-6))

        return v


class GoalConditionedQuasiPhiValue(nn.Module):
    hidden_dims: tuple = (256, 256)
    readout_size: tuple = (256,)
    latent_dim: int = 2
    quasi_dim: int = 0
    layer_norm: bool = True
    ensemble: bool = True
    encoder: nn.Module = None

    def setup(self) -> None:
        repr_class = LayerNormRepresentation if self.layer_norm else Representation
        phi = repr_class((*self.hidden_dims, self.latent_dim + self.quasi_dim), activate_final=False, ensemble=self.ensemble)
        if self.encoder is not None:
            phi = nn.Sequential([self.encoder(), phi])
        self.phi = phi

    def get_phi(self, observations):
        return self.phi(observations)[0][..., :self.latent_dim]  # Use the first vf

    def __call__(self, observations, goals=None, info=False):
        phif_s = self.phi(observations)
        phif_g = self.phi(goals)
        phi_s = phif_s[..., :self.latent_dim]
        phi_g = phif_g[..., :self.latent_dim]
        f_s = phif_s[..., self.latent_dim:]
        f_g = phif_g[..., self.latent_dim:]
        squared_dist = ((phi_s - phi_g) ** 2).sum(axis=-1)
        quasi = jax.nn.relu((f_s - f_g).max(axis=-1))
        v = -jnp.sqrt(jnp.maximum(squared_dist, 1e-6)) - quasi

        return v


class GoalConditionedCritic(nn.Module):
    hidden_dims: tuple = (256, 256)
    readout_size: tuple = (256,)
    layer_norm: bool = True
    ensemble: bool = True
    value_exp: bool = False
    encoder: nn.Module = None

    def setup(self) -> None:
        repr_class = LayerNormRepresentation if self.layer_norm else Representation
        critic_net = repr_class((*self.hidden_dims, 1), activate_final=False, ensemble=self.ensemble)
        if self.encoder is not None:
            critic_net = nn.Sequential([self.encoder(), critic_net])
        self.critic_net = critic_net

    def __call__(self, observations, goals=None, actions=None, return_log=False, info=False):
        if goals is None:
            q = self.critic_net(jnp.concatenate([observations, actions], axis=-1)).squeeze(-1)
        else:
            q = self.critic_net(jnp.concatenate([observations, goals, actions], axis=-1)).squeeze(-1)
        if self.value_exp and not return_log:
            q = jnp.exp(q)
        elif not self.value_exp and return_log:
            q = jnp.log(jnp.maximum(q, 1e-6))

        return q


class GoalConditionedBilinearCritic(nn.Module):
    hidden_dims: tuple = (256, 256)
    readout_size: tuple = (256,)
    latent_dim: int = 2
    layer_norm: bool = True
    ensemble: bool = True
    value_exp: bool = False
    encoder: nn.Module = None

    def setup(self) -> None:
        repr_class = LayerNormRepresentation if self.layer_norm else Representation
        phi = repr_class((*self.hidden_dims, self.latent_dim), activate_final=False, ensemble=self.ensemble)
        psi = repr_class((*self.hidden_dims, self.latent_dim), activate_final=False, ensemble=self.ensemble)
        if self.encoder is not None:
            phi = nn.Sequential([self.encoder(), phi])
            psi = nn.Sequential([self.encoder(), psi])
        self.phi = phi
        self.psi = psi

    def __call__(self, observations, goals=None, actions=None, return_log=False, info=False):
        phi = self.phi(jnp.concatenate([observations, actions], axis=-1))
        psi = self.psi(goals)
        q = (phi * psi / jnp.sqrt(self.latent_dim)).sum(axis=-1)
        if self.value_exp and not return_log:
            q = jnp.exp(q)
        elif not self.value_exp and return_log:
            q = jnp.log(jnp.maximum(q, 1e-6))

        if info:
            return q, phi, psi
        else:
            return q


def get_rep(
        encoder: nn.Module, targets: jnp.ndarray, bases: jnp.ndarray = None,
):
    if encoder is None:
        return targets
    else:
        if bases is None:
            return encoder(targets)
        else:
            return encoder(targets, bases)


class ActorCritic(nn.Module):
    networks: Dict[str, nn.Module]
    goal_conditioned: int = 1

    def value(self, observations, goals=None, **kwargs):
        return self.networks['value'](observations, goals, **kwargs)

    def target_value(self, observations, goals=None, **kwargs):
        return self.networks['target_value'](observations, goals, **kwargs)

    def actor(self, observations, goals=None, **kwargs):
        if self.goal_conditioned:
            return self.networks['actor'](jnp.concatenate([observations, goals], axis=-1), **kwargs)
        else:
            return self.networks['actor'](observations, **kwargs)

    def __call__(self, observations, goals=None, actions=None):
        # Only for initialization
        rets = {
            'value': self.value(observations, goals),
            'target_value': self.target_value(observations, goals),
            'actor': self.actor(observations, goals),
        }
        return rets


class TRLNetwork(nn.Module):
    networks: Dict[str, nn.Module]
    goal_conditioned: int = 1
    value_only: int = 1

    def unsqueeze_context(self, observations, contexts):
        if len(observations.shape) <= 2:
            return contexts
        else:
            # observations: (H, W, D) or (B, H, W, D)
            # contexts: (Z) -> (H, W, Z) or (B, Z) -> (B, H, W, Z)
            assert len(observations.shape) == len(contexts.shape) + 2
            return jnp.expand_dims(jnp.expand_dims(contexts, axis=-2), axis=-2).repeat(observations.shape[-3], axis=-3).repeat(observations.shape[-2], axis=-2)

    def value(self, observations, goals=None, **kwargs):
        return self.networks['value'](observations, goals, **kwargs)

    def target_value(self, observations, goals=None, **kwargs):
        return self.networks['target_value'](observations, goals, **kwargs)

    def critic(self, observations, goals, actions=None, **kwargs):
        actions = self.unsqueeze_context(observations, actions)
        return self.networks['critic'](observations, goals, actions, **kwargs)

    def target_critic(self, observations, goals, actions=None, **kwargs):
        actions = self.unsqueeze_context(observations, actions)
        return self.networks['target_critic'](observations, goals, actions, **kwargs)

    def actor(self, observations, goals=None, temp=None, **kwargs):
        if self.goal_conditioned:
            return self.networks[f'actor_{temp}'](jnp.concatenate([observations, goals], axis=-1), **kwargs)
        else:
            return self.networks[f'actor_{temp}'](observations, **kwargs)

    def bc_actor(self, observations, goals=None, **kwargs):
        if self.goal_conditioned:
            return self.networks[f'bc_actor'](jnp.concatenate([observations, goals], axis=-1), **kwargs)
        else:
            return self.networks[f'bc_actor'](observations, **kwargs)

    def __call__(self, observations, goals=None, actions=None, temperatures=None):
        # Only for initialization
        rets = {}
        rets.update({
            f'actor_{temp}': self.actor(observations, goals, temp, info=True) for temp in temperatures
        })
        rets.update({
            'bc_actor': self.bc_actor(observations, goals, info=True),
        })
        rets.update({
            'value': self.value(observations, goals),
            'target_value': self.target_value(observations, goals),
        })
        if not self.value_only:
            rets.update({
                'critic': self.critic(observations, goals, actions),
                'target_critic': self.target_critic(observations, goals, actions),
            })
        return rets
