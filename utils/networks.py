from typing import Sequence, Optional, Any

import distrax
import flax.linen as nn
import jax.numpy as jnp


def default_init(scale=1.0):
    return nn.initializers.variance_scaling(scale, 'fan_avg', 'uniform')


def ensemblize(cls, num_qs, out_axes=0, **kwargs):
    return nn.vmap(
        cls,
        variable_axes={'params': 0},
        split_rngs={'params': True},
        in_axes=None,
        out_axes=out_axes,
        axis_size=num_qs,
        **kwargs
    )


class MLP(nn.Module):
    hidden_dims: Sequence[int]
    activations: Any = nn.relu
    activate_final: int = False
    kernel_init: Any = default_init()

    def setup(self):
        self.layers = [
            nn.Dense(size, kernel_init=self.kernel_init) for size in self.hidden_dims
        ]

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i + 1 < len(self.layers) or self.activate_final:
                x = self.activations(x)
        return x


class LayerNormMLP(nn.Module):
    hidden_dims: Sequence[int]
    activations: Any = nn.gelu
    activate_final: int = False
    kernel_init: Any = default_init()

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=self.kernel_init)(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                x = self.activations(x)
                x = nn.LayerNorm()(x)
        return x


class Critic(nn.Module):
    hidden_dims: Sequence[int]
    activations: Any = nn.relu
    layer_norm: bool = True

    @nn.compact
    def __call__(self, observations: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        inputs = jnp.concatenate([observations, actions], -1)
        if self.layer_norm:
            critic = LayerNormMLP((*self.hidden_dims, 1), activations=self.activations)(inputs)
        else:
            critic = MLP((*self.hidden_dims, 1), activations=self.activations)(inputs)
        return jnp.squeeze(critic, -1)


class ValueCritic(nn.Module):
    hidden_dims: Sequence[int]
    layer_norm: bool = True

    @nn.compact
    def __call__(self, observations: jnp.ndarray) -> jnp.ndarray:
        if self.layer_norm:
            critic = LayerNormMLP((*self.hidden_dims, 1))(observations)
        else:
            critic = MLP((*self.hidden_dims, 1))(observations)
        return jnp.squeeze(critic, -1)


class Actor(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    log_std_min: Optional[float] = -20
    log_std_max: Optional[float] = 2
    state_dependent_std: bool = False
    const_std: bool = True
    final_fc_init_scale: float = 1e-2
    encoder: nn.Module = None

    def setup(self) -> None:
        actor_module = MLP(self.hidden_dims, activate_final=True)
        if self.encoder is not None:
            actor_module = nn.Sequential([self.encoder(), actor_module])
        self.actor_module = actor_module

        self.mean_module = nn.Dense(
            self.action_dim, kernel_init=default_init(self.final_fc_init_scale)
        )
        if self.state_dependent_std:
            self.log_std_module = nn.Dense(
                self.action_dim, kernel_init=default_init(self.final_fc_init_scale)
            )
        else:
            if not self.const_std:
                self.log_stds = self.param("log_stds", nn.initializers.zeros, (self.action_dim,))

    def __call__(
            self, observations, goals=None, temperature=1.0,
    ):
        if goals is None:
            outputs = self.actor_module(observations)
        else:
            outputs = self.actor_module(jnp.concatenate([observations, goals], axis=-1))

        means = self.mean_module(outputs)
        if self.state_dependent_std:
            log_stds = self.log_std_module(outputs)
        else:
            if not self.const_std:
                log_stds = self.log_stds
            else:
                log_stds = jnp.zeros_like(means)

        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)

        distribution = distrax.MultivariateNormalDiag(
            loc=means, scale_diag=jnp.exp(log_stds) * temperature
        )

        return distribution


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
