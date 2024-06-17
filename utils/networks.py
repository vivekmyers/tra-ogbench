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
    activations: Any = nn.gelu
    activate_final: bool = False
    kernel_init: Any = default_init()

    def setup(self):
        self.layers = [
            nn.Dense(size, kernel_init=self.kernel_init) for size in self.hidden_dims
        ]

    def __call__(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i + 1 < len(self.layers) or self.activate_final:
                x = self.activations(x)
        return x


class LayerNormMLP(nn.Module):
    hidden_dims: Sequence[int]
    activations: Any = nn.gelu
    activate_final: bool = False
    kernel_init: Any = default_init()

    @nn.compact
    def __call__(self, x):
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=self.kernel_init)(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                x = self.activations(x)
                x = nn.LayerNorm()(x)
        return x


class Actor(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    activations: Any = nn.gelu
    log_std_min: Optional[float] = -5
    log_std_max: Optional[float] = 2
    state_dependent_std: bool = False
    const_std: bool = True
    final_fc_init_scale: float = 1e-2
    encoder: nn.Module = None

    def setup(self):
        actor_module = MLP(self.hidden_dims, activations=self.activations, activate_final=True)
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
                self.log_stds = self.param('log_stds', nn.initializers.zeros, (self.action_dim,))

    def __call__(
            self, observations, goals=None, temperature=1.0,
    ):
        if goals is None:
            inputs = observations
        else:
            inputs = jnp.concatenate([observations, goals], axis=-1)
        outputs = self.actor_module(inputs)

        means = self.mean_module(outputs)
        if self.state_dependent_std:
            log_stds = self.log_std_module(outputs)
        else:
            if self.const_std:
                log_stds = jnp.zeros_like(means)
            else:
                log_stds = self.log_stds

        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)

        distribution = distrax.MultivariateNormalDiag(loc=means, scale_diag=jnp.exp(log_stds) * temperature)

        return distribution


class GoalConditionedValue(nn.Module):
    hidden_dims: Sequence[int]
    activations: Any = nn.gelu
    layer_norm: bool = True
    ensemble: bool = True
    value_exp: bool = False
    encoder: nn.Module = None

    def setup(self):
        mlp_module = LayerNormMLP if self.layer_norm else MLP
        if self.ensemble:
            mlp_module = ensemblize(mlp_module, 2)
        value_net = mlp_module((*self.hidden_dims, 1), activations=self.activations, activate_final=False)

        if self.encoder is not None:
            value_net = nn.Sequential([self.encoder(), value_net])

        self.value_net = value_net

    def __call__(self, observations, goals=None, actions=None, info=False):
        inputs = [observations]
        if goals is not None:
            inputs.append(goals)
        if actions is not None:
            inputs.append(actions)
        inputs = jnp.concatenate(inputs, axis=-1)

        v = self.value_net(inputs).squeeze(-1)

        if self.value_exp:
            v = jnp.exp(v)

        return v


class GoalConditionedBilinearValue(nn.Module):
    hidden_dims: Sequence[int]
    latent_dim: int
    activations: Any = nn.gelu
    layer_norm: bool = True
    ensemble: bool = True
    value_exp: bool = False
    encoder: nn.Module = None

    def setup(self) -> None:
        mlp_module = LayerNormMLP if self.layer_norm else MLP
        if self.ensemble:
            mlp_module = ensemblize(mlp_module, 2)
        phi = mlp_module((*self.hidden_dims, self.latent_dim), activations=self.activations, activate_final=False)
        psi = mlp_module((*self.hidden_dims, self.latent_dim), activations=self.activations, activate_final=False)

        if self.encoder is not None:
            phi = nn.Sequential([self.encoder(), phi])
            psi = nn.Sequential([self.encoder(), psi])

        self.phi = phi
        self.psi = psi

    def __call__(self, observations, goals=None, actions=None, info=False):
        if actions is None:
            phi_inputs = observations
        else:
            phi_inputs = jnp.concatenate([observations, actions], axis=-1)

        phi = self.phi(phi_inputs)
        psi = self.psi(goals)

        v = (phi * psi / jnp.sqrt(self.latent_dim)).sum(axis=-1)

        if self.value_exp:
            v = jnp.exp(v)

        if info:
            return v, phi, psi
        else:
            return v
