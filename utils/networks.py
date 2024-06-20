from typing import Sequence, Optional, Any

import distrax
import flax.linen as nn
import jax
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


class LogParam(nn.Module):
    init_value: float = 1.0

    @nn.compact
    def __call__(self):
        log_value = self.param('log_value', init_fn=lambda key: jnp.full((), jnp.log(self.init_value)))
        return jnp.exp(log_value)


class TransformedWithMode(distrax.Transformed):
    def mode(self):
        return self.bijector.forward(self.distribution.mode())


class GCActor(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    log_std_min: Optional[float] = -5
    log_std_max: Optional[float] = 2
    tanh_squash: bool = False
    state_dependent_std: bool = False
    const_std: bool = True
    final_fc_init_scale: float = 1e-2
    goal_encoder: nn.Module = None
    state_encoder: nn.Module = None

    def setup(self):
        self.actor_net = MLP(self.hidden_dims, activate_final=True)

        self.mean_net = nn.Dense(
            self.action_dim, kernel_init=default_init(self.final_fc_init_scale)
        )
        if self.state_dependent_std:
            self.log_std_net = nn.Dense(
                self.action_dim, kernel_init=default_init(self.final_fc_init_scale)
            )
        else:
            if not self.const_std:
                self.log_stds = self.param('log_stds', nn.initializers.zeros, (self.action_dim,))

    def __call__(
            self, observations, goals=None, goal_encoded=False, temperature=1.0,
    ):
        if self.goal_encoder is not None and not goal_encoded:
            goals = self.goal_encoder(targets=goals, bases=observations)
        if self.state_encoder is not None:
            observations = self.state_encoder(observations)

        inputs = [observations]
        if goals is not None:
            inputs.append(goals)
        inputs = jnp.concatenate(inputs, axis=-1)
        outputs = self.actor_net(inputs)

        means = self.mean_net(outputs)
        if self.state_dependent_std:
            log_stds = self.log_std_net(outputs)
        else:
            if self.const_std:
                log_stds = jnp.zeros_like(means)
            else:
                log_stds = self.log_stds

        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)

        distribution = distrax.MultivariateNormalDiag(loc=means, scale_diag=jnp.exp(log_stds) * temperature)
        if self.tanh_squash:
            distribution = TransformedWithMode(distribution, distrax.Block(distrax.Tanh(), ndims=1))

        return distribution


class GCValue(nn.Module):
    hidden_dims: Sequence[int]
    layer_norm: bool = True
    ensemble: bool = True
    value_exp: bool = False
    goal_encoder: nn.Module = None
    state_encoder: nn.Module = None

    def setup(self):
        mlp_module = LayerNormMLP if self.layer_norm else MLP
        if self.ensemble:
            mlp_module = ensemblize(mlp_module, 2)
        value_net = mlp_module((*self.hidden_dims, 1), activate_final=False)

        self.value_net = value_net

    def __call__(self, observations, goals=None, actions=None, info=False):
        if self.goal_encoder is not None:
            goals = self.goal_encoder(targets=goals, bases=observations)
        if self.state_encoder is not None:
            observations = self.state_encoder(observations)

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


class GCBilinearValue(nn.Module):
    hidden_dims: Sequence[int]
    latent_dim: int
    layer_norm: bool = True
    ensemble: bool = True
    value_exp: bool = False
    goal_encoder: nn.Module = None
    state_encoder: nn.Module = None

    def setup(self) -> None:
        mlp_module = LayerNormMLP if self.layer_norm else MLP
        if self.ensemble:
            mlp_module = ensemblize(mlp_module, 2)

        self.phi = mlp_module((*self.hidden_dims, self.latent_dim), activate_final=False)
        self.psi = mlp_module((*self.hidden_dims, self.latent_dim), activate_final=False)

    def __call__(self, observations, goals, actions=None, info=False):
        if self.goal_encoder is not None:
            goals = self.goal_encoder(goals)
        if self.state_encoder is not None:
            observations = self.state_encoder(observations)

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


class RelativeRepresentation(nn.Module):
    hidden_dims: Sequence[int]
    rep_dim: int
    layer_norm: bool = True
    kernel_init: Any = default_init()
    rep_type: str = 'state'
    encoder: nn.Module = None

    @nn.compact
    def __call__(self, targets, bases=None):
        if bases is None:
            inputs = targets
        else:
            if self.rep_type == 'state':
                inputs = targets
            elif self.rep_type == 'diff':
                inputs = jax.tree_util.tree_map(lambda t, b: t - b + jnp.ones_like(t) * 1e-9, targets, bases)
            elif self.rep_type == 'concat':
                inputs = jax.tree_util.tree_map(lambda t, b: jnp.concatenate([t, b], axis=-1), targets, bases)
            else:
                raise NotImplementedError

        if self.encoder is not None:
            inputs = self.encoder()(inputs)

        mlp_module = LayerNormMLP if self.layer_norm else MLP
        reps = mlp_module((*self.hidden_dims, self.rep_dim), activate_final=False)(inputs)
        reps = reps / jnp.linalg.norm(reps, axis=-1, keepdims=True) * jnp.sqrt(self.rep_dim)

        return reps
