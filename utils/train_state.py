import functools
from typing import Any

import flax
import jax
import jax.numpy as jnp
import optax

nonpytree_field = functools.partial(flax.struct.field, pytree_node=False)


class TrainState(flax.struct.PyTreeNode):
    step: int
    apply_fn: Any = nonpytree_field()
    model_def: Any = nonpytree_field()
    params: Any
    tx: Any = nonpytree_field()
    opt_state: Any

    @classmethod
    def create(cls, model_def, params, tx=None, **kwargs):
        if tx is not None:
            opt_state = tx.init(params)
        else:
            opt_state = None

        return cls(
            step=1,
            apply_fn=model_def.apply,
            model_def=model_def,
            params=params,
            tx=tx,
            opt_state=opt_state,
            **kwargs,
        )

    def __call__(self, method=None):
        def inner(*args, params=None, **kwargs):
            if params is None:
                params = self.params
            variables = {"params": params}
            if method is not None:
                method_name = getattr(self.model_def, method)
            else:
                method_name = None

            return self.apply_fn(variables, *args, method=method_name, **kwargs)
        return inner

    def apply_gradients(self, grads, **kwargs):
        updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)

        return self.replace(
            step=self.step + 1,
            params=new_params,
            opt_state=new_opt_state,
            **kwargs,
        )

    def apply_loss_fn(self, loss_fn):
        grads, info = jax.grad(loss_fn, has_aux=True)(self.params)

        # Compute grad stats
        grad_max = jax.tree_util.tree_map(jnp.max, grads)
        grad_min = jax.tree_util.tree_map(jnp.min, grads)
        grad_norm = jax.tree_util.tree_map(jnp.linalg.norm, grads)

        grad_max_flat = jnp.concatenate([jnp.reshape(x, -1) for x in jax.tree_util.tree_leaves(grad_max)], axis=0)
        grad_min_flat = jnp.concatenate([jnp.reshape(x, -1) for x in jax.tree_util.tree_leaves(grad_min)], axis=0)
        grad_norm_flat = jnp.concatenate([jnp.reshape(x, -1) for x in jax.tree_util.tree_leaves(grad_norm)], axis=0)

        final_grad_max = jnp.max(grad_max_flat)
        final_grad_min = jnp.min(grad_min_flat)
        final_grad_norm = jnp.linalg.norm(grad_norm_flat)

        info.update({
            'grad/max': final_grad_max,
            'grad/min': final_grad_min,
            'grad/norm': final_grad_norm,
        })

        return self.apply_gradients(grads=grads), info

    def save(self):
        return {
            'params': self.params,
            'opt_state': self.opt_state,
            'step': self.step,
        }

    def load(self, data):
        return self.replace(**data)
