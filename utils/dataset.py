import dataclasses
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax.core import freeze
from flax.core.frozen_dict import FrozenDict
from jax import tree_util


def get_size(data):
    sizes = tree_util.tree_map(lambda arr: len(arr), data)
    return max(tree_util.tree_leaves(sizes))


@partial(jax.jit, static_argnames=('padding',))
def random_crop(img, crop_from, padding):
    padded_img = jnp.pad(img, ((padding, padding), (padding, padding), (0, 0)), mode='edge')
    return jax.lax.dynamic_slice(padded_img, crop_from, img.shape)


@partial(jax.jit, static_argnames=('padding',))
def batched_random_crop(imgs, crop_froms, padding):
    return jax.vmap(random_crop, (0, 0, None))(imgs, crop_froms, padding)


class Dataset(FrozenDict):
    @classmethod
    def create(
            cls,
            observations,
            actions,
            rewards,
            masks,
            next_observations,
            freeze=True,
            **extra_fields,
    ):
        data = {
            'observations': observations,
            'actions': actions,
            'rewards': rewards,
            'masks': masks,
            'next_observations': next_observations,
            **extra_fields,
        }
        # Force freeze
        if freeze:
            tree_util.tree_map(lambda arr: arr.setflags(write=False), data)
        return cls(data)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.size = get_size(self._dict)

    def sample(self, batch_size: int, indices=None):
        if indices is None:
            indices = np.random.randint(self.size, size=batch_size)
        return self.get_subset(indices)

    def get_subset(self, indices):
        return tree_util.tree_map(lambda arr: arr[indices], self._dict)


@dataclasses.dataclass
class GCDataset:
    dataset: Dataset
    config: Any

    def __post_init__(self):
        self.size = self.dataset.size
        self.terminal_locs, = np.nonzero(self.dataset['dones_float'] > 0)
        assert np.isclose(self.config['value_p_curgoal'] + self.config['value_p_trajgoal'] + self.config['value_p_randomgoal'], 1.0)
        assert np.isclose(self.config['actor_p_curgoal'] + self.config['actor_p_trajgoal'] + self.config['actor_p_randomgoal'], 1.0)

    def sample_goals(self, indices, p_curgoal, p_trajgoal, p_randomgoal, geom_sample):
        batch_size = len(indices)

        # Random goals
        random_goal_indices = np.random.randint(self.size, size=batch_size)

        # Goals from the same trajectory (excluding the current state, unless it is the final state)
        final_state_indices = self.terminal_locs[np.searchsorted(self.terminal_locs, indices)]
        if geom_sample:
            offsets = np.random.geometric(p=1 - self.config['discount'], size=batch_size)  # in [1, inf)
            middle_goal_indices = np.minimum(indices + offsets, final_state_indices)
        else:
            distances = np.random.rand(batch_size)  # in [0, 1)
            middle_goal_indices = np.round((np.minimum(indices + 1, final_state_indices) * distances + final_state_indices * (1 - distances))).astype(int)
        goal_indices = np.where(np.random.rand(batch_size) < p_trajgoal / (1.0 - p_curgoal + 1e-9), middle_goal_indices, random_goal_indices)

        # Goals at the current state
        goal_indices = np.where(np.random.rand(batch_size) < p_curgoal, indices, goal_indices)

        return goal_indices

    def sample(self, batch_size: int, indices=None, evaluation=False):
        if indices is None:
            indices = np.random.randint(self.dataset.size - 1, size=batch_size)

        batch = self.dataset.sample(batch_size, indices)

        value_goal_indices = self.sample_goals(indices, self.config['value_p_curgoal'], self.config['value_p_trajgoal'], self.config['value_p_randomgoal'], self.config['value_geom_sample'])
        actor_goal_indices = self.sample_goals(indices, self.config['actor_p_curgoal'], self.config['actor_p_trajgoal'], self.config['actor_p_randomgoal'], self.config['actor_geom_sample'])

        batch['value_goals'] = jax.tree_util.tree_map(lambda arr: arr[value_goal_indices], self.dataset['observations'])
        batch['actor_goals'] = jax.tree_util.tree_map(lambda arr: arr[actor_goal_indices], self.dataset['observations'])
        successes = (indices == value_goal_indices).astype(float)
        batch['masks'] = 1.0 - successes
        batch['rewards'] = successes - (1.0 if self.config['gc_negative'] else 0.0)

        if self.config['p_aug'] is not None and not evaluation:
            if np.random.rand() < self.config['p_aug']:
                aug_keys = ['observations', 'next_observations', 'goals', 'actor_goals']
                padding = 3
                crop_froms = np.random.randint(0, 2 * padding + 1, (batch_size, 2))
                crop_froms = np.concatenate([crop_froms, np.zeros((batch_size, 1), dtype=np.int32)], axis=1)
                for key in aug_keys:
                    batch[key] = jax.tree_util.tree_map(
                        lambda arr: np.array(batched_random_crop(arr, crop_froms, padding)) if len(
                            arr.shape) == 4 else arr, batch[key])

        if isinstance(batch['value_goals'], FrozenDict):
            # Freeze the other observations
            batch['observations'] = freeze(batch['observations'])
            batch['next_observations'] = freeze(batch['next_observations'])

        return batch
