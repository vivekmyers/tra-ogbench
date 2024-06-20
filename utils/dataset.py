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
    def create(cls, freeze=True, **fields):
        data = fields
        # Force freeze
        if freeze:
            tree_util.tree_map(lambda arr: arr.setflags(write=False), data)
        return cls(data)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.size = get_size(self._dict)

    def sample(self, batch_size: int, idxs=None):
        if idxs is None:
            idxs = np.random.randint(self.size, size=batch_size)
        return self.get_subset(idxs)

    def get_subset(self, idxs):
        return tree_util.tree_map(lambda arr: arr[idxs], self._dict)


@dataclasses.dataclass
class GCDataset:
    dataset: Dataset
    config: Any

    def __post_init__(self):
        self.size = self.dataset.size
        self.terminal_locs, = np.nonzero(self.dataset['terminals'] > 0)
        assert np.isclose(self.config['value_p_curgoal'] + self.config['value_p_trajgoal'] + self.config['value_p_randomgoal'], 1.0)
        assert np.isclose(self.config['actor_p_curgoal'] + self.config['actor_p_trajgoal'] + self.config['actor_p_randomgoal'], 1.0)

    def augment(self, batch, keys):
        padding = 3
        batch_size = len(batch[keys[0]])
        crop_froms = np.random.randint(0, 2 * padding + 1, (batch_size, 2))
        crop_froms = np.concatenate([crop_froms, np.zeros((batch_size, 1), dtype=np.int32)], axis=1)
        for key in keys:
            batch[key] = jax.tree_util.tree_map(lambda arr: np.array(batched_random_crop(arr, crop_froms, padding)) if len(arr.shape) == 4 else arr, batch[key])

    def sample_goals(self, idxs, p_curgoal, p_trajgoal, p_randomgoal, geom_sample):
        batch_size = len(idxs)

        # Random goals
        random_goal_idxs = np.random.randint(self.size, size=batch_size)

        # Goals from the same trajectory (excluding the current state, unless it is the final state)
        final_state_idxs = self.terminal_locs[np.searchsorted(self.terminal_locs, idxs)]
        if geom_sample:
            offsets = np.random.geometric(p=1 - self.config['discount'], size=batch_size)  # in [1, inf)
            middle_goal_idxs = np.minimum(idxs + offsets, final_state_idxs)
        else:
            distances = np.random.rand(batch_size)  # in [0, 1)
            middle_goal_idxs = np.round((np.minimum(idxs + 1, final_state_idxs) * distances + final_state_idxs * (1 - distances))).astype(int)
        goal_idxs = np.where(np.random.rand(batch_size) < p_trajgoal / (1.0 - p_curgoal + 1e-9), middle_goal_idxs, random_goal_idxs)

        # Goals at the current state
        goal_idxs = np.where(np.random.rand(batch_size) < p_curgoal, idxs, goal_idxs)

        return goal_idxs

    def sample(self, batch_size: int, idxs=None, evaluation=False):
        if idxs is None:
            idxs = np.random.randint(self.dataset.size - 1, size=batch_size)

        batch = self.dataset.sample(batch_size, idxs)

        value_goal_idxs = self.sample_goals(idxs, self.config['value_p_curgoal'], self.config['value_p_trajgoal'], self.config['value_p_randomgoal'], self.config['value_geom_sample'])
        actor_goal_idxs = self.sample_goals(idxs, self.config['actor_p_curgoal'], self.config['actor_p_trajgoal'], self.config['actor_p_randomgoal'], self.config['actor_geom_sample'])

        batch['value_goals'] = jax.tree_util.tree_map(lambda arr: arr[value_goal_idxs], self.dataset['observations'])
        batch['actor_goals'] = jax.tree_util.tree_map(lambda arr: arr[actor_goal_idxs], self.dataset['observations'])
        successes = (idxs == value_goal_idxs).astype(float)
        batch['masks'] = 1.0 - successes
        batch['rewards'] = successes - (1.0 if self.config['gc_negative'] else 0.0)

        if self.config['p_aug'] is not None and not evaluation:
            if np.random.rand() < self.config['p_aug']:
                self.augment(batch, ['observations', 'next_observations', 'value_goals', 'actor_goals'])

        if isinstance(batch['value_goals'], FrozenDict):
            # Freeze the other observations
            batch['observations'] = freeze(batch['observations'])
            batch['next_observations'] = freeze(batch['next_observations'])

        return batch


@dataclasses.dataclass
class HGCDataset(GCDataset):
    def sample(self, batch_size: int, idxs=None, evaluation=False):
        if idxs is None:
            idxs = np.random.randint(self.dataset.size - 1, size=batch_size)

        batch = self.dataset.sample(batch_size, idxs)

        # Sample value goals
        value_goal_idxs = self.sample_goals(idxs, self.config['value_p_curgoal'], self.config['value_p_trajgoal'], self.config['value_p_randomgoal'], self.config['value_geom_sample'])
        batch['value_goals'] = jax.tree_util.tree_map(lambda arr: arr[value_goal_idxs], self.dataset['observations'])

        successes = (idxs == value_goal_idxs).astype(float)
        batch['masks'] = 1.0 - successes
        batch['rewards'] = successes - (1.0 if self.config['gc_negative'] else 0.0)

        # Set low actor goals
        final_state_idxs = self.terminal_locs[np.searchsorted(self.terminal_locs, idxs)]
        low_goal_idxs = np.minimum(idxs + self.config['subgoal_steps'], final_state_idxs)
        batch['low_actor_goals'] = jax.tree_map(lambda arr: arr[low_goal_idxs], self.dataset['observations'])

        # Sample high actor goals and prediction targets
        if self.config['actor_geom_sample']:
            offsets = np.random.geometric(p=1 - self.config['discount'], size=batch_size)  # in [1, inf)
            high_traj_goal_idxs = np.minimum(idxs + offsets, final_state_idxs)
        else:
            distances = np.random.rand(batch_size)  # in [0, 1)
            high_traj_goal_idxs = np.round((np.minimum(idxs + 1, final_state_idxs) * distances + final_state_idxs * (1 - distances))).astype(int)
        high_traj_target_idxs = np.minimum(idxs + self.config['subgoal_steps'], high_traj_goal_idxs)

        high_random_goal_idxs = np.random.randint(self.dataset.size, size=batch_size)
        high_random_target_idxs = np.minimum(idxs + self.config['subgoal_steps'], final_state_idxs)

        pick_random = (np.random.rand(batch_size) < self.config['actor_p_randomgoal'])
        high_goal_idxs = np.where(pick_random, high_random_goal_idxs, high_traj_goal_idxs)
        high_target_idxs = np.where(pick_random, high_random_target_idxs, high_traj_target_idxs)

        batch['high_actor_goals'] = jax.tree_map(lambda arr: arr[high_goal_idxs], self.dataset['observations'])
        batch['high_actor_targets'] = jax.tree_map(lambda arr: arr[high_target_idxs], self.dataset['observations'])

        if self.config['p_aug'] is not None and not evaluation:
            if np.random.rand() < self.config['p_aug']:
                self.augment(batch, ['observations', 'next_observations', 'value_goals', 'low_actor_goals', 'high_actor_goals', 'high_actor_targets'])

        if isinstance(batch['value_goals'], FrozenDict):
            # Freeze the other observations
            batch['observations'] = freeze(batch['observations'])
            batch['next_observations'] = freeze(batch['next_observations'])

        return batch


class ReplayBuffer(Dataset):
    @classmethod
    def create(cls, transition, size):
        def create_buffer(example):
            example = np.array(example)
            return np.zeros((size, *example.shape), dtype=example.dtype)

        buffer_dict = tree_util.tree_map(create_buffer, transition)
        return cls(buffer_dict)

    @classmethod
    def create_from_initial_dataset(cls, init_dataset, size):
        def create_buffer(init_buffer):
            buffer = np.zeros((size, *init_buffer.shape[1:]), dtype=init_buffer.dtype)
            buffer[: len(init_buffer)] = init_buffer
            return buffer

        buffer_dict = tree_util.tree_map(create_buffer, init_dataset)
        dataset = cls(buffer_dict)
        dataset.size = dataset.pointer = get_size(init_dataset)
        return dataset

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.max_size = get_size(self._dict)
        self.size = 0
        self.pointer = 0

    def add_transition(self, transition):
        def set_idx(buffer, new_element):
            buffer[self.pointer] = new_element

        tree_util.tree_map(set_idx, self._dict, transition)
        self.pointer = (self.pointer + 1) % self.max_size
        self.size = max(self.pointer, self.size)

    def clear(self):
        self.size = self.pointer = 0