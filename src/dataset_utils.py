import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
from flax.core import freeze
from flax.core.frozen_dict import FrozenDict

from jaxrl_m.dataset import Dataset


def merge_datasets(dataset1, dataset2):
    return Dataset({
        k: np.concatenate([dataset1[k], dataset2[k]], axis=0)
        for k in dataset1.keys()
    })


def random_crop(img, crop_from, padding):
    padded_img = jnp.pad(img, ((padding, padding), (padding, padding), (0, 0)), mode='edge')
    return jax.lax.dynamic_slice(padded_img, crop_from, img.shape)


random_crop = jax.jit(random_crop, static_argnames=('padding',))


def batched_random_crop(imgs, crop_froms, padding):
    return jax.vmap(random_crop, (0, 0, None))(imgs, crop_froms, padding)


batched_random_crop = jax.jit(batched_random_crop, static_argnames=('padding',))


@dataclasses.dataclass
class GCDataset:
    dataset: Dataset
    p_randomgoal: float
    p_trajgoal: float
    p_currgoal: float
    geom_sample: int
    policy_geom_sample: int
    discount: float
    terminal_key: str = 'dones_float'
    policy_p_randomgoal: float = 0.0
    p_aug: float = None
    color_aug: int = 0

    def __post_init__(self):
        self.terminal_locs, = np.nonzero(self.dataset[self.terminal_key] > 0)
        assert np.isclose(self.p_randomgoal + self.p_trajgoal + self.p_currgoal, 1.0)

    def sample_goals(self, indx, p_randomgoal=None, p_trajgoal=None, p_currgoal=None):
        if p_randomgoal is None:
            p_randomgoal = self.p_randomgoal
        if p_trajgoal is None:
            p_trajgoal = self.p_trajgoal
        if p_currgoal is None:
            p_currgoal = self.p_currgoal

        batch_size = len(indx)
        # Random goals
        goal_indx = np.random.randint(self.dataset.size, size=batch_size)

        # Goals from the same trajectory
        final_state_indx = self.terminal_locs[np.searchsorted(self.terminal_locs, indx)]

        distance = np.random.rand(batch_size)
        if self.geom_sample:
            us = np.random.rand(batch_size)
            middle_goal_indx = np.minimum(indx + np.ceil(np.log(1 - us) / np.log(self.discount)).astype(int), final_state_indx)
        else:
            middle_goal_indx = np.round((np.minimum(indx + 1, final_state_indx) * distance + final_state_indx * (1 - distance))).astype(int)

        goal_indx = np.where(np.random.rand(batch_size) < p_trajgoal / (1.0 - p_currgoal), middle_goal_indx, goal_indx)

        # Goals at the current state
        goal_indx = np.where(np.random.rand(batch_size) < p_currgoal, indx, goal_indx)
        return goal_indx

    def sample(self, batch_size: int, indx=None, evaluation=False):
        if indx is None:
            indx = np.random.randint(self.dataset.size - 1, size=batch_size)

        batch = self.dataset.sample(batch_size, indx)
        batch['ori_rewards'] = batch['rewards']
        goal_indx = self.sample_goals(indx)

        success = (indx == goal_indx)

        batch['rewards'] = success.astype(float)
        batch['masks'] = (1.0 - success.astype(float))

        batch['goals'] = jax.tree_map(lambda arr: arr[goal_indx], self.dataset['observations'])

        final_state_indx = self.terminal_locs[np.searchsorted(self.terminal_locs, indx)]

        distance = np.random.rand(batch_size)
        if self.policy_geom_sample:
            us = np.random.rand(batch_size)
            policy_traj_goal_indx = np.minimum(indx + np.ceil(np.log(1 - us) / np.log(self.discount)).astype(int), final_state_indx)
        else:
            policy_traj_goal_indx = np.round((np.minimum(indx + 1, final_state_indx) * distance + final_state_indx * (1 - distance))).astype(int)
        policy_random_goal_indx = np.random.randint(self.dataset.size, size=batch_size)
        pick_random = (np.random.rand(batch_size) < self.policy_p_randomgoal)
        policy_goal_idx = np.where(pick_random, policy_random_goal_indx, policy_traj_goal_indx)

        batch['policy_goals'] = jax.tree_map(lambda arr: arr[policy_goal_idx], self.dataset['observations'])

        if self.p_aug is not None and not evaluation:
            if np.random.rand() < self.p_aug:
                aug_keys = ['observations', 'next_observations', 'goals', 'policy_goals']
                padding = 3
                crop_froms = np.random.randint(0, 2 * padding + 1, (batch_size, 2))
                crop_froms = np.concatenate([crop_froms, np.zeros((batch_size, 1), dtype=np.int32)], axis=1)
                for key in aug_keys:
                    batch[key] = jax.tree_map(lambda arr: np.array(batched_random_crop(arr, crop_froms, padding)) if len(arr.shape) == 4 else arr, batch[key])

                if self.color_aug:
                    import tensorflow as tf
                    for key in aug_keys:
                        batch[key] = tf.image.random_brightness(batch[key], 0.2)
                        batch[key] = tf.image.random_contrast(batch[key], 0.8, 1.2)
                        batch[key] = tf.image.random_saturation(batch[key], 0.8, 1.2)
                        batch[key] = tf.image.random_hue(batch[key], 0.1)
                        batch[key] = np.array(batch[key])

        if isinstance(batch['goals'], FrozenDict):
            # Freeze the other observations
            batch['observations'] = freeze(batch['observations'])
            batch['next_observations'] = freeze(batch['next_observations'])

        return batch
