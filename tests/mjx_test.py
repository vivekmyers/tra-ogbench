import functools
import os
import warnings

import jax
import jax.numpy as jnp
import mujoco
from brax import envs
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
from brax.training.agents.ppo import train as ppo

_DEFAULT_VALUE_AT_MARGIN = 0.1
_STAND_HEIGHT = 1.4
_WALK_SPEED = 1
_RUN_SPEED = 10


def _sigmoids(x, value_at_1, sigmoid):
    if sigmoid in ('cosine', 'linear', 'quadratic'):
        if not 0 <= value_at_1 < 1:
            raise ValueError('`value_at_1` must be nonnegative and smaller than 1, ' 'got {}.'.format(value_at_1))
    else:
        if not 0 < value_at_1 < 1:
            raise ValueError('`value_at_1` must be strictly between 0 and 1, ' 'got {}.'.format(value_at_1))

    if sigmoid == 'gaussian':
        scale = jnp.sqrt(-2 * jnp.log(value_at_1))
        return jnp.exp(-0.5 * (x * scale) ** 2)

    elif sigmoid == 'hyperbolic':
        scale = jnp.arccosh(1 / value_at_1)
        return 1 / jnp.cosh(x * scale)

    elif sigmoid == 'long_tail':
        scale = jnp.sqrt(1 / value_at_1 - 1)
        return 1 / ((x * scale) ** 2 + 1)

    elif sigmoid == 'reciprocal':
        scale = 1 / value_at_1 - 1
        return 1 / (abs(x) * scale + 1)

    elif sigmoid == 'cosine':
        scale = jnp.arccos(2 * value_at_1 - 1) / jnp.pi
        scaled_x = x * scale
        with warnings.catch_warnings():
            warnings.filterwarnings(action='ignore', message='invalid value encountered in cos')
            cos_pi_scaled_x = jnp.cos(jnp.pi * scaled_x)
        return jnp.where(abs(scaled_x) < 1, (1 + cos_pi_scaled_x) / 2, 0.0)

    elif sigmoid == 'linear':
        scale = 1 - value_at_1
        scaled_x = x * scale
        return jnp.where(abs(scaled_x) < 1, 1 - scaled_x, 0.0)

    elif sigmoid == 'quadratic':
        scale = jnp.sqrt(1 - value_at_1)
        scaled_x = x * scale
        return jnp.where(abs(scaled_x) < 1, 1 - scaled_x**2, 0.0)

    elif sigmoid == 'tanh_squared':
        scale = jnp.arctanh(jnp.sqrt(1 - value_at_1))
        return 1 - jnp.tanh(x * scale) ** 2

    else:
        raise ValueError('Unknown sigmoid type {!r}.'.format(sigmoid))


def tolerance(x, bounds=(0.0, 0.0), margin=0.0, sigmoid='gaussian', value_at_margin=_DEFAULT_VALUE_AT_MARGIN):
    lower, upper = bounds
    if lower > upper:
        raise ValueError('Lower bound must be <= upper bound.')
    if margin < 0:
        raise ValueError('`margin` must be non-negative.')

    in_bounds = jnp.logical_and(lower <= x, x <= upper)
    if margin == 0:
        value = jnp.where(in_bounds, 1.0, 0.0)
    else:
        d = jnp.where(x < lower, lower - x, x - upper) / margin
        value = jnp.where(in_bounds, 1.0, _sigmoids(d, value_at_margin, sigmoid))

    return float(value) if jnp.isscalar(x) else value


class Humanoid(PipelineEnv):
    def __init__(
        self,
        task='walk',
        **kwargs,
    ):
        path = os.path.join(os.path.dirname(__file__), '../envs/locomotion/assets/humanoid.xml')
        mj_model = mujoco.MjModel.from_xml_path(path)

        self._move_speed = {
            'walk': _WALK_SPEED,
            'run': _RUN_SPEED,
        }[task]

        sys = mjcf.load_model(mj_model)

        physics_steps_per_control_step = 5
        kwargs['n_frames'] = kwargs.get('n_frames', physics_steps_per_control_step)
        kwargs['backend'] = 'mjx'

        super().__init__(sys, **kwargs)

    def reset(self, rng):
        qvel = jnp.zeros(self.sys.nv)

        qpos_list = []
        qpos_list.append(self.sys.qpos0[:3])

        rng, subrng = jax.random.split(rng)
        quat = jax.random.normal(key=subrng, shape=(4,))
        quat /= jnp.linalg.norm(quat)
        qpos_list.append(quat)

        for joint_id in range(1, self.sys.njnt):
            rng, subrng = jax.random.split(rng)
            range_min, range_max = self.sys.jnt_range[joint_id]
            qpos_list.append(jnp.array([jax.random.uniform(key=subrng, minval=range_min, maxval=range_max)]))

        qpos = jnp.concatenate(qpos_list)
        data = self.pipeline_init(qpos, qvel)

        obs = self._get_obs(data, data)
        reward, done, zero = jnp.zeros(3)
        metrics = self._get_info(data, data)

        return State(data, obs, reward, done, metrics)

    def step(self, state, action):
        data0 = state.pipeline_state
        data = self.pipeline_step(data0, action)
        obs = self._get_obs(data0, data)
        reward = self._get_reward(data0, data)
        done = 0.0
        state.metrics.update(self._get_info(data0, data))

        return state.replace(
            pipeline_state=data,
            obs=obs,
            reward=reward,
            done=done,
        )

    def _get_obs(self, data0, data):
        joint_angles = data.qpos[7:]
        head_height = data.xpos[2, 2]
        torso_frame = data.xmat[1].reshape(3, 3)
        torso_pos = data.xpos[1]
        positions = []
        for idx in [16, 10, 13, 7]:
            torso_to_limb = data.xpos[idx] - torso_pos
            positions.append(torso_to_limb.dot(torso_frame))
        extremities = jnp.hstack(positions)
        torso_vertical_orientation = data.xmat[1].reshape((9,))[6:9]
        center_of_mass_velocity = (data.subtree_com[0] - data0.subtree_com[0]) / self.dt
        velocity = data.qvel

        return jnp.concatenate(
            [
                joint_angles,
                jnp.array([head_height]),
                extremities,
                torso_vertical_orientation,
                center_of_mass_velocity,
                velocity,
            ]
        )

    def _get_reward(self, data0, data):
        head_height = data.xpos[2, 2]
        torso_upright = data.xmat[1].reshape(9)[8]
        center_of_mass_velocity = (data.subtree_com[0] - data0.subtree_com[0]) / self.dt
        control = data.ctrl.copy()

        standing = tolerance(head_height, bounds=(_STAND_HEIGHT, float('inf')), margin=_STAND_HEIGHT / 4)
        upright = tolerance(torso_upright, bounds=(0.9, float('inf')), margin=1.9, sigmoid='linear', value_at_margin=0)
        stand_reward = standing * upright
        small_control = tolerance(control, margin=1, value_at_margin=0, sigmoid='quadratic').mean()
        small_control = (4 + small_control) / 5
        if self._move_speed == 0:
            horizontal_velocity = center_of_mass_velocity[0:2]
            dont_move = tolerance(horizontal_velocity, margin=2).mean()
            return small_control * stand_reward * dont_move
        else:
            com_velocity = jnp.linalg.norm(center_of_mass_velocity[0:2])
            move = tolerance(
                com_velocity,
                bounds=(self._move_speed, float('inf')),
                margin=self._move_speed,
                value_at_margin=0,
                sigmoid='linear',
            )
            move = (5 * move + 1) / 6
            return small_control * stand_reward * move

    def _get_info(self, data0, data):
        return dict(
            x=data.subtree_com[0][0],
            y=data.subtree_com[0][1],
        )


envs.register_environment('humanoid', Humanoid)


def main():
    env = envs.get_environment('humanoid')

    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    state = jit_reset(jax.random.PRNGKey(0))
    rollout = [state.pipeline_state]

    # for i in range(1000):
    #     print(i)
    #     ctrl = -0.1 * jnp.ones(env.sys.nu)
    #     state = jit_step(state, ctrl)
    #     rollout.append(state.pipeline_state)

    train_fn = functools.partial(
        # ppo.train, num_timesteps=30_000_000, num_evals=5, reward_scaling=0.1,
        # episode_length=1000, normalize_observations=True, action_repeat=1,
        # unroll_length=10, num_minibatches=32, num_updates_per_batch=8,
        # discounting=0.97, learning_rate=3e-4, entropy_cost=1e-3, num_envs=2048,
        # batch_size=1024, seed=0,
        ppo.train,
        num_timesteps=30_000_000,
        num_evals=5,
        reward_scaling=0.1,
        episode_length=1000,
        normalize_observations=True,
        action_repeat=1,
        unroll_length=10,
        num_minibatches=32,
        num_updates_per_batch=8,
        discounting=0.97,
        learning_rate=3e-4,
        entropy_cost=1e-3,
        num_envs=1,
        batch_size=256,
        seed=0,
    )

    x_data = []
    y_data = []
    ydataerr = []

    def progress(num_steps, metrics):
        x_data.append(num_steps)
        y_data.append(metrics['eval/episode_reward'])
        ydataerr.append(metrics['eval/episode_reward_std'])
        print(num_steps, metrics['eval/episode_reward'])

    make_inference_fn, params, _ = train_fn(environment=env, progress_fn=progress)


if __name__ == '__main__':
    main()
