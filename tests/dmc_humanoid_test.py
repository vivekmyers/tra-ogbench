import envs.online_locomotion  # noqa
import gymnasium
from dm_control import suite
import numpy as np


def main():
    dmc_env = suite.load(domain_name='humanoid', task_name='walk', environment_kwargs=dict(flat_observation=True))
    dmc_ob = dmc_env.reset().observation['observations']

    qpos, qvel = dmc_env.physics.data.qpos, dmc_env.physics.data.qvel

    custom_env = gymnasium.make('HumanoidCustom-v0', render_mode='rgb_array', task='walk')
    custom_env.reset()
    custom_env.step(custom_env.action_space.sample())

    custom_env.unwrapped.set_state(qpos, qvel)
    custom_ob = custom_env.unwrapped._get_obs()

    print(np.linalg.norm(dmc_ob - custom_ob))

    action = custom_env.action_space.sample()
    dmc_env.step(action)
    dmc_ob_dict = dmc_env.step(action)
    dmc_ob = dmc_ob_dict.observation['observations']
    dmc_reward = dmc_ob_dict.reward
    custom_ob, custom_reward, *_ = custom_env.step(action)
    custom_ob, custom_reward, *_ = custom_env.step(action)

    print(np.linalg.norm(dmc_ob - custom_ob))
    print(abs(dmc_reward - custom_reward))


if __name__ == '__main__':
    main()
