import envs.locomotion  # noqa
import gymnasium
import numpy as np

from envs.robomanip import oracles
from envs.robomanip.pick_place import RoboManipEnv


def main():
    env = RoboManipEnv(
        absolute_action_space=True,
        physics_timestep=0.004,
        control_timestep=0.04,
    )
    obs, info = env.reset(seed=12345)
    agent = oracles.PickPlaceOracle(segment_dt=0.32)
    agent.reset(obs, info)
    obs, info = env.reset()
    agent.reset(obs, info)
    step = 0
    for _ in range(1000):
        action = agent.select_action(obs)
        obs, *_ = env.step(action)
        step += 1

        if agent.done:
            obs, info = env.set_new_target()
            agent.reset(obs, info)
            print('done', step)


if __name__ == '__main__':
    main()
