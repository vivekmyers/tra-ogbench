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
    agent = oracles.PickPlaceOracle()
    agent.reset(obs, info)
    obs, info = env.reset()
    agent.reset(obs, info)
    for _ in range(1000):
        action = agent.select_action(obs)
        # action = env.action_space.sample()
        obs, *_ = env.step(action)


if __name__ == '__main__':
    main()
