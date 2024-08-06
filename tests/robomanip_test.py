import envs.locomotion  # noqa
import gymnasium
import numpy as np

from envs.robomanip import oracles
from envs.robomanip.pick_place import RoboManipEnv


def main():
    env = RoboManipEnv(
        absolute_action_space=True,
        randomize_object_orientation=True,
        randomize_object_position=True,
        randomize_target_orientation=True,
        randomize_target_position=True,
    )
    obs, _ = env.reset(seed=12345)
    agent = oracles.PickPlaceOracle()
    agent.reset(obs)
    obs, _ = env.reset()
    agent.reset(obs)
    for _ in range(1000):
        action = agent.select_action(obs)
        obs, *_ = env.step(action)


if __name__ == '__main__':
    main()
