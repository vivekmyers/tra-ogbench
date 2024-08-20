import numpy as np

from envs.robomanip import oracles
from envs.robomanip.cube import CubeEnv


def main():
    env = CubeEnv(
        env_type='cube_single',
        terminate_at_goal=False,
        mode='data_collection',
        # mode='evaluation',
        visualize_info=True,
    )
    ob, info = env.reset(seed=12345)
    agent = oracles.ClosedLoopCubeOracle()
    agent.reset(ob, info)
    ob, info = env.reset()
    agent.reset(ob, info)
    step = 0
    obs = []
    for _ in range(1000):
        action = agent.select_action(ob, info)
        action = np.array(action)
        ob, _, _, _, info = env.step(action)
        obs.append(ob)
        step += 1

        if agent.done:
            ob, info = env.set_new_target()
            agent.reset(ob, info)
            print('done', step)


if __name__ == '__main__':
    main()
