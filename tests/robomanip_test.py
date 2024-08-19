import numpy as np

from envs.robomanip import oracles
from envs.robomanip.robomanip import RoboManipEnv


def main():
    use_oracle = True
    oracle_type = 'learned'
    env = RoboManipEnv(
        env_type='cube_single',
        absolute_action_space=(oracle_type == 'open'),
        terminate_at_goal=False,
        mode='data_collection',
        visualize_info=True,
    )
    ob, info = env.reset(seed=12345)
    if oracle_type == 'open':
        agent = oracles.OpenLoopCubeOracle(segment_dt=0.32)
    elif oracle_type == 'closed':
        agent = oracles.ClosedLoopCubeOracle()
    else:
        agent = oracles.LearnedCubeOracle(
            'exp/restore/sd320002',
            4000000,
            env.observation_space.shape[0] + 5,
            env.action_space.shape[0],
        )
    agent.reset(ob, info)
    ob, info = env.reset()
    agent.reset(ob, info)
    step = 0
    obs = []
    for _ in range(1000):
        action = agent.select_action(ob, info)
        if oracle_type == 'open':
            action = env.normalize_action(action)
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
