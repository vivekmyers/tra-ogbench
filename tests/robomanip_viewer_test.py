import time

import numpy as np

from envs.robomanip import oracles, viewer_utils
from envs.robomanip.robomanip import RoboManipEnv

SPEED_UP = 3.0


def main() -> None:
    use_oracle = True
    oracle_type = 'closed'
    env = RoboManipEnv(
        env_type='cubes',
        absolute_action_space=(oracle_type == 'open'),
        terminate_at_goal=False,
        mode='data_collection',
        # mode='evaluation',
        visualize_info=True,
    )

    obs, info = env.reset(seed=12345)
    key_callback = viewer_utils.KeyCallback(pause=True)
    if use_oracle:
        if oracle_type == 'open':
            agent = oracles.OpenLoopCubeOracle(segment_dt=0.32)
        else:
            agent = oracles.ClosedLoopCubeOracle()
        agent.reset(obs, info)
    step = 0
    with env.passive_viewer(key_callback=key_callback) as viewer:
        while viewer.is_running():
            step_start = time.time()
            if key_callback.reset:
                obs, info = env.reset()
                if use_oracle:
                    agent.reset(obs, info)
                step = 0
                key_callback.reset = False
                # key_callback.pause = True
            else:
                if not key_callback.pause:
                    if use_oracle:
                        action = agent.select_action(obs, info)
                        if oracle_type == 'open':
                            action = env.normalize_action(action)
                        # action = action + np.random.uniform(-0.25, 0.25, size=action.shape)
                        action = np.clip(action, -1, 1)
                    else:
                        action = env.action_space.sample()
                        action[:] = 0
                    obs, _, _, _, info = env.step(action)
                    step += 1
            viewer.sync()
            time_until_next_step = env.control_timestep() / SPEED_UP - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

            if use_oracle:
                if agent.done:
                    obs, info = env.set_new_target()
                    agent.reset(obs, info)
                    print(step)


if __name__ == '__main__':
    main()
