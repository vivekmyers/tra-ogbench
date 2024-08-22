import os
import time

import numpy as np

from envs.robomanip import viewer_utils
from envs.robomanip.button_env import ButtonEnv
from envs.robomanip.cube_env import CubeEnv
from envs.robomanip.oracles.button import ButtonOracle
from envs.robomanip.oracles.cube import CubeOracle
from envs.robomanip.oracles.drawer import DrawerOracle
from envs.robomanip.oracles.window import WindowOracle
from envs.robomanip.scene_env import SceneEnv

SPEED_UP = 3.0


def main():
    use_oracle = True
    use_viewer = (os.environ.get('USE_VIEWER', 'False') == 'True')
    env_type = 'button-game-3x3'
    mode = 'data_collection'
    # mode = 'evaluation'
    if 'cube' in env_type:
        env = CubeEnv(
            env_type=env_type,
            terminate_at_goal=False,
            mode=mode,
            visualize_info=True,
        )
    elif 'button' in env_type:
        env = ButtonEnv(
            env_type=env_type,
            terminate_at_goal=False,
            mode=mode,
            visualize_info=True,
        )
    elif 'scene' in env_type:
        env = SceneEnv(
            env_type=env_type,
            terminate_at_goal=False,
            mode=mode,
            visualize_info=True,
        )

    ob, info = env.reset(seed=0)
    if use_oracle:
        if 'cube' in env_type:
            agents = {
                'cube': CubeOracle(env=env),
            }
        elif 'button' in env_type:
            agents = {
                'button': ButtonOracle(env=env, gripper_always_closed=True),
            }
        else:
            agents = {
                'cube': CubeOracle(env=env),
                'button': ButtonOracle(env=env),
                'drawer': DrawerOracle(env=env),
                'window': WindowOracle(env=env),
            }
        agent = agents[info['privileged/target_task']]
        agent.reset(ob, info)

    if use_viewer:
        key_callback = viewer_utils.KeyCallback(pause=True)
        step = 0
        with env.passive_viewer(key_callback=key_callback) as viewer:
            while viewer.is_running():
                step_start = time.time()
                if key_callback.reset:
                    ob, info = env.reset()
                    if use_oracle:
                        agent = agents[info['privileged/target_task']]
                        agent.reset(ob, info)
                    step = 0
                    key_callback.reset = False
                else:
                    if not key_callback.pause:
                        if use_oracle:
                            action = agent.select_action(ob, info)
                            action = np.array(action)
                            action = np.clip(action, -1, 1)
                        else:
                            action = env.action_space.sample()
                            action[:] = 0
                        ob, _, _, _, info = env.step(action)
                        step += 1
                viewer.sync()
                time_until_next_step = env.control_timestep() / SPEED_UP - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

                if use_oracle and agent.done:
                    ob, info = env.set_new_target()
                    agent = agents[info['privileged/target_task']]
                    agent.reset(ob, info)
                    print(step)
    else:
        step = 0
        obs = []
        for _ in range(1000):
            if use_oracle:
                action = agent.select_action(ob, info)
                action = np.array(action)
                action = np.clip(action, -1, 1)
            else:
                action = env.action_space.sample()
                action[:] = 0
            ob, _, _, _, info = env.step(action)
            obs.append(ob)
            step += 1

            if use_oracle and agent.done:
                ob, info = env.set_new_target()
                agent = agents[info['privileged/target_task']]
                agent.reset(ob, info)
                print('done', step)
        obs = np.array(obs)
        print('done')


if __name__ == '__main__':
    main()
