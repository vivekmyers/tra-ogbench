import time
from envs.robomanip import oracles
from envs.robomanip import viewer_utils
from envs.robomanip.pick_place import RoboManipEnv


SPEED_UP = 2.0


def main() -> None:
    env = RoboManipEnv(
        absolute_action_space=True,
        physics_timestep=0.004,
        control_timestep=0.04,
    )
    obs, info = env.reset(seed=12345)
    key_callback = viewer_utils.KeyCallback(pause=True)
    agent = oracles.PickPlaceOracle(segment_dt=0.32)
    agent.reset(obs, info)
    step = 0
    with env.passive_viewer(key_callback=key_callback) as viewer:
        while viewer.is_running():
            step_start = time.time()
            if key_callback.reset:
                obs, info = env.reset()
                agent.reset(obs, info)
                step = 0
                key_callback.reset = False
                # key_callback.pause = True
            else:
                if not key_callback.pause:
                    action = agent.select_action(obs)
                    # action = env.action_space.sample()
                    obs, *_ = env.step(action)
                    step += 1
            viewer.sync()
            time_until_next_step = env.control_timestep() / SPEED_UP - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

            if agent.done:
                print(step)
                step = 0
                key_callback.reset = True


if __name__ == "__main__":
    main()
