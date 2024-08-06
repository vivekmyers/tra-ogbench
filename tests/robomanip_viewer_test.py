import time
from envs.robomanip import oracles
from envs.robomanip import viewer_utils
from envs.robomanip.pick_place import RoboManipEnv


def main() -> None:
    env = RoboManipEnv(
        absolute_action_space=True,
        randomize_object_orientation=True,
        randomize_object_position=True,
        randomize_target_orientation=True,
        randomize_target_position=True,
    )
    obs, _ = env.reset(seed=12345)
    key_callback = viewer_utils.KeyCallback(pause=True)
    agent = oracles.PickPlaceOracle()
    agent.reset(obs)
    with env.passive_viewer(key_callback=key_callback) as viewer:
        while viewer.is_running():
            step_start = time.time()
            if key_callback.reset:
                obs, _ = env.reset()
                agent.reset(obs)
                key_callback.reset = False
                key_callback.pause = True
            else:
                if not key_callback.pause:
                    action = agent.select_action(obs)
                    obs, *_ = env.step(action)
            viewer.sync()
            time_until_next_step = env.physics_timestep() - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()
