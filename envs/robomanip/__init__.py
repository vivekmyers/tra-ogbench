from gymnasium.envs.registration import register

register(
    id='cubes-v0',
    entry_point='envs.robomanip.robomanip:RoboManipEnv',
    max_episode_steps=1000,
    kwargs=dict(
        absolute_action_space=True,
    ),
)
