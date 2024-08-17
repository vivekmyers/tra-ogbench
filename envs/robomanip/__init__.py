from gymnasium.envs.registration import register

register(
    id='cube-v0',
    entry_point='envs.robomanip.robomanip:RoboManipEnv',
    max_episode_steps=200,
    kwargs=dict(
        env_type='cube',
    ),
)
register(
    id='cubes-v0',
    entry_point='envs.robomanip.robomanip:RoboManipEnv',
    max_episode_steps=1000,
    kwargs=dict(
        env_type='cubes',
    ),
)
