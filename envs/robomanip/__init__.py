from gymnasium.envs.registration import register

register(
    id='cube-v0',
    entry_point='envs.robomanip.robomanip:RoboManipEnv',
    max_episode_steps=200,
    kwargs=dict(
        env_type='cube',
        absolute_action_space=False,
    ),
)
register(
    id='cubes-v0',
    entry_point='envs.robomanip.robomanip:RoboManipEnv',
    max_episode_steps=1000,
    kwargs=dict(
        env_type='cubes',
        absolute_action_space=False,
    ),
)
