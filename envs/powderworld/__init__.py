from gymnasium.envs.registration import register

register(
    id='powderworld-2-v0',
    entry_point='envs.powderworld.powderworld_env:PowderworldEnv',
    max_episode_steps=500,
    kwargs=dict(num_elems=2),
)

register(
    id='powderworld-5-v0',
    entry_point='envs.powderworld.powderworld_env:PowderworldEnv',
    max_episode_steps=500,
    kwargs=dict(num_elems=5),
)

register(
    id='powderworld-8-v0',
    entry_point='envs.powderworld.powderworld_env:PowderworldEnv',
    max_episode_steps=500,
    kwargs=dict(num_elems=8),
)
