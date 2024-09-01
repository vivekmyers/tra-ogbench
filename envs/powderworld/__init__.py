from gymnasium.envs.registration import register

register(
    id='powderworld-v0',
    entry_point='envs.powderworld.env:PowderworldEnv',
    max_episode_steps=200,
    kwargs=dict(),
)

register(
    id='powderworld-2-v0',
    entry_point='envs.powderworld.env:PowderworldEnv',
    max_episode_steps=200,
    kwargs=dict(num_elems=2),
)

register(
    id='powderworld-5-v0',
    entry_point='envs.powderworld.env:PowderworldEnv',
    max_episode_steps=200,
    kwargs=dict(num_elems=5),
)
