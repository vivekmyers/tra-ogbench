from gymnasium.envs.registration import register

register(
    id='powderworld-v0',
    entry_point='envs.powderworld.env:PowderworldEnv',
    max_episode_steps=200,
    kwargs=dict(),
)
