from gymnasium.envs.registration import register

register(
    id='crafter-v0',
    entry_point='envs.crafter.crafter_env:CrafterEnv',
    kwargs=dict(),
)
