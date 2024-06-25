from gymnasium.envs.registration import register

register(
    id="AntCustom-v0",
    entry_point="envs.locomotion.ant:AntEnv",
    max_episode_steps=1000,
    reward_threshold=6000.0,
)
register(
    id="HumanoidCustom-v0",
    entry_point="envs.locomotion.humanoid:HumanoidEnv",
    max_episode_steps=1000,
)
