from gymnasium.envs.registration import register

register(
    id="AntCustom-v0",
    entry_point="envs.locomotion.ant:AntEnv",
    max_episode_steps=1000,
)
register(
    id="AntBall-v0",
    entry_point="envs.locomotion.ant_ball:AntBallEnv",
    max_episode_steps=1000,
)
register(
    id="HumanoidCustom-v0",
    entry_point="envs.locomotion.humanoid:HumanoidEnv",
    max_episode_steps=1000,
)
