from gymnasium.envs.registration import register


register(
    id='quadmaze-large-v0',
    entry_point='envs.locomaze.maze:make_maze_env',
    max_episode_steps=1000,
    kwargs=dict(
        loco_env_type='quad',
        maze_env_type='maze',
        maze_type='large',
    )
)
for friction_type in ['low', 'medium', 'high']:
    register(
        id=f'quadball-arena-fric{friction_type}-v0',
        entry_point='envs.locomaze.maze:make_maze_env',
        max_episode_steps=1000,
        kwargs=dict(
            loco_env_type='quad',
            maze_env_type='ball',
            maze_type='arena',
            friction_type=friction_type,
        )
    )
register(
    id='quadball-medium-v0',
    entry_point='envs.locomaze.maze:make_maze_env',
    max_episode_steps=1000,
    kwargs=dict(
        loco_env_type='quad',
        maze_env_type='ball',
        maze_type='medium',
    )
)
