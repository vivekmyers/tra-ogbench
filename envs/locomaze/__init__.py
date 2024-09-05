from gymnasium.envs.registration import register

visual_dict = dict(
    ob_type='pixels',
    render_mode='rgb_array',
    width=64,
    height=64,
    camera_name='back',
)


register(
    id='quadmaze-medium-v0',
    entry_point='envs.locomaze.maze:make_maze_env',
    max_episode_steps=1000,
    kwargs=dict(
        loco_env_type='quad',
        maze_env_type='maze',
        maze_type='medium',
    ),
)
register(
    id='visual-quadmaze-medium-v0',
    entry_point='envs.locomaze.maze:make_maze_env',
    max_episode_steps=1000,
    kwargs=dict(
        loco_env_type='quad',
        maze_env_type='maze',
        maze_type='medium',
        **visual_dict,
    ),
)
register(
    id='quadmaze-large-v0',
    entry_point='envs.locomaze.maze:make_maze_env',
    max_episode_steps=1000,
    kwargs=dict(
        loco_env_type='quad',
        maze_env_type='maze',
        maze_type='large',
    ),
)
register(
    id='visual-quadmaze-large-v0',
    entry_point='envs.locomaze.maze:make_maze_env',
    max_episode_steps=1000,
    kwargs=dict(
        loco_env_type='quad',
        maze_env_type='maze',
        maze_type='large',
        **visual_dict,
    ),
)
register(
    id='quadmaze-giant-v0',
    entry_point='envs.locomaze.maze:make_maze_env',
    max_episode_steps=2000,
    kwargs=dict(
        loco_env_type='quad',
        maze_env_type='maze',
        maze_type='giant',
    ),
)
register(
    id='visual-quadmaze-giant-v0',
    entry_point='envs.locomaze.maze:make_maze_env',
    max_episode_steps=2000,
    kwargs=dict(
        loco_env_type='quad',
        maze_env_type='maze',
        maze_type='giant',
        **visual_dict,
    ),
)

register(
    id='quadteleport-large-v0',
    entry_point='envs.locomaze.maze:make_maze_env',
    max_episode_steps=1000,
    kwargs=dict(
        loco_env_type='quad',
        maze_env_type='teleport',
        maze_type='large_variant',
    ),
)
register(
    id='visual-quadteleport-large-v0',
    entry_point='envs.locomaze.maze:make_maze_env',
    max_episode_steps=1000,
    kwargs=dict(
        loco_env_type='quad',
        maze_env_type='teleport',
        maze_type='large_variant',
        **visual_dict,
    ),
)

register(
    id=f'quadball-arena-v0',
    entry_point='envs.locomaze.maze:make_maze_env',
    max_episode_steps=1000,
    kwargs=dict(
        loco_env_type='quad',
        maze_env_type='ball',
        maze_type='arena',
    ),
)
register(
    id='quadball-medium-v0',
    entry_point='envs.locomaze.maze:make_maze_env',
    max_episode_steps=1000,
    kwargs=dict(
        loco_env_type='quad',
        maze_env_type='ball',
        maze_type='medium',
    ),
)

register(
    id='humanoidmaze-medium-v0',
    entry_point='envs.locomaze.maze:make_maze_env',
    max_episode_steps=2000,
    kwargs=dict(
        loco_env_type='humanoid',
        maze_env_type='maze',
        maze_type='medium',
    ),
)
register(
    id='visual-humanoidmaze-medium-v0',
    entry_point='envs.locomaze.maze:make_maze_env',
    max_episode_steps=2000,
    kwargs=dict(
        loco_env_type='humanoid',
        maze_env_type='maze',
        maze_type='medium',
        **visual_dict,
    ),
)
register(
    id='humanoidmaze-large-v0',
    entry_point='envs.locomaze.maze:make_maze_env',
    max_episode_steps=2000,
    kwargs=dict(
        loco_env_type='humanoid',
        maze_env_type='maze',
        maze_type='large',
    ),
)
register(
    id='visual-humanoidmaze-large-v0',
    entry_point='envs.locomaze.maze:make_maze_env',
    max_episode_steps=2000,
    kwargs=dict(
        loco_env_type='humanoid',
        maze_env_type='maze',
        maze_type='large',
        **visual_dict,
    ),
)
