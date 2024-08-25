from gymnasium.envs.registration import register

visual_dict = dict(
    ob_type='pixels',
    width=64,
    height=64,
    visualize_info=False,
)

register(
    id='cube-single-v0',
    entry_point='envs.manipspace.envs.cube_env:CubeEnv',
    max_episode_steps=200,
    kwargs=dict(
        env_type='cube_single',
    ),
)
register(
    id='visual-cube-single-v0',
    entry_point='envs.manipspace.envs.cube_env:CubeEnv',
    max_episode_steps=200,
    kwargs=dict(
        env_type='cube_single',
        **visual_dict,
    ),
)
register(
    id='cube-double-v0',
    entry_point='envs.manipspace.envs.cube_env:CubeEnv',
    max_episode_steps=500,
    kwargs=dict(
        env_type='cube_double',
    ),
)
register(
    id='visual-cube-double-v0',
    entry_point='envs.manipspace.envs.cube_env:CubeEnv',
    max_episode_steps=500,
    kwargs=dict(
        env_type='cube_double',
        **visual_dict,
    ),
)
register(
    id='cube-triple-v0',
    entry_point='envs.manipspace.envs.cube_env:CubeEnv',
    max_episode_steps=1000,
    kwargs=dict(
        env_type='cube_triple',
    ),
)
register(
    id='visual-cube-triple-v0',
    entry_point='envs.manipspace.envs.cube_env:CubeEnv',
    max_episode_steps=1000,
    kwargs=dict(
        env_type='cube_triple',
        **visual_dict,
    ),
)
register(
    id='cube-quadruple-v0',
    entry_point='envs.manipspace.envs.cube_env:CubeEnv',
    max_episode_steps=1000,
    kwargs=dict(
        env_type='cube_quadruple',
    ),
)
register(
    id='visual-cube-quadruple-v0',
    entry_point='envs.manipspace.envs.cube_env:CubeEnv',
    max_episode_steps=1000,
    kwargs=dict(
        env_type='cube_quadruple',
        **visual_dict,
    ),
)

register(
    id='puzzle-3x3-v0',
    entry_point='envs.manipspace.envs.puzzle_env:PuzzleEnv',
    max_episode_steps=500,
    kwargs=dict(
        env_type='puzzle_3x3',
    ),
)
register(
    id='visual-puzzle-3x3-v0',
    entry_point='envs.manipspace.envs.puzzle_env:PuzzleEnv',
    max_episode_steps=500,
    kwargs=dict(
        env_type='puzzle_3x3',
        **visual_dict,
    ),
)
register(
    id='puzzle-4x4-v0',
    entry_point='envs.manipspace.envs.puzzle_env:PuzzleEnv',
    max_episode_steps=500,
    kwargs=dict(
        env_type='puzzle_4x4',
    ),
)
register(
    id='visual-puzzle-4x4-v0',
    entry_point='envs.manipspace.envs.puzzle_env:PuzzleEnv',
    max_episode_steps=500,
    kwargs=dict(
        env_type='puzzle_4x4',
        **visual_dict,
    ),
)
register(
    id='puzzle-4x5-v0',
    entry_point='envs.manipspace.envs.puzzle_env:PuzzleEnv',
    max_episode_steps=1000,
    kwargs=dict(
        env_type='puzzle_4x5',
    ),
)
register(
    id='visual-puzzle-4x5-v0',
    entry_point='envs.manipspace.envs.puzzle_env:PuzzleEnv',
    max_episode_steps=1000,
    kwargs=dict(
        env_type='puzzle_4x5',
        **visual_dict,
    ),
)
register(
    id='puzzle-4x6-v0',
    entry_point='envs.manipspace.envs.puzzle_env:PuzzleEnv',
    max_episode_steps=1000,
    kwargs=dict(
        env_type='puzzle_4x6',
    ),
)
register(
    id='visual-puzzle-4x6-v0',
    entry_point='envs.manipspace.envs.puzzle_env:PuzzleEnv',
    max_episode_steps=1000,
    kwargs=dict(
        env_type='puzzle_4x6',
        **visual_dict,
    ),
)

register(
    id='scene-v0',
    entry_point='envs.manipspace.envs.scene_env:SceneEnv',
    max_episode_steps=750,
    kwargs=dict(
        env_type='scene',
    ),
)
register(
    id='visual-scene-v0',
    entry_point='envs.manipspace.envs.scene_env:SceneEnv',
    max_episode_steps=750,
    kwargs=dict(
        env_type='scene',
        **visual_dict,
    ),
)
