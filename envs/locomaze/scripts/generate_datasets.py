import gymnasium
from envs.locomaze import maze
from envs.locomaze.maze import make_maze_env


def main():
    env = make_maze_env('quad', maze_type='large', render_mode='rgb_array', width=200, height=200)
    env.reset()
    env.step(env.action_space.sample())


if __name__ == '__main__':
    main()
