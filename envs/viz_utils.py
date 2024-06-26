import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

matplotlib.use('Agg')


def get_2d_colors(points, min_point, max_point):
    points = np.array(points)
    min_point = np.array(min_point)
    max_point = np.array(max_point)

    colors = (points - min_point) / (max_point - min_point)
    colors = np.hstack((colors, (2 - np.sum(colors, axis=1, keepdims=True)) / 2))
    colors = np.clip(colors, 0, 1)
    colors = np.c_[colors, np.full(len(colors), 0.8)]

    return colors


def visualize_trajs(env_name, trajs):
    fig = plt.figure(tight_layout=True)
    canvas = FigureCanvas(fig)
    if env_name in ['ant-xy', 'gymhum-xy', 'humanoid-xy']:
        ax = fig.add_subplot()

        max_xy = 0.
        for traj in trajs:
            xy = np.array([info['xy'] for info in traj['info']])
            direction = np.array([info['direction'] for info in traj['info']])
            color = get_2d_colors(direction, [-1, -1], [1, 1])
            for i in range(len(xy) - 1):
                ax.plot(xy[i:i + 2, 0], xy[i:i + 2, 1], color=color[i], linewidth=0.7)
            max_xy = max(max_xy, np.abs(xy).max() * 1.2)

        plot_axis = [-max_xy, max_xy, -max_xy, max_xy]
        ax.axis(plot_axis)
        ax.set_aspect('equal')
    else:
        return None

    plt.tight_layout()
    canvas.draw()
    out_image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    out_image = out_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return out_image
