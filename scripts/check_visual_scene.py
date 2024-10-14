import glob
import pickle

import numpy as np
from absl import app, flags
from tqdm import trange

FLAGS = flags.FLAGS


def main(_):
    dataset_paths = sorted(glob.glob('data/manipspace/*scene-*.npz'))
    for dataset_path in dataset_paths:
        print(dataset_path, flush=True)
        dataset = np.load(dataset_path)
        qpos = dataset['qpos']
        print((qpos[:, 15] >= 0.29).any())  # Should be False
        print(((qpos[:, 15] <= -0.3) & ((qpos[:, 16] < 0.06) | (qpos[:, 16] > 0.08))).any())  # Should be False
        print(qpos[qpos[:, 15] <= -0.3][:, 16].min(), qpos[qpos[:, 15] <= -0.3][:, 16].max())


if __name__ == '__main__':
    app.run(main)
