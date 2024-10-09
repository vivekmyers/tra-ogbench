import glob
import pickle

import numpy as np
from absl import app, flags
from tqdm import trange

FLAGS = flags.FLAGS


def main(_):
    dataset_paths = sorted(glob.glob('data/*/*.npz'))
    results = dict()
    for i in trange(len(dataset_paths)):
        dataset_path = dataset_paths[i]
        data = np.load(dataset_path)
        if 'qpos' not in data:
            continue
        qpos = data['qpos']
        if len(qpos) == 0:
            continue
        min_qpos = np.min(qpos, axis=0)
        max_qpos = np.max(qpos, axis=0)
        diff_max = (max_qpos - min_qpos).max()
        results[dataset_path] = dict(
            min_qpos=min_qpos,
            max_qpos=max_qpos,
            diff_max=diff_max,
        )
    with open('data/results.pkl', 'wb') as f:
        pickle.dump(results, f)


if __name__ == '__main__':
    app.run(main)
