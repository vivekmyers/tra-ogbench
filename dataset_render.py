import os
import gzip
import pickle
from pathlib import Path

from absl import app, flags
import numpy as np
import tqdm


FLAGS = flags.FLAGS
flags.DEFINE_string('env_name', 'calvin', '')
flags.DEFINE_string('dataset_path', '../calvin/dataset/calvin_debug_dataset', '')
flags.DEFINE_integer('dataset_size', None, '')
flags.DEFINE_string('save_name', 'calvin', '')


def main(_):
    if 'calvin' in FLAGS.env_name:
        ep_start_end_ids = np.load(os.path.join(FLAGS.dataset_path, "training", "ep_start_end_ids.npy"))
        scene_info = np.load(os.path.join(FLAGS.dataset_path, "training", "scene_info.npy"), allow_pickle=True)
        scene_info = scene_info.item()
        function_inputs = []
        A_ctr, B_ctr, C_ctr, D_ctr = 0, 0, 0, 0
        for idx_range in ep_start_end_ids:
            start_idx = idx_range[0].item()
            if start_idx <= scene_info["calvin_scene_D"][1]:
                ctr = D_ctr
                D_ctr += 1
                letter = "D"
            elif start_idx <= scene_info["calvin_scene_B"][1]:  # This is actually correct. In ascending order we have D, B, C, A
                ctr = B_ctr
                B_ctr += 1
                letter = "B"
            elif start_idx <= scene_info["calvin_scene_C"][1]:
                ctr = C_ctr
                C_ctr += 1
                letter = "C"
            else:
                ctr = A_ctr
                A_ctr += 1
                letter = "A"

            function_inputs.append((idx_range, letter, ctr, "training"))

        from calvin_env.envs.play_table_env import get_env
        env = get_env(Path('../calvin-my/env_settings/small'), show_gui=False)
        num_steps = sum(ep_start_end_ids[:, 1] - ep_start_end_ids[:, 0])
        if FLAGS.dataset_size is not None:
            num_steps = min(num_steps, FLAGS.dataset_size)
        dataset = dict(
            states=np.zeros((num_steps, 15 + 24), dtype=np.float32),
            images=np.zeros((num_steps, 64, 64, 3), dtype=np.uint8),
            actions=np.zeros((num_steps, 7), dtype=np.float32),
            rewards=np.zeros(num_steps, dtype=np.float32),
            terminals=np.zeros(num_steps, dtype=bool),
        )

        def make_seven_characters(id):
            id = str(id)
            while len(id) < 7:
                id = "0" + id
            return id

        cur_idx = 0
        for function_data in tqdm.tqdm(function_inputs):
            idx_range, letter, ctr, split = function_data

            start_id, end_id = idx_range[0].item(), idx_range[1].item()

            for ep_id in tqdm.tqdm(range(start_id, end_id + 1)):  # end_id is inclusive
                int_ep_id = ep_id
                ep_id = make_seven_characters(ep_id)
                timestep_data = np.load(os.path.join(FLAGS.dataset_path, split, "episode_" + ep_id + ".npz"))

                env.reset(robot_obs=timestep_data['robot_obs'], scene_obs=timestep_data['scene_obs'])
                obs = env.get_obs()
                dataset['states'][cur_idx] = np.concatenate([obs['robot_obs'], obs['scene_obs']], axis=-1)
                dataset['images'][cur_idx] = obs['rgb_obs']['rgb_static']
                dataset['actions'][cur_idx] = timestep_data['rel_actions']
                if int_ep_id == end_id:
                    dataset['terminals'][cur_idx] = True

                cur_idx += 1
                if cur_idx >= num_steps:
                    break
            if cur_idx >= num_steps:
                break
        dataset['terminals'][-1] = True

        np.savez_compressed(f'data/calvin_visual/{FLAGS.save_name}.npz', **dataset)


if __name__ == '__main__':
    app.run(main)
