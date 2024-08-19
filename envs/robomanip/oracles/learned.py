import glob
import json
import pickle

import flax
import numpy as np

from algos import SACAgent
from envs.robomanip.oracles import oracle
from utils.evaluation import supply_rng


class LearnedCubeOracle(oracle.Oracle):
    def __init__(self, cube_restore_path, cube_restore_epoch, ob_dim, action_dim):
        self.agent = self.load_agent(cube_restore_path, cube_restore_epoch, ob_dim, action_dim)
        self.actor_fn = supply_rng(self.agent.sample_actions, rng=self.agent.rng)

    def load_agent(self, restore_path, restore_epoch, ob_dim, action_dim):
        candidates = glob.glob(restore_path)
        assert len(candidates) == 1
        restore_path = candidates[0]

        with open(restore_path + '/flags.json', 'r') as f:
            agent_config = json.load(f)['agent']

        agent = SACAgent.create(
            0,
            np.zeros(ob_dim),
            np.zeros(action_dim),
            agent_config,
        )

        if restore_epoch is None:
            param_path = restore_path + '/params.pkl'
        else:
            param_path = restore_path + f'/params_{restore_epoch}.pkl'
        with open(param_path, 'rb') as f:
            load_dict = pickle.load(f)
        agent = flax.serialization.from_state_dict(agent, load_dict['agent'])

        return agent

    def shortest_yaw(self, eff_yaw, obj_yaw):
        symmetries = np.array([i * np.pi / 2 + obj_yaw for i in range(-4, 5)])
        d = np.argmin(np.abs(eff_yaw - symmetries))
        return symmetries[d]

    def reset(self, obs, info):
        self._done = False
        self._step = 0
        self._max_step = 75
        self._xyz_center = np.array([0.425, 0.0, 0.0])
        self._xyz_scaler = 10
        self._final_pos = np.random.uniform([0.25, -0.35, 0.20], [0.6, 0.35, 0.35])
        self._final_yaw = np.random.uniform(-np.pi, np.pi)
        self._block_above_offset = np.array([0, 0, 0.18])

        self._phase = 'reach_midpoint'
        target_block = info['target_block']
        block_pos = info[f'privileged/block_{target_block}_pos']
        target_pos = info['privileged/target_pos']
        self._target_pos = (block_pos + target_pos) / 2
        self._target_pos[2] = 0.18
        self._target_yaw = info['privileged/target_yaw']

    def select_action(self, obs, info):
        if self._phase == 'reach_midpoint':
            block_pos = info[f'privileged/block_{info["target_block"]}_pos']
            success = np.linalg.norm(block_pos - self._target_pos) < 0.04
        elif self._phase == 'reach_target':
            block_pos = info[f'privileged/block_{info["target_block"]}_pos']
            success = np.linalg.norm(block_pos - self._target_pos) < 0.03
        elif self._phase == 'reinitialize':
            effector_pos = info['proprio/effector_pos']
            success = np.linalg.norm(effector_pos - self._target_pos) < 0.04

        if success:
            if self._phase == 'reach_midpoint':
                self._phase = 'reach_target'
                self._target_pos = info['privileged/target_pos']
            elif self._phase == 'reach_target':
                self._phase = 'reinitialize'
                self._target_pos = np.random.uniform([0.25, -0.35, 0.20], [0.6, 0.35, 0.35])
                self._target_yaw = np.random.uniform(-np.pi, np.pi)
            else:
                self._done = True

        if self._phase in ['reach_midpoint', 'reach_target']:
            target_block_idx = 19 + 9 * info['target_block']
            agent_obs = np.concatenate(
                [
                    obs[:19],
                    obs[target_block_idx:target_block_idx + 9],
                    (self._target_pos - self._xyz_center) * self._xyz_scaler,
                    np.cos(self._target_yaw),
                    np.sin(self._target_yaw),
                ]
            )
            action = self.actor_fn(agent_obs, temperature=0)
        elif self._phase == 'reinitialize':
            diff_pos = self._target_pos - info['proprio/effector_pos']
            diff_yaw = self._target_yaw - info['proprio/effector_yaw'][0]
            action = np.zeros(5)
            action[:3] = diff_pos / np.linalg.norm(diff_pos + 1e-6)
            action[3] = diff_yaw * 3
            action[4] = -1

        self._step += 1
        if self._step == self._max_step:
            self._done = True

        return action

    @property
    def done(self):
        return self._done
