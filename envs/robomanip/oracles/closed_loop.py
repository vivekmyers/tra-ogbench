import numpy as np

from envs.robomanip.oracles import oracle


class ClosedLoopCubeOracle(oracle.Oracle):
    def __init__(self):
        pass

    def shortest_yaw(self, eff_yaw, obj_yaw):
        symmetries = np.array([i * np.pi / 2 + obj_yaw for i in range(-4, 5)])
        d = np.argmin(np.abs(eff_yaw - symmetries))
        return symmetries[d]

    def reset(self, obs, info):
        self._success = False
        self._done = False
        self._step = 0
        self._max_step = 250
        self._final_pos = info['privileged/target_pos'] + np.array([0, 0, 0.18])

        self._block_above_offset = np.array([0, 0, 0.23]) + np.random.uniform([-0.01, -0.01, -0.02], [0.01, 0.01, 0.02])

    def select_action(self, obs, info):
        debug = False

        effector_pos = info['proprio/effector_pos']
        effector_yaw = info['proprio/effector_yaw'][0]
        gripper_opening = info['proprio/gripper_opening']

        target_block = info['target_block']
        block_pos = info[f'privileged/block_{target_block}_pos']
        block_yaw = self.shortest_yaw(effector_yaw, info[f'privileged/block_{target_block}_yaw'][0])
        target_pos = info['privileged/target_pos']
        target_yaw = self.shortest_yaw(effector_yaw, info['privileged/target_yaw'][0])

        gripper_closed = gripper_opening > 0.3
        gripper_open = gripper_opening <= 0.02
        above = effector_pos[2] > 0.18
        xy_aligned = np.linalg.norm(block_pos[:2] - effector_pos[:2]) <= 0.03
        yaw_aligned = np.linalg.norm(block_yaw - effector_yaw) <= 0.04
        pos_aligned = np.linalg.norm(block_pos - effector_pos) <= 0.015
        target_xy_aligned = np.linalg.norm(target_pos[:2] - block_pos[:2]) <= 0.03
        target_yaw_aligned = np.linalg.norm(target_yaw - block_yaw) <= 0.04
        target_pos_aligned = np.linalg.norm(target_pos - effector_pos) <= 0.015
        final_pos_aligned = np.linalg.norm(self._final_pos - effector_pos) <= 0.03

        def print_phase(step):
            if debug:
                print(f'Phase {step}', end=' ')

        gain_pos = 4
        gain_pos_small = 3
        gain_yaw = 3
        action = np.zeros(5)
        if not self._success:
            if not xy_aligned or not yaw_aligned:
                print_phase(1)
                action = np.zeros(5)
                diff = block_pos + self._block_above_offset - effector_pos
                action[:3] = diff[:3] * gain_pos
                action[3] = (block_yaw - effector_yaw) * gain_yaw
                action[4] = -1
            elif not pos_aligned:
                print_phase(2)
                diff = block_pos - effector_pos
                action[:3] = diff[:3] * gain_pos_small
                action[3] = (block_yaw - effector_yaw) * gain_yaw
                action[4] = -1
            elif pos_aligned and not gripper_closed:
                print_phase(3)
                diff = block_pos - effector_pos
                action[:3] = diff[:3] * gain_pos_small
                action[3] = (block_yaw - effector_yaw) * gain_yaw
                action[4] = 1
            elif pos_aligned and gripper_closed and not above and (not target_xy_aligned or not target_yaw_aligned):
                print_phase(4)
                diff = np.array([block_pos[0], block_pos[1], self._block_above_offset[2]]) - effector_pos
                action[:3] = diff[:3] * gain_pos
                action[3] = (target_yaw - block_yaw) * gain_yaw
                action[4] = 1
            elif pos_aligned and gripper_closed and above and (not target_xy_aligned or not target_yaw_aligned):
                print_phase(5)
                diff = target_pos + self._block_above_offset - effector_pos
                action[:3] = diff[:3] * gain_pos
                action[3] = (target_yaw - block_yaw) * gain_yaw
                action[4] = 1
            elif pos_aligned and gripper_closed and not target_pos_aligned:
                print_phase(6)
                diff = target_pos - effector_pos
                action[:3] = diff[:3] * gain_pos
                action[3] = (target_yaw - block_yaw) * gain_yaw
                action[4] = 1
            else:
                print_phase(7)
                action[4] = -1
                self._success = True
        else:
            if not gripper_open:
                print_phase(8)
                action[4] = -1
            elif gripper_open and not above:
                print_phase(9)
                diff = np.array([block_pos[0], block_pos[1], self._block_above_offset[2]]) - effector_pos
                action[:3] = diff[:3] * gain_pos
                action[4] = -1
            elif gripper_open and above:
                print_phase(10)
                diff = self._final_pos - effector_pos
                action[:3] = diff[:3] * gain_pos
                action[4] = -1

            if final_pos_aligned:
                self._done = True

        action = np.clip(action, -1, 1)
        if debug:
            print(action)

        self._step += 1
        if self._step == self._max_step:
            self._done = True

        return action

    @property
    def done(self):
        return self._done
