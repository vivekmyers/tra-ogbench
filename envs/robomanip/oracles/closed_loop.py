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
        self._done = False
        self._step = 0
        self._max_step = 250
        self._final_pos = np.random.uniform([0.25, -0.45, 0.18], [0.6, 0.45, 0.35])
        self._final_yaw = np.random.uniform(-np.pi, np.pi)

        self._block_above_offset = np.array([0, 0, 0.16])

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

        gripper_closed = info['proprio/gripper_contact'] > 0.5
        gripper_open = info['proprio/gripper_contact'] < 0.1
        above = effector_pos[2] > 0.14
        xy_aligned = np.linalg.norm(block_pos[:2] - effector_pos[:2]) <= 0.04
        pos_aligned = np.linalg.norm(block_pos - effector_pos) <= 0.02
        target_xy_aligned = np.linalg.norm(target_pos[:2] - block_pos[:2]) <= 0.04
        target_pos_aligned = np.linalg.norm(target_pos - block_pos) <= 0.02
        final_pos_aligned = np.linalg.norm(self._final_pos - effector_pos) <= 0.04

        def print_phase(step):
            if debug:
                print(f'Phase {step}', end=' ')

        def shape_diff(diff):
            diff_norm = np.linalg.norm(diff)
            if diff_norm >= 0.3:
                return diff
            else:
                return diff / (diff_norm + 1e-6) * 0.3

        gain_pos = 5
        gain_yaw = 3
        action = np.zeros(5)
        if not target_pos_aligned:
            if not xy_aligned:
                print_phase(1)
                action = np.zeros(5)
                diff = block_pos + self._block_above_offset - effector_pos
                diff = shape_diff(diff)
                action[:3] = diff[:3] * gain_pos
                action[3] = (block_yaw - effector_yaw) * gain_yaw
                action[4] = -1
            elif not pos_aligned:
                print_phase(2)
                diff = block_pos - effector_pos
                diff = shape_diff(diff)
                action[:3] = diff[:3] * gain_pos
                action[3] = (block_yaw - effector_yaw) * gain_yaw
                action[4] = -1
            elif pos_aligned and not gripper_closed:
                print_phase(3)
                diff = block_pos - effector_pos
                diff = shape_diff(diff)
                action[:3] = diff[:3] * gain_pos
                action[3] = (block_yaw - effector_yaw) * gain_yaw
                action[4] = 1
            elif pos_aligned and gripper_closed and not above and not target_xy_aligned:
                print_phase(4)
                diff = np.array([block_pos[0], block_pos[1], self._block_above_offset[2] * 2]) - effector_pos
                diff = shape_diff(diff)
                action[:3] = diff[:3] * gain_pos
                action[3] = (target_yaw - block_yaw) * gain_yaw
                action[4] = 1
            elif pos_aligned and gripper_closed and above and not target_xy_aligned:
                print_phase(5)
                diff = target_pos + self._block_above_offset - effector_pos
                diff = shape_diff(diff)
                action[:3] = diff[:3] * gain_pos
                action[3] = (target_yaw - block_yaw) * gain_yaw
                action[4] = 1
            else:
                print_phase(6)
                diff = target_pos - effector_pos
                diff = shape_diff(diff)
                action[:3] = diff[:3] * gain_pos
                action[3] = (target_yaw - block_yaw) * gain_yaw
                action[4] = 1
        else:
            if not gripper_open:
                print_phase(7)
                diff = target_pos - effector_pos
                diff = shape_diff(diff)
                action[:3] = diff[:3] * gain_pos
                action[3] = (target_yaw - block_yaw) * gain_yaw
                action[4] = -1
            elif gripper_open and not above:
                print_phase(8)
                diff = np.array([block_pos[0], block_pos[1], self._block_above_offset[2] * 2]) - effector_pos
                diff = shape_diff(diff)
                action[:3] = diff[:3] * gain_pos
                action[3] = (self._final_yaw - effector_yaw) * gain_yaw
                action[4] = -1
            elif gripper_open and above:
                print_phase(9)
                diff = self._final_pos - effector_pos
                diff = shape_diff(diff)
                action[:3] = diff[:3] * gain_pos
                action[3] = (self._final_yaw - effector_yaw) * gain_yaw
                action[4] = -1

            if final_pos_aligned:
                self._done = True

        action = np.clip(action, -1, 1)
        if debug:
            np.set_printoptions(suppress=True)
            print(action)

        self._step += 1
        if self._step == self._max_step:
            self._done = True

        return action

    @property
    def done(self):
        return self._done
