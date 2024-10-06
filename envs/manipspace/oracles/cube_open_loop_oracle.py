import numpy as np
from scipy.ndimage import gaussian_filter1d

from envs.manipspace import lie
from envs.manipspace.oracles.oracle import Oracle
from envs.manipspace.oracles.traj import piecewise_pose, piecewise_polynomial


class CubeOpenLoopOracle(Oracle):
    def __init__(
            self,
            segment_dt=0.4,
            noise=0.1,
            noise_smoothing=0.5,
            *args,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._dt = segment_dt
        self._noise = noise
        self._noise_smoothing = noise_smoothing

    def multiply(self, pose: lie.SE3, z: float) -> lie.SE3:
        return (
            lie.SE3.from_rotation_and_translation(
                rotation=lie.SO3.identity(),
                translation=np.array([0.0, 0.0, z]),
            )
            @ pose
        )

    def to_pose(self, pos: np.ndarray, yaw: float) -> lie.SE3:
        return lie.SE3.from_rotation_and_translation(
            rotation=lie.SO3.from_z_radians(yaw),
            translation=pos,
        )

    def get_yaw(self, pose: lie.SE3) -> float:
        yaw = pose.rotation().compute_yaw_radians()
        if yaw < 0.0:
            return yaw + 2 * np.pi
        return yaw

    def shortest_yaw(self, eff_yaw, obj_yaw, translation, n=4):
        symmetries = np.array([i * 2 * np.pi / n + obj_yaw for i in range(-n, n + 1)])
        d = np.argmin(np.abs(eff_yaw - symmetries))
        return lie.SE3.from_rotation_and_translation(
            rotation=lie.SO3.from_z_radians(symmetries[d]),
            translation=translation,
        )

    def construct_trajectory(self, X_WG: dict[str, lie.SE3], X_WO: dict[str, lie.SE3]):
        # Pick
        X_WO['initial'] = self.shortest_yaw(
            eff_yaw=self.get_yaw(X_WG['initial']),
            obj_yaw=self.get_yaw(X_WO['initial']),
            translation=X_WO['initial'].translation(),
        )
        X_WG['pick'] = self.multiply(X_WO['initial'], 0.1 + np.random.uniform(0, 0.1))
        X_WG['pick_start'] = X_WO['initial']
        X_WG['pick_end'] = X_WG['pick_start']
        X_WG['postpick'] = X_WG['pick']

        # Place
        X_WO['goal'] = self.shortest_yaw(
            eff_yaw=self.get_yaw(X_WG['postpick']),
            obj_yaw=self.get_yaw(X_WO['goal']),
            translation=X_WO['goal'].translation(),
        )
        X_WG['place'] = self.multiply(X_WO['goal'], 0.1 + np.random.uniform(0, 0.1))
        X_WG['place_start'] = X_WO['goal']
        X_WG['place_end'] = X_WG['place_start']
        X_WG['postplace'] = X_WG['place']
        X_WG['final'] = X_WG['initial']

        # Clearance
        midway = lie.interpolate(X_WG['postpick'], X_WG['place'])
        X_WG['clearance'] = lie.SE3.from_rotation_and_translation(
            rotation=midway.rotation(),
            translation=np.array([*midway.translation()[:2], X_WG['initial'].translation()[-1]])
            + np.random.uniform([-0.1, -0.1, 0], [0.1, 0.1, 0.2]),
        )

        times = {'initial': 0.0}
        times['pick'] = times['initial'] + self._dt
        times['pick_start'] = times['pick'] + 1.5 * self._dt
        times['pick_end'] = times['pick_start'] + self._dt
        times['postpick'] = times['pick_end'] + self._dt
        times['clearance'] = times['postpick'] + self._dt
        times['place'] = times['clearance'] + self._dt
        times['place_start'] = times['place'] + 1.5 * self._dt
        times['place_end'] = times['place_start'] + self._dt
        times['postplace'] = times['place_end'] + self._dt
        times['final'] = times['postplace'] + self._dt
        for time in times.keys():
            if time != 'initial':
                times[time] += np.random.uniform(-1, 1) * self._dt * 0.2

        g = 0.0
        grasps = {}
        for name in times.keys():
            if name in {'pick_end', 'place_end'}:
                g = 1.0 - g
            grasps[name] = g

        return times, X_WG, grasps

    def reset(self, ob, info) -> None:
        target_block = info['privileged/target_block']
        X_O = {
            'initial': self.to_pose(
                pos=info[f'privileged/block_{target_block}_pos'],
                yaw=info[f'privileged/block_{target_block}_yaw'][0],
            ),
            'goal': self.to_pose(
                pos=info['privileged/target_block_pos'],
                yaw=info['privileged/target_block_yaw'][0],
            ),
        }
        X_G = {
            'initial': self.to_pose(
                pos=info['proprio/effector_pos'],
                yaw=info['proprio/effector_yaw'][0],
            )
        }
        times, X_G, Gs = self.construct_trajectory(X_G, X_O)

        sample_times = []
        poses = []
        grasps = []
        for name in times.keys():
            sample_times.append(times[name])
            poses.append(X_G[name])
            grasps.append(Gs[name])

        self._t_init = info['time'][0]
        self._t_max = sample_times[-1]
        self._done = False
        self._traj = piecewise_pose.PiecewisePose.make_linear(sample_times, poses)
        self._gripper_traj = piecewise_polynomial.FirstOrderHold(sample_times, grasps)
        self._plan = []
        t = 0.0
        while t <= self._t_max:
            self._plan.append(self.get_absolute_action(t))
            t += self._env.unwrapped._control_timestep
        self._plan = np.array(self._plan)

        noise = np.random.normal(0, 1, size=(len(self._plan), 5)) * np.array([0.05, 0.05, 0.05, 0.3, 1.0]) * self._noise
        noise = gaussian_filter1d(noise, axis=0, sigma=self._noise_smoothing)
        self._plan += noise

    def get_absolute_action(self, t):
        action = np.zeros(5)
        pose = self._traj.value(t)
        action[:3] = pose.translation()
        action[3] = pose.rotation().compute_yaw_radians()
        action[4] = self._gripper_traj.value(t)
        return action

    def select_action(self, ob, info):
        cur_plan_idx = int((info['time'][0] - self._t_init + 1e-7) // self._env.unwrapped._control_timestep)
        if cur_plan_idx >= len(self._plan):
            cur_plan_idx = len(self._plan) - 1
            self._done = True

        ab_action = self._plan[cur_plan_idx]
        if self._env.unwrapped._action_space_type == 'absolute':
            action = self._env.normalize_action(ab_action)
        else:
            action = np.zeros(5)
            action[:3] = ab_action[:3] - info['proprio/effector_pos']
            action[3] = ab_action[3] - info['proprio/effector_yaw'][0]
            action[4] = ab_action[4] - info['proprio/gripper_opening'][0]
            action = self._env.normalize_action(action)

        return action

    @property
    def done(self):
        return self._done
