from envs.robomanip import lie
import numpy as np

from envs.robomanip.traj import piecewise_polynomial, piecewise_pose
from envs.robomanip.oracles import oracle


class PickPlaceOracle(oracle.Oracle):
    def __init__(self, segment_dt: float = 1.0) -> None:
        self._dt = segment_dt

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

    def shortest_yaw(
        self,
        eff_yaw: float,
        obj_yaw: float,
        translation: np.ndarray,
        s: float = np.pi / 2,
    ) -> lie.SE3:
        symmetries = np.array([i * s + obj_yaw for i in range(-4, 5)])
        d = np.argmin(np.abs(eff_yaw - symmetries))
        return lie.SE3.from_rotation_and_translation(
            rotation=lie.SO3.from_z_radians(symmetries[d]),
            translation=translation,
        )

    def construct_trajectory(self, X_WG: dict[str, lie.SE3], X_WO: dict[str, lie.SE3]):
        # Pick.
        X_WO["initial"] = self.shortest_yaw(
            eff_yaw=self.get_yaw(X_WG["initial"]),
            obj_yaw=self.get_yaw(X_WO["initial"]),
            translation=X_WO["initial"].translation(),
        )
        X_WG["pick"] = self.multiply(X_WO["initial"], 0.1)
        X_WG["pick_start"] = X_WO["initial"]
        X_WG["pick_end"] = X_WG["pick_start"]
        X_WG["postpick"] = X_WG["pick"]

        # Place.
        X_WO["goal"] = self.shortest_yaw(
            eff_yaw=self.get_yaw(X_WG["postpick"]),
            obj_yaw=self.get_yaw(X_WO["goal"]),
            translation=X_WO["goal"].translation(),
        )
        X_WG["place"] = self.multiply(X_WO["goal"], 0.1)
        X_WG["place_start"] = X_WO["goal"]
        X_WG["place_end"] = X_WG["place_start"]
        X_WG["postplace"] = X_WG["place"]
        X_WG["final"] = X_WG["initial"]

        # Clearance.
        midway = lie.interpolate(X_WG["postpick"], X_WG["place"])
        X_WG["clearance"] = lie.SE3.from_rotation_and_translation(
            rotation=midway.rotation(),
            translation=np.array(
                [*midway.translation()[:2], X_WG["initial"].translation()[-1]]
            ),
        )

        times = {"initial": 0.0}
        times["pick"] = times["initial"] + self._dt
        times["pick_start"] = times["pick"] + 1.5 * self._dt
        times["pick_end"] = times["pick_start"] + self._dt
        times["postpick"] = times["pick_end"] + self._dt
        times["clearance"] = times["postpick"] + self._dt
        times["place"] = times["clearance"] + self._dt
        times["place_start"] = times["place"] + 1.5 * self._dt
        times["place_end"] = times["place_start"] + self._dt
        times["postplace"] = times["place_end"] + self._dt
        times["final"] = times["postplace"] + self._dt

        g = 0.0
        grasps = {}
        for name in times.keys():
            if name in {"pick_end", "place_end"}:
                g = not g
            grasps[name] = g

        return times, X_WG, grasps

    def reset(self, obs, info) -> None:
        target_block = info["target_block"]
        X_O = {
            "initial": self.to_pose(
                pos=obs[f"privileged/block_{target_block}_pos"],
                yaw=obs[f"privileged/block_{target_block}_yaw"][0],
            ),
            "goal": self.to_pose(
                pos=obs["privileged/target_pos"],
                yaw=obs["privileged/target_yaw"][0],
            ),
        }
        X_G = {
            "initial": self.to_pose(
                pos=obs["proprio/effector_pos"],
                yaw=obs["proprio/effector_yaw"][0],
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

        self._t_max = sample_times[-1]
        self._done = False
        self._traj = piecewise_pose.PiecewisePose.make_linear(sample_times, poses)
        self._gripper_traj = piecewise_polynomial.FirstOrderHold(sample_times, grasps)

    def select_action(self, obs):
        t = np.clip(obs["time"][0], 0.0, self._t_max)
        self._done = obs["time"][0] >= self._t_max
        pose = self._traj.value(t)
        action = np.zeros(5)
        action[:3] = pose.translation()
        action[3] = pose.rotation().compute_yaw_radians()
        action[4] = self._gripper_traj.value(t)
        return action

    @property
    def done(self):
        return self._done
