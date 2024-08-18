import abc
import contextlib
from typing import Any, Callable, Optional, SupportsFloat

import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np
from dm_control import mjcf

from envs.robomanip import mjcf_utils


class CustomMuJoCoEnv(gym.Env, abc.ABC):
    """Base class for Mujoco environments."""

    def __init__(
        self,
        physics_timestep: float = 0.002,
        control_timestep: float = 0.002,
        render_mode: Optional[str] = None,
        width: int = 200,
        height: int = 200,
    ):
        """Initializes the Mujoco environment.

        Args:
            physics_timestep: The timestep used for physics simulation. The default
                value is Mujoco's default of 2ms.
            control_timestep: The timestep used for the control signal. By default,
                this is the same as the physics timestep.
            width: The width of the rendered image in pixels.
            height: The height of the rendered image in pixels.
        """
        self._dirty = True
        self._passive_viewer_handle = None
        self._never_compiled = True
        self._reset_next_step = True
        self._mjcf_model: Optional[mjcf.RootElement] = None
        self._model: Optional[mujoco.MjModel] = None
        self._data: Optional[mujoco.MjData] = None
        self._renderer: Optional[mujoco.Renderer] = None
        self._scene_option = mujoco.MjvOption()
        self._camera = mujoco.MjvCamera()
        self._render_height = height
        self._render_width = width

        self.set_timesteps(
            physics_timestep=float(physics_timestep),
            control_timestep=float(control_timestep),
        )

    @abc.abstractmethod
    def build_mjcf_model(self) -> mjcf.RootElement:
        """Builds the MJCF model for the environment using the `mjcf` library.

        Returns:
            The root element of the MJCF model.
        """
        raise NotImplementedError

    def modify_mjcf_model(self, mjcf_model: mjcf.RootElement) -> mjcf.RootElement:
        """Modifies the MJCF model at the beginning of each episode.

        This is useful for domain randomization or other forms of model modifications
        that may require recompilation of the MjModel and MjData objects. If the
        operation performed requires recompilation, call `mark_dirty` to force
        recompilation.

        Args:
            mjcf: The root element of the MJCF model.

        Returns:
            The root element of the modified MJCF model.
        """
        return mjcf_model

    @abc.abstractmethod
    def initialize_episode(self) -> None:
        """Initializes the environment at the beginning of each episode."""
        raise NotImplementedError

    @abc.abstractmethod
    def compute_observation(self) -> Any:
        """Computes the observation at each timestep.

        Returns:
            A dictionary of observation arrays.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def compute_reward(self, obs, action) -> SupportsFloat:
        """Computes the reward at each timestep."""
        raise NotImplementedError

    def set_control(self, action) -> None:
        """Sets the control signal for the actuators at each timestep.

        This simply forwards the action to the underlying actuators. Override this
        method to provide custom control logic such as end-effector Cartesian control.
        """
        # TODO(kevin): Check for NaNs and check space compatibility.
        self._data.ctrl[:] = action

    def post_compilation(self) -> None:
        """Performs any post-compilation operations.

        This can be useful for caching references to commonly accessed model or
        data fields. By default, this method does nothing.
        """
        pass

    def terminate_episode(self) -> bool:
        """Determines whether the episode should be terminated.

        Can be used to implement custom termination conditions such as task success,
        and task failure.
        """
        return False

    def truncate_episode(self) -> bool:
        """Determines whether the episode should be truncated.

        Can be used to implement custom truncation conditions such as time limits.
        """
        return False

    def get_reset_info(self) -> dict:
        """Returns a dictionary of information to be included in the reset return."""
        return {}

    def get_step_info(self) -> dict:
        """Returns a dictionary of information to be included in the step return."""
        return {}

    def pre_step(self) -> None:
        """Performs any pre-step operations.

        This can be useful for saving information. By default, this method does nothing.
        """
        pass

    def post_step(self) -> None:
        """Performs any post-step operations.

        This can be useful for updating the environment state after the simulation has
        been stepped. By default, this method does nothing.
        """
        pass

    def compile_model_and_data(self):
        """Compiles the MJCF model into MjModel and MjData objects."""
        getattr(self._mjcf_model.visual, 'global').offwidth = self._render_width
        getattr(self._mjcf_model.visual, 'global').offheight = self._render_height

        self._model = mujoco.MjModel.from_xml_string(
            xml=mjcf_utils.to_string(self._mjcf_model),
            assets=mjcf_utils.get_assets(self._mjcf_model),
        )
        self._data = mujoco.MjData(self._model)

        # Assign the physics timestep.
        self._model.opt.timestep = self._physics_timestep

        mujoco.mj_resetData(self._model, self._data)
        mujoco.mj_forward(self._model, self._data)

        # Make sure the passive viewer is up-to-date.
        if self._passive_viewer_handle is not None:
            self._passive_viewer_handle._sim().load(self._model, self._data, '')

        # Re-initialize the renderer.
        if self._renderer is not None:
            self._renderer.close()
            self._initialize_renderer()

        # Perform any post-compilation operations.
        self.post_compilation()

        # Mark the environment as clean.
        self._dirty = False

    def mark_dirty(self):
        """Marks the environment as dirty, requiring recompilation of the model."""
        self._dirty = True

    def reset(self, seed: int | None = None, options=None, **kwargs):
        """Resets the environment to the initial state.

        - If this is the first call to `reset`, builds the MJCF model with
            `build_mjcf_model`.
        - Modifies the MJCF model by calling `modify_mjcf`.
        - If the environment is dirty, MjModel and MjData objects will be recompiled.
            Otherwise, compilation will be skipped unless this is the first call to
            `reset`.
        - Resets the simulation with `mujoco.mj_resetData`.
        - Initializes the episode with `initialize_episode`.
        - Computes the first observation with `compute_observation`.
        """
        super().reset(seed=seed, options=options, **kwargs)
        if self._mjcf_model is None:
            self._mjcf_model = self.build_mjcf_model()
        self._mjcf_model = self.modify_mjcf_model(self._mjcf_model)
        if self._dirty or self._never_compiled:
            self.compile_model_and_data()
            self._never_compiled = False
        else:
            mujoco.mj_resetData(self._model, self._data)
            mujoco.mj_forward(self._model, self._data)
        self._reset_next_step = False
        self.initialize_episode()
        mujoco.mj_forward(self._model, self._data)
        obs = self.compute_observation()
        info = self.get_reset_info()
        return obs, info

    def set_state(self, qpos, qvel):
        """Resets the environment to a specific state."""
        assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)
        self._data.qpos[:] = np.copy(qpos)
        self._data.qvel[:] = np.copy(qvel)
        if self._model.na == 0:
            self._data.act[:] = None
        mujoco.mj_forward(self._model, self._data)

    def step(self, action):
        """Steps the environment forward by one timestep.

        - Sets the control signal with `set_control`.
        - Steps the simulation with `mujoco.mj_step`.
        - Computes the observation with `compute_observation`.
        - Computes the reward with `compute_reward`.
        - Checks if the time limit has been exceeded.
        """
        if self._reset_next_step:
            return self.reset()
        prev_qpos = self._data.qpos.copy()
        prev_qvel = self._data.qvel.copy()
        self.set_control(action)
        self.pre_step()
        mujoco.mj_step(self._model, self._data, nstep=self._n_steps)
        mujoco.mj_rnePostConstraint(self._model, self._data)  # Compute contact forces.
        self.post_step()
        terminated = self.terminate_episode()
        truncated = self.truncate_episode()
        # NOTE(kevin): We explicitly check for termination/truncation before computing
        # the observation so that any logic that potentially modifies the model/data in
        # the termination / truncation methods (e.g., changing the color of a geom to
        # indicate success) is reflected in the final observation (e.g., when using
        # pixel observations).
        obs = self.compute_observation()
        reward = self.compute_reward(obs, action)
        info = self.get_step_info()
        qpos = self._data.qpos.copy()
        qvel = self._data.qvel.copy()
        info.update(
            {
                'prev_qpos': prev_qpos,
                'prev_qvel': prev_qvel,
                'qpos': qpos,
                'qvel': qvel,
            }
        )
        return obs, reward, terminated, truncated, info

    @property
    def action_space(self):
        """Returns the action space for the environment.

        By default, this returns a Box matching the actuators defined in the
        model. Override this method to provide a custom action space.
        """
        if self._model is None:
            self.reset()
        is_limited = self._model.actuator_ctrllimited.ravel().astype(bool)
        ctrlrange = self._model.actuator_ctrlrange
        return gym.spaces.Box(
            low=np.where(is_limited, ctrlrange[:, 0], -mujoco.mjMAXVAL),
            high=np.where(is_limited, ctrlrange[:, 1], mujoco.mjMAXVAL),
            dtype=np.float32,
        )

    @property
    def observation_space(self):
        """Returns the observation space for the environment.

        By default, this returns an empty Dict.
        """
        return gym.spaces.Dict({})

    def set_timesteps(self, physics_timestep: float, control_timestep: float) -> None:
        """Sets the physics and control timesteps for the environment.

        The physics timestep will be assigned to the MjModel during compilation. The
        control timestep is used to determine the number of physics steps to take per
        control step.
        """
        # Check timesteps divisible.
        n_steps = control_timestep / physics_timestep
        rounded_n_steps = int(round(n_steps))
        if abs(n_steps - rounded_n_steps) > 1e-6:
            raise ValueError(
                f'Control timestep {control_timestep} should be an integer multiple of '
                f'physics timestep {physics_timestep}.'
            )

        self._physics_timestep = physics_timestep
        self._control_timestep = control_timestep
        self._n_steps = rounded_n_steps

    # Accessors.

    @property
    def model(self) -> mujoco.MjModel:
        """Returns the MjModel object."""
        if self._model is None:
            raise ValueError('MjModel object not yet initialized. Call `reset` to initialize.')
        return self._model

    @property
    def data(self) -> mujoco.MjData:
        """Returns the MjData object."""
        if self._data is None:
            raise ValueError('MjData object not yet initialized. Call `reset` to initialize.')
        return self._data

    @property
    def mjcf_model(self) -> mjcf.RootElement:
        """Returns the root element of the MJCF model."""
        if self._mjcf_model is None:
            raise ValueError('MJCF model not yet initialized. Call `reset` to initialize.')
        return self._mjcf_model

    def physics_timestep(self) -> float:
        """Returns the simulation timestep in seconds."""
        return self._physics_timestep

    def control_timestep(self) -> float:
        """Returns the control timestep in seconds."""
        return self._control_timestep

    # Visualization.

    def launch_passive_viewer(self, *args, **kwargs):
        if self._passive_viewer_handle is not None:
            raise ValueError('Passive viewer already launched.')
        if self._model is None or self._data is None:
            raise ValueError('Call `reset` before launching the passive viewer.')
        self._passive_viewer_handle = mujoco.viewer.launch_passive(
            self._model,
            self._data,
            show_left_ui=kwargs.pop('show_left_ui', False),
            show_right_ui=kwargs.pop('show_right_ui', False),
            *args,
            **kwargs,
        )

    def sync_passive_viewer(self):
        if self._passive_viewer_handle is None:
            raise ValueError('Passive viewer not launched.')
        self._passive_viewer_handle.sync()

    def close_passive_viewer(self):
        if self._passive_viewer_handle is not None:
            self._passive_viewer_handle.close()
            self._passive_viewer_handle = None

    @contextlib.contextmanager
    def passive_viewer(self, *args, **kwargs):
        self.launch_passive_viewer(*args, **kwargs)
        mujoco.mjv_defaultFreeCamera(self._model, self._passive_viewer_handle.cam)
        with self._passive_viewer_handle.lock():
            self._passive_viewer_handle.opt.flags |= self._scene_option.flags
            self._passive_viewer_handle.opt.geomgroup = self._scene_option.geomgroup
            self._passive_viewer_handle.opt.sitegroup = self._scene_option.sitegroup
            self._passive_viewer_handle.opt.frame = self._scene_option.frame
        yield self._passive_viewer_handle
        self.close_passive_viewer()

    def _initialize_renderer(self):
        if self._model is None:
            raise ValueError('Call `reset` before rendering.')
        self._renderer = mujoco.Renderer(model=self._model, height=self._render_height, width=self._render_width)
        mujoco.mjv_defaultFreeCamera(self._model, self._camera)

    def render(
        self,
        camera: int | str | mujoco.MjvCamera | None = None,
        depth: bool = False,
        segmentation: bool = False,
        scene_option: Optional[mujoco.MjvOption] = None,
        scene_callback: Optional[Callable[[mujoco.MjvScene], None]] = None,
    ) -> np.ndarray:
        if self._model is None or self._data is None:
            raise ValueError('Call `reset` before render.')

        if self._renderer is None:
            self._initialize_renderer()

        if depth and segmentation:
            raise ValueError('Only one of depth or segmentation can be enabled.')
        if depth:
            self._renderer.enable_depth_rendering()
        elif segmentation:
            self._renderer.enable_segmentation_rendering()
        else:
            self._renderer.disable_depth_rendering()
            self._renderer.disable_segmentation_rendering()

        self._renderer.update_scene(
            data=self._data,
            camera=camera or self._camera,
            scene_option=scene_option or self._scene_option,
        )
        if scene_callback is not None:
            scene_callback(self._renderer.scene)

        return self._renderer.render()
