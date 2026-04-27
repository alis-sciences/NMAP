#!/usr/bin/env python3
"""
ncap_progressive_env.py
=======================
A clean progressive water/land environment for A/B testing the Simple NCAP
model against the water-only baseline.

Design rules
------------
  1. SAME MODEL  — SimpleNCAPSwimmer, unchanged.
  2. SAME REWARD — pure dm_control forward-velocity tolerance reward,
                   identical to the water-only simple_swim task.
                   NO transition bonuses, NO Bayesian terms, NO mismatch
                   penalties, NO navigation targets.
  3. DIFFERENT PHYSICS — land zones progressively appear and change the
                         fluid viscosity under the swimmer, forcing it to cope
                         with mechanically harder conditions using the same
                         motor circuit.

Phase schedule (step-driven, 0-100% of training_steps):
  Phase 0  0-30%   Pure water    (water_viscosity everywhere)
  Phase 1  30-60%  Single island (one circular land zone, radius 1.8 m)
  Phase 2  60-80%  Two islands   (left + right zones, radius 1.5 m each)
  Phase 3  80-100% Four islands  (left / right / north / south)

The viscosity switches per-step based on the swimmer's head position.
No episode resets are needed to change phases -- set_training_progress()
is called by the training loop every N steps.

Classes
-------
  NCAPProgressiveTask            dm_control task  (reward-only; no extras)
  NCAPProgressiveGymEnv          thin gym wrapper (same API as SimpleSwimmerEnv)
  TonicNCAPProgressiveWrapper    Tonic wrapper    (same API as TonicSimpleSwimmerWrapper)

Training command
----------------
  python main.py --mode train_simple --progressive \\
      --algorithm ppo --training_steps 3000000 --save_steps 5000 \\
      --output_dir outputs/ncap_progressive_ppo_v1/
"""

import numpy as np
import collections

from dm_control.suite import swimmer
from dm_control.rl import control
from dm_control.utils import rewards
from dm_control import suite

try:
    import gymnasium as _gym_module
    import gymnasium.spaces as spaces
    _GYM_BASE = _gym_module.Env
except ImportError:
    import gym as _gym_module
    import gym.spaces as spaces
    _GYM_BASE = _gym_module.Env

# physics constants
_DESIRED_SPEED   = 0.1     # same as simple_swim (paper default)
_WATER_VISCOSITY = 0.001   # low viscosity - swimming
_LAND_VISCOSITY  = 0.05    # higher viscosity - mechanically harder


# ─────────────────────────────────────────────────────────────────────────────
# 1.  dm_control task
# ─────────────────────────────────────────────────────────────────────────────

class NCAPProgressiveTask(swimmer.Swimmer):
    """
    Progressive mixed-medium task with ONLY the base dm_control velocity reward.

    get_reward() is identical to swimmer.Swimmer.get_reward() (forward velocity
    tolerance).  The only difference from simple_swim is that viscosity varies
    spatially: inside a land zone it is _land_viscosity, otherwise _water_viscosity.
    """

    def __init__(self,
                 desired_speed=_DESIRED_SPEED,
                 water_viscosity=_WATER_VISCOSITY,
                 land_viscosity=_LAND_VISCOSITY,
                 training_progress=0.0,
                 **kwargs):
        super().__init__(**kwargs)
        self._desired_speed      = desired_speed
        self._water_viscosity    = water_viscosity
        self._land_viscosity     = land_viscosity
        self._training_progress  = float(training_progress)
        self._current_land_zones = self._compute_land_zones()

    # ── land zone geometry ────────────────────────────────────────────────────

    def _compute_land_zones(self):
        p = self._training_progress
        if p < 0.30:
            return []
        elif p < 0.60:
            return [
                {'center': np.array([3.0,  0.0]), 'radius': 1.8},
            ]
        elif p < 0.80:
            return [
                {'center': np.array([-2.0, 0.0]), 'radius': 1.5},
                {'center': np.array([ 3.5, 0.0]), 'radius': 1.5},
            ]
        else:
            return [
                {'center': np.array([-3.0,  0.0]), 'radius': 1.2},
                {'center': np.array([ 3.0,  0.0]), 'radius': 1.2},
                {'center': np.array([ 0.0,  2.0]), 'radius': 1.0},
                {'center': np.array([ 0.0, -2.0]), 'radius': 1.0},
            ]

    def _in_land_zone(self, pos_xy):
        for z in self._current_land_zones:
            if np.linalg.norm(pos_xy - z['center']) < z['radius']:
                return True
        return False

    # ── dm_control interface ──────────────────────────────────────────────────

    def initialize_episode(self, physics):
        super().initialize_episode(physics)
        self._current_land_zones = self._compute_land_zones()
        for mat in ('target', 'target_default', 'target_highlight'):
            try:
                physics.named.model.mat_rgba[mat, 'a'] = 0.0
            except Exception:
                pass
        physics.model.opt.viscosity = self._water_viscosity

    def get_observation(self, physics):
        """Identical to simple_swim: joint positions + body velocities only."""
        obs = collections.OrderedDict()
        obs['joints']          = physics.joints()
        obs['body_velocities'] = physics.body_velocities()
        return obs

    def get_reward(self, physics):
        """
        Forward-velocity tolerance reward -- identical to simple_swim.
        Side-effect: updates viscosity based on head position (physics only,
        not a reward signal).
        """
        head_pos = physics.named.data.xpos['head'][:2]
        in_land  = self._in_land_zone(head_pos)
        physics.model.opt.viscosity = (
            self._land_viscosity if in_land else self._water_viscosity
        )
        forward_velocity = -physics.named.data.sensordata['head_vel'][1]
        return rewards.tolerance(
            forward_velocity,
            bounds=(self._desired_speed, float('inf')),
            margin=self._desired_speed,
            value_at_margin=0.0,
            sigmoid='linear',
        )


# Register with dm_control; guard against double-registration on re-import
try:
    @swimmer.SUITE.add()
    def ncap_progressive(n_links=6,
                         desired_speed=_DESIRED_SPEED,
                         training_progress=0.0,
                         time_limit=swimmer._DEFAULT_TIME_LIMIT,
                         random=None,
                         environment_kwargs=None):
        model_string, assets = swimmer.get_model_and_assets(n_links)
        physics = swimmer.Physics.from_xml_string(model_string, assets=assets)
        task    = NCAPProgressiveTask(
            desired_speed=desired_speed,
            training_progress=training_progress,
            random=random,
        )
        return control.Environment(
            physics, task,
            time_limit=time_limit,
            control_timestep=swimmer._CONTROL_TIMESTEP,
            **(environment_kwargs or {}),
        )
except ValueError:
    pass   # already registered


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Gym-style wrapper  (mirrors SimpleSwimmerEnv)
# ─────────────────────────────────────────────────────────────────────────────

class NCAPProgressiveGymEnv:
    """
    Thin gym wrapper around NCAPProgressiveTask.
    Mirrors SimpleSwimmerEnv so _VideoEnv and ncap_analysis.py work unchanged.
    """

    def __init__(self, n_links=6, desired_speed=_DESIRED_SPEED):
        self._n_links = n_links
        self._dm_env  = suite.load(
            'swimmer', 'ncap_progressive',
            task_kwargs={'random': 1, 'n_links': n_links,
                         'desired_speed': desired_speed}
        )
        self.physics     = self._dm_env.physics
        self.action_spec = self._dm_env.action_spec()
        self._done       = False

    @property
    def training_progress(self):
        return self._dm_env.task._training_progress

    @training_progress.setter
    def training_progress(self, value):
        p = float(np.clip(value, 0.0, 1.0))
        self._dm_env.task._training_progress = p
        self._dm_env.task._current_land_zones = \
            self._dm_env.task._compute_land_zones()

    def reset(self):
        ts         = self._dm_env.reset()
        self._done = False
        return ts.observation

    def step(self, action):
        ts         = self._dm_env.step(action)
        self._done = ts.last()
        return ts.observation, ts.reward or 0.0, self._done, {}

    def render(self, mode='rgb_array', height=480, width=640):
        return self.physics.render(camera_id=0, height=height, width=width)

    def close(self):
        pass


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Tonic wrapper  (mirrors TonicSimpleSwimmerWrapper exactly)
# ─────────────────────────────────────────────────────────────────────────────

class TonicNCAPProgressiveWrapper(_GYM_BASE):
    """
    Tonic-compatible wrapper for the progressive environment.

    Observation layout -- IDENTICAL to TonicSimpleSwimmerWrapper:
        [0 : n_joints]              joint positions       (n_links-1 values)
        [n_joints : n_joints*4+1]   body velocities       (n_links*3 values)
        [-1]                        time feature (step/max_steps in [0,1])

    For n_links=6: obs_dim = 5 + 18 + 1 = 24  <- same as water-only baseline.

    step() returns the Tonic (obs, infos) 2-tuple where infos contains:
        observations, rewards, resets, terminations

    The training loop controls the phase via set_training_progress(p)
    where p = current_step / total_training_steps (0.0 to 1.0).
    """

    def __init__(self, n_links=6, time_feature=True,
                 desired_speed=_DESIRED_SPEED):
        super().__init__()
        self.n_links      = n_links
        self.time_feature = time_feature

        self._env = NCAPProgressiveGymEnv(n_links=n_links,
                                          desired_speed=desired_speed)

        spec = self._env.action_spec
        self.action_space = spaces.Box(
            low=spec.minimum, high=spec.maximum, dtype=np.float32
        )

        n_joints = n_links - 1
        obs_dim  = n_joints + n_links * 3 + (1 if time_feature else 0)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self.step_count = 0
        self.max_steps  = 1000
        self._progress  = 0.0

    # ── phase control (called by training loop) ───────────────────────────────

    def set_training_progress(self, progress: float):
        """Advance the phase. progress = current_step / total_steps (0-1)."""
        self._progress              = float(np.clip(progress, 0.0, 1.0))
        self._env.training_progress = self._progress

    # ── Tonic entry-points ────────────────────────────────────────────────────

    def start(self):
        return [self._reset_flat()]

    def reset(self):
        return [self._reset_flat()]

    def step(self, action):
        if isinstance(action, np.ndarray) and action.ndim == 2:
            action = action[0]

        obs_dict, reward, done, _ = self._env.step(action)
        self.step_count += 1
        if self.step_count >= self.max_steps:
            done = True

        obs      = self._flatten(obs_dict)
        next_obs = self._reset_flat() if done else obs

        infos = {
            'observations':  [next_obs],
            'rewards':       np.array([reward], dtype=np.float32),
            'resets':        np.array([done],   dtype=bool),
            'terminations':  np.array([done],   dtype=bool),
        }
        return obs, infos

    def render(self, mode='rgb_array'):
        return self._env.render(mode)

    def close(self):
        self._env.close()

    @property
    def name(self):
        labels = {0: 'water_only', 1: 'one_island',
                  2: 'two_islands', 3: 'four_islands'}
        phase  = int(self._progress * 4)
        return f"ncap_progressive_{self.n_links}links_{labels.get(phase, 'p' + str(phase))}"

    # ── obs helpers ───────────────────────────────────────────────────────────

    def _reset_flat(self):
        obs_dict        = self._env.reset()
        self.step_count = 0
        return self._flatten(obs_dict)

    def _flatten(self, obs_dict):
        if isinstance(obs_dict, dict):
            joints   = obs_dict.get('joints',
                           np.zeros(self.n_links - 1, dtype=np.float32))
            body_vel = obs_dict.get('body_velocities',
                           np.zeros(self.n_links * 3, dtype=np.float32))
        else:
            n        = self.n_links - 1
            joints   = obs_dict[:n]
            body_vel = obs_dict[n:]

        flat = np.concatenate([
            np.asarray(joints,   dtype=np.float32).ravel(),
            np.asarray(body_vel, dtype=np.float32).ravel(),
        ])
        if self.time_feature:
            flat = np.append(flat,
                             np.float32(self.step_count / self.max_steps))
        return flat.astype(np.float32)
