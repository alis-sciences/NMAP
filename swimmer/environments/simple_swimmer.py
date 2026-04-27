#!/usr/bin/env python3
"""
Simple Swimmer Environment - Based on Notebook Implementation
Uses basic dm_control physics without complex land/water zones.
"""

import numpy as np
import collections
from dm_control import suite
from dm_control.suite import swimmer
from dm_control.rl import control
from dm_control.utils import rewards
try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces

_SWIM_SPEED = 0.1

class SimpleSwim(swimmer.Swimmer):
    """Simple forward swimming task - matches notebook implementation."""
    
    def __init__(self, desired_speed=_SWIM_SPEED, **kwargs):
        super().__init__(**kwargs)
        self._desired_speed = desired_speed

    def initialize_episode(self, physics):
        super().initialize_episode(physics)
        # Hide target
        physics.named.model.mat_rgba['target', 'a'] = 1
        physics.named.model.mat_rgba['target_default', 'a'] = 1
        physics.named.model.mat_rgba['target_highlight', 'a'] = 1

    def get_observation(self, physics):
        """Returns observation of joint angles and body velocities."""
        obs = collections.OrderedDict()
        obs['joints'] = physics.joints()
        obs['body_velocities'] = physics.body_velocities()
        return obs

    def get_reward(self, physics):
        """Simple forward swimming reward."""
        forward_velocity = -physics.named.data.sensordata['head_vel'][1]
        return rewards.tolerance(
            forward_velocity,
            bounds=(self._desired_speed, float('inf')),
            margin=self._desired_speed,
            value_at_margin=0.,
            sigmoid='linear',
        )

@swimmer.SUITE.add()
def simple_swim(
    n_links=6,
    desired_speed=_SWIM_SPEED,
    time_limit=swimmer._DEFAULT_TIME_LIMIT,
    random=None,
    environment_kwargs={},
):
    """Simple swim task matching notebook implementation."""
    model_string, assets = swimmer.get_model_and_assets(n_links)
    physics = swimmer.Physics.from_xml_string(model_string, assets=assets)
    task = SimpleSwim(desired_speed=desired_speed, random=random)
    return control.Environment(
        physics,
        task,
        time_limit=time_limit,
        control_timestep=swimmer._CONTROL_TIMESTEP,
        **environment_kwargs,
    )

class SimpleSwimmerEnv:
    """Simple wrapper for the basic swimmer environment."""
    
    def __init__(self, n_links=5, desired_speed=_SWIM_SPEED, time_limit=3000):
        # Create base environment
        self.env = suite.load('swimmer', 'simple_swim', 
                             task_kwargs={'random': 1, 'n_links': n_links})
        self.physics = self.env.physics
        self.action_spec = self.env.action_spec()
        self.observation_spec = self.env.observation_spec()
        self.done = False
        
    def reset(self):
        """Reset the environment."""
        time_step = self.env.reset()
        self.done = False
        return time_step.observation
    
    def step(self, action):
        """Take a step in the environment."""
        time_step = self.env.step(action)
        self.done = time_step.last()
        return time_step.observation, time_step.reward, self.done, {}
    
    def render(self, mode='rgb_array', height=480, width=640):
        """Render the environment."""
        return self.physics.render(camera_id=0, height=height, width=width)
    
    def close(self):
        """Close the environment."""
        pass

class TonicSimpleSwimmerWrapper(gym.Env):
    """
    Gym wrapper for the simple swimmer to make it compatible with Tonic.
    """
    
    def __init__(self, n_links=5, time_feature=True, desired_speed=_SWIM_SPEED):
        super().__init__()
        
        # Create the underlying environment
        self.env = SimpleSwimmerEnv(n_links=n_links, desired_speed=desired_speed)
        
        # Get action space from environment
        action_spec = self.env.action_spec
        self.action_space = spaces.Box(
            low=action_spec.minimum,
            high=action_spec.maximum,
            dtype=np.float32
        )
        
        # Create observation space
        # Joint positions + body velocities + optional time
        obs_dim = n_links - 1  # joint positions
        obs_dim += (n_links * 3)  # body velocities (3 per link)
        if time_feature:
            obs_dim += 1  # Add time feature

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        self.time_feature = time_feature
        self.step_count = 0
        self.max_steps = 1000  # Maximum episode length
        
    def reset(self):
        """Reset the environment and return a flat observation array."""
        obs = self.env.reset()
        self.step_count = 0
        return self._process_observation(obs)

    def start(self):
        """
        Tonic entry-point: called once at the beginning of training instead
        of reset().  Must return a *list* of observations (one per worker).
        """
        return [self.reset()]

    def step(self, action):
        """
        Take one step.  Returns the Tonic-expected tuple:
            (current_obs, infos)

        infos is a dict with:
            observations  – [next_obs]  (post-reset if episode ended)
            rewards       – np.array([reward], float32)
            resets        – np.array([done],   bool)
            terminations  – np.array([done],   bool)

        Tonic reads infos['observations'] as the next observation to feed
        to the agent, so we auto-reset and put the fresh obs there when
        the episode ends.
        """
        obs, reward, done, _info = self.env.step(action)

        self.step_count += 1
        if self.step_count >= self.max_steps:
            done = True

        gym_obs = self._process_observation(obs)

        # Auto-reset so Tonic always receives a valid next observation
        next_obs = self.reset() if done else gym_obs

        infos = {
            'observations': [next_obs],
            'rewards':      np.array([reward], dtype=np.float32),
            'resets':       np.array([done],   dtype=bool),
            'terminations': np.array([done],   dtype=bool),
        }
        return gym_obs, infos
    
    def _process_observation(self, obs):
        """Process observation to gym format."""
        if isinstance(obs, dict):
            # Extract joint positions and body velocities
            joint_pos = obs.get('joints', np.zeros(self.env.action_spec.shape[0]))
            body_vel = obs.get('body_velocities', np.zeros(len(joint_pos) * 3 + 3))
        else:
            # Assume obs is concatenated array
            n_joints = self.env.action_spec.shape[0]
            joint_pos = obs[:n_joints]
            body_vel = obs[n_joints:]
        
        # Combine into single observation vector
        gym_obs = np.concatenate([joint_pos, body_vel])
        
        # Add time feature if requested
        if self.time_feature:
            time_feature = np.array([self.step_count / self.max_steps], dtype=np.float32)
            gym_obs = np.concatenate([gym_obs, time_feature])
        
        return gym_obs.astype(np.float32)
    
    def render(self, mode='rgb_array'):
        """Render the environment."""
        if hasattr(self.env, 'render'):
            return self.env.render(mode)
        return None
    
    def close(self):
        """Close the environment."""
        if hasattr(self.env, 'close'):
            self.env.close()
    
    @property
    def name(self):
        """Environment name for Tonic."""
        return f"simple-swimmer-{self.env.action_spec.shape[0]}links" 
