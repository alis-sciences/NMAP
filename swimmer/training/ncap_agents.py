#!/usr/bin/env python3
"""
ncap_agents.py
──────────────
Algorithm-specific agent wrappers for SimpleNCAPTrainer.

Supported algorithms
────────────────────
  a2c   – Advantage Actor-Critic        (on-policy, default)
  ppo   – Proximal Policy Optimisation  (on-policy, clipped surrogate)
  ddpg  – Deep Deterministic PG         (off-policy, deterministic actor)
  es    – Evolution Strategies          (gradient-free, CMA-ES via tonic)

All agents expose the same interface used by the training loop:
    agent.initialize(observation_space, action_space, seed)
    agent.step(observations, step)          → actions np.ndarray
    agent.test_step(observations, steps)    → actions np.ndarray
    agent.update(observations, rewards, resets, terminations, steps)
    agent.save(path)

Each wrapper inherits from the matching Tonic base and applies the same
device-aware tensor conversion used in CustomA2C.
"""
import os
import torch
import numpy as np
from tonic.torch.agents import a2c as _a2c

# ── shared tensor helper (same as custom_tonic_agent.py) ─────────────────────
def _to_tensor(arr, device, dtype=torch.float32):
    if isinstance(arr, list):
        arr = np.asarray(arr, dtype=np.float32)
    return torch.as_tensor(arr, dtype=dtype, device=device)


# ─────────────────────────────────────────────────────────────────────────────
# A2C  (existing CustomA2C, re-exported here for uniform import)
# ─────────────────────────────────────────────────────────────────────────────
try:
    from .custom_tonic_agent import CustomA2C
except ImportError:
    from custom_tonic_agent import CustomA2C


# ─────────────────────────────────────────────────────────────────────────────
# PPO
# ─────────────────────────────────────────────────────────────────────────────
try:
    from tonic.torch.agents import ppo as _ppo

    class CustomPPO(_ppo.PPO):
        """
        PPO with device-aware tensor conversion.

        Key difference from A2C: uses a clipped surrogate objective and
        runs multiple epochs over each collected segment.  Suitable when
        you want more stable, sample-efficient on-policy learning than A2C.
        """

        def __init__(self, model=None, replay=None,
                     actor_updater=None, critic_updater=None):
            if replay is None:
                from tonic import replays
                # PPO benefits from larger segments and more epochs
                replay = replays.Segment(size=2048, batch_iterations=10)
            super().__init__(model=model, replay=replay,
                             actor_updater=actor_updater,
                             critic_updater=critic_updater)

        def step(self, observations, steps):
            actions, log_probs = self._step(observations)
            actions   = actions.cpu().numpy()
            log_probs = log_probs.cpu().numpy()
            self.last_observations = observations.copy()
            self.last_actions      = actions.copy()
            self.last_log_probs    = log_probs.copy()
            return actions

        def test_step(self, observations, steps):
            return self._test_step(observations).cpu().numpy()

        def _step(self, observations):
            device = next(self.model.parameters()).device
            obs_t  = _to_tensor(observations, device)
            with torch.no_grad():
                dist = self.model.actor(obs_t)
                if hasattr(dist, 'sample_with_log_prob'):
                    actions, log_probs = dist.sample_with_log_prob()
                else:
                    actions   = dist.sample()
                    log_probs = dist.log_prob(actions).sum(-1)
            return actions, log_probs

        def _test_step(self, observations):
            device = next(self.model.parameters()).device
            obs_t  = _to_tensor(observations, device)
            with torch.no_grad():
                return self.model.actor(obs_t).sample()

        def _evaluate(self, observations, next_observations):
            device = next(self.model.parameters()).device
            with torch.no_grad():
                values      = self.model.critic(_to_tensor(observations,      device))
                next_values = self.model.critic(_to_tensor(next_observations, device))
            # PPO's _update calls .numpy() directly — must be on CPU first
            return values.cpu(), next_values.cpu()

        def update(self, observations=None, rewards=None, resets=None,
                   terminations=None, steps=None, **kwargs):
            # Accept both positional and keyword args (mirrors CustomA2C.update)
            if observations is None:  observations  = kwargs.get('observations')
            if rewards      is None:  rewards       = kwargs.get('rewards')
            if resets       is None:  resets        = kwargs.get('resets')
            if terminations is None:  terminations  = kwargs.get('terminations')
            self.replay.store(
                observations=self.last_observations,
                actions=self.last_actions,
                next_observations=observations,
                rewards=rewards,
                resets=resets,
                terminations=terminations,
                log_probs=self.last_log_probs,
            )
            if self.model.observation_normalizer:
                self.model.observation_normalizer.record(self.last_observations)
            if self.model.return_normalizer:
                self.model.return_normalizer.record(rewards)
            if self.replay.ready():
                self._update()

        def _update(self):
            # Move every replay batch to device before Tonic PPO updater
            # touches it. actions and old_log_probs from the CPU replay buffer
            # would otherwise be passed to dist.log_prob() whose loc/scale
            # live on CUDA, causing a device mismatch.
            device = next(self.model.parameters()).device

            # Compute lambda returns (Tonic expects CPU numpy arrays here)
            batch = self.replay.get_full('observations', 'next_observations')
            values, next_values = self._evaluate(**batch)
            self.replay.compute_returns(values.numpy(), next_values.numpy())

            # PPO epochs: move every tensor in the batch to device
            for batch in self.replay.get(
                'observations', 'actions', 'advantages', 'log_probs', 'returns'
            ):
                batch = {k: torch.as_tensor(v, dtype=torch.float32, device=device)
                         for k, v in batch.items()}
                self._update_actor_critic(**batch)

            if self.model.observation_normalizer:
                self.model.observation_normalizer.update()
            if self.model.return_normalizer:
                self.model.return_normalizer.update()

    _HAS_PPO = True

except Exception as _e:
    CustomPPO = None
    _HAS_PPO  = False
    print(f'[ncap_agents] PPO not available: {_e}')


# ─────────────────────────────────────────────────────────────────────────────
# DDPG
# ─────────────────────────────────────────────────────────────────────────────
try:
    from tonic.torch.agents import ddpg as _ddpg

    class CustomDDPG(_ddpg.DDPG):
        """
        DDPG with device-aware tensor conversion.

        Off-policy algorithm that suits the NCAP because the deterministic
        actor maps directly to torque outputs — no stochastic sampling needed
        during exploitation.  Uses an experience replay buffer, making it
        more sample-efficient than A2C/PPO for long training runs.

        Note: DDPG requires a *deterministic* actor model.  The _NCAPActor
        wraps the NCAP output in a Normal distribution; DDPG will call
        actor(obs).mean directly via its own updater, so this is compatible.
        """

        def __init__(self, model=None, replay=None,
                     actor_updater=None, critic_updater=None,
                     noise_start: float = 0.3,
                     noise_end:   float = 0.05,
                     noise_decay: int   = 100_000):
            if replay is None:
                from tonic import replays
                # Large ring buffer with increased warmup (10k steps of random
                # exploration before any gradient update)
                replay = replays.Buffer(size=200_000)
                # Warmup: DDPG base class has a start_steps attribute
                # that controls when updates begin. We set it after super().__init__.
            super().__init__(model=model, replay=replay,
                             actor_updater=actor_updater,
                             critic_updater=critic_updater)
            # Delay updates until 10k steps of data have been collected
            if hasattr(self, 'start_steps'):
                self.start_steps = 10_000
            # Linearly decay exploration noise from noise_start → noise_end
            self._noise_start = noise_start
            self._noise_end   = noise_end
            self._noise_decay = noise_decay

        def step(self, observations, steps):
            actions = self._step(observations, steps)
            actions = actions.cpu().numpy()
            self.last_observations = observations.copy()
            self.last_actions      = actions.copy()
            return actions

        def test_step(self, observations, steps):
            return self._test_step(observations).cpu().numpy()

        def _step(self, observations, steps):
            device  = next(self.model.parameters()).device
            obs_t   = _to_tensor(observations, device)
            with torch.no_grad():
                actions = self.model.actor(obs_t)
                if not isinstance(actions, torch.Tensor):
                    actions = actions.mean
                # Linearly decaying exploration noise
                frac    = min(1.0, steps / max(1, self._noise_decay))
                sigma   = self._noise_start + frac * (self._noise_end - self._noise_start)
                noise   = torch.randn_like(actions) * sigma
                actions = (actions + noise).clamp(-1.0, 1.0)
            return actions

        def _test_step(self, observations):
            device  = next(self.model.parameters()).device
            obs_t   = _to_tensor(observations, device)
            with torch.no_grad():
                actions = self.model.actor(obs_t)
                if not isinstance(actions, torch.Tensor):
                    actions = actions.mean
                return actions

        def update(self, observations=None, rewards=None, resets=None,
                   terminations=None, steps=None, **kwargs):
            # Accept keyword args from our training loop
            if observations is None:  observations  = kwargs.get('observations')
            if rewards      is None:  rewards       = kwargs.get('rewards')
            if resets       is None:  resets        = kwargs.get('resets')
            if terminations is None:  terminations  = kwargs.get('terminations')
            if steps        is None:  steps         = kwargs.get('steps', 0)

            # DDPG replay stores (obs, actions, next_obs, rewards, resets,
            # terminations) — no log_probs
            self.replay.store(
                observations=self.last_observations,
                actions=self.last_actions,
                next_observations=observations,
                rewards=rewards,
                resets=resets,
                terminations=terminations,
            )
            if self.model.observation_normalizer:
                self.model.observation_normalizer.record(self.last_observations)
            if self.model.return_normalizer:
                self.model.return_normalizer.record(rewards)
            if self.replay.ready(steps):
                self._update(steps)

        def _update(self, steps=0):
            # Move every replay batch to device before the DDPG updater runs
            device = next(self.model.parameters()).device
            # DDPG._update_actor_critic expects:
            # observations, actions, next_observations, rewards, discounts
            # where discounts = gamma * (1 - terminations)
            _gamma = getattr(self, 'discount_factor', 0.99)
            for batch in self.replay.get(
                'observations', 'actions', 'next_observations',
                'rewards', 'resets', 'terminations',
                steps=steps
            ):
                terminations = np.asarray(batch.get('terminations',
                               np.zeros(len(batch['rewards']))), dtype=np.float32)
                discounts = _gamma * (1.0 - terminations)
                clean = {k: torch.as_tensor(
                             np.asarray(v), dtype=torch.float32, device=device)
                         for k, v in batch.items()
                         if k in ('observations', 'actions',
                                  'next_observations', 'rewards')}
                clean['discounts'] = torch.as_tensor(
                    discounts, dtype=torch.float32, device=device)
                self._update_actor_critic(**clean)

            if self.model.observation_normalizer:
                self.model.observation_normalizer.update()
            if self.model.return_normalizer:
                self.model.return_normalizer.update()

    _HAS_DDPG = True

except Exception as _e:
    CustomDDPG = None
    _HAS_DDPG  = False
    print(f'[ncap_agents] DDPG not available: {_e}')



# ─────────────────────────────────────────────────────────────────────────────
# ES  — Self-contained OpenAI Evolution Strategies (Salimans et al. 2017)
#        Independent of Tonic; works with any Tonic version.
# ─────────────────────────────────────────────────────────────────────────────

class CustomES:
    # OpenAI-style ES with antithetic sampling and Adam.
    # Same step/test_step/update/save/load interface as A2C/PPO.

    def __init__(self, model=None, pop_size=32, sigma=0.02,
                 lr=3e-4, rollout_steps=200):
        self.model, self.pop_size  = model, pop_size
        self.sigma, self.lr        = sigma, lr
        self.rollout_steps         = rollout_steps
        self._m = self._v = self._theta = None
        self._shapes = self._sizes = None
        self._t = 0
        self._step_rewards = []
        self._ep_returns   = []
        self.last_observations = self.last_actions = None

    def initialize(self, observation_space, action_space, seed=42):
        torch.manual_seed(seed); np.random.seed(seed)
        self.model.initialize(observation_space, action_space)
        params = [p.data.cpu().numpy().ravel() for p in self.model.parameters()]
        self._theta  = np.concatenate(params).astype(np.float64)
        self._shapes = [p.data.shape for p in self.model.parameters()]
        self._sizes  = [p.data.numel() for p in self.model.parameters()]
        self._m = np.zeros_like(self._theta)
        self._v = np.zeros_like(self._theta)
        print(f'[CustomES] OpenAI-ES | param vector: {len(self._theta):,}')

    def step(self, observations, steps):
        actions = self._get_actions(observations)
        self.last_observations = observations.copy()
        self.last_actions      = actions.copy()
        return actions

    def test_step(self, observations, steps):
        return self._get_actions(observations)

    def update(self, observations=None, rewards=None, resets=None,
               terminations=None, steps=None, **kwargs):
        if rewards is None:
            rewards = kwargs.get('rewards', np.zeros(1))
        self._step_rewards.append(float(np.mean(rewards)))
        if len(self._step_rewards) >= self.rollout_steps:
            self._ep_returns.append(float(np.sum(self._step_rewards)))
            self._step_rewards = []
            if len(self._ep_returns) >= self.pop_size:
                self._es_update()
                self._ep_returns = []

    def save(self, path):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        np.save(path + '_es_theta.npy', self._theta)
        print(f'[CustomES] saved -> {path}_es_theta.npy')

    def load(self, path):
        for c in [path + '_es_theta.npy', path]:
            if os.path.exists(c):
                self._theta = np.load(c)
                self._set_params(self._theta)
                return
        raise FileNotFoundError(f'ES checkpoint not found at {path}')

    def _get_actions(self, observations):
        device = next(self.model.parameters()).device
        obs_t  = _to_tensor(observations, device)
        with torch.no_grad():
            out = self.model.actor(obs_t)
            return (out if isinstance(out, torch.Tensor) else out.mean).cpu().numpy()

    def _set_params(self, theta):
        device = next(self.model.parameters()).device
        offset = 0
        with torch.no_grad():
            for param, shape, size in zip(
                    self.model.parameters(), self._shapes, self._sizes):
                chunk = theta[offset:offset + size].reshape(shape)
                param.data.copy_(
                    torch.tensor(chunk, dtype=torch.float32, device=device))
                offset += size

    def _es_update(self):
        half    = self.pop_size // 2
        noises  = [np.random.randn(len(self._theta)) for _ in range(half)]
        returns = np.array(self._ep_returns[:self.pop_size], dtype=np.float64)
        if returns.std() > 1e-8:
            returns = (returns - returns.mean()) / returns.std()
        all_noise = np.array(noises + [-n for n in noises])
        grad      = np.dot(all_noise[:len(returns)].T, returns) / (len(returns) * self.sigma)
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        self._t += 1
        self._m  = beta1 * self._m + (1 - beta1) * grad
        self._v  = beta2 * self._v + (1 - beta2) * grad ** 2
        m_hat    = self._m / (1 - beta1 ** self._t)
        v_hat    = self._v / (1 - beta2 ** self._t)
        self._theta += self.lr * m_hat / (np.sqrt(v_hat) + eps)
        self._set_params(self._theta)


_HAS_ES  = True
_ES_NAME = 'OpenAI-ES'


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

#: Default replay buffer sizes per algorithm
_DEFAULT_REPLAY = {
    'a2c':  dict(size=1024,   batch_iterations=20),
    'ppo':  dict(size=2048,   batch_iterations=10),
    'ddpg': dict(size=100_000),
    'es':   {},
}

#: Human-readable notes shown at startup
ALGORITHM_NOTES = {
    'a2c':  'Advantage Actor-Critic (on-policy, small buffer, fast updates)',
    'ppo':  'Proximal Policy Optimisation (on-policy, clipped surrogate, stable)',
    'ddpg': 'Deep Deterministic PG (off-policy, large replay buffer, sample-efficient)',
    'es':   f'Evolution Strategies / {_ES_NAME} (gradient-free, population-based)',
}


def build_agent(algorithm: str, model) -> object:
    """
    Instantiate and return the appropriate agent for the given algorithm.

    Parameters
    ----------
    algorithm : 'a2c' | 'ppo' | 'ddpg' | 'es'
    model     : _NCAPModel instance

    Returns
    -------
    Tonic-compatible agent with .initialize() / .step() / .update() / .save()
    """
    algorithm = algorithm.lower()

    if algorithm == 'a2c':
        return CustomA2C(model=model)

    elif algorithm == 'ppo':
        if not _HAS_PPO:
            raise RuntimeError('PPO not available in this Tonic installation. '
                               'Try --algorithm a2c instead.')
        return CustomPPO(model=model)

    elif algorithm == 'ddpg':
        if not _HAS_DDPG:
            raise RuntimeError('DDPG not available in this Tonic installation. '
                               'Try --algorithm a2c instead.')
        return CustomDDPG(model=model)

    elif algorithm == 'es':
        if not _HAS_ES:
            print('[ncap_agents] ⚠️  ES not available in this Tonic installation '
                  '— falling back to A2C.')
            return CustomA2C(model=model)
        return CustomES(model=model)

    else:
        raise ValueError(f"Unknown algorithm '{algorithm}'. "
                         f"Choose from: a2c, ppo, ddpg, es")
