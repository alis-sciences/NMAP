#!/usr/bin/env python3
"""
NCAP Swimmer — Three-Layer Architecture
========================================
Layer 1  NCAP CPG Circuit          B/D neuron oscillator, E/I constraints,
                                    proprioceptive coupling, phase-lag φ(s,t)
Layer 2  CCMN / FiLM               ContextEncoder (GRU on φ(s,t)) → z=[z_DA,z_5HT]
                                    FiLMGenerator → (γ,β) applied at B-neuron step
Layer 3  HRL Manager               Slow-timescale policy that observes accumulated
                                    mechanical context and issues a continuous
                                    mode signal m ∈ ℝᵈ to the ContextEncoder,
                                    conditioned on which the GRU updates z.
                                    Trained with PPO via an energy-hierarchical
                                    reward: worker reward = locomotion efficiency;
                                    manager reward = energy-efficient gait selection.

Key changes over ncap_ccmn.py
------------------------------
1.  φ(s,t) is now the PRIMARY and ONLY input to the ContextEncoder (biologically
    faithful — C. elegans has no viscosity sensor).  viscosity_norm is accepted
    for legacy/evaluation but is NOT passed to the GRU unless use_viscosity_input=True.

2.  HRLManager — new module.  Operates every manager_period steps.  Observes a
    slow summary of recent φ(s,t) history and outputs a mode vector m that is
    concatenated with z before the FiLMGenerator.  This implements the Options
    Framework: the manager selects a locomotor context, the worker (Layers 1+2)
    executes it.

3.  HRLBuffer — lightweight rollout buffer for manager PPO updates.  Stores
    (state, action, reward, value, log_prob) tuples at manager timescale.

4.  HierarchicalReward — computes dual reward:
        worker_reward  = forward velocity / muscle energy  (dense, every step)
        manager_reward = cumulative worker_reward over manager interval (sparse)
    The manager is only updated on manager_reward, preventing it from micro-managing
    the CPG and matching the biological timescale separation.

All Layers 1 and 2 code is unchanged from ncap_ccmn.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import collections
from dataclasses import dataclass, field
from typing import Optional

# ==============================================================================
# Weight / activation constraints  (Layer 1 — unchanged)
# ==============================================================================

def excitatory(w, upper=None):
    return w.clamp(min=0, max=upper)

def inhibitory(w, lower=None):
    return w.clamp(min=lower, max=0)

def unsigned(w, lower=None, upper=None):
    return w if lower is None and upper is None else w.clamp(min=lower, max=upper)

def graded(x):
    return x.clamp(min=0, max=1)

def excitatory_constant(shape=(1,), value=1.):
    return nn.Parameter(torch.full(shape, value))

def inhibitory_constant(shape=(1,), value=-1.):
    return nn.Parameter(torch.full(shape, value))


# ==============================================================================
# Layer 2 — ContextEncoder
# ==============================================================================

class ContextEncoder(nn.Module):
    """
    Slow-timescale GRU encoder.

    Primary input  : φ(s,t) — phase lags between adjacent joints.
                     This is the ONLY input in the biologically faithful mode.
                     φ(s,t) carries all substrate-resistance information:
                     high drag → slow wave propagation → large φ.

    Optional input : mode vector m from HRLManager (Layer 3), concatenated
                     with φ(s,t).  When the manager is present, z is jointly
                     determined by bottom-up proprioception and top-down context.
                     This models how DA/5-HT release is modulated by descending
                     command signals in addition to local mechanosensory input.

    Output         : z = [z_DA, z_5HT] ∈ [-1,1]²
    """

    def __init__(self, n_joints: int, hidden_size: int = 16,
                 manager_mode_dim: int = 0,
                 use_viscosity_input: bool = False):
        super().__init__()
        self.use_viscosity_input = use_viscosity_input
        self.manager_mode_dim    = manager_mode_dim

        # Input channels:
        #   (n_joints-1) phase-lag channels
        #   + manager_mode_dim  (0 if no manager)
        #   + 1 viscosity channel (only if use_viscosity_input)
        input_size = (n_joints - 1) + manager_mode_dim + (1 if use_viscosity_input else 0)

        self.gru  = nn.GRU(input_size=input_size, hidden_size=hidden_size,
                           num_layers=1, batch_first=True)
        self.proj = nn.Linear(hidden_size, 2)
        self._init_weights()

    def _init_weights(self):
        for name, p in self.gru.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(p)
            elif 'bias' in name:
                nn.init.constant_(p, 0.)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.constant_(self.proj.bias, 0.)

    def forward(self, phase_lags: torch.Tensor,
                viscosity_norm: Optional[torch.Tensor] = None,
                mode: Optional[torch.Tensor] = None,
                hidden: Optional[torch.Tensor] = None):
        """
        Parameters
        ----------
        phase_lags    : (B, T, n_joints-1)
        viscosity_norm: (B, T, 1) | None   — only used when use_viscosity_input=True
        mode          : (B, T, manager_mode_dim) | None  — HRL manager signal
        hidden        : (1, B, H) | None

        Returns
        -------
        z      : (B, 2)
        hidden : (1, B, H)
        """
        parts = [phase_lags]
        if self.manager_mode_dim > 0 and mode is not None:
            parts.append(mode)
        if self.use_viscosity_input and viscosity_norm is not None:
            parts.append(viscosity_norm)

        x = torch.cat(parts, dim=-1)          # (B, T, input_size)
        out, hidden = self.gru(x, hidden)      # out: (B, T, H)
        z_raw = self.proj(out[:, -1, :])       # (B, 2)

        z_DA  = torch.tanh(z_raw[:, 0:1])
        z_5HT = torch.tanh(-z_raw[:, 0:1] + z_raw[:, 1:2])
        z = torch.cat([z_DA, z_5HT], dim=-1)  # (B, 2)
        return z, hidden


# ==============================================================================
# Layer 2 — FiLM Generator
# ==============================================================================

class FiLMGenerator(nn.Module):
    """
    Maps z ∈ ℝ² → per-joint (γ, β) for FiLM modulation of B-neurons.
    γ constrained positive (gain); β kept small (broadcast bias).
    """

    def __init__(self, n_joints: int, context_dim: int = 2):
        super().__init__()
        self.gamma_head = nn.Linear(context_dim, n_joints)
        self.beta_head  = nn.Linear(context_dim, n_joints)
        nn.init.constant_(self.gamma_head.weight, 0.)
        nn.init.constant_(self.gamma_head.bias,   1.)
        nn.init.constant_(self.beta_head.weight,  0.)
        nn.init.constant_(self.beta_head.bias,    0.)

    def forward(self, z: torch.Tensor):
        gamma = torch.clamp(F.softplus(self.gamma_head(z)), max=2.0)
        beta  = torch.clamp(self.beta_head(z), min=-0.3, max=0.3)
        return gamma, beta


# ==============================================================================
# Layer 2 — Gait Period Scheduler
# ==============================================================================

class GaitPeriodScheduler:
    CRAWL_PERIOD = 60
    SWIM_PERIOD  = 15
    STEEPNESS    = 10.0

    @classmethod
    def period(cls, z_DA: float) -> int:
        w = torch.sigmoid(torch.tensor(z_DA * cls.STEEPNESS)).item()
        p = w * cls.CRAWL_PERIOD + (1 - w) * cls.SWIM_PERIOD
        return max(cls.SWIM_PERIOD, int(round(p)))


# ==============================================================================
# Layer 3 — HRL Manager
# ==============================================================================

class HRLManager(nn.Module):
    """
    Slow-timescale manager (Layer 3).

    Biological analog
    -----------------
    Corresponds to the descending neuromodulatory signal that integrates
    accumulated proprioceptive evidence and biases DA/5-HT release.
    In C. elegans this maps to interneurons (AVB, AVA) that receive
    environmental context and modulate the neuromodulatory broadcast.

    Architecture
    ------------
    State   : slow summary of recent φ(s,t) history
              = mean and std of the phase-lag buffer  → (2*(n_joints-1),)
    Actor   : MLP → mode vector m ∈ ℝ^{mode_dim}    (continuous action)
    Critic  : MLP → scalar value V(state)            (for PPO baseline)

    The manager fires every `manager_period` environment steps.
    Between firings the mode vector is held constant (zero-order hold).

    Training
    --------
    PPO on manager_reward = cumulative worker_reward over the manager interval.
    This sparse, energy-efficient reward prevents the manager from interfering
    with fast CPG dynamics and matches the biological timescale separation.

    mode_dim
    --------
    The mode vector m is concatenated with φ(s,t) inside the ContextEncoder
    (see ContextEncoder.forward).  m ∈ [-1,1]^{mode_dim} (tanh output).
    Recommended mode_dim = 4: enough expressivity, small enough to stay
    interpretable.  mode_dim=0 disables the manager pathway (ablation).
    """

    def __init__(self, n_joints: int, mode_dim: int = 4,
                 hidden_size: int = 32):
        super().__init__()
        self.mode_dim = mode_dim
        # State: mean + std of phase lags over slow window → 2*(n_joints-1) dims
        state_dim = 2 * (n_joints - 1)

        # Actor — outputs mode vector in ℝ^{mode_dim}
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, mode_dim),
            nn.Tanh(),          # m ∈ [-1, 1]^{mode_dim}
        )

        # Critic — scalar value estimate for PPO baseline
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

        # Log-std for Gaussian policy (learned, not state-dependent)
        self.log_std = nn.Parameter(torch.zeros(mode_dim))

        self._init_weights()

    def _init_weights(self):
        for module in [self.actor, self.critic]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, gain=0.01)
                    nn.init.constant_(layer.bias, 0.)
        # Final actor layer: very small init so manager starts near-neutral
        nn.init.orthogonal_(self.actor[-2].weight, gain=0.01)

    def _make_state(self, phase_lag_history: np.ndarray) -> torch.Tensor:
        """
        Compress phase-lag history into a fixed-size state vector.
        state = [mean(φ), std(φ)] over the slow window.

        Parameters
        ----------
        phase_lag_history : (T, n_joints-1)  numpy array

        Returns
        -------
        state : (1, 2*(n_joints-1))  tensor on manager's device
        """
        if phase_lag_history.ndim == 1:
            phase_lag_history = phase_lag_history[np.newaxis, :]
        mean = phase_lag_history.mean(axis=0)         # (n_joints-1,)
        std  = phase_lag_history.std(axis=0) + 1e-6   # (n_joints-1,)
        state = np.concatenate([mean, std])            # (2*(n_joints-1),)
        device = next(self.parameters()).device
        return torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

    def act(self, phase_lag_history: np.ndarray):
        """
        Sample a mode vector from the manager policy.

        Returns
        -------
        mode      : (1, mode_dim)  tensor — to be passed to ContextEncoder
        log_prob  : scalar tensor  — for PPO loss
        value     : scalar tensor  — critic estimate
        """
        state   = self._make_state(phase_lag_history)   # (1, state_dim)
        mu      = self.actor(state)                      # (1, mode_dim)
        std     = self.log_std.exp().clamp(1e-4, 1.0)
        dist    = torch.distributions.Normal(mu, std)
        raw     = dist.sample()
        mode    = torch.tanh(raw)                        # squash to [-1,1]
        # log_prob with tanh squashing correction
        log_prob = (dist.log_prob(raw) - torch.log(1 - mode.pow(2) + 1e-6)).sum(-1)
        value   = self.critic(state).squeeze(-1)
        return mode.detach(), log_prob.detach(), value.detach()

    def evaluate(self, states: torch.Tensor, modes: torch.Tensor):
        """
        Evaluate log-probs and values for PPO update.

        Parameters
        ----------
        states : (B, state_dim)
        modes  : (B, mode_dim)   — tanh-squashed actions already stored

        Returns
        -------
        log_probs : (B,)
        values    : (B,)
        entropy   : scalar
        """
        mu    = self.actor(states)
        std   = self.log_std.exp().clamp(1e-4, 1.0)
        dist  = torch.distributions.Normal(mu, std)
        # Invert tanh to recover raw action
        modes_clamped = modes.clamp(-1 + 1e-6, 1 - 1e-6)
        raw   = torch.atanh(modes_clamped)
        log_prob = (dist.log_prob(raw) - torch.log(1 - modes.pow(2) + 1e-6)).sum(-1)
        value = self.critic(states).squeeze(-1)
        entropy = dist.entropy().sum(-1).mean()
        return log_prob, value, entropy


# ==============================================================================
# Layer 3 — HRL Rollout Buffer
# ==============================================================================

@dataclass
class HRLBuffer:
    """
    Lightweight rollout buffer for HRL manager PPO updates.
    Stores transitions at the manager timescale (every manager_period steps).
    """
    states:    list = field(default_factory=list)
    modes:     list = field(default_factory=list)
    rewards:   list = field(default_factory=list)
    values:    list = field(default_factory=list)
    log_probs: list = field(default_factory=list)
    dones:     list = field(default_factory=list)

    def push(self, state, mode, reward, value, log_prob, done):
        self.states.append(state)
        self.modes.append(mode)
        self.rewards.append(float(reward))
        self.values.append(float(value))
        self.log_probs.append(float(log_prob))
        self.dones.append(float(done))

    def compute_returns(self, gamma: float = 0.99,
                        lam: float = 0.95,
                        last_value: float = 0.0) -> torch.Tensor:
        """GAE-λ returns."""
        returns, gae = [], 0.0
        values_ext = self.values + [last_value]
        for t in reversed(range(len(self.rewards))):
            delta = (self.rewards[t]
                     + gamma * values_ext[t + 1] * (1 - self.dones[t])
                     - values_ext[t])
            gae = delta + gamma * lam * (1 - self.dones[t]) * gae
            returns.insert(0, gae + values_ext[t])
        return torch.tensor(returns, dtype=torch.float32)

    def clear(self):
        self.__init__()

    def __len__(self):
        return len(self.rewards)


# ==============================================================================
# Layer 3 — Hierarchical Reward
# ==============================================================================

class HierarchicalReward:
    """
    Dual reward signal.

    worker_reward  (dense, every step)
        = forward_velocity / (muscle_energy + ε)
        Biologically grounded: Fang-Yen et al. showed muscle power is
        approximately constant across gaits; efficiency requires minimising
        energy for a given velocity.

    manager_reward (sparse, every manager_period steps)
        = mean(worker_reward over interval)
        The manager is rewarded for selecting gaits that maximise sustained
        locomotion efficiency, not individual step performance.

    Parameters
    ----------
    energy_weight : float
        Scales energy penalty relative to velocity reward.
        Higher → stronger pressure toward low-amplitude, efficient gaits.
    """

    def __init__(self, energy_weight: float = 0.1):
        self.energy_weight   = energy_weight
        self._worker_history: list = []

    def worker_reward(self,
                      forward_velocity: float,
                      joint_torques: np.ndarray) -> float:
        muscle_energy = float(np.sum(np.square(joint_torques)))
        reward = forward_velocity / (self.energy_weight * muscle_energy + 1e-6)
        reward = float(np.clip(reward, -10.0, 10.0))
        self._worker_history.append(reward)
        return reward

    def manager_reward(self) -> float:
        if not self._worker_history:
            return 0.0
        r = float(np.mean(self._worker_history))
        self._worker_history.clear()
        return r


# ==============================================================================
# Main model: CCMNSwimmerHRL  (all three layers)
# ==============================================================================

class CCMNSwimmerHRL(nn.Module):
    """
    Three-layer NMAP architecture.

    Layer 1 — NCAP CPG  (fastest timescale, ~10–50 ms per step)
    Layer 2 — CCMN/FiLM (intermediate, ~100–500 ms per context update)
    Layer 3 — HRL Manager (slowest, fires every manager_period steps)

    Parameters
    ----------
    n_joints         : number of swimmer joints
    oscillator_period: CPG oscillator base period (steps)
    context_hidden   : GRU hidden size for ContextEncoder
    manager_mode_dim : dimensionality of manager mode vector m.
                       0 disables the manager pathway (Layer 2 only).
    manager_period   : steps between manager decisions (slow timescale)
    manager_hidden   : hidden size for manager MLP
    use_weight_sharing, use_weight_constraints, include_proprioception,
    include_head_oscillators : passed to NCAP CPG (Layer 1)
    use_viscosity_input : if True passes viscosity_norm to ContextEncoder
                          (ablation / legacy mode; default False = bio-faithful)
    use_amplitude_scaling : z_DA modulates output torque amplitude
    """

    def __init__(self,
                 n_joints: int,
                 oscillator_period: int = 60,
                 context_hidden: int = 32,
                 manager_mode_dim: int = 4,
                 manager_period: int = 20,
                 manager_hidden: int = 32,
                 use_weight_sharing: bool = True,
                 use_weight_constraints: bool = True,
                 include_proprioception: bool = True,
                 include_head_oscillators: bool = True,
                 use_viscosity_input: bool = False,
                 use_amplitude_scaling: bool = True):
        super().__init__()

        self.n_joints          = n_joints
        self.oscillator_period = oscillator_period
        self.manager_mode_dim  = manager_mode_dim
        self.manager_period    = manager_period
        self._use_viscosity_input    = use_viscosity_input
        self._use_amplitude_scaling  = use_amplitude_scaling
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # ------------------------------------------------------------------
        # Layer 1 — NCAP CPG weights
        # ------------------------------------------------------------------
        self.ws = lambda ns, s: s if use_weight_sharing else ns

        if use_weight_constraints:
            self.exc = excitatory
            self.inh = inhibitory
            exc_p    = excitatory_constant
            inh_p    = inhibitory_constant
        else:
            self.exc = unsigned
            self.inh = unsigned
            exc_p    = lambda **kw: nn.Parameter(torch.tensor(1.0))
            inh_p    = lambda **kw: nn.Parameter(torch.tensor(-1.0))

        self.include_proprioception   = include_proprioception
        self.include_head_oscillators = include_head_oscillators

        self.params = nn.ParameterDict()
        if use_weight_sharing:
            if include_proprioception:
                self.params['bneuron_prop']  = exc_p()
            if include_head_oscillators:
                self.params['bneuron_osc']   = exc_p()
            self.params['muscle_ipsi']       = exc_p()
            self.params['muscle_contra']     = inh_p()
        else:
            for i in range(n_joints):
                if include_proprioception and i > 0:
                    self.params[f'bneuron_d_prop_{i}'] = exc_p()
                    self.params[f'bneuron_v_prop_{i}'] = exc_p()
                if include_head_oscillators and i == 0:
                    self.params[f'bneuron_d_osc_{i}']  = exc_p()
                    self.params[f'bneuron_v_osc_{i}']  = exc_p()
                self.params[f'muscle_d_d_{i}'] = exc_p()
                self.params[f'muscle_d_v_{i}'] = inh_p()
                self.params[f'muscle_v_v_{i}'] = exc_p()
                self.params[f'muscle_v_d_{i}'] = inh_p()

        # ------------------------------------------------------------------
        # Layer 2 — CCMN / FiLM
        # ------------------------------------------------------------------
        self.context_encoder = ContextEncoder(
            n_joints,
            hidden_size=context_hidden,
            manager_mode_dim=manager_mode_dim,
            use_viscosity_input=use_viscosity_input,
        )
        self.film_generator  = FiLMGenerator(n_joints, context_dim=2)
        self.gait_scheduler  = GaitPeriodScheduler()

        # ------------------------------------------------------------------
        # Layer 3 — HRL Manager
        # ------------------------------------------------------------------
        if manager_mode_dim > 0:
            self.manager = HRLManager(n_joints,
                                      mode_dim=manager_mode_dim,
                                      hidden_size=manager_hidden)
            self.hrl_buffer        = HRLBuffer()
            self.hierarchical_reward = HierarchicalReward()
        else:
            self.manager             = None   # ablation: Layer 2 only
            self.hrl_buffer          = None
            self.hierarchical_reward = None

        # ------------------------------------------------------------------
        # Episode state
        # ------------------------------------------------------------------
        self.timestep        = 0
        self._gru_hidden: Optional[torch.Tensor] = None
        self._phase_lag_buf  = collections.deque(maxlen=64)   # Layer 2 context window
        self._viscosity_buf  = collections.deque(maxlen=64)
        self._slow_phase_buf = collections.deque(maxlen=manager_period)  # Layer 3 window

        # Current manager mode (held constant between manager decisions)
        self._current_mode: Optional[torch.Tensor] = None
        self._manager_log_prob: Optional[torch.Tensor] = None
        self._manager_value: Optional[torch.Tensor] = None
        self._manager_state_snapshot: Optional[np.ndarray] = None

        # Cached outputs for logging
        self.last_z             = torch.zeros(1, 2, device=self._device)
        self.last_gamma         = torch.ones(1, n_joints, device=self._device)
        self.last_beta          = torch.zeros(1, n_joints, device=self._device)
        self.last_amplitude_scale = 1.0
        self.last_mode          = np.zeros(manager_mode_dim)

        self.to(self._device)
        print(f"CCMNSwimmerHRL initialised on {self._device}")
        print(f"  Layer 3 HRL Manager: {'ENABLED (mode_dim=' + str(manager_mode_dim) + ')' if manager_mode_dim > 0 else 'DISABLED (ablation)'}")
        print(f"  Manager fires every {manager_period} steps")
        print(f"  Phase-lag φ(s,t) wired directly into ContextEncoder (bio-faithful)")

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def reset(self, worker_id: int = None):
        """Reset episode state.  Pass worker_id to reset only that worker."""
        if worker_id is not None:
            if hasattr(self, '_gru_hidden_per_worker'):
                self._gru_hidden_per_worker[worker_id] = None
            return
        # Full reset
        self.timestep      = 0
        self._gru_hidden   = None
        if hasattr(self, '_gru_hidden_per_worker'):
            self._gru_hidden_per_worker.clear()
        self._phase_lag_buf.clear()
        self._viscosity_buf.clear()
        self._slow_phase_buf.clear()
        self._current_mode          = None
        self._manager_log_prob      = None
        self._manager_value         = None
        self._manager_state_snapshot = None
        self.last_z    = torch.zeros(1, 2, device=self._device)
        self.last_gamma = torch.ones(1, self.n_joints, device=self._device)
        self.last_beta  = torch.zeros(1, self.n_joints, device=self._device)
        self.last_amplitude_scale = 1.0
        self.last_mode = np.zeros(self.manager_mode_dim)
        if self.hrl_buffer is not None:
            self.hrl_buffer.clear()

    # ------------------------------------------------------------------
    # Layer 1 helpers
    # ------------------------------------------------------------------
    def _constrain_parameters(self):
        with torch.no_grad():
            for name, p in self.params.items():
                if 'contra' in name or 'v_d_' in name or 'muscle_contra' in name:
                    p.data = p.data.clamp(-1.0, 0.0)
                else:
                    p.data = p.data.clamp(0.0, 1.0)

    # ------------------------------------------------------------------
    # Phase-lag computation  (φ(s,t) — primary resistance signal)
    # ------------------------------------------------------------------
    def _compute_phase_lags(self, joint_pos_norm: torch.Tensor) -> torch.Tensor:
        """
        φ(s,t) = joint_pos_norm[i] − joint_pos_norm[i-1]

        High drag → slow bending-wave propagation → large |φ|.
        This is the ONLY substrate-resistance signal; C. elegans has no
        viscosity sensor and infers mechanical context from proprioception.

        Returns: (n_joints-1,) 1-D tensor
        """
        diff = joint_pos_norm[..., 1:] - joint_pos_norm[..., :-1]
        while diff.dim() > 1:
            diff = diff.mean(dim=0)
        return diff   # (n_joints-1,)

    # ------------------------------------------------------------------
    # Layer 3 — Manager step
    # ------------------------------------------------------------------
    def _manager_step(self, phase_lags: torch.Tensor,
                      worker_reward: float = 0.0,
                      done: bool = False) -> torch.Tensor:
        """
        Called every manager_period steps.

        1. Records manager transition into HRLBuffer.
        2. Queries manager policy for next mode vector m.

        Parameters
        ----------
        phase_lags    : (n_joints-1,) current phase-lag tensor
        worker_reward : cumulative worker reward since last manager step
        done          : episode done flag

        Returns
        -------
        mode : (1, manager_mode_dim) tensor — new mode vector
        """
        if self.manager is None:
            return None

        # Build slow-window state from accumulated phase-lag history
        self._slow_phase_buf.append(phase_lags.detach().cpu().numpy())
        slow_hist = np.array(self._slow_phase_buf, dtype=np.float32)  # (T, n_joints-1)

        # --- Record previous transition (if any) ---
        if self._manager_state_snapshot is not None:
            self.hrl_buffer.push(
                state    = self._manager_state_snapshot,
                mode     = self._current_mode.cpu().numpy().squeeze(0),
                reward   = worker_reward,
                value    = float(self._manager_value),
                log_prob = float(self._manager_log_prob),
                done     = float(done),
            )

        # --- Query policy for new mode ---
        mode, log_prob, value = self.manager.act(slow_hist)  # all detached tensors

        # Store for next transition recording
        self._manager_state_snapshot = self.manager._make_state(slow_hist).cpu().numpy()
        self._manager_log_prob = log_prob
        self._manager_value    = value
        self._current_mode     = mode.to(self._device)       # (1, mode_dim)
        self.last_mode         = mode.cpu().numpy().squeeze(0)

        return self._current_mode

    # ------------------------------------------------------------------
    # Layer 2 — Context update
    # ------------------------------------------------------------------
    def _update_context(self,
                        phase_lags: torch.Tensor,
                        viscosity_norm: float,
                        mode: Optional[torch.Tensor],
                        worker_id: int = 0):
        """
        Push φ(s,t) into sliding window, optionally concatenate manager
        mode, and run ContextEncoder → (z, γ, β).

        Phase lag is passed as the PRIMARY input to the GRU (biologically
        faithful).  Mode from Layer 3 is concatenated when present.
        """
        self._phase_lag_buf.append(phase_lags.detach().cpu().numpy())

        # Per-worker hidden state
        if not hasattr(self, '_gru_hidden_per_worker'):
            self._gru_hidden_per_worker = {}
        hidden_in = self._gru_hidden_per_worker.get(worker_id, None)

        pl_arr = np.array(self._phase_lag_buf, dtype=np.float32)
        if pl_arr.ndim == 1:
            pl_arr = pl_arr[np.newaxis, :]
        pl_seq = torch.tensor(pl_arr, dtype=torch.float32,
                              device=self._device).unsqueeze(0)    # (1, T, n_joints-1)

        # Manager mode: broadcast across sequence length T
        mode_seq = None
        if self.manager_mode_dim > 0 and mode is not None:
            T = pl_seq.shape[1]
            mode_seq = mode.unsqueeze(1).expand(1, T, -1)          # (1, T, mode_dim)

        # Viscosity (ablation/legacy only)
        vis_seq = None
        if self._use_viscosity_input:
            self._viscosity_buf.append([viscosity_norm])
            vis_seq = torch.tensor(
                np.array(self._viscosity_buf, dtype=np.float32),
                dtype=torch.float32, device=self._device).unsqueeze(0)  # (1, T, 1)

        ctx = torch.enable_grad() if self.training else torch.no_grad()
        with ctx:
            z, new_hidden = self.context_encoder(
                pl_seq, vis_seq, mode_seq, hidden_in
            )
            # Store per-worker hidden state
            self._gru_hidden_per_worker[worker_id] = new_hidden.detach()
            if worker_id == 0:
                self._gru_hidden = new_hidden.detach()   # legacy compat
            gamma, beta = self.film_generator(z)

        return z, gamma, beta

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def forward(self,
                joint_pos,
                viscosity_norm: float = 0.0,
                timesteps=None,
                worker_reward: float = 0.0,
                done: bool = False,
                worker_id: int = 0,
                **kwargs):
        """
        Parameters
        ----------
        joint_pos      : (n_joints,) or (B, n_joints)
        viscosity_norm : scalar [0,1] — only used if use_viscosity_input=True
        timesteps      : optional external clock
        worker_reward  : locomotion reward from previous step (for manager)
        done           : episode done flag (for manager buffer)

        Returns
        -------
        final_torques : (n_joints,) or (B, n_joints) ∈ [-1, 1]
        """
        # ---- Input handling ----
        # Use non_blocking=True so the CPU→GPU transfer overlaps with
        # other CPU work (MuJoCo step in the next worker) rather than
        # stalling the Python thread.  When joint_pos is already on the
        # correct device (persistent GPU buffer path) this is a fast no-op.
        if not isinstance(joint_pos, torch.Tensor):
            joint_pos = torch.tensor(joint_pos, dtype=torch.float32,
                                     device=self._device)
        elif joint_pos.device != self._device:
            joint_pos = joint_pos.to(self._device, non_blocking=True)

        squeeze_output = joint_pos.dim() == 1
        if squeeze_output:
            joint_pos = joint_pos.unsqueeze(0)
        batch_size = joint_pos.shape[0]

        if timesteps is None:
            timesteps = torch.tensor([self.timestep], dtype=torch.float32,
                                     device=self._device)
        elif not isinstance(timesteps, torch.Tensor):
            timesteps = torch.tensor(timesteps, dtype=torch.float32,
                                     device=self._device)
        if timesteps.dim() == 1:
            timesteps = timesteps.unsqueeze(-1)
        elif timesteps.dim() == 0:
            timesteps = timesteps.unsqueeze(0).unsqueeze(0)

        # ---- Layer 1: Normalise joint positions ----
        joint_limit    = 2 * np.pi / (self.n_joints + 1)
        joint_pos_norm = torch.clamp(joint_pos / joint_limit, -1., 1.)
        joint_pos_d    = joint_pos_norm.clamp(min=0,  max=1)
        joint_pos_v    = joint_pos_norm.clamp(min=-1, max=0).neg()

        # ---- φ(s,t): Phase-lag signal ----
        # PRIMARY input to ContextEncoder — must be computed first
        # so it can be passed to BOTH Layer 2 and Layer 3.
        phase_lags = self._compute_phase_lags(joint_pos_norm)   # (n_joints-1,)

        # ---- Layer 3: Manager decision (every manager_period steps) ----
        if self.manager is not None and self.timestep % self.manager_period == 0:
            self._manager_step(phase_lags, worker_reward, done)

        # Retrieve current (held) mode vector
        mode = self._current_mode   # (1, mode_dim) or None

        # ---- Layer 2: Context encoder → FiLM parameters ----
        z, gamma, beta = self._update_context(phase_lags, viscosity_norm, mode, worker_id=worker_id)
        gamma = gamma.expand(batch_size, -1)
        beta  = beta.expand(batch_size, -1)
        self.last_z, self.last_gamma, self.last_beta = z, gamma, beta

        # ---- Gait period from z_DA ----
        z_DA_val        = z[0, 0].item()
        effective_period = self.gait_scheduler.period(z_DA_val)

        # ---- Layer 1: CPG forward pass ----
        exc = self.exc
        inh = self.inh
        ws  = self.ws

        joint_torques = []
        for i in range(self.n_joints):

            bneuron_d = bneuron_v = torch.zeros_like(joint_pos_norm[..., 0:1])

            if self.include_proprioception and i > 0:
                bneuron_d = bneuron_d + joint_pos_d[..., i-1:i] * exc(
                    self.params[ws(f'bneuron_d_prop_{i}', 'bneuron_prop')])
                bneuron_v = bneuron_v + joint_pos_v[..., i-1:i] * exc(
                    self.params[ws(f'bneuron_v_prop_{i}', 'bneuron_prop')])

            if self.include_head_oscillators and i == 0:
                phase = timesteps.round().remainder(effective_period)
                mask  = phase < effective_period // 2
                osc_d = torch.zeros_like(timesteps)
                osc_v = torch.zeros_like(timesteps)
                osc_d[mask]  = 1.0
                osc_v[~mask] = 1.0
                bneuron_d = bneuron_d + osc_d * exc(
                    self.params[ws(f'bneuron_d_osc_{i}', 'bneuron_osc')])
                bneuron_v = bneuron_v + osc_v * exc(
                    self.params[ws(f'bneuron_v_osc_{i}', 'bneuron_osc')])

            # B-neuron graded activation
            bneuron_d = graded(bneuron_d)
            bneuron_v = graded(bneuron_v)

            # ---- FiLM modulation (Layer 2 → Layer 1 interface) ----
            # γᵢ · h + βᵢ applied at B-neuron step (bio-faithful placement)
            g_i = gamma[:, i:i+1]
            b_i = beta[:,  i:i+1]
            bneuron_d = graded(g_i * bneuron_d + b_i)
            bneuron_v = graded(g_i * bneuron_v + b_i)

            # Muscle activation (antagonistic pairs)
            muscle_d = graded(
                bneuron_d * exc(self.params[ws(f'muscle_d_d_{i}', 'muscle_ipsi')]) +
                bneuron_v * inh(self.params[ws(f'muscle_d_v_{i}', 'muscle_contra')])
            )
            muscle_v = graded(
                bneuron_v * exc(self.params[ws(f'muscle_v_v_{i}', 'muscle_ipsi')]) +
                bneuron_d * inh(self.params[ws(f'muscle_v_d_{i}', 'muscle_contra')])
            )
            joint_torques.append(muscle_d - muscle_v)

        final_torques = torch.cat(joint_torques, dim=-1)
        final_torques = torch.clamp(final_torques, -1., 1.)

        # z_DA amplitude scaling (bio-faithful: crawl ≈ 70% of swim)
        if self._use_amplitude_scaling:
            _CRAWL_SCALE   = 0.70
            amp_weight     = torch.sigmoid(
                torch.tensor(z_DA_val * 3.0, dtype=torch.float32, device=self._device))
            amplitude_scale = 1.0 - (1.0 - _CRAWL_SCALE) * amp_weight
            final_torques   = final_torques * amplitude_scale
            self.last_amplitude_scale = float(amplitude_scale.item())
        else:
            self.last_amplitude_scale = 1.0

        if self.training:
            final_torques = final_torques + 0.05 * torch.randn_like(final_torques)

        if torch.isnan(final_torques).any():
            print("WARNING: NaN in CCMNSwimmerHRL output — replacing with zeros")
            final_torques = torch.zeros_like(final_torques)

        self.timestep += 1
        self._constrain_parameters()

        if squeeze_output:
            final_torques = final_torques.squeeze(0)

        return final_torques

    # ------------------------------------------------------------------
    # PPO update for manager (Layer 3)
    # ------------------------------------------------------------------
    def update_manager(self,
                       optimizer: torch.optim.Optimizer,
                       gamma: float = 0.99,
                       lam: float   = 0.95,
                       clip_eps: float = 0.2,
                       n_epochs: int   = 4,
                       last_value: float = 0.0) -> dict:
        """
        Run PPO update on the HRL manager using the stored HRLBuffer.

        Should be called at episode end (or periodically when buffer is full).

        Parameters
        ----------
        optimizer  : optimizer for self.manager parameters only
        gamma, lam : GAE discount and lambda
        clip_eps   : PPO clipping coefficient
        n_epochs   : number of PPO update epochs over the buffer
        last_value : bootstrap value (0 if terminal)

        Returns
        -------
        dict with 'policy_loss', 'value_loss', 'entropy' scalars for logging
        """
        if self.manager is None or len(self.hrl_buffer) == 0:
            return {}

        device = self._device

        # Compute GAE returns
        returns = self.hrl_buffer.compute_returns(gamma, lam, last_value).to(device)

        # Convert buffer to tensors
        states_np = np.array([s.squeeze(0) for s in self.hrl_buffer.states], dtype=np.float32)
        modes_np  = np.array(self.hrl_buffer.modes, dtype=np.float32)
        old_lp    = torch.tensor(self.hrl_buffer.log_probs, dtype=torch.float32, device=device)

        states_t = torch.tensor(states_np, device=device)
        modes_t  = torch.tensor(modes_np,  device=device)

        advantages = (returns - returns.mean()) / (returns.std() + 1e-8)

        metrics = {'policy_loss': 0., 'value_loss': 0., 'entropy': 0.}

        for _ in range(n_epochs):
            log_probs, values, entropy = self.manager.evaluate(states_t, modes_t)

            ratio        = (log_probs - old_lp).exp()
            surr1        = ratio * advantages
            surr2        = ratio.clamp(1 - clip_eps, 1 + clip_eps) * advantages
            policy_loss  = -torch.min(surr1, surr2).mean()
            value_loss   = F.mse_loss(values, returns)
            loss         = policy_loss + 0.5 * value_loss - 0.01 * entropy

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.manager.parameters(), 0.5)
            optimizer.step()

            metrics['policy_loss'] += policy_loss.item()
            metrics['value_loss']  += value_loss.item()
            metrics['entropy']     += entropy.item()

        n = n_epochs
        metrics = {k: v / n for k, v in metrics.items()}

        self.hrl_buffer.clear()
        return metrics

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    def neuromod_state(self) -> dict:
        return {
            'z_DA':            self.last_z[0, 0].item(),
            'z_5HT':           self.last_z[0, 1].item(),
            'gamma':           self.last_gamma[0].tolist(),
            'beta':            self.last_beta[0].tolist(),
            'period':          self.gait_scheduler.period(self.last_z[0, 0].item()),
            'amplitude_scale': self.last_amplitude_scale,
            'manager_mode':    self.last_mode.tolist(),
        }


# ==============================================================================
# Actor wrapper
# ==============================================================================

class CCMNSwimmerHRLActor(nn.Module):
    """
    Drop-in actor wrapper for CCMNSwimmerHRL.
    Matches the observation dict contract of CCMNSwimmerActor.
    """
    _VIS_MIN_LOG = np.log10(1e-3)
    _VIS_MAX_LOG = np.log10(28.0)

    def __init__(self, n_joints: int = 2, **kwargs):
        super().__init__()
        self.swimmer = CCMNSwimmerHRL(n_joints=n_joints, **kwargs)
        self.to(self.swimmer._device)

    def _viscosity_to_norm(self, viscosity) -> float:
        vis  = float(viscosity[0]) if isinstance(viscosity, (list, np.ndarray)) else float(viscosity)
        vis  = max(vis, 1e-6)
        norm = (np.log10(vis) - self._VIS_MIN_LOG) / (self._VIS_MAX_LOG - self._VIS_MIN_LOG)
        return float(np.clip(norm, 0., 1.))

    def reset(self):
        self.swimmer.reset()

    def forward(self, observations, worker_reward: float = 0.0, done: bool = False):
        viscosity_norm = 0.0
        if isinstance(observations, dict):
            joint_pos = observations['joints']
            if 'fluid_viscosity' in observations:
                viscosity_norm = self._viscosity_to_norm(observations['fluid_viscosity'])
        else:
            joint_pos = observations[:self.swimmer.n_joints]

        if not isinstance(joint_pos, torch.Tensor):
            joint_pos = torch.tensor(joint_pos, dtype=torch.float32,
                                     device=self.swimmer._device)
        else:
            joint_pos = joint_pos.to(self.swimmer._device)

        return self.swimmer(joint_pos,
                            viscosity_norm=viscosity_norm,
                            worker_reward=worker_reward,
                            done=done)

    def __call__(self, observations, worker_reward: float = 0.0, done: bool = False):
        with torch.no_grad():
            actions = self.forward(observations, worker_reward, done)
            if torch.is_tensor(actions):
                actions = actions.cpu().numpy()
        return actions


# ==============================================================================
# Smoke test
# ==============================================================================

if __name__ == '__main__':
    print("=" * 65)
    print("CCMNSwimmerHRL — Three-Layer Architecture Smoke Test")
    print("=" * 65)

    actor = CCMNSwimmerHRLActor(
        n_joints=2,
        context_hidden=16,
        manager_mode_dim=4,
        manager_period=5,   # short period for smoke test
        manager_hidden=32,
    )

    manager_opt = torch.optim.Adam(actor.swimmer.manager.parameters(), lr=3e-4)

    for env_label, vis in [('water (swim)', 0.001),
                            ('intermediate', 0.5),
                            ('high-drag (crawl)', 25.0)]:
        actor.reset()
        fake_reward = 0.0
        for t in range(20):
            obs = {
                'joints':          np.array([0.1 * np.sin(t), 0.1 * np.cos(t)]),
                'fluid_viscosity': vis,
            }
            done = (t == 19)
            action = actor(obs, worker_reward=fake_reward, done=done)
            # Simulate a worker reward (would come from environment in practice)
            fake_reward = float(np.abs(action).mean())

        # Run manager PPO update at episode end
        metrics = actor.swimmer.update_manager(manager_opt, last_value=0.0)

        state = actor.swimmer.neuromod_state()
        print(f"\nEnvironment : {env_label}  (viscosity={vis} Pa·s)")
        print(f"  z_DA  = {state['z_DA']:.3f}   z_5HT = {state['z_5HT']:.3f}")
        print(f"  γ     = {[f'{g:.3f}' for g in state['gamma']]}")
        print(f"  OSC period = {state['period']} steps  "
              f"({'crawl' if state['period'] > 30 else 'swim'})")
        print(f"  Manager mode = {[f'{m:.3f}' for m in state['manager_mode']]}")
        if metrics:
            print(f"  Manager PPO — policy_loss={metrics['policy_loss']:.4f} "
                  f"value_loss={metrics['value_loss']:.4f} "
                  f"entropy={metrics['entropy']:.4f}")

    print("\nSmoke test passed.")
