#!/usr/bin/env python3
"""
Simple NCAP Trainer
-------------------
Trains SimpleNCAPSwimmer (simple_ncap.py) on the plain dm_control swimmer
(TonicSimpleSwimmerWrapper from simple_swimmer.py) using Tonic's A2C via
the custom device-aware CustomA2C agent (custom_tonic_agent.py).

Drop this file alongside simple_ncap.py, simple_swimmer.py, and
custom_tonic_agent.py (all at the same directory / package level).

Usage (from repo root):
    python main.py --mode train_simple --n_links 6 \\
                   --training_steps 1000000 --save_steps 50000
"""

import os
import torch
import numpy as np
import tonic
import tonic.torch

# ---------------------------------------------------------------------------
# Local imports – try relative first (package), fall back to bare (direct run)
# ---------------------------------------------------------------------------
try:
    from swimmer.models.simple_ncap import SimpleNCAPSwimmer
    from swimmer.environments.simple_swimmer import TonicSimpleSwimmerWrapper
    from swimmer.training.custom_tonic_agent import CustomA2C
    from swimmer.utils.simple_ncap_logger import SimpleNCAPLogger, make_prefix
    from swimmer.training.ncap_agents import build_agent, ALGORITHM_NOTES
except ImportError:
    from swimmer.models.simple_ncap import SimpleNCAPSwimmer
    from swimmer.environments.simple_swimmer import TonicSimpleSwimmerWrapper
    from swimmer.training.custom_tonic_agent import CustomA2C
    from swimmer.utils.simple_ncap_logger import SimpleNCAPLogger, make_prefix
    from swimmer.training.ncap_agents import build_agent, ALGORITHM_NOTES

# Progressive environment — lazy import, only used with --progressive
_TonicNCAPProgressiveWrapper = None
def _get_progressive_wrapper():
    global _TonicNCAPProgressiveWrapper
    if _TonicNCAPProgressiveWrapper is None:
        try:
            from .ncap_progressive_env import TonicNCAPProgressiveWrapper as _W
        except ImportError:
            from ncap_progressive_env import TonicNCAPProgressiveWrapper as _W
        _TonicNCAPProgressiveWrapper = _W
    return _TonicNCAPProgressiveWrapper


# ---------------------------------------------------------------------------
# Tonic module protocol helper
# ---------------------------------------------------------------------------

class _TonicModule(torch.nn.Module):
    """
    Base mixin that satisfies Tonic's two-stage initialisation protocol.

    Tonic calls  model.initialize(observation_space, action_space)
    immediately after  agent.initialize(...).  It then cascades the same
    call down to every sub-module that has an `initialize` method
    (actor, critic, observation_normalizer, return_normalizer).

    Sub-classes that need the spaces for lazy layer creation should
    override `_build(observation_space, action_space)` instead of
    `initialize` directly.
    """

    def initialize(self, observation_space, action_space):
        self._build(observation_space, action_space)

    def _build(self, observation_space, action_space):
        """Override in sub-classes that need lazy initialisation."""
        pass


# ---------------------------------------------------------------------------
# Tonic distribution helper
# ---------------------------------------------------------------------------

def _sample_with_log_prob(dist):
    """sample + summed log_prob helper for torch.distributions.Normal."""
    action   = dist.sample()
    log_prob = dist.log_prob(action).sum(dim=-1)
    return action, log_prob


# ---------------------------------------------------------------------------
# Actor
# ---------------------------------------------------------------------------

class _NCAPActor(_TonicModule):
    """
    Wraps SimpleNCAPSwimmer as a Tonic-compatible actor.

    forward(observations) → torch.distributions.Normal  (augmented with
    sample_with_log_prob so CustomA2C._step works without modification).

    Tonic's A2C calls:
        dist = model.actor(obs)
        actions, log_probs = dist.sample_with_log_prob()
    """

    def __init__(self, ncap: SimpleNCAPSwimmer):
        super().__init__()
        self.ncap = ncap
        # NOTE: do NOT pre-assign _action_low/_action_high as plain Python
        # attributes here – torch.register_buffer() raises KeyError if a
        # regular attribute with the same name already exists.
        # They are registered lazily in _build() once action_space is known.
        # Learnable per-joint log-std (log(0.3) ≈ −1.2)
        self.log_std = torch.nn.Parameter(
            torch.full((ncap.n_joints,), -1.2)
        )

    def _build(self, observation_space, action_space):
        # Guard against double-call (Tonic may cascade initialize more than once)
        if "_action_low" in dict(self.named_buffers()):
            return
        low  = torch.as_tensor(action_space.low,  dtype=torch.float32)
        high = torch.as_tensor(action_space.high, dtype=torch.float32)
        self.register_buffer("_action_low",  low)
        self.register_buffer("_action_high", high)

    def forward(self, observations: torch.Tensor):
        """
        observations: (batch, obs_dim)

        TonicSimpleSwimmerWrapper layout:
          [0 : n_joints]          joint positions  (radians)
          [n_joints : ...]        body velocities  (3 per link)
          [-1]                    time feature     (0 → 1, max_steps=1000)
        """
        n         = self.ncap.n_joints
        joint_pos = observations[..., :n]           # (batch, n_joints)
        time_feat = observations[..., -1]           # (batch,)
        timesteps = (time_feat * 1000).round()      # approximate integer step

        # Ensure joint_pos and timesteps are on the same device as the NCAP
        # parameters. observations may arrive on CPU even when the model is
        # on CUDA (PPO's updater does not move batches automatically).
        device    = next(self.ncap.parameters()).device
        joint_pos = joint_pos.to(device)
        timesteps = timesteps.to(device)

        # NCAP forward → deterministic mean in [−1, 1]
        mean = self.ncap(joint_pos, timesteps=timesteps)

        # Scale mean into action range if bounds have been built
        if "_action_low" in dict(self.named_buffers()):
            scale  = (self._action_high - self._action_low) / 2.0
            offset = (self._action_high + self._action_low) / 2.0
            mean   = mean * scale + offset

        std  = self.log_std.exp().expand_as(mean)
        dist = torch.distributions.Normal(mean, std)

        # Attach sample_with_log_prob so CustomA2C._step works unchanged
        dist.sample_with_log_prob = lambda: _sample_with_log_prob(dist)
        return dist


# ---------------------------------------------------------------------------
# Deterministic actor for DDPG
# ---------------------------------------------------------------------------

class _DeterministicNCAPActor(_NCAPActor):
    # DDPG variant: forward() returns a plain Tensor (mean action),
    # not a Distribution.  Tonic DDPG calls actor(obs) expecting a tensor
    # to differentiate through, and target_actor(obs) the same way.

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        n         = self.ncap.n_joints
        joint_pos = observations[..., :n]
        time_feat = observations[..., -1]
        timesteps = (time_feat * 1000).round()
        device    = next(self.ncap.parameters()).device
        joint_pos = joint_pos.to(device)
        timesteps = timesteps.to(device)
        mean = self.ncap(joint_pos, timesteps=timesteps)
        if '_action_low' in dict(self.named_buffers()):
            scale  = (self._action_high - self._action_low) / 2.0
            offset = (self._action_high + self._action_low) / 2.0
            mean   = mean * scale + offset
        return mean


# ---------------------------------------------------------------------------
# Critic
# ---------------------------------------------------------------------------

class _ValueCritic(_TonicModule):
    # V(s) critic for on-policy algorithms (A2C, PPO).
    # Input: observations only.

    def __init__(self, hidden: int = 256):
        super().__init__()
        self.hidden = hidden
        self.net    = None

    def _build(self, observation_space, action_space):
        obs_dim = int(np.prod(observation_space.shape))
        self.net = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, self.hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(self.hidden, self.hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(self.hidden, 1),
        )

    def forward(self, observations: torch.Tensor,
                actions: torch.Tensor = None):
        if self.net is None:
            raise RuntimeError("_ValueCritic not initialised — call agent.initialize() first.")
        return self.net(observations).squeeze(-1)


class _QCritic(_TonicModule):
    # Q(s,a) critic for off-policy algorithms (DDPG).
    # Input: concat(observations, actions).
    # Keeping obs and action inputs completely separate avoids the
    # zero-padding corruption that occurred with the shared _NCAPCritic.

    def __init__(self, hidden: int = 256):
        super().__init__()
        self.hidden = hidden
        self.net    = None

    def _build(self, observation_space, action_space):
        obs_dim = int(np.prod(observation_space.shape))
        act_dim = int(np.prod(action_space.shape))
        in_dim  = obs_dim + act_dim
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, self.hidden),
            torch.nn.ReLU(),           # ReLU works better than Tanh for Q-functions
            torch.nn.Linear(self.hidden, self.hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden, 1),
        )

    def forward(self, observations: torch.Tensor,
                actions: torch.Tensor = None):
        if self.net is None:
            raise RuntimeError("_QCritic not initialised — call agent.initialize() first.")
        if actions is None:
            raise ValueError("_QCritic requires actions — pass actions=... to forward()")
        x = torch.cat([observations, actions], dim=-1)
        return self.net(x).squeeze(-1)


# Keep _NCAPCritic as an alias for backward compatibility
_NCAPCritic = _ValueCritic


# ---------------------------------------------------------------------------
# Combined model
# ---------------------------------------------------------------------------

class _NCAPModel(_TonicModule):
    """
    Actor + critic bundle satisfying the full Tonic model contract:

        model.actor                  → callable returning a Distribution
        model.critic                 → callable returning scalar values
        model.observation_normalizer → None  (disabled)
        model.return_normalizer      → None  (disabled)
        model.initialize(obs_sp, act_sp)
    """

    def __init__(self, ncap: SimpleNCAPSwimmer, algorithm: str = 'a2c'):
        super().__init__()
        alg = algorithm.lower()
        # DDPG: deterministic actor + Q-critic (obs+action input)
        # A2C/PPO: stochastic actor + value critic (obs-only input)
        if alg == 'ddpg':
            self.actor  = _DeterministicNCAPActor(ncap)
            self.critic = _QCritic(hidden=256)
        else:
            self.actor  = _NCAPActor(ncap)
            self.critic = _ValueCritic(hidden=256)
        # Tonic checks for these; None disables both normalizers
        self.observation_normalizer = None
        self.return_normalizer      = None

    def initialize(self, observation_space, action_space):
        # Cascade Tonic two-stage init down to actor and critic.
        # Also builds target_actor / target_critic required by DDPG:
        # Tonic's DDPG updater calls model.target_actor(next_obs) and
        # model.target_critic(next_obs, next_actions) for TD targets.
        self.actor.initialize(observation_space, action_space)
        self.critic.initialize(observation_space, action_space)

        # Deep copies updated by Polyak averaging, not by the optimiser
        import copy
        self.target_actor  = copy.deepcopy(self.actor)
        self.target_critic = copy.deepcopy(self.critic)
        for p in self.target_actor.parameters():
            p.requires_grad_(False)
        for p in self.target_critic.parameters():
            p.requires_grad_(False)

    def update_targets(self, tau: float = 0.005):
        # Polyak averaging: target ← τ * online + (1-τ) * target
        # Called by Tonic's DDPG after every _update_actor_critic step.
        if not hasattr(self, 'target_actor'):
            return
        for p_tgt, p_src in zip(self.target_actor.parameters(),
                                self.actor.parameters()):
            p_tgt.data.mul_(1.0 - tau).add_(tau * p_src.data)
        for p_tgt, p_src in zip(self.target_critic.parameters(),
                                self.critic.parameters()):
            p_tgt.data.mul_(1.0 - tau).add_(tau * p_src.data)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class SimpleNCAPTrainer:
    """
    End-to-end trainer: SimpleNCAPSwimmer on the plain dm_control swimmer.

    Parameters
    ----------
    n_links           : total link count (joints = n_links − 1)
    oscillator_period : CPG period in control steps  (default 60)
    training_steps    : total environment interaction steps
    save_steps        : checkpoint / test frequency
    log_episodes      : (reserved) episodes between console progress logs
    output_dir        : directory for checkpoints and logs
    """

    def __init__(
        self,
        n_links: int = 6,
        oscillator_period: int = 60,
        algorithm: str = 'a2c',
        num_workers: int = 1,
        training_steps: int = 1_000_000,
        save_steps: int = 50_000,
        log_episodes: int = 10,
        output_dir: str = None,
        water_only: bool = False,
        progressive: bool = False,
        resume_checkpoint: str = None,
    ):
        self.n_links            = n_links
        self.oscillator_period  = oscillator_period
        self.algorithm          = algorithm.lower()
        self.num_workers        = num_workers
        self.training_steps     = training_steps
        self.save_steps         = save_steps
        self.log_episodes       = log_episodes
        self.water_only         = water_only
        self.progressive        = progressive
        self.resume_checkpoint  = resume_checkpoint

        # Output directory: explicit > auto-generated from algorithm + mode
        if output_dir is None:
            if progressive:
                _mt = 'progressive'
            elif water_only:
                _mt = 'water_only'
            else:
                _mt = 'simple'
            output_dir = (f"outputs/simple_ncap_{algorithm}_{n_links}links"
                          f"_osc{oscillator_period}_{_mt}")
        self.output_dir = output_dir

        os.makedirs(output_dir, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[SimpleNCAPTrainer] device = {self.device}")

    # ------------------------------------------------------------------
    def _build_env(self):
        if self.progressive:
            Wrapper = _get_progressive_wrapper()
            return Wrapper(n_links=self.n_links, time_feature=True)
        return TonicSimpleSwimmerWrapper(
            n_links=self.n_links,
            time_feature=True,
        )

    def _build_model(self) -> _NCAPModel:
        """Build NCAP model.  n_joints = n_links − 1 (dm_control convention)."""
        n_joints = self.n_links - 1
        ncap = SimpleNCAPSwimmer(
            n_joints=n_joints,
            oscillator_period=self.oscillator_period,
            use_weight_sharing=True,
            use_weight_constraints=True,
            include_proprioception=True,
            include_head_oscillators=True,
        )
        model = _NCAPModel(ncap=ncap, algorithm=self.algorithm)
        model.to(self.device)
        return model

    # ------------------------------------------------------------------
    def train(self):
        """Run full training and save final checkpoint."""
        print("=" * 60)
        print("  Simple NCAP Swimmer – Training")
        print(f"  n_links           = {self.n_links}")
        print(f"  oscillator_period = {self.oscillator_period}")
        print(f"  training_steps    = {self.training_steps:,}")
        print(f"  save_steps        = {self.save_steps:,}")
        print(f"  output_dir        = {self.output_dir}")
        print("=" * 60)

        # ---- environments -----------------------------------------------
        env      = self._build_env()
        test_env = self._build_env()
        obs_dim  = env.observation_space.shape[0]
        act_dim  = env.action_space.shape[0]
        print(f"[env] obs_dim={obs_dim}  act_dim={act_dim}")

        # ---- Tonic logger -----------------------------------------------
        log_dir = os.path.join(
            "outputs", "training_logs",
            f"simple_ncap_{self.n_links}links_tonic",
        )
        tonic.logger.initialize(path=log_dir)

        # ---- model + agent ----------------------------------------------
        model = self._build_model()
        agent = CustomA2C(model=model)

        # agent.initialize() → a2c.A2C.initialize() → model.initialize()
        # → actor._build()  (registers action buffers)
        # → critic._build() (constructs the lazy MLP net)
        agent.initialize(
            observation_space=env.observation_space,
            action_space=env.action_space,
            seed=42,
        )

        # critic._build() created new linear layers after the initial .to(device),
        # so move the whole model again to make sure everything is on the GPU.
        model.to(self.device)

        # ---- Tonic trainer ----------------------------------------------
        trainer = tonic.Trainer(
            steps=self.training_steps,
            save_steps=self.save_steps,
            test_episodes=5,
        )
        trainer.initialize(
            agent=agent,
            environment=env,
            test_environment=test_env,
        )

        print("[SimpleNCAPTrainer] Starting Tonic training loop …")
        trainer.run()

        # ---- final checkpoint -------------------------------------------
        save_path = os.path.join(self.output_dir, "final_model")
        agent.save(save_path)
        print(f"[SimpleNCAPTrainer] ✔ Done.  Final model → {save_path}")

        return agent, env


# ─────────────────────────────────────────────────────────────────────────────
# PATCHED train() — replaces the original above; Python's method resolution
# uses the last definition in the class body, but since we're appending to
# the module we instead monkey-patch the class after definition.
# ─────────────────────────────────────────────────────────────────────────────

def _train_with_logging(self):
    """Full training with SimpleNCAPLogger hooks (replaces original train())."""
    if self.progressive:
        mode_tag = 'progressive'
    elif self.water_only:
        mode_tag = 'water_only'
    else:
        mode_tag = 'simple'
    alg_note = ALGORITHM_NOTES.get(self.algorithm, self.algorithm.upper())

    print("=" * 60)
    print("  Simple NCAP Swimmer – Training")
    print(f"  algorithm         = {self.algorithm.upper()}  ({alg_note})")
    print(f"  n_links           = {self.n_links}")
    print(f"  oscillator_period = {self.oscillator_period}")
    print(f"  training_steps    = {self.training_steps:,}")
    print(f"  save_steps        = {self.save_steps:,}")
    print(f"  num_workers       = {self.num_workers}")
    print(f"  output_dir        = {self.output_dir}")
    print("=" * 60)

    os.makedirs(self.output_dir, exist_ok=True)

    env      = self._build_env()
    test_env = self._build_env()
    obs_dim  = env.observation_space.shape[0]
    act_dim  = env.action_space.shape[0]
    print(f"[env] obs_dim={obs_dim}  act_dim={act_dim}")

    log_dir = os.path.join(self.output_dir, "tonic_logs")
    tonic.logger.initialize(path=log_dir)

    model = self._build_model()
    # ── select agent based on --algorithm ────────────────────────────────
    agent = build_agent(self.algorithm, model)
    print(f"[agent] {type(agent).__name__} initialised")
    agent.initialize(
        observation_space=env.observation_space,
        action_space=env.action_space,
        seed=42,
    )
    model.to(self.device)

    # ── build our logger ──────────────────────────────────────────────────
    prefix = make_prefix(
        n_links=self.n_links,
        algorithm=self.algorithm,
        oscillator_period=self.oscillator_period,
        training_mode=mode_tag,
    )
    vis_logger = SimpleNCAPLogger(
        output_dir=self.output_dir,
        prefix=prefix,
        n_links=self.n_links,
        oscillator_period=self.oscillator_period,
        log_every=8,
        eval_every=self.save_steps,
    )

    # ── resume from checkpoint if requested ──────────────────────────────
    start_step = 0
    if self.resume_checkpoint is not None:
        print(f"[resume] loading checkpoint: {self.resume_checkpoint}")
        start_step = self._load_checkpoint(agent, self.resume_checkpoint)
        remaining  = self.training_steps - start_step
        if remaining <= 0:
            print(f"[resume] checkpoint step {start_step:,} >= "
                  f"training_steps {self.training_steps:,} — nothing to do.")
            return agent, env
        print(f"[resume] resuming from step {start_step:,}, "
              f"{remaining:,} steps remaining")

    # ── instrumented loop ─────────────────────────────────────────────────
    VIZ_EVERY = 25_000   # video + plots saved every this many steps

    print(f"[SimpleNCAPTrainer] Starting instrumented training loop "
          f"(steps {start_step:,} → {self.training_steps:,}) …")
    observations = np.array(env.start())   # (1, obs_dim)
    last_ckpt    = start_step - 1
    last_viz     = start_step - 1

    for step in range(start_step, self.training_steps):
        # Advance phase schedule for progressive env (no-op otherwise)
        if self.progressive and hasattr(env, 'set_training_progress'):
            env.set_training_progress(step / max(1, self.training_steps))

        actions = agent.step(observations, step)         # (1, act_dim)
        _obs, infos = env.step(actions[0])

        next_obs     = np.array(infos['observations'])   # (1, obs_dim)
        rewards      = infos['rewards']
        resets       = infos['resets']
        terminations = infos['terminations']

        vis_logger.on_step(step, float(rewards[0]))

        agent.update(
            observations=next_obs,
            rewards=rewards,
            resets=resets,
            terminations=terminations,
            steps=step,
        )
        observations = next_obs

        # ── checkpoint (model save + eval) ───────────────────────────────
        if step > start_step and step % self.save_steps == 0 and step != last_ckpt:
            last_ckpt = step
            eval_returns = _evaluate_agent(agent, self._build_env(), n_episodes=5,
                                           max_steps=test_env.max_steps)
            vis_logger.on_eval(step, eval_returns)
            self._save_checkpoint(agent, step)
            print(f"[step {step:>8,}] eval_mean={np.mean(eval_returns):.3f}")

        # ── visualisations + video every VIZ_EVERY steps ─────────────────
        if step > start_step and step % VIZ_EVERY == 0 and step != last_viz:
            last_viz = step
            vis_logger.on_checkpoint(step, agent, self._build_env(), phase=0)

    print("[SimpleNCAPTrainer] Training loop complete.")

    # ── final save + plots ────────────────────────────────────────────────
    save_path = os.path.join(self.output_dir, "final_model")
    agent.save(save_path)
    print(f"[SimpleNCAPTrainer] ✔ Done.  Final model → {save_path}")
    vis_logger.finalize(agent, self._build_env(), phase=0)

    return agent, env


def _evaluate_agent(agent, env, n_episodes: int = 5, max_steps: int = 1000) -> list:
    """Run n_episodes of deterministic rollouts; return list of total returns."""
    returns = []
    for _ in range(n_episodes):
        obs       = env.reset()
        ep_return = 0.0
        done      = False
        steps     = 0
        while not done and steps < max_steps:
            action = agent.test_step(np.array([obs]), steps=0)
            result = env.step(action[0])
            if isinstance(result[1], dict):
                obs, infos = result
                ep_return += float(infos['rewards'][0])
                done       = bool(infos['resets'][0])
            else:
                obs, r, done, _ = result
                ep_return += r
            steps += 1
        returns.append(ep_return)
    return returns



# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint save / load helpers — added to SimpleNCAPTrainer
# ─────────────────────────────────────────────────────────────────────────────

def _save_checkpoint_method(self, agent, step: int) -> str:
    import json
    path = os.path.join(self.output_dir, f"checkpoint_{step}")
    agent.save(path)
    meta = {"step": step, "algorithm": self.algorithm,
            "n_links": self.n_links, "oscillator_period": self.oscillator_period}
    with open(path + "_meta.json", "w") as f:
        json.dump(meta, f)
    print(f"[checkpoint] saved -> {path}  (step {step:,})")
    return path


def _load_checkpoint_method(self, agent, path: str) -> int:
    import json
    meta_path = path + "_meta.json"
    start_step = 0
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            start_step = json.load(f).get("step", 0)
    loaded = False
    for candidate in [path, path + ".pt"]:
        if os.path.exists(candidate):
            agent.load(candidate)
            loaded = True
            break
    if not loaded:
        raise FileNotFoundError(
            f"Checkpoint not found at '{path}' or '{path}.pt'. "
            "Check --resume_checkpoint path."
        )
    print(f"[checkpoint] resumed from '{path}'  (step {start_step:,})")
    return start_step


SimpleNCAPTrainer._save_checkpoint  = _save_checkpoint_method
SimpleNCAPTrainer._load_checkpoint  = _load_checkpoint_method

# Monkey-patch the class so the new method is used
SimpleNCAPTrainer.train = _train_with_logging
