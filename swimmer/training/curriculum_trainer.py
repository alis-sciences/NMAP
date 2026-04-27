#!/usr/bin/env python3
"""
Curriculum Trainer for Swimming and Crawling
Manages progressive training from simple swimming to complex mixed environments.
"""

import torch
import torch.nn as nn
import numpy as np
import os
import time
import tonic
import warnings
from tqdm import tqdm

# Suppress the harmless gym Box precision warning
warnings.filterwarnings("ignore", message=".*Box bound precision lowered by casting to.*")
from ..models.biological_ncap import BiologicalNCAPSwimmer, BiologicalNCAPActor
from ..models.enhanced_biological_ncap import EnhancedBiologicalNCAPSwimmer
from ..models.simple_ncap import SimpleNCAPSwimmer
#from ..models.ncap_ccmn import CCMNSwimmer, GaitPeriodScheduler
from ..models.ncap_ccmn_hrl import CCMNSwimmerHRL as CCMNSwimmer, GaitPeriodScheduler
from ..environments.progressive_mixed_env import TonicProgressiveMixedWrapper
from ..utils.training_logger import TrainingLogger
from ..utils.curriculum_visualization import create_curriculum_plots, create_test_video, create_phase_comparison_video, save_training_summary, create_trajectory_analysis, create_body_profile_figure
from ..utils.ccmn_visualizations import create_ccmn_neuromodulatory_analysis
from ..utils.ccmn_diagnostics import run_ccmn_sanity_check, run_film_ablation_comparison, plot_ccmn_diagnostic_summary
from ..utils.ccmn_visualizations import create_ccmn_neuromodulatory_analysis
from ..utils.artifact_naming import ArtifactNamer, detect_model_type

# Force basic logging to avoid background monitoring overhead
ADVANCED_LOGGING_AVAILABLE = False


class CurriculumNCAPTrainer:
    """
    Curriculum trainer for NCAP swimmer with progressive complexity.
    
    Designed for 1M episode training with curriculum learning:
    - Phase 1 (0-30%): Pure swimming in simple environment
    - Phase 2 (30-60%): Introduction of single land zone
    - Phase 3 (60-80%): Two land zones for complex navigation
    - Phase 4 (80-100%): Full mixed environment complexity
    """
    
    # Phase duration configuration (easily modifiable)
    PHASE_DURATION_CONFIG = {
        'evaluation_steps': [400, 600, 800, 1200],     # **INCREASED** Steps per episode for each phase (was 200,200,200,400)
        'video_steps': [800, 1000, 1200, 1500],         # **INCREASED** Steps per video for each phase (was 500,500,500,1000)
        'trajectory_multiplier': [1.5, 2.0, 2.5, 3.0] # **INCREASED** Multiplier for trajectory analysis (was 1.0,1.0,1.0,2.0)
    }
    
    def __init__(self, 
                 n_links=5,
                 learning_rate=3e-5,
                 training_steps=1000000,
                 save_steps=50000,
                 log_episodes=50,
                 device='cuda' if torch.cuda.is_available() else 'cpu',
                 oscillator_period=60,
                 min_oscillator_strength=0.8,  # **REDUCED** from 1.2 to 0.8 for speed flexibility
                 min_coupling_strength=0.5,  # **REDUCED** from 0.8 to 0.5 for speed flexibility  
                 biological_constraint_frequency=25000,  # **REDUCED** frequency: every 25k steps
                 resume_from_checkpoint=None,  # Path to checkpoint to resume from
                  model_type='enhanced_ncap',  # Model type: biological_ncap, enhanced_ncap, ccmn
                 algorithm='ppo',  # Algorithm for naming
                 num_workers=8,    # NEW: Number of parallel environments
                 use_multi_gpu=True, # NEW: Use all available GPUs
                 use_locomotion_only_early_training=True,
                 expose_environment_observation=True,
                 expose_viscosity_observation=True,
                 anisotropic_drag_mode='off',
                 anisotropic_drag_ratio=10.0,
                 anisotropic_drag_gain=0.02,
                 anisotropic_drag_land_only=True,
                 ccmn_phase1_land_fraction=0.15,
                 ccmn_sequential_training=True,
                 film_ablation=False,
                 run_sanity_check=False,
                 sanity_check_steps=600,
                 bistability_regulariser=False,
                 bistability_lambda=0.01,
                 ccmn_use_viscosity_input=False,
                 use_amplitude_scaling=True,
                 use_metabolic_bonus=True,
                 transition_bonus=5.0,
                 mismatch_penalty_weight=0.3,
                 water_velocity_scale=6.0,
                 training_mode='progressive',
                 forced_segment_steps=200,
                 sustained_swim_bonus_per_step=0.05,
                 sustained_swim_min_bout=50,
                 transition_cooldown_steps=100,
                 bayes_likelihood_sharpness=4.0,
                 bayes_reward_weight=0.3,
                 bayes_epistemic_penalty_weight=0.2,
                 use_ext_env=False,
                 swim_only=False,
                 output_dir='outputs'):
        
        self.output_dir = output_dir
        self.n_links = n_links
        self.learning_rate = learning_rate
        self.training_steps = training_steps
        self.save_steps = save_steps
        self.log_episodes = log_episodes
        self.device = device
        self.num_workers = num_workers
        self.use_multi_gpu = use_multi_gpu
        self.oscillator_period = oscillator_period
        self.min_oscillator_strength = min_oscillator_strength
        self.min_coupling_strength = min_coupling_strength
        self.biological_constraint_frequency = biological_constraint_frequency
        self.resume_from_checkpoint = resume_from_checkpoint
        self.model_type = model_type
        self.algorithm = algorithm
        self.use_locomotion_only_early_training = use_locomotion_only_early_training
        self.expose_environment_observation = expose_environment_observation
        self.expose_viscosity_observation = expose_viscosity_observation
        self.anisotropic_drag_mode = anisotropic_drag_mode
        self.anisotropic_drag_ratio = anisotropic_drag_ratio
        self.anisotropic_drag_gain = anisotropic_drag_gain
        self.anisotropic_drag_land_only    = anisotropic_drag_land_only
        self.ccmn_phase1_land_fraction     = ccmn_phase1_land_fraction
        self.ccmn_sequential_training      = ccmn_sequential_training
        self.film_ablation                 = film_ablation
        self.run_sanity_check              = run_sanity_check
        self.sanity_check_steps            = sanity_check_steps
        self.bistability_regulariser       = bistability_regulariser
        self.bistability_lambda            = bistability_lambda
        self.ccmn_use_viscosity_input      = ccmn_use_viscosity_input
        self.use_amplitude_scaling         = use_amplitude_scaling
        self.use_metabolic_bonus           = use_metabolic_bonus
        self.transition_bonus              = transition_bonus
        self.mismatch_penalty_weight       = mismatch_penalty_weight
        self.water_velocity_scale          = water_velocity_scale
        self.training_mode                 = training_mode
        self.forced_segment_steps          = forced_segment_steps
        self.sustained_swim_bonus_per_step = sustained_swim_bonus_per_step
        self.sustained_swim_min_bout       = sustained_swim_min_bout
        self.transition_cooldown_steps     = transition_cooldown_steps
        self.bayes_likelihood_sharpness    = bayes_likelihood_sharpness
        self.bayes_reward_weight           = bayes_reward_weight
        self.bayes_epistemic_penalty_weight = bayes_epistemic_penalty_weight
        self.use_ext_env                   = use_ext_env
        self.swim_only                     = swim_only

        # Select environment wrapper based on use_ext_env flag.
        # use_ext_env=True  → TonicProgressiveMixedWrapperExt (composite reward:
        #                       dm_control base velocity + switching components,
        #                       episode returns in 200-1000 range matching NCAP paper)
        # use_ext_env=False → TonicProgressiveMixedWrapper (original custom reward)
        if use_ext_env:
            from ..environments.progressive_ext_env import (
                TonicProgressiveMixedWrapperExt as _EnvWrapper
            )
            print('🧪 Using extended environment (progressive_ext_env): '
                  'dm_control base reward + switching components')
        else:
            from ..environments.progressive_mixed_env import (
                TonicProgressiveMixedWrapper as _EnvWrapper
            )
        self._EnvWrapper = _EnvWrapper
        
        # Initialize artifact namer for consistent naming across all outputs
        self.artifact_namer = ArtifactNamer(
            model_type=model_type,
            n_links=n_links,
            algorithm=algorithm,
            additional_config={
                'oscillator_period': oscillator_period,
                'training_mode': 'curriculum'
            }
        )
        
        # Training state
        self.current_step = 0
        self.current_episode = 0
        self.phase_rewards = {0: [], 1: [], 2: [], 3: []}
        self.phase_distances = {0: [], 1: [], 2: [], 3: []}

        # Per-step reward log — flushed to CSV every REWARD_LOG_INTERVAL steps.
        # Stores (step, mean_reward_across_workers, phase) tuples accumulated
        # between flushes.  Kept small so memory cost is negligible.
        self._REWARD_LOG_INTERVAL = 10_000
        self._reward_log_buffer   = []
        self._reward_csv_path     = None
        self._last_reward_flush   = 0   # tracks step of last flush
        
        # Initialize components with advanced logging if available
        log_dir = os.path.dirname(self.artifact_namer.training_log_dir())
        experiment_name = self.artifact_namer.base_id
        
        if ADVANCED_LOGGING_AVAILABLE:
            self.logger = AdvancedTrainingLogger(
                log_dir=log_dir, 
                experiment_name=experiment_name
            )
            print("🔬 Using advanced logging with hardware monitoring")
        else:
            self.logger = TrainingLogger(
                log_dir=log_dir,
                experiment_name=experiment_name
            )
            print("📊 Using standard logging")
        
        print(f"🎓 Initialized Curriculum {model_type.upper()} Trainer")
        print(f"   Model: {model_type} with {n_links} links")
        print(f"   Algorithm: {algorithm}")
        print(f"   Device: {device}")
        print(f"   Total training: {training_steps:,} steps")
        print(f"   Artifact ID: {self.artifact_namer.base_id}")
        print(f"   Phase progression:")
        print(f"     Phase 1 (0-30%): Pure swimming")
        print(f"     Phase 2 (30-60%): Single land zone")
        print(f"     Phase 3 (60-80%): Two land zones")
        print(f"     Phase 4 (80-100%): Full complexity")
        if model_type == 'ccmn':
            print(f"   ── CCMN options ──")
            print(f"     Phase-1 land fraction:   {ccmn_phase1_land_fraction:.0%}")
            print(f"     Sequential GRU training: {ccmn_sequential_training}")
            print(f"     FiLM ablation:           {film_ablation}")
            print(f"     Sanity check:            {run_sanity_check}")
            print(f"     Bistability regulariser: {bistability_regulariser} (λ={bistability_lambda})")
            vis_mode = "legacy (viscosity passed)" if ccmn_use_viscosity_input else "bio-faithful (φ(s,t) only)"
            print(f"     ContextEncoder input:    {vis_mode}")
            print(f"     Amplitude scaling:       {use_amplitude_scaling}")
            print(f"     Metabolic bonus:         {use_metabolic_bonus}")
        
    def create_environment(self):
        """Create progressive mixed environment (ext or standard based on use_ext_env flag)."""
        env = self._EnvWrapper(
            n_links=self.n_links,
            time_feature=True,
            desired_speed=0.15,
            expose_environment_observation=self.expose_environment_observation,
            expose_viscosity_observation=self.expose_viscosity_observation,
            anisotropic_drag_mode=self.anisotropic_drag_mode,
            anisotropic_drag_ratio=self.anisotropic_drag_ratio,
            anisotropic_drag_gain=self.anisotropic_drag_gain,
            anisotropic_drag_land_only=self.anisotropic_drag_land_only,
            use_metabolic_bonus=self.use_metabolic_bonus,
            transition_bonus=self.transition_bonus,
            mismatch_penalty_weight=self.mismatch_penalty_weight,
            water_velocity_scale=self.water_velocity_scale,
            training_mode=self.training_mode,
            forced_segment_steps=self.forced_segment_steps,
            sustained_swim_bonus_per_step=self.sustained_swim_bonus_per_step,
            sustained_swim_min_bout=self.sustained_swim_min_bout,
            transition_cooldown_steps=self.transition_cooldown_steps,
            bayes_likelihood_sharpness=self.bayes_likelihood_sharpness,
            bayes_reward_weight=self.bayes_reward_weight,
            bayes_epistemic_penalty_weight=self.bayes_epistemic_penalty_weight,
        )
        return env
    
    def create_vectorized_environment(self):
        """Create multiple environments running in parallel."""
        from ..environments.vectorized_env import SubprocVecEnv
        print(f"🌊 Creating {self.num_workers} parallel environments...")
        # IMPORTANT: use a default-argument capture (env_fn=self.create_environment)
        # to avoid the classic Python lambda closure bug where every lambda in a
        # list comprehension captures the same late-bound 'self' reference.
        # Without this, all workers would share one environment instance.
        # Each element must be a zero-argument callable that CREATES and
        # RETURNS a new environment instance when called.
        # The default-argument capture (create_fn=self.create_environment)
        # freezes the reference so each lambda is independent.
        env_fns = [
            (lambda create_fn=self.create_environment: create_fn())
            for _ in range(self.num_workers)
        ]
        return SubprocVecEnv(env_fns)
    
    def create_model(self):
        """Create NCAP model optimized for curriculum learning based on model_type."""
        n_joints = self.n_links - 1  # 4 joints for 5-link swimmer
        
        # **NEW**: Determine if we should use locomotion-only mode for early training
        # Handle evaluation mode where training_steps=0
        if self.swim_only:
            # swim_only: always Phase 0, always locomotion-only
            training_progress = 0.0
        elif self.training_steps > 0:
            training_progress = self.current_step / self.training_steps
        else:
            # Evaluation mode - use full progress (1.0) to enable all features
            training_progress = 1.0
            
        use_locomotion_only = (self.use_locomotion_only_early_training and 
                             training_progress < 0.3)  # First 30% of training
        
        if self.model_type == 'enhanced_ncap':
            model = EnhancedBiologicalNCAPSwimmer(
                n_joints=n_joints,
                oscillator_period=self.oscillator_period,
                include_environment_adaptation=True,  # Dramatic frequency adaptation
                include_goal_direction=not use_locomotion_only,  # **DISABLED** for early training
                locomotion_only_mode=use_locomotion_only,  # **NEW**: Pure swimming mode
                action_scaling_factor=1.8  # **NEW**: Increased scaling for stronger swimming
            ).to(self.device)
            
            print(f"🚀 Created ENHANCED Biological NCAP model with {sum(p.numel() for p in model.parameters())} parameters")
            print(f"🔬 Relaxation oscillator: Asymmetric (60/40 phase) with 5x frequency adaptation")
            if use_locomotion_only:
                print(f"🏊 TRAINING MODE: Pure locomotion (first 30% of training)")
            else:
                print(f"🎯 Goal-directed navigation: Target-seeking with anti-tail-chasing fixes")
            print(f"📄 Based on C. elegans research: https://elifesciences.org/articles/69905")
            
        elif self.model_type == 'biological_ncap':
            model = BiologicalNCAPSwimmer(
                n_joints=n_joints,
                oscillator_period=self.oscillator_period,
                include_environment_adaptation=True  # Enable biological adaptation
            ).to(self.device)
            
            print(f"🧬 Created standard Biological NCAP model with {sum(p.numel() for p in model.parameters())} parameters")
            print(f"🔬 Biological adaptation: ENABLED (no LSTM - pure neuromodulation)")
            
        elif self.model_type == 'ccmn_hrl':
            model = CCMNSwimmer(
                n_joints=n_joints,
                oscillator_period=self.oscillator_period,
                context_hidden=32, #previously 16
                use_weight_sharing=True,
                use_weight_constraints=True,
                include_proprioception=True,
                include_head_oscillators=True,
                use_viscosity_input=self.ccmn_use_viscosity_input,
                use_amplitude_scaling=self.use_amplitude_scaling,
            ).to(self.device)

            print(f"🧠 Created CCMN (Context-Conditioned Modular Network) model "
                  f"with {sum(p.numel() for p in model.parameters())} parameters")
            print(f"🔬 FiLM neuromodulation: DA/5-HT gating via ContextEncoder + FiLMGenerator")
            if self.ccmn_use_viscosity_input:
                print(f"⚠️  ContextEncoder: LEGACY mode — viscosity passed directly")
            else:
                print(f"🧬 ContextEncoder: BIO-FAITHFUL — infers substrate from φ(s,t) only")
            print(f"🔀 Gait switching: bistable oscillator period "
                  f"({GaitPeriodScheduler.SWIM_PERIOD}–{GaitPeriodScheduler.CRAWL_PERIOD} steps) "
                  f"driven by z_DA")

        elif self.model_type == 'simple_ncap':
            # use_weight_constraints=True with use_weight_constant_init=True
            # initialises all weights at the exact boundary of their clamp
            # range (muscle_ipsi=1.0, muscle_contra=-1.0).  Any gradient step
            # that pushes in the wrong direction is immediately clamped back,
            # so weights never change under PPO/DDPG.
            # Fix: use uniform random init within the valid range so gradients
            # have room to move, and remove hard clamping so the optimiser is
            # not fighting a wall on every step.  Biological sign constraints
            # (excitatory/inhibitory) are still enforced via the architecture
            # structure — muscle_ipsi excites and muscle_contra inhibits by
            # construction through the antagonistic torque computation.
            model = SimpleNCAPSwimmer(
                n_joints=n_joints,
                oscillator_period=self.oscillator_period,
                use_weight_sharing=True,
                use_weight_constraints=False,   # remove clamp — weights free to move
                use_weight_constant_init=False,  # random init in (-1, 1) not at boundary
                include_proprioception=True,
                include_head_oscillators=True,
            ).to(self.device)
            print(f"🧬 Created SimpleNCAPSwimmer (Bhattasali et al. baseline) "
                  f"with {sum(p.numel() for p in model.parameters())} parameters")
            print(f"   period={self.oscillator_period}  |  No ContextEncoder  |  No FiLM")
            print(f"   Weight constraints: OFF (random init, unclamped — required for gradient learning)")

        else:
            raise ValueError(f"Unsupported model type: {self.model_type}. "
                           f"Supported: 'biological_ncap', 'enhanced_ncap', 'ccmn', 'ccmn_hrl', 'simple_ncap'")
        
        # Enable Multi-GPU if requested and available
        if self.use_multi_gpu and torch.cuda.device_count() > 1:
            print(f"⚡ Using {torch.cuda.device_count()} GPUs with DataParallel")
            model = torch.nn.DataParallel(model)

        # ── GPU performance knobs ─────────────────────────────────────────
        # 1. cuDNN auto-tuner: benchmarks convolution algorithms on first
        #    batch and picks the fastest for this input shape.  Safe to
        #    enable whenever input shapes are fixed (they are here).
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark     = True
            torch.backends.cudnn.deterministic = False   # required for benchmark=True
            print("⚡ cuDNN benchmark enabled (deterministic=False)")

        # 2. torch.compile: fuses GRU + FiLM + CPG ops into a single CUDA
        #    kernel, eliminating Python-level per-op dispatch overhead.
        #    mode='reduce-overhead' is the right choice for a training loop
        #    that repeats the same forward/backward graph every step.
        #    Falls back silently on PyTorch < 2.0 or when CUDA is absent.
        if torch.cuda.is_available() and hasattr(torch, 'compile'):
            try:
                actual = model.module if isinstance(model, torch.nn.DataParallel) else model
                actual = torch.compile(actual, mode='reduce-overhead')
                if isinstance(model, torch.nn.DataParallel):
                    model.module = actual
                else:
                    model = actual
                print("⚡ torch.compile applied (mode='reduce-overhead')")
            except Exception as _e:
                print(f"⚠️  torch.compile skipped: {_e}")

        return model
    
    def create_agent(self, model, env):
        """Create simplified agent for curriculum training."""
        trainer_lr = self.learning_rate
        
        # Create biological NCAP agent wrapper with environment adaptation
        class BiologicalNCAPAgent:
            def __init__(self, ncap_model, environment, n_links):
                self.ncap_model = ncap_model
                self.n_links = n_links
                self.n_joints = n_links - 1
                self.step_count = 0  # Global step count for evaluation
                self.use_stable_init = False
                self.is_ccmn = isinstance(
                    ncap_model.module if isinstance(ncap_model, torch.nn.DataParallel) else ncap_model,
                    CCMNSwimmer
                )
                self.is_simple_ncap = isinstance(
                    ncap_model.module if isinstance(ncap_model, torch.nn.DataParallel) else ncap_model,
                    SimpleNCAPSwimmer
                )
                # Expose .swimmer for collect_ncap_episode_data compatibility
                if self.is_simple_ncap:
                    self.swimmer = (ncap_model.module
                                   if isinstance(ncap_model, torch.nn.DataParallel)
                                   else ncap_model)
                
                # Initialize RL training components
                self.optimizer = torch.optim.Adam(ncap_model.parameters(), lr=trainer_lr)
                self.num_workers = len(environment.remotes) if hasattr(environment, 'remotes') else 1
                self.episode_buffers = [{'obs': [], 'actions': [], 'rewards': [], 'timesteps': []} for _ in range(self.num_workers)]
                self.training_enabled = True
                self.step_counts = np.zeros(self.num_workers, dtype=int)
                
                # Precompute offsets for observation processing
                self.body_vel_size = self.n_links * 3 + 3
                self.env_features_start = self.n_joints + self.body_vel_size
                self.goal_features_start = self.env_features_start + 5
                
            def step(self, obs):
                """Batch training step - returns actions for all workers."""
                # obs shape: (num_workers, obs_dim)
                actions = self.test_step(obs)

                # Store experience for each worker
                if self.training_enabled:
                    algorithm = getattr(self, '_algorithm', 'ppo')
                    for i in range(self.num_workers):
                        self.episode_buffers[i]['obs'].append(obs[i])
                        self.episode_buffers[i]['timesteps'].append(self.step_counts[i])

                        # ── Exploration noise for PPO and DDPG ────────────────
                        # SimpleNCAPSwimmer is deterministic: stored actions ==
                        # re-predicted actions, so mse_loss gradient is zero and
                        # weights never update.  Adding noise to the *stored*
                        # target (not to the action sent to the environment)
                        # creates a genuine learning signal without degrading
                        # the evaluation trajectory.
                        # ES is immune — it ignores stored actions entirely.
                        if algorithm in ('ppo', 'a2c', 'ddpg'):
                            noise_scale = getattr(self, '_exploration_noise', 0.1)
                            noisy = (np.asarray(actions[i], dtype=np.float32)
                                     + np.random.normal(
                                         0, noise_scale,
                                         size=np.asarray(actions[i]).shape
                                     ).astype(np.float32))
                            stored_action = np.clip(noisy, -1.0, 1.0)
                        else:
                            # ES: use clean action (noise applied at weight level)
                            stored_action = np.asarray(actions[i], dtype=np.float32)

                        self.episode_buffers[i]['actions'].append(stored_action.copy())

                # Return un-noised actions to the environment so the
                # evaluation trajectory is not corrupted by exploration.
                return actions
            
            def add_rewards(self, rewards):
                """Add rewards for all workers."""
                if self.training_enabled:
                    for i, reward in enumerate(rewards):
                        self.episode_buffers[i]['rewards'].append(reward)
            
            def end_episodes(self, indices=None):
                """End episodes for specified workers and train."""
                if not self.training_enabled:
                    return

                if indices is None:
                    indices = range(self.num_workers)

                # Per-worker GRU hidden state reset.
                # Each worker maintains its own slice of the GRU hidden state
                # stored in _gru_hidden_per_worker.  We only clear the workers
                # whose episode just ended — other workers' GRU context is
                # preserved so their episode history is not wiped.
                if self.is_ccmn:
                    actual_model = (self.ncap_model.module
                                    if isinstance(self.ncap_model, torch.nn.DataParallel)
                                    else self.ncap_model)
                    # Initialise per-worker hidden dict on first call
                    if not hasattr(actual_model, '_gru_hidden_per_worker'):
                        actual_model._gru_hidden_per_worker = {}
                    for idx in indices:
                        # Wipe only the finishing worker's hidden state
                        actual_model.reset(worker_id=idx)

                for i in indices:
                    if len(self.episode_buffers[i]['rewards']) >= 5:
                        self._train_on_episode(i)
                    self._reset_buffer(i)
            
            def _train_on_episode(self, worker_idx):
                """Train on stored episode buffer using the selected algorithm.

                'ppo'/'a2c' : REINFORCE-style policy gradient
                'es'        : Gaussian weight perturbation — gradient-free
                'ddpg'      : Regression toward highest-return stored action
                """
                buffer = self.episode_buffers[worker_idx]
                if len(buffer['rewards']) == 0:
                    return

                algorithm = getattr(self, '_algorithm', 'ppo')

                try:
                    actual_model = (self.ncap_model.module
                                    if isinstance(self.ncap_model, torch.nn.DataParallel)
                                    else self.ncap_model)
                    device = next(actual_model.parameters()).device

                    # ── discounted returns ────────────────────────────────────
                    returns = []
                    running_return = 0
                    for reward in reversed(buffer['rewards']):
                        running_return = reward + 0.99 * running_return
                        returns.insert(0, running_return)
                    if len(returns) == 0:
                        return
                    returns_t = torch.FloatTensor(returns).to(device)
                    if returns_t.std() > 1e-6:
                        returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

                    # ── ES: gradient-free perturbation ────────────────────────
                    if algorithm == 'es':
                        episode_return = float(sum(buffer['rewards']))
                        sigma = 0.02
                        lr_es = getattr(self, '_es_lr', 0.01)
                        with torch.no_grad():
                            for param in self.ncap_model.parameters():
                                noise = torch.randn_like(param) * sigma
                                param.data += lr_es * episode_return * noise / (sigma ** 2 + 1e-8)
                            if hasattr(actual_model, '_apply_weight_constraints'):
                                actual_model._apply_weight_constraints()
                        return

                    # ── build batches (ppo/a2c/ddpg) ─────────────────────────
                    obs_batch, action_batch, timestep_batch = [], [], []
                    for i in range(min(len(buffer['obs']),
                                      len(buffer['actions']),
                                      len(buffer['timesteps']))):
                        obs_batch.append(buffer['obs'][i])
                        action_batch.append(buffer['actions'][i])
                        timestep_batch.append(buffer['timesteps'][i])
                    if len(obs_batch) == 0:
                        return

                    # ── DDPG: regress toward highest-return stored action ─────
                    if algorithm == 'ddpg':
                        best_idx    = int(torch.FloatTensor(buffer['rewards']).argmax())
                        best_obs    = np.array([obs_batch[best_idx]])
                        best_act    = torch.FloatTensor(action_batch[best_idx]).to(device).unsqueeze(0)
                        best_ts     = torch.FloatTensor([timestep_batch[best_idx]]).to(device)
                        joint_pos_b = torch.FloatTensor(best_obs[:, :self.n_joints]).to(device)
                        vis_b = (float(best_obs[0, self.env_features_start])
                                 if best_obs.shape[1] > self.env_features_start else 0.0)

                        if self.is_simple_ncap:
                            pred = actual_model(joint_pos_b, timesteps=best_ts)
                        elif self.is_ccmn:
                            pred = actual_model(joint_pos_b, viscosity_norm=vis_b, timesteps=best_ts)
                        else:
                            env_t = None
                            if best_obs.shape[1] >= self.env_features_start + 3:
                                wf  = torch.FloatTensor(best_obs[:, self.env_features_start+1:self.env_features_start+2]).to(device)
                                lf  = torch.FloatTensor(best_obs[:, self.env_features_start+2:self.env_features_start+3]).to(device)
                                vn  = torch.FloatTensor(best_obs[:, self.env_features_start:self.env_features_start+1]).to(device)
                                env_t = torch.cat([wf, lf, vn], dim=1)
                            pred = actual_model(joint_pos_b, environment_type=env_t, timesteps=best_ts)

                        ddpg_loss = torch.nn.functional.mse_loss(pred, best_act)
                        self.optimizer.zero_grad()
                        ddpg_loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.ncap_model.parameters(), 0.5)
                        self.optimizer.step()
                        return

                    # ── PPO / A2C: REINFORCE policy gradient ──────────────────
                    batch_size = min(512, len(obs_batch))  # increased from 64 for GPU utilisation
                    for start_idx in range(0, len(obs_batch), batch_size):
                        end_idx = min(start_idx + batch_size, len(obs_batch))

                        batch_obs       = np.array(obs_batch[start_idx:end_idx])
                        batch_actions   = torch.FloatTensor(np.array(action_batch[start_idx:end_idx])).to(device)
                        batch_timesteps = torch.FloatTensor(np.array(timestep_batch[start_idx:end_idx])).to(device)
                        batch_returns   = returns_t[start_idx:end_idx]

                        joint_pos = torch.FloatTensor(batch_obs[:, :self.n_joints]).to(device)

                        environment_type = None
                        vis_norm = None
                        if batch_obs.shape[1] >= self.env_features_start + 3:
                            water_flag = torch.FloatTensor(batch_obs[:, self.env_features_start+1:self.env_features_start+2]).to(device)
                            land_flag  = torch.FloatTensor(batch_obs[:, self.env_features_start+2:self.env_features_start+3]).to(device)
                            vis_norm   = torch.FloatTensor(batch_obs[:, self.env_features_start:self.env_features_start+1]).to(device)
                            environment_type = torch.cat([water_flag, land_flag, vis_norm], dim=1)

                        target_direction = None
                        if batch_obs.shape[1] >= self.goal_features_start + 3:
                            target_direction = torch.FloatTensor(batch_obs[:, self.goal_features_start+1:self.goal_features_start+3]).to(device)

                        if self.is_simple_ncap:
                            sn_preds = []
                            for w_i in range(joint_pos.shape[0]):
                                p_w = actual_model(joint_pos[w_i:w_i+1],
                                                   timesteps=batch_timesteps[w_i:w_i+1])
                                sn_preds.append(p_w)
                            predicted_actions = torch.cat(sn_preds, dim=0)
                        elif self.is_ccmn:
                            ccmn_preds = []
                            for w_i in range(joint_pos.shape[0]):
                                vis_w = float(vis_norm[w_i, 0].item()) if vis_norm is not None else 0.0
                                p_w = actual_model(joint_pos[w_i:w_i+1],
                                                   viscosity_norm=vis_w,
                                                   timesteps=batch_timesteps[w_i:w_i+1])
                                ccmn_preds.append(p_w)
                            predicted_actions = torch.cat(ccmn_preds, dim=0)
                        elif hasattr(actual_model, 'include_goal_direction') and actual_model.include_goal_direction:
                            predicted_actions = self.ncap_model(
                                joint_pos, environment_type=environment_type,
                                target_direction=target_direction, timesteps=batch_timesteps)
                        else:
                            predicted_actions = self.ncap_model(
                                joint_pos, environment_type=environment_type,
                                timesteps=batch_timesteps)

                        loss = torch.nn.functional.mse_loss(
                            predicted_actions, batch_actions, reduction='none')
                        policy_loss = (loss.mean(dim=1) * batch_returns[:len(loss)]).mean()

                        self.optimizer.zero_grad()
                        policy_loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.ncap_model.parameters(), 0.5)
                        has_bad_grad = any(
                            p.grad is not None and
                            (torch.isnan(p.grad).any() or torch.isinf(p.grad).any())
                            for p in self.ncap_model.parameters()
                        )
                        if has_bad_grad:
                            self.optimizer.zero_grad()
                            continue
                        self.optimizer.step()
                        with torch.no_grad():
                            for param in self.ncap_model.parameters():
                                if torch.isnan(param).any() or torch.isinf(param).any():
                                    param.data = torch.nan_to_num(
                                        param.data, nan=0.0, posinf=1.0, neginf=-1.0)

                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    print(f"⚠️ Training step failed: {e}")
            
            def _reset_buffer(self, worker_idx):
                """Reset specific worker's episode buffer."""
                self.episode_buffers[worker_idx] = {'obs': [], 'actions': [], 'rewards': [], 'timesteps': []}
                self.step_counts[worker_idx] = 0
            
            def _get_model_action(self, obs):
                """Get action from NCAP model for a single observation (e.g. during evaluation)."""
                # CRITICAL: Always use the underlying module for single samples to avoid DataParallel scattering errors
                actual_model = self.ncap_model.module if isinstance(self.ncap_model, torch.nn.DataParallel) else self.ncap_model
                device = next(actual_model.parameters()).device
                
                if not isinstance(obs, torch.Tensor):
                    obs = torch.tensor(obs, dtype=torch.float32, device=device)
                elif obs.device != device:
                    obs = obs.to(device)
                
                # Single observation handling
                if obs.dim() == 0:
                    return torch.zeros(self.n_joints, device=device)
                
                # Use dynamic offsets
                joint_pos = obs[:self.n_joints]
                
                # Environment adaptation [water, land, viscosity_norm]
                environment_type = None
                if len(obs) >= self.env_features_start + 3:
                    water_flag = obs[self.env_features_start + 1]
                    land_flag = obs[self.env_features_start + 2]
                    vis_norm = obs[self.env_features_start]
                    environment_type = torch.tensor([water_flag, land_flag, vis_norm], dtype=torch.float32, device=device)
                
                # Goal direction
                target_direction = None
                if len(obs) >= self.goal_features_start + 3 and hasattr(actual_model, 'include_goal_direction') and actual_model.include_goal_direction:
                    target_direction = obs[self.goal_features_start + 1:self.goal_features_start + 3]
                
                # Get action from the ALREADY UNWRAPPED model
                with torch.no_grad():
                    # We MUST increment evaluation step count separately
                    t = torch.tensor([self.step_count], dtype=torch.float32, device=device)
                    if self.is_simple_ncap:
                        # SimpleNCAPSwimmer only needs joint_pos + timesteps
                        action = actual_model(joint_pos, timesteps=t)
                    elif self.is_ccmn:
                        # CCMNSwimmer takes joint_pos + viscosity_norm (scalar float)
                        vis_scalar = float(obs[self.env_features_start].item()) if len(obs) >= self.env_features_start + 1 else 0.0
                        action = actual_model(
                            joint_pos,
                            viscosity_norm=vis_scalar,
                            timesteps=t
                        )
                    elif hasattr(actual_model, 'include_goal_direction') and actual_model.include_goal_direction:
                        action = actual_model(
                            joint_pos, 
                            environment_type=environment_type,
                            target_direction=target_direction,
                            timesteps=t
                        )
                    else:
                        action = actual_model(
                            joint_pos, 
                            environment_type=environment_type,
                            timesteps=t
                        )
                    self.step_count += 1 # Increment for next call
                
                return action
            
            def test_step(self, obs):
                """Test step - returns action without training."""
                # Get device from model parameters
                device = next(self.ncap_model.parameters()).device
                
                # NCAP processing
                with torch.no_grad():
                    # Handle batch observations from SubprocVecEnv
                    if obs.ndim == 2:
                        batch_size = obs.shape[0]

                        # ── Persistent GPU buffer ─────────────────────────────
                        # Avoid re-allocating a new tensor every step.
                        # On the first call (or when shape changes) we allocate
                        # a pinned-memory numpy array and a matching GPU tensor.
                        # Subsequent calls copy in-place via copy_() which is
                        # async and avoids the Python allocation overhead.
                        if (not hasattr(self, '_obs_gpu') or
                                self._obs_gpu.shape != obs.shape):
                            self._obs_np  = np.ascontiguousarray(obs, dtype=np.float32)
                            self._obs_gpu = torch.empty(
                                obs.shape, dtype=torch.float32,
                                device=device, pin_memory=False
                            )
                        np.copyto(self._obs_np, obs, casting='unsafe')
                        self._obs_gpu.copy_(
                            torch.from_numpy(self._obs_np), non_blocking=True
                        )
                        obs_gpu = self._obs_gpu

                        joint_pos = obs_gpu[:, :self.n_joints]

                        environment_type = None
                        target_direction = None
                        vis_norm         = None

                        # Use precomputed offsets for batch extraction
                        if obs.shape[1] >= self.env_features_start + 3:
                            vis_norm        = obs_gpu[:, self.env_features_start:self.env_features_start+1]
                            water_flag      = obs_gpu[:, self.env_features_start+1:self.env_features_start+2]
                            land_flag       = obs_gpu[:, self.env_features_start+2:self.env_features_start+3]
                            environment_type = torch.cat([water_flag, land_flag, vis_norm], dim=1)

                        if obs.shape[1] >= self.goal_features_start + 3:
                            target_direction = obs_gpu[:, self.goal_features_start+1:self.goal_features_start+3]

                        timesteps = torch.as_tensor(
                            self.step_counts, dtype=torch.float32, device=device
                        )
                        
                        # Forward pass - self.ncap_model might be DataParallel
                        if self.is_simple_ncap:
                            # SimpleNCAPSwimmer: joint_pos + timesteps only, no env features
                            actual_sn = (self.ncap_model.module
                                        if isinstance(self.ncap_model, torch.nn.DataParallel)
                                        else self.ncap_model)
                            sn_actions = []
                            for w_i in range(batch_size):
                                t_w = timesteps[w_i:w_i+1]
                                a_w = actual_sn(joint_pos[w_i:w_i+1], timesteps=t_w)
                                sn_actions.append(a_w)
                            action = torch.cat(sn_actions, dim=0)
                        elif self.is_ccmn:
                            # CCMNSwimmer has a single shared GRU state, so we run
                            # each worker observation independently and stack results.
                            # This keeps the GRU trajectory coherent and avoids any
                            # batch-vs-hidden-state shape conflicts.
                            ccmn_actions = []
                            actual_ccmn = self.ncap_model.module if isinstance(self.ncap_model, torch.nn.DataParallel) else self.ncap_model
                            for w_i in range(batch_size):
                                vis_w = float(vis_norm[w_i, 0].item()) if vis_norm is not None else 0.0
                                t_w = timesteps[w_i:w_i+1]  # (1,) — forward will unsqueeze to (1,1)
                                a_w = actual_ccmn(
                                    joint_pos[w_i:w_i+1],   # (1, n_joints)
                                    viscosity_norm=vis_w,
                                    timesteps=t_w,
                                    worker_id=w_i,           # per-worker GRU state
                                )  # (1, n_joints)
                                ccmn_actions.append(a_w)
                            action = torch.cat(ccmn_actions, dim=0)  # (B, n_joints)
                        elif hasattr(self.ncap_model, 'module') and hasattr(self.ncap_model.module, 'include_goal_direction') and self.ncap_model.module.include_goal_direction:
                            action = self.ncap_model(
                                joint_pos, 
                                environment_type=environment_type,
                                target_direction=target_direction,
                                timesteps=timesteps
                            )
                        elif not hasattr(self.ncap_model, 'module') and hasattr(self.ncap_model, 'include_goal_direction') and self.ncap_model.include_goal_direction:
                            action = self.ncap_model(
                                joint_pos, 
                                environment_type=environment_type,
                                target_direction=target_direction,
                                timesteps=timesteps
                            )
                        else:
                            action = self.ncap_model(
                                joint_pos, 
                                environment_type=environment_type,
                                timesteps=timesteps
                            )
                        self.step_counts += 1
                        return action.cpu().numpy()
                    else:
                        # Single observation (fallback to _get_model_action which handles unwrap)
                        return self._get_model_action(obs).cpu().numpy()
        
        agent = BiologicalNCAPAgent(model, env, self.n_links)
        agent._use_sequential_training = (self.model_type in ('ccmn', 'ccmn_hrl') and self.ccmn_sequential_training)
        agent._use_bistability_reg     = (self.model_type in ('ccmn', 'ccmn_hrl') and self.bistability_regulariser)
        agent._bistab_lambda           = self.bistability_lambda
        # Algorithm selection — governs _train_on_episode and step() noise
        agent._algorithm               = self.algorithm
        agent._es_lr                   = self.learning_rate
        # Exploration noise added to *stored* targets (not env actions) for
        # PPO/DDPG so deterministic models like SimpleNCAPSwimmer have a
        # non-zero loss gradient.  0.1 is conservative; raise to 0.2 if
        # weights still fail to update after 500k steps.
        agent._exploration_noise       = 0.1
        if self.model_type in ('ccmn', 'ccmn_hrl') and self.film_ablation:
            import math
            actual_m = model.module if isinstance(model, torch.nn.DataParallel) else model
            with torch.no_grad():
                for p in actual_m.film_generator.parameters(): p.requires_grad_(False)
                _spi = math.log(math.exp(1.0) - 1.0)
                torch.nn.init.constant_(actual_m.film_generator.gamma_head.bias, _spi)
                torch.nn.init.constant_(actual_m.film_generator.gamma_head.weight, 0.0)
                torch.nn.init.constant_(actual_m.film_generator.beta_head.bias,  0.0)
                torch.nn.init.constant_(actual_m.film_generator.beta_head.weight, 0.0)
            print('🔬 FiLM ABLATION: γ=1, β=0 frozen')
        
        print(f"🧬 Created biological NCAP agent with environment adaptation for curriculum learning")
        
        return agent, model
    
    def _flush_reward_log(self) -> None:
        """Append buffered per-step reward entries to the CSV log and clear
        the buffer.  The CSV has three columns: step, mean_reward, phase.
        Appending (mode='a') means the file grows incrementally across the
        run and survives a crash — no data is lost from previous flushes.
        """
        if not self._reward_log_buffer:
            return

        if self._reward_csv_path is None:
            csv_dir = os.path.join(self.output_dir, 'curriculum_training', 'plots')
            os.makedirs(csv_dir, exist_ok=True)
            self._reward_csv_path = os.path.join(
                csv_dir,
                f'{self.artifact_namer.base_id}_step_rewards.csv'
            )
            # Write header only when the file does not yet exist
            if not os.path.exists(self._reward_csv_path):
                with open(self._reward_csv_path, 'w', newline='') as f:
                    import csv as _csv
                    _csv.writer(f).writerow(['step', 'mean_reward', 'phase'])

        with open(self._reward_csv_path, 'a', newline='') as f:
            import csv as _csv
            writer = _csv.writer(f)
            writer.writerows(self._reward_log_buffer)

        self._reward_log_buffer.clear()

    def _log_eval_to_csv(self, eval_results: dict, step: int) -> None:
        """
        Append one row per checkpoint to <artifact_id>_eval_returns.csv.

        Each row contains the training step and, for each of the four phases,
        the mean and std of the *summed* episode return (not per-step reward).
        This matches what the NCAP paper plots: total episode return on the
        y-axis, training timesteps on the x-axis.

        Columns:
          step,
          ph0_mean, ph0_std,
          ph1_mean, ph1_std,
          ph2_mean, ph2_std,
          ph3_mean, ph3_std
        """
        import csv as _csv

        csv_dir = os.path.join(self.output_dir, 'curriculum_training', 'plots')
        os.makedirs(csv_dir, exist_ok=True)
        path = os.path.join(csv_dir,
                            f'{self.artifact_namer.base_id}_eval_returns.csv')

        write_header = not os.path.exists(path)
        with open(path, 'a', newline='') as f:
            writer = _csv.writer(f)
            if write_header:
                writer.writerow([
                    'step',
                    'ph0_mean', 'ph0_std',
                    'ph1_mean', 'ph1_std',
                    'ph2_mean', 'ph2_std',
                    'ph3_mean', 'ph3_std',
                ])
            writer.writerow([
                step,
                eval_results.get(0, {}).get('mean_reward', 0.0),
                eval_results.get(0, {}).get('std_reward',  0.0),
                eval_results.get(1, {}).get('mean_reward', 0.0),
                eval_results.get(1, {}).get('std_reward',  0.0),
                eval_results.get(2, {}).get('mean_reward', 0.0),
                eval_results.get(2, {}).get('std_reward',  0.0),
                eval_results.get(3, {}).get('mean_reward', 0.0),
                eval_results.get(3, {}).get('std_reward',  0.0),
            ])
        tqdm.write(f'   📈 Eval returns logged → {path}')

    def save_checkpoint(self, model, step, eval_results=None):
        """Save training checkpoint with model-specific naming."""
        checkpoint_path = self.artifact_namer.checkpoint_name(
            step=step, 
            base_dir=os.path.join(self.output_dir, "curriculum_training/checkpoints")
        )
        
        checkpoint_data = {
            'model_state_dict': model.state_dict(),
            'current_step': self.current_step,
            'current_episode': self.current_episode,
            'phase_rewards': self.phase_rewards,
            'phase_distances': self.phase_distances,
            'training_config': {
                'n_links': self.n_links,
                'learning_rate': self.learning_rate,
                'training_steps': self.training_steps,
                'oscillator_period': self.oscillator_period,
                'min_oscillator_strength': self.min_oscillator_strength,
                'min_coupling_strength': self.min_coupling_strength,
                'biological_constraint_frequency': self.biological_constraint_frequency,
                'expose_environment_observation': self.expose_environment_observation,
                'expose_viscosity_observation': self.expose_viscosity_observation,
                'anisotropic_drag_mode': self.anisotropic_drag_mode,
                'anisotropic_drag_ratio': self.anisotropic_drag_ratio,
                'anisotropic_drag_gain': self.anisotropic_drag_gain,
                'anisotropic_drag_land_only': self.anisotropic_drag_land_only,
            },
            'eval_results': eval_results,
        }
        
        torch.save(checkpoint_data, checkpoint_path)
        print(f"💾 Checkpoint saved: {checkpoint_path}")
        return checkpoint_path
    
    def load_checkpoint(self, model, checkpoint_path):
        """Load training checkpoint with backward compatibility."""
        print(f"📂 Loading checkpoint: {checkpoint_path}")
        
        checkpoint_data = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # Load model state with compatibility for missing parameters
        try:
            model.load_state_dict(checkpoint_data['model_state_dict'])
        except RuntimeError as e:
            if "Missing key(s)" in str(e):
                print(f"⚠️ Checkpoint compatibility issue: {e}")
                print("🔧 Attempting to load compatible parameters only...")
                
                # Load only the parameters that exist in both model and checkpoint
                model_state = model.state_dict()
                checkpoint_state = checkpoint_data['model_state_dict']
                
                # Filter out missing parameters and load the rest
                compatible_state = {}
                missing_params = []
                extra_params = []
                
                for key, value in checkpoint_state.items():
                    if key in model_state:
                        compatible_state[key] = value
                    else:
                        extra_params.append(key)
                
                for key in model_state.keys():
                    if key not in checkpoint_state:
                        missing_params.append(key)
                
                # Load compatible parameters
                model.load_state_dict(compatible_state, strict=False)
                
                print(f"✅ Loaded {len(compatible_state)} compatible parameters")
                if missing_params:
                    print(f"⚠️ Missing parameters (will use defaults): {missing_params}")
                if extra_params:
                    print(f"ℹ️ Extra parameters in checkpoint (ignored): {extra_params}")
            else:
                raise
        
        # Load training state with backward compatibility
        if 'current_step' in checkpoint_data:
            # New checkpoint format
            self.current_step = checkpoint_data['current_step']
            self.current_episode = checkpoint_data['current_episode']
            self.phase_rewards = checkpoint_data.get('phase_rewards', {0: [], 1: [], 2: [], 3: []})
            self.phase_distances = checkpoint_data.get('phase_distances', {0: [], 1: [], 2: [], 3: []})
        else:
            # Old checkpoint format (legacy compatibility)
            self.current_step = checkpoint_data.get('step', 0)
            self.current_episode = checkpoint_data.get('episode', 0)
            self.phase_rewards = {0: [], 1: [], 2: [], 3: []}  # Reset for old checkpoints
            self.phase_distances = {0: [], 1: [], 2: [], 3: []}
            print("⚠️ Legacy checkpoint format detected - phase history reset")
        
        print(f"✅ Checkpoint loaded successfully!")
        print(f"   Resuming from step: {self.current_step:,}")
        print(f"   Episode: {self.current_episode:,}")
        
        return checkpoint_data.get('eval_results', {})
    
    def apply_biological_constraints(self, model):
        """Apply biological constraints to maintain realism."""
        # simple_ncap is trained with use_weight_constraints=False and random
        # init so its weights can move freely under gradient descent.  Applying
        # hard clamps here would undo every optimizer step and prevent learning.
        if self.model_type == 'simple_ncap':
            return False

        constraints_applied = []
        
        # CCMNSwimmer uses the same weight-sharing params dict; skip silently if
        # any key is absent (e.g. when weight sharing is disabled).
        required_keys = {'bneuron_osc', 'bneuron_prop', 'muscle_ipsi', 'muscle_contra'}
        if not required_keys.issubset(model.params.keys()):
            return False
        
        with torch.no_grad():
            # Ensure oscillator strength minimum
            if model.params['bneuron_osc'].item() < self.min_oscillator_strength:
                old_val = model.params['bneuron_osc'].item()
                model.params['bneuron_osc'].data.fill_(self.min_oscillator_strength)
                constraints_applied.append(f"oscillator {old_val:.3f} → {self.min_oscillator_strength}")
            
            # Ensure coupling strength minimum
            if model.params['bneuron_prop'].item() < self.min_coupling_strength:
                old_val = model.params['bneuron_prop'].item()
                model.params['bneuron_prop'].data.fill_(self.min_coupling_strength)
                constraints_applied.append(f"coupling {old_val:.3f} → {self.min_coupling_strength}")
            
            # **RELAXED**: Ensure ipsilateral muscle is positive (less restrictive)
            if model.params['muscle_ipsi'].item() < 0.5:  # **REDUCED** from 0.8 to 0.5
                old_val = model.params['muscle_ipsi'].item()
                model.params['muscle_ipsi'].data.fill_(0.5)
                constraints_applied.append(f"ipsi {old_val:.3f} → 0.5")
            
            # **RELAXED**: Ensure contralateral muscle is negative (less restrictive)
            if model.params['muscle_contra'].item() > -0.5:  # **REDUCED** from -0.8 to -0.5
                old_val = model.params['muscle_contra'].item()
                model.params['muscle_contra'].data.fill_(-0.5)
                constraints_applied.append(f"contra {old_val:.3f} → -0.5")
        
        if constraints_applied:
            print(f"🧬 Applied biological constraints: {', '.join(constraints_applied)}")
        
        return len(constraints_applied) > 0
    
    def get_current_phase(self, progress):
        if progress < 0.3: return 0
        elif progress < 0.6: return 1
        elif progress < 0.8: return 2
        else: return 3

    def _ccmn_phase1_should_use_land(self) -> bool:
        if self.model_type != 'ccmn' or self.ccmn_phase1_land_fraction <= 0: return False
        period = max(1, int(round(1.0 / self.ccmn_phase1_land_fraction)))
        return (self.current_step // self.num_workers) % period == 0
    
    def evaluate_performance(self, agent, env_ignored, num_episodes=5, progress_bar=None):
        """Evaluate current performance across different phases."""
        evaluation_results = {}
        
        for phase in range(4):
            # swim_only: only evaluate Phase 0 — skip land-zone phases
            if self.swim_only and phase > 0:
                continue
            # Create temporary environment for this phase
            temp_progress = (phase + 0.5) * 0.25  # Middle of each phase
            
            # Get phase-specific episode duration from configuration
            steps_per_episode = self.PHASE_DURATION_CONFIG['evaluation_steps'][phase]
            phase_names = ["Pure Swimming", "Single Land Zone", "Two Land Zones", "Full Complexity"]
            
            if steps_per_episode != 200:  # Log when using non-standard duration
                print(f"🎯 {phase_names[phase]}: Using {steps_per_episode} steps per episode")
            
            distances = []
            rewards = []
            
            for episode in range(num_episodes):
                # Set environment to specific phase
                from ..environments.progressive_mixed_env import TonicProgressiveMixedWrapper
                eval_env = self._EnvWrapper(n_links=self.n_links)
                eval_env.env.training_progress = temp_progress
                eval_env.env._create_environment()
                
                obs = eval_env.reset()
                episode_reward = 0
                initial_pos = eval_env.head_position
                
                for _ in range(steps_per_episode):  # Data-driven steps per episode
                    action = agent.test_step(obs)
                    obs, reward, done, _ = eval_env.step(action)
                    episode_reward += reward
                    
                    if done:
                        break
                
                final_pos = eval_env.head_position
                distance = np.linalg.norm(final_pos - initial_pos)
                
                distances.append(distance)
                rewards.append(episode_reward)
                
                # Update progress bar if provided
                if progress_bar is not None:
                    phase_names = ["Pure Swimming", "Single Land Zone", "Two Land Zones", "Full Complexity"]
                    progress_bar.set_description(f"🔬 Evaluating {phase_names[phase]} ({episode+1}/{num_episodes})")
                    progress_bar.update(1)
                eval_env.close()
            
            evaluation_results[phase] = {
                'mean_distance': np.mean(distances),
                'mean_reward': np.mean(rewards),
                'std_distance': np.std(distances),
                'std_reward': np.std(rewards)
            }
        
        return evaluation_results
    
    def train(self):
        """Run curriculum training for 1M episodes."""
        print(f"\n🎓 Starting Curriculum NCAP Training...")
        print(f"   Target: {self.training_steps:,} steps")
        print(f"   Biological constraints every {self.biological_constraint_frequency:,} steps")
        
        # Create vectorized environment
        env = self.create_vectorized_environment()
        model = self.create_model()
        agent, tonic_model = self.create_agent(model, env)
        
        # Load checkpoint if resuming
        if self.resume_from_checkpoint:
            self.load_checkpoint(model, self.resume_from_checkpoint)
        
        if self.model_type == 'ccmn' and self.run_sanity_check:
            print('\n🔬 Running CCMN sanity-check...')
            sanity_env = self.create_environment()
            for sc_label, sc_progress, sc_land in [('water-only', 0.05, False), ('land-present', 0.45, True)]:
                sanity_env.env.set_manual_progress(sc_progress, force_land_start=sc_land)
                sanity_log = run_ccmn_sanity_check(agent=agent, env=sanity_env, num_steps=self.sanity_check_steps, label=sc_label)
                diag_dir = os.path.join(self.output_dir, 'curriculum_training/plots/ccmn_diagnostics', f'{self.artifact_namer.base_id}_sanity_{sc_label.replace(" ","_")}')
                plot_ccmn_diagnostic_summary(sanity_log, save_dir=diag_dir)
                print(f'   ✅ ({sc_label}): z_DA range=[{sanity_log["z_DA"].min():.3f}, {sanity_log["z_DA"].max():.3f}], γ_mean={sanity_log["gamma"].mean():.3f}')
            sanity_env.close()
            print('🔬 Sanity check complete.\n')

        # Training loop with advanced monitoring
        start_time = time.time()
        self.logger.start_time = start_time
        last_phase = -1
        
        # Start hardware monitoring if available
        if ADVANCED_LOGGING_AVAILABLE:
            self.logger.start_monitoring()
        
        # Initialize progress bars
        main_pbar = tqdm(
            total=self.training_steps,
            desc="🎓 Curriculum Training",
            unit="steps",
            unit_scale=True,
            position=0,
            leave=True,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        )
        
        # Phase progress tracking
        phase_names = ["🏊 Pure Swimming", "🏝️ Single Land Zone", "🏝️🏝️ Two Land Zones", "🌍 Full Complexity"]
        
        # Reset vectorized environment
        obs = env.reset()
        episode_rewards = np.zeros(self.num_workers)
        episode_distances = np.zeros(self.num_workers)
        initial_positions = np.array(env.get_attr('head_position')).copy()
        
        while self.current_step < self.training_steps:
            # Get current training progress
            # swim_only pins to Phase 0 (pure swimming) for the full run,
            # reproducing the original NCAP paper single-environment baseline.
            progress = 0.0 if self.swim_only else self.current_step / self.training_steps
            current_phase = self.get_current_phase(progress)
            
            # Update environment progress across all workers (Reduced frequency to optimize speed)
            if self.current_step % (self.num_workers * 100) < self.num_workers:
                env.set_attr('env.training_progress', progress)
            if current_phase == 0 and self._ccmn_phase1_should_use_land():
                env.set_attr('env.training_progress', 0.32)
                env.set_attr('env.training_progress', progress)
            
            # Check for phase transitions
            if current_phase != last_phase:
                main_pbar.set_description(f"🎓 Curriculum Training - {phase_names[current_phase]}")
                tqdm.write(f"\n🎓 PHASE TRANSITION: {last_phase} → {current_phase}")
                last_phase = current_phase
            
            # Apply biological constraints periodically
            if self.current_step % self.biological_constraint_frequency < self.num_workers:
                actual_model = model.module if isinstance(model, torch.nn.DataParallel) else model
                self.apply_biological_constraints(actual_model)
            
            # Step all environments in parallel
            actions = agent.step(obs)
            next_obs, rewards, dones, infos = env.step(actions)
            rewards = np.asarray(rewards, dtype=np.float32)
            rewards = np.nan_to_num(rewards, nan=0.0, posinf=0.0, neginf=0.0)
            dones = np.asarray(dones, dtype=bool)
            agent.add_rewards(rewards)

            # ── Per-step reward logging (flush every 10k steps) ───────────────
            self._reward_log_buffer.append((
                self.current_step,
                float(np.mean(rewards)),
                current_phase,
            ))
            if self.current_step - self._last_reward_flush >= self._REWARD_LOG_INTERVAL:
                self._flush_reward_log()
                self._last_reward_flush = self.current_step

            episode_rewards += rewards
            self.current_step += self.num_workers
            
            # Update progress bar periodically to reduce overhead
            if self.current_step % (self.num_workers * 10) < self.num_workers:
                main_pbar.update(self.num_workers * 10)
            
            # Handle episode completions
            if np.any(dones):
                done_indices = np.where(dones)[0]
                
                # Calculate distances for completed episodes
                current_head_positions = np.array(env.get_attr('head_position')).copy()
                
                for idx in done_indices:
                    episode_distance = np.linalg.norm(current_head_positions[idx] - initial_positions[idx])
                    episode_reward_value = float(np.nan_to_num(episode_rewards[idx], nan=0.0, posinf=0.0, neginf=0.0))
                    episode_distance_value = float(np.nan_to_num(episode_distance, nan=0.0, posinf=0.0, neginf=0.0))
                    
                    self.current_episode += 1
                    self.phase_rewards[current_phase].append(episode_reward_value)
                    self.phase_distances[current_phase].append(episode_distance_value)
                    
                    # Log to file periodically
                    if self.current_episode % self.log_episodes == 0:
                        self.logger.log_training_step({
                            'step': self.current_step,
                            'episode': self.current_episode,
                            'phase': current_phase,
                            'reward': episode_reward_value,
                            'distance': episode_distance_value,
                        })
                    
                    # Reset worker state
                    episode_rewards[idx] = 0
                    episode_distances[idx] = 0
                    initial_positions[idx] = current_head_positions[idx]
                
                # Train on completed episodes
                agent.end_episodes(done_indices)
            
            obs = next_obs
            
            # Periodic logging
            if self.current_episode % self.log_episodes == 0 and np.any(dones):
                elapsed_time = time.time() - start_time
                steps_per_sec = self.current_step / elapsed_time if elapsed_time > 0 else 0
                
                recent_reward = np.mean(self.phase_rewards[current_phase][-10:]) if self.phase_rewards[current_phase] else 0
                recent_distance = np.mean(self.phase_distances[current_phase][-10:]) if self.phase_distances[current_phase] else 0
                
                main_pbar.set_postfix({
                    'Phase': current_phase,
                    'Episode': f"{self.current_episode:,}",
                    'Reward': f"{recent_reward:.1f}",
                    'Distance': f"{recent_distance:.3f}m",
                    'Steps/s': f"{steps_per_sec:.1f}"
                })
                
                # Calculate ETA if advanced logging is available
                eta_str = ""
                if ADVANCED_LOGGING_AVAILABLE:
                    eta = self.logger.calculate_eta(self.current_step, self.training_steps)
                    eta_str = f" | ETA: {eta}"
                
                # Detailed logging (less frequent to avoid clutter)
                if self.current_episode % (self.log_episodes * 4) == 0:  # Every 200 episodes instead of 50
                    tqdm.write(f"[{self.current_step:7d}/{self.training_steps:7d}] "
                              f"Phase {current_phase} | "
                              f"Episode {self.current_episode:6d} | "
                              f"Reward: {recent_reward:6.2f} | "
                              f"Distance: {recent_distance:6.3f}m | "
                              f"Steps/s: {steps_per_sec:.1f}{eta_str}")
                
                # Log to file
                self.logger.log_training_step({
                    'step': self.current_step,
                    'episode': self.current_episode,
                    'phase': current_phase,
                    'progress': progress,
                    'reward': recent_reward,
                    'distance': recent_distance,
                    'mean_reward_10': recent_reward,   # recent_reward is already a mean of 10
                    'mean_distance_10': recent_distance,
                })
            
            # Periodic saves and evaluation
            if self.current_step % self.save_steps == 0:
                tqdm.write(f"\n💾 Checkpoint at step {self.current_step:,}")
                
                # Comprehensive evaluation first
                eval_results = self.evaluate_performance(agent, env, num_episodes=10)

                # ── Log eval episode returns to CSV ───────────────────────────
                # Each row = one checkpoint. Columns: step + per-phase mean and
                # std of summed episode return (not per-step reward).
                # This is what the NCAP paper plots on the y-axis.
                self._log_eval_to_csv(eval_results, self.current_step)
                
                # Save comprehensive checkpoint with eval results
                checkpoint_path = self.save_checkpoint(tonic_model, self.current_step, eval_results)
                tqdm.write(f"📊 Performance across all phases:")
                for phase, results in eval_results.items():
                    tqdm.write(f"   Phase {phase}: {results['mean_distance']:.3f}m ± {results['std_distance']:.3f} "
                              f"(reward: {results['mean_reward']:.2f})")
                
                # Advanced checkpoint logging
                if ADVANCED_LOGGING_AVAILABLE:
                    checkpoint_data = self.logger.log_checkpoint(
                        step=self.current_step,
                        model=tonic_model,
                        performance_metrics=eval_results
                    )
                    
                    # Show training dashboard
                    dashboard = self.logger.create_training_dashboard()
                    tqdm.write(dashboard)
                
                # Create visualizations
                if self.current_step >= 50000:  # After some training
                    plot_path = self.artifact_namer.analysis_plot_name(
                        "curriculum_progress", 
                        step=self.current_step,
                        base_dir=os.path.join(self.output_dir, "curriculum_training/plots")
                    )
                    create_curriculum_plots(
                        phase_rewards=self.phase_rewards,
                        phase_distances=self.phase_distances,
                        eval_results=eval_results,
                        save_path=plot_path
                    )
                
                # Create trajectory analysis
                current_phase = 0 if self.swim_only else min(int(self.current_step / (self.training_steps / 4)), 3)
                phase_names = ["Pure Swimming", "Single Land Zone", "Two Land Zones", "Full Complexity"]
                trajectory_path = self.artifact_namer.analysis_plot_name(
                    "trajectory_analysis", 
                    step=self.current_step,
                    phase=f"phase{current_phase}",
                    base_dir=os.path.join(self.output_dir, "curriculum_training/plots")
                )
                
                # Need a single environment for visualizations
                eval_env = self.create_environment()
                
                trajectory_stats = create_trajectory_analysis(
                    agent=agent,
                    env=eval_env,
                    save_path=trajectory_path,
                    num_steps=500,
                    phase_name=f"Step {self.current_step} - {phase_names[current_phase]}",
                    trajectory_multiplier=self.PHASE_DURATION_CONFIG['trajectory_multiplier'][current_phase]
                )
                
                tqdm.write(f"📊 Trajectory stats: distance={trajectory_stats['final_distance']:.3f}m, "
                          f"transitions={trajectory_stats['transitions']}, "
                          f"mean water bout={trajectory_stats.get('mean_water_bout', 0.0):.1f} steps")

                # 2-D body profile figure
                try:
                    body_profile_path = trajectory_path.replace(
                        'trajectory_analysis', 'body_profile')
                    create_body_profile_figure(
                        agent=agent,
                        env=eval_env,
                        save_path=body_profile_path,
                        num_steps=400,
                        phase_name=f"Step {self.current_step} — {phase_names[current_phase]}"
                    )
                except Exception as _bp_e:
                    tqdm.write(f"⚠️  Body profile skipped: {_bp_e}")

                # Create test video
                video_path = self.artifact_namer.training_video_name(
                    step=self.current_step,
                    phase=f"phase{current_phase}",
                    base_dir=os.path.join(self.output_dir, "curriculum_training/videos")
                )
                create_test_video(
                    agent=agent,
                    env=eval_env,
                    save_path=video_path,
                    num_steps=300,
                    episode_name=f"Curriculum Step {self.current_step}"
                )
                eval_env.close()

                # CCMN neuromodulatory analysis (only for ccmn model type)
                if self.model_type == 'ccmn_hrl':
                    try:
                        tqdm.write(f"\n🧠 Generating CCMN neuromodulatory analysis...")
                        ccmn_env = self.create_environment()
                        # Set progress to Phase 2 (single land zone) WITHOUT
                        # force_land_start so the episode begins in water and
                        # naturally encounters land — this ensures figH
                        # (speed_zDA_environment) shows genuine water AND land
                        # segments rather than all-land.
                        ccmn_env.env.set_manual_progress(0.45, force_land_start=False)
                        ccmn_plot_dir = self.artifact_namer.analysis_plot_name(
                            "ccmn_neuromod",
                            step=self.current_step,
                            base_dir=os.path.join(self.output_dir, "curriculum_training/plots/ccmn_neuromod")
                        ).replace(".png", "")   # use as directory prefix
                        create_ccmn_neuromodulatory_analysis(
                            agent=agent,
                            env=ccmn_env,
                            base_dir=ccmn_plot_dir,
                            num_steps=1200,
                            name_prefix=f"step{self.current_step}"
                        )
                        ccmn_env.close()
                        tqdm.write(f"✅ CCMN neuromod plots saved to: {ccmn_plot_dir}")
                    except Exception as _ccmn_e:
                        tqdm.write(f"⚠️  CCMN neuromod analysis skipped: {_ccmn_e}")
        
        # Close progress bar
        main_pbar.close()

        # Flush any remaining per-step reward entries not yet written to CSV
        self._flush_reward_log()

        # Final evaluation and save
        tqdm.write(f"\n🏁 Training Complete!")
        total_time_hours = (time.time() - start_time) / 3600
        tqdm.write(f"   Total time: {total_time_hours:.2f} hours")
        
        # Stop hardware monitoring with indicator
        if ADVANCED_LOGGING_AVAILABLE:
            self.logger.stop_monitoring()  # Advanced logger handles its own progress messages
        else:
            tqdm.write(f"🖥️ Hardware monitoring stopped")
        
        # Generate final performance summary from training data
        tqdm.write(f"\n📊 Generating final performance summary from training data...")
        
        # Convert training data to evaluation format for model saving and artifacts
        final_eval = {}
        for phase in range(4):
            if phase in self.phase_rewards and len(self.phase_rewards[phase]) > 0:
                # Use actual training data from this phase
                phase_rewards = self.phase_rewards[phase]
                phase_distances = self.phase_distances[phase]
                
                final_eval[phase] = {
                    'mean_distance': np.mean(phase_distances),
                    'std_distance': np.std(phase_distances) if len(phase_distances) > 1 else 0.0,
                    'mean_reward': np.mean(phase_rewards),
                    'std_reward': np.std(phase_rewards) if len(phase_rewards) > 1 else 0.0
                }
            else:
                # Fallback for phases not trained yet (shouldn't happen in normal training)
                final_eval[phase] = {
                    'mean_distance': 0.0,
                    'std_distance': 0.0,
                    'mean_reward': 0.0,
                    'std_reward': 0.0
                }
        
        total_episodes = sum(len(self.phase_rewards[p]) for p in range(4) if p in self.phase_rewards)
        active_phases = len([p for p in range(4) if p in self.phase_rewards and len(self.phase_rewards[p]) > 0])
        tqdm.write(f"✅ Training performance summary: {total_episodes} episodes across {active_phases} phases")
        
        tqdm.write(f"\n📊 Final Performance Summary:")
        for phase, results in final_eval.items():
            phase_names_final = ["Pure Swimming", "Single Land Zone", "Two Land Zones", "Full Complexity"]
            tqdm.write(f"   {phase_names_final[phase]}: {results['mean_distance']:.3f}m ± {results['std_distance']:.3f}")
        
        # Save final model
        final_path = self.artifact_namer.final_model_name(
            base_dir=os.path.join(self.output_dir, "curriculum_training/models")
        )
        torch.save({
            'model_state_dict': model.state_dict(),
            'final_evaluation': final_eval,
            'training_history': {
                'phase_rewards': self.phase_rewards,
                'phase_distances': self.phase_distances,
            }
        }, final_path)
        
        tqdm.write(f"💾 Final model saved to: {final_path}")
        
        # Create final visualizations
        tqdm.write(f"\n🎨 Creating final training visualizations...")
        
        # Create progress bar for final visualizations
        final_tasks = [
            "Creating final training plots",
            "Trajectory analysis: Pure Swimming", 
            "Trajectory analysis: Single Land Zone",
            "Trajectory analysis: Two Land Zones", 
            "Trajectory analysis: Full Complexity",
            "Creating phase comparison video",
            "Generating training summary",
            "Creating comprehensive report",
            "CCMN neuromodulatory analysis"
        ]
        
        with tqdm(total=len(final_tasks), desc="🎬 Final Analysis", unit="task", 
                 bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:
            
            # Final training plots
            pbar.set_description("📊 Creating training plots")
            final_plot_path = self.artifact_namer.analysis_plot_name(
                "curriculum_final", 
                base_dir=os.path.join(self.output_dir, "curriculum_training/plots")
            )
            create_curriculum_plots(
                phase_rewards=self.phase_rewards,
                phase_distances=self.phase_distances,
                eval_results=final_eval,
                save_path=final_plot_path
            )
            pbar.update(1)
            tqdm.write(f"✅ Training plots saved to: {final_plot_path}")
            
            # Final trajectory analysis for each phase
            phase_names = ["Pure Swimming", "Single Land Zone", "Two Land Zones", "Full Complexity"]
            final_trajectory_stats = {}
            
            for phase in range(4):
                pbar.set_description(f"📊 Analyzing {phase_names[phase]}")
                
                # Set environment to specific phase using manual override
                eval_env = self.create_environment()
                temp_progress = (phase + 0.5) * 0.25  # Middle of each phase
                force_land_for_evaluation = phase >= 1 and not getattr(eval_env.env, 'prefer_transition_evaluation', False)
                eval_env.env.set_manual_progress(temp_progress, force_land_start=force_land_for_evaluation)
                
                trajectory_path = self.artifact_namer.analysis_plot_name(
                    "final_trajectory", 
                    phase=f"phase{phase}",
                    base_dir=os.path.join(self.output_dir, "curriculum_training/plots")
                )
                stats = create_trajectory_analysis(
                    agent=agent,
                    env=eval_env,
                    save_path=trajectory_path,
                    num_steps=1000,  # Longer analysis for final evaluation
                    phase_name=f"Final - {phase_names[phase]}",
                    trajectory_multiplier=self.PHASE_DURATION_CONFIG['trajectory_multiplier'][phase]
                )
                
                final_trajectory_stats[phase] = stats
                pbar.update(1)
                tqdm.write(f"   ✅ {phase_names[phase]}: {stats['final_distance']:.3f}m, "
                          f"{stats['transitions']} transitions, "
                          f"mean water bout {stats.get('mean_water_bout', 0.0):.1f} steps")

                # 2-D body profile figure
                try:
                    body_profile_path = trajectory_path.replace(
                        'final_trajectory', 'final_body_profile')
                    create_body_profile_figure(
                        agent=agent,
                        env=eval_env,
                        save_path=body_profile_path,
                        num_steps=600,
                        phase_name=f"Final — {phase_names[phase]}"
                    )
                except Exception as _bp_e:
                    tqdm.write(f"⚠️  Body profile skipped: {_bp_e}")
            
            # Final test video with phase comparisons
            pbar.set_description("🎬 Creating phase comparison video")
            final_video_path = self.artifact_namer.evaluation_video_name(
                evaluation_type="phase_comparison_final",
                base_dir=os.path.join(self.output_dir, "curriculum_training/videos")
            )
            eval_env = self.create_environment()
            create_phase_comparison_video(
                agent=agent,
                env=eval_env,
                save_path=final_video_path,
                phases_to_test=[0, 1, 2, 3],
                phase_video_steps=self.PHASE_DURATION_CONFIG['video_steps']
            )
            eval_env.close()
            pbar.update(1)
            tqdm.write(f"✅ Phase comparison video: {final_video_path}")
            
            # Training summary
            pbar.set_description("📄 Generating training summary")
            summary_path = self.artifact_namer.experiment_summary_name(
                base_dir=os.path.join(self.output_dir, "curriculum_training/summaries")
            )
            save_training_summary(
                eval_results=final_eval,
                training_history={
                    'phase_rewards': self.phase_rewards,
                    'phase_distances': self.phase_distances,
                    'trajectory_stats': final_trajectory_stats,
                },
                save_path=summary_path
            )
            pbar.update(1)
            tqdm.write(f"✅ Training summary: {summary_path}")
            
            # Generate comprehensive report with advanced metrics
            if ADVANCED_LOGGING_AVAILABLE:
                pbar.set_description("📊 Creating comprehensive report")
                comprehensive_report = self.logger.save_comprehensive_report()
                pbar.update(1)
                tqdm.write(f"✅ Advanced training analysis complete")
            else:
                pbar.update(1)  # Skip if not available

            # CCMN final neuromodulatory analysis
            pbar.set_description("🧠 CCMN neuromodulatory analysis")
            if self.model_type == 'ccmn':
                try:
                    ccmn_final_env = self.create_environment()
                    # Use full-complexity phase for the final analysis
                    ccmn_final_env.env.set_manual_progress(
                        0.9, force_land_start=True
                    )
                    ccmn_final_dir = os.path.join(
                        os.path.join(self.output_dir, "curriculum_training/plots/ccmn_neuromod"),
                        f"{self.artifact_namer.base_id}_final"
                    )
                    create_ccmn_neuromodulatory_analysis(
                        agent=agent,
                        env=ccmn_final_env,
                        base_dir=ccmn_final_dir,
                        num_steps=1800,   # longer episode for richer statistics
                        name_prefix="final"
                    )
                    ccmn_final_env.close()
                    tqdm.write(f"✅ CCMN final neuromod analysis: {ccmn_final_dir}")
                except Exception as _ccmn_e:
                    tqdm.write(f"⚠️  CCMN final neuromod analysis skipped: {_ccmn_e}")
            pbar.update(1)
        
        env.close()
        return model, final_eval
    
    def evaluate_only(self, eval_episodes=20, video_steps=400):
        """Run evaluation and visualization only (no training) from a checkpoint."""
        print(f"\n📊 Starting Curriculum Evaluation (No Training)")
        print(f"   Checkpoint: {self.resume_from_checkpoint}")
        print(f"   Links: {self.n_links}")
        print(f"   Episodes per phase: {eval_episodes}")
        print(f"   Video length: {video_steps} steps")
        
        # Create environment and model based on model_type
        env = self.create_environment()
        model = self.create_model()
        agent, tonic_model = self.create_agent(model, env)
        
        # Load checkpoint
        if self.resume_from_checkpoint:
            checkpoint_results = self.load_checkpoint(tonic_model, self.resume_from_checkpoint)
        else:
            print("⛔ No checkpoint provided for evaluation!")
            return
        
        start_time = time.time()
        
        print(f"\n📊 Generating performance summary from checkpoint training data...")
        
        # Convert training data to evaluation format for visualization artifacts
        final_eval = {}
        for phase in range(4):
            if phase in self.phase_rewards and len(self.phase_rewards[phase]) > 0:
                # Use actual training data from this phase
                phase_rewards = self.phase_rewards[phase]
                phase_distances = self.phase_distances[phase]
                
                final_eval[phase] = {
                    'mean_distance': np.mean(phase_distances),
                    'std_distance': np.std(phase_distances) if len(phase_distances) > 1 else 0.0,
                    'mean_reward': np.mean(phase_rewards),
                    'std_reward': np.std(phase_rewards) if len(phase_rewards) > 1 else 0.0
                }
            else:
                # Fallback for phases not trained yet (shouldn't happen in normal training)
                final_eval[phase] = {
                    'mean_distance': 0.0,
                    'std_distance': 0.0,
                    'mean_reward': 0.0,
                    'std_reward': 0.0
                }
        
        total_episodes = sum(len(self.phase_rewards[p]) for p in range(4) if p in self.phase_rewards)
        active_phases = len([p for p in range(4) if p in self.phase_rewards and len(self.phase_rewards[p]) > 0])
        print(f"✅ Checkpoint performance summary: {total_episodes} episodes across {active_phases} phases")
        

        print(f"\n📊 Performance Summary:")
        phase_names_final = ["Pure Swimming", "Single Land Zone", "Two Land Zones", "Full Complexity"]
        for phase, results in final_eval.items():
            print(f"   {phase_names_final[phase]}: {results['mean_distance']:.3f}m ± {results['std_distance']:.3f}")
        
        print(f"\n🎨 Creating comprehensive visualizations...")
        with tqdm(total=8, desc="📊 Creating Visualizations", unit="task",
                 bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as vis_pbar:
            
            # Training plots
            vis_pbar.set_description("📊 Creating final training plots")
            eval_plot_path = self.artifact_namer.analysis_plot_name(
                "evaluation_final", 
                base_dir=os.path.join(self.output_dir, "curriculum_training/plots")
            )
            create_curriculum_plots(
                phase_rewards=self.phase_rewards,
                phase_distances=self.phase_distances,
                eval_results=final_eval,
                save_path=eval_plot_path
            )
            vis_pbar.update(1)
            print(f"✅ Training plots saved to: {eval_plot_path}")
            
            # Trajectory analysis for each phase
            phase_names = ["Pure Swimming", "Single Land Zone", "Two Land Zones", "Full Complexity"]
            final_trajectory_stats = {}
            
            for phase in range(4):
                vis_pbar.set_description(f"📊 Analyzing {phase_names[phase]}")
                
                # Set environment to specific phase
                eval_env = self.create_environment()
                temp_progress = (phase + 0.5) * 0.25  # Middle of each phase
                force_land_for_evaluation = phase >= 1 and not getattr(eval_env.env, 'prefer_transition_evaluation', False)
                eval_env.env.set_manual_progress(temp_progress, force_land_start=force_land_for_evaluation)
                
                eval_trajectory_path = self.artifact_namer.analysis_plot_name(
                    "evaluation_trajectory", 
                    phase=f"phase{phase}",
                    base_dir=os.path.join(self.output_dir, "curriculum_training/plots")
                )
                trajectory_stats = create_trajectory_analysis(
                    agent=agent,
                    env=eval_env,
                    save_path=eval_trajectory_path,
                    num_steps=video_steps,
                    phase_name=f"Evaluation - {phase_names[phase]}",
                    trajectory_multiplier=self.PHASE_DURATION_CONFIG['trajectory_multiplier'][phase]
                )
                
                final_trajectory_stats[phase] = trajectory_stats

                # 2-D body profile figure
                try:
                    body_profile_path = eval_trajectory_path.replace(
                        'evaluation_trajectory', 'evaluation_body_profile')
                    create_body_profile_figure(
                        agent=agent,
                        env=eval_env,
                        save_path=body_profile_path,
                        num_steps=600,
                        phase_name=f"Evaluation — {phase_names[phase]}"
                    )
                except Exception as _bp_e:
                    print(f"⚠️  Body profile skipped: {_bp_e}")

                eval_env.close()
                vis_pbar.update(1)
                print(f"   ✅ {phase_names[phase]}: {trajectory_stats['final_distance']:.3f}m, "
                      f"{trajectory_stats['transitions']} transitions, "
                      f"mean water bout {trajectory_stats.get('mean_water_bout', 0.0):.1f} steps")
            
            # Phase comparison video
            vis_pbar.set_description("🎬 Creating phase comparison video")
            eval_comparison_video_path = self.artifact_namer.evaluation_video_name(
                evaluation_type="phase_comparison",
                base_dir=os.path.join(self.output_dir, "curriculum_training/videos")
            )
            eval_env = self.create_environment()
            create_phase_comparison_video(
                agent=agent,
                env=eval_env,
                save_path=eval_comparison_video_path,
                phases_to_test=[0, 1, 2, 3],
                phase_video_steps=self.PHASE_DURATION_CONFIG['video_steps']
            )
            eval_env.close()
            vis_pbar.update(1)
            print(f"✅ Phase comparison video: {eval_comparison_video_path}")
            
            # Individual test videos for each phase
            for phase in range(4):
                vis_pbar.set_description(f"🎬 Creating {phase_names[phase]} video")
                
                eval_env = self.create_environment()
                temp_progress = (phase + 0.5) * 0.25
                force_land_for_evaluation = phase >= 1 and not getattr(eval_env.env, 'prefer_transition_evaluation', False)
                eval_env.env.set_manual_progress(temp_progress, force_land_start=force_land_for_evaluation)
                
                phase_video_path = self.artifact_namer.evaluation_video_name(
                    evaluation_type=f"phase{phase}_{phase_names[phase].lower().replace(' ', '_')}",
                    base_dir=os.path.join(self.output_dir, "curriculum_training/videos")
                )
                create_test_video(
                    agent=agent,
                    env=eval_env,
                    save_path=phase_video_path,
                    num_steps=video_steps,
                    episode_name=f"Evaluation - {phase_names[phase]}"
                )
                eval_env.close()
                print(f"   ✅ {phase_names[phase]} video: {phase_video_path}")
            vis_pbar.update(1)
            
            # Training summary
            vis_pbar.set_description("📄 Generating evaluation summary")
            eval_summary_path = self.artifact_namer.experiment_summary_name(
                base_dir=os.path.join(self.output_dir, "curriculum_training/summaries")
            ).replace("_experiment_summary.md", "_evaluation_summary.md")
            save_training_summary(
                eval_results=final_eval,
                training_history={
                    'phase_rewards': self.phase_rewards,
                    'phase_distances': self.phase_distances,
                    'trajectory_stats': final_trajectory_stats,
                },
                save_path=eval_summary_path
            )
            vis_pbar.update(1)
            print(f"✅ Evaluation summary: {eval_summary_path}")
        
        total_time = time.time() - start_time
        print(f"\n🏁 Evaluation Complete!")
        print(f"   Total time: {total_time/60:.1f} minutes")
        print(f"   Step: {self.current_step:,}")
        print(f"   Episode: {self.current_episode:,}")
        
        env.close()
        return final_eval 
