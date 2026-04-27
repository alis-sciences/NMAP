#!/usr/bin/env python3
"""
simple_ncap_logger.py
─────────────────────
Training-loop logger for SimpleNCAPTrainer.

Reuses existing project utilities:
  • TrainingLogger  (swimmer/utils/training_logger.py)  – metrics, CSVs, JSON
  • plot_training_interval_rewards (swimmer/utils/visualization.py) – reward curve
  • flatten_observation (swimmer/utils/helpers.py) – obs flattening in adapter

Produces files matching the curriculum-trainer naming convention:
  <prefix>_training_final.png
  <prefix>_final_trajectory_phase_phase0.png
  <prefix>_body_profile_phase_phase<N>_step_<S>.png
  <prefix>_trajectory_analysis_phase_phase<N>_step_<S>.png
  <prefix>_step_rewards.csv   (step | mean_reward | phase)
  <prefix>_eval_returns.csv   (step | ph0_mean    | ph0_std)
"""

import os
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── reuse existing project utils ─────────────────────────────────────────────
try:
    from ..utils.training_logger import TrainingLogger
    from ..utils.visualization import plot_training_interval_rewards
    from ..utils.helpers import flatten_observation
except ImportError:
    try:
        from swimmer.utils.training_logger import TrainingLogger
        from swimmer.utils.visualization import plot_training_interval_rewards
        from swimmer.utils.helpers import flatten_observation
    except ImportError:
        TrainingLogger               = None
        plot_training_interval_rewards = None
        flatten_observation          = lambda obs: (
            obs.flatten() if hasattr(obs, 'flatten') else np.asarray(obs).flatten()
        )

# ── ccmn_visualizations for episode rollout plots ────────────────────────────
try:
    from ..ccmn_visualizations import (
        collect_ncap_episode_data,
        plot_ccmn_gait_trajectory,
        plot_ccmn_gait_kymograph,
        plot_ccmn_speed_zda_environment,
        save_log_to_csv,
        SWIM_COLOR, CRAWL_COLOR,
    )
    _HAS_CCMN = True
except ImportError:
    try:
        from ccmn_visualizations import (
            collect_ncap_episode_data,
            plot_ccmn_gait_trajectory,
            plot_ccmn_gait_kymograph,
            plot_ccmn_speed_zda_environment,
            save_log_to_csv,
            SWIM_COLOR, CRAWL_COLOR,
        )
        _HAS_CCMN = True
    except ImportError:
        _HAS_CCMN  = False
        SWIM_COLOR  = '#E07B2A'
        CRAWL_COLOR = '#2A5FA5'


# ── curriculum_visualization for video generation ────────────────────────────
try:
    from ..utils.curriculum_visualization import (
        create_test_video,
        create_body_profile_figure,
        _save_video_file,
        _save_video_previews,
    )
    _HAS_VIZ = True
except ImportError:
    try:
        from swimmer.utils.curriculum_visualization import (
            create_test_video,
            create_body_profile_figure,
            _save_video_file,
            _save_video_previews,
        )
        _HAS_VIZ = True
    except ImportError:
        _HAS_VIZ = False


# ─────────────────────────────────────────────────────────────────────────────
# Plain video environment — wraps SimpleSwimmerEnv with a 4-tuple step()
# and working render(), exactly matching what create_test_video expects.
# ─────────────────────────────────────────────────────────────────────────────

class _VideoEnv:
    """
    Thin wrapper around SimpleSwimmerEnv that:
      • exposes step(action) → (obs, reward, done, info)  plain 4-tuple
      • exposes render(mode='rgb_array') → np.ndarray via physics.render()
      • exposes reset() → flat np.ndarray obs
    Used exclusively for video generation so create_test_video works unchanged.
    """

    def __init__(self, n_links: int):
        try:
            from swimmer.environments.simple_swimmer import SimpleSwimmerEnv
        except ImportError:
            from swimmer.environments.simple_swimmer import SimpleSwimmerEnv
        self._env   = SimpleSwimmerEnv(n_links=n_links)
        self._step  = 0
        self._max   = 1000

    # ── obs flattening (matches TonicSimpleSwimmerWrapper layout) ────────────
    def _flat_obs(self, obs_dict):
        if isinstance(obs_dict, dict):
            joints   = obs_dict.get('joints',         np.zeros(self._env.action_spec.shape[0]))
            body_vel = obs_dict.get('body_velocities', np.zeros(len(joints) * 3 + 3))
        else:
            joints   = obs_dict
            body_vel = np.zeros(len(joints) * 3 + 3)
        flat = np.concatenate([joints, body_vel]).astype(np.float32)
        # append time feature
        return np.append(flat, self._step / self._max).astype(np.float32)

    def reset(self):
        obs_dict = self._env.reset()
        self._step = 0
        return self._flat_obs(obs_dict)

    def step(self, action):
        obs_dict, reward, done, info = self._env.step(action)
        self._step += 1
        if self._step >= self._max:
            done = True
        return self._flat_obs(obs_dict), reward if reward is not None else 0.0, done, info

    def render(self, mode='rgb_array'):
        """Direct physics render — works headlessly with MUJOCO_GL=egl."""
        return self._env.physics.render(camera_id=0, height=480, width=640)

    def close(self):
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Video-compatible agent adapter
# ─────────────────────────────────────────────────────────────────────────────

class _VideoAgent:
    """
    Adapter with the .test_step(obs) signature that create_test_video calls.
    Mirrors BiologicalNCAPAgent.test_step from curriculum_trainer.
    """

    def __init__(self, ncap_model, n_joints: int):
        self._model  = ncap_model
        self.n_joints = n_joints

    def test_step(self, obs):
        import torch
        if not isinstance(obs, np.ndarray):
            obs = np.asarray(obs, dtype=np.float32)
        device  = next(self._model.parameters()).device
        obs_t   = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            out = self._model.actor(obs_t)
            # Deterministic actor returns Tensor; stochastic returns Distribution
            if isinstance(out, torch.Tensor):
                action = out.squeeze(0).cpu().numpy()
            else:
                action = out.mean.squeeze(0).cpu().numpy()
        return action


# ─────────────────────────────────────────────────────────────────────────────
# Agent adapter  (bridges CustomA2C/_NCAPModel → collect_ncap_episode_data API)
# ─────────────────────────────────────────────────────────────────────────────

class _NCAPAgentAdapter:
    """
    Thin shim so collect_ncap_episode_data can drive _NCAPModel / CustomA2C.

    collect_ncap_episode_data expects:
        agent.swimmer            : SimpleNCAPSwimmer
        agent.n_joints           : int
        agent.env_features_start : int
        agent.test_step(obs)     : → np.ndarray action
    """

    def __init__(self, tonic_agent, ncap_model, n_joints: int):
        self._agent  = tonic_agent
        self._model  = ncap_model
        self.swimmer = ncap_model.actor.ncap          # SimpleNCAPSwimmer
        self.n_joints = n_joints
        # Point past the end of the obs vector so env-feature lookups return 0
        self.env_features_start = n_joints + n_joints * 3 + 1

    def test_step(self, obs):
        """Deterministic action = mean of the actor Normal distribution."""
        import torch
        # Use flatten_observation from helpers.py so dict/array obs both work
        obs_np = flatten_observation(obs) if not isinstance(obs, np.ndarray) else obs
        device = next(self._model.parameters()).device
        obs_t  = torch.as_tensor(obs_np, dtype=torch.float32,
                                 device=device).unsqueeze(0)
        with torch.no_grad():
            out = self._model.actor(obs_t)
            if isinstance(out, torch.Tensor):
                action = out.squeeze(0).cpu().numpy()
            else:
                action = out.mean.squeeze(0).cpu().numpy()
        return action


# ─────────────────────────────────────────────────────────────────────────────
# Main logger
# ─────────────────────────────────────────────────────────────────────────────

class SimpleNCAPLogger:
    """
    Attach to SimpleNCAPTrainer to get curriculum-style plots & CSVs.

    Delegates all metric storage / JSON / base plots to the existing
    TrainingLogger from swimmer/utils/training_logger.py.
    Adds ccmn_visualizations episode plots on top.

    Parameters
    ----------
    output_dir        : root directory for outputs
    prefix            : canonical filename prefix
    n_links           : total link count
    oscillator_period : CPG period (informational only)
    log_every         : env steps between reward CSV rows
    eval_every        : env steps between checkpoint plots  (= save_steps)
    experiment_name   : passed to TrainingLogger; defaults to prefix
    """

    def __init__(
        self,
        output_dir: str,
        prefix: str,
        n_links: int,
        oscillator_period: int = 60,
        log_every: int = 8,
        eval_every: int = 5000,
        experiment_name: str = None,
    ):
        self.output_dir        = output_dir
        self.prefix            = prefix
        self.n_links           = n_links
        self.oscillator_period = oscillator_period
        self.log_every         = log_every
        self.eval_every        = eval_every

        os.makedirs(output_dir, exist_ok=True)

        # ── TrainingLogger (existing util) ───────────────────────────────────
        exp_name = experiment_name or prefix
        if TrainingLogger is not None:
            self._tl = TrainingLogger(
                log_dir=output_dir,
                experiment_name=exp_name,
            )
            self._tl.start_training()
        else:
            self._tl = None
            print('[SimpleNCAPLogger] TrainingLogger not found – falling back to raw CSV')

        # ── step_rewards.csv  (curriculum-style naming) ──────────────────────
        self._rewards_csv = os.path.join(output_dir, f'{prefix}_step_rewards.csv')
        self._eval_csv    = os.path.join(output_dir, f'{prefix}_eval_returns.csv')
        self._init_csvs()

        # In-memory buffers for the summary plot
        self._interval_rewards: list[float] = []   # one entry per log_every block
        self._eval_returns:     list[dict]  = []

    # ── CSV initialisation ────────────────────────────────────────────────────

    def _init_csvs(self):
        with open(self._rewards_csv, 'w', newline='') as f:
            csv.writer(f).writerow(['step', 'mean_reward', 'phase'])
        with open(self._eval_csv, 'w', newline='') as f:
            csv.writer(f).writerow(['step', 'ph0_mean', 'ph0_std'])

    # ── Public hooks ──────────────────────────────────────────────────────────

    def on_step(self, step: int, reward: float):
        """Call after every environment step."""
        reward = float(reward)

        # Delegate to TrainingLogger for JSON / internal tracking
        if self._tl is not None:
            self._tl.log_training_step({'step': step, 'reward': reward})

        # Write to curriculum-style CSV every log_every steps
        if step % self.log_every == 0:
            with open(self._rewards_csv, 'a', newline='') as f:
                csv.writer(f).writerow([step, reward, 0])   # phase=0 always
            self._interval_rewards.append(reward)

    def on_eval(self, step: int, episode_returns: list):
        """Call at each checkpoint with a list of episode total returns."""
        arr  = np.asarray(episode_returns, dtype=np.float32)
        mean = float(arr.mean()) if len(arr) else 0.0
        std  = float(arr.std())  if len(arr) else 0.0

        self._eval_returns.append({'step': step, 'mean': mean, 'std': std})

        # Curriculum-style eval CSV
        with open(self._eval_csv, 'a', newline='') as f:
            csv.writer(f).writerow([step, mean, std])

        # TrainingLogger episode logging (reuses its reward tracking)
        if self._tl is not None:
            self._tl.log_episode(
                episode_reward=mean,
                episode_length=0,
                episode_distance=0.0,
            )

    def on_checkpoint(self, step: int, agent, env, phase: int = 0):
        """Checkpoint: save kymograph + speed/z_DA/env plots + video."""
        if _HAS_CCMN:
            try:
                log = self._rollout(self._make_adapter(agent), env, num_steps=600)
                tag = f'phase_phase{phase}_step_{step}'
                plot_ccmn_gait_kymograph(
                    log,
                    os.path.join(self.output_dir, f'{self.prefix}_body_profile_phase_{tag}.png')
                )
                plot_ccmn_speed_zda_environment(
                    log,
                    os.path.join(self.output_dir, f'{self.prefix}_trajectory_analysis_{tag}.png')
                )
            except Exception as e:
                print(f'[SimpleNCAPLogger] on_checkpoint failed at step {step}: {e}')

        # video is generated only at finalize(), not at every checkpoint

    def generate_video(self, agent, env=None, step: int = 0,
                       num_steps: int = 500, tag: str = ""):
        """
        Render an MP4 of the agent swimming.

        Uses _VideoEnv (plain 4-tuple step + direct physics.render) so that
        create_test_video from curriculum_visualization works without any
        monkey-patching.  Mirrors exactly how curriculum_trainer.py does it.
        Falls back to a direct imageio loop if curriculum_visualization is
        not importable.
        """
        label    = tag or (f"step_{step}" if step else "final")
        vid_path = os.path.join(self.output_dir, f"{self.prefix}_video_{label}.mp4")
        os.makedirs(self.output_dir, exist_ok=True)

        # Build a fresh plain video env and a compatible video agent
        vid_env   = _VideoEnv(n_links=self.n_links)
        vid_agent = _VideoAgent(ncap_model=agent.model, n_joints=self.n_links - 1)

        try:
            if _HAS_VIZ:
                create_test_video(
                    agent=vid_agent,
                    env=vid_env,
                    save_path=vid_path,
                    num_steps=num_steps,
                    episode_name=f"SimpleNCAPSwimmer {label}",
                    show_minimap=False,   # water-only: no land zones
                )
            else:
                # Direct imageio fallback
                import imageio
                frames = []
                obs = vid_env.reset()
                for _ in range(num_steps):
                    frame = vid_env.render()
                    if frame is not None:
                        frames.append(np.asarray(frame, dtype=np.uint8))
                    action = vid_agent.test_step(obs)
                    obs, _, done, _ = vid_env.step(action)
                    if done:
                        obs = vid_env.reset()
                if frames:
                    _save_video_file(vid_path, frames, fps=30)
                    _save_video_previews(vid_path, frames, fps=30)
                    print(f'🎬 Video saved (fallback): {vid_path}')
                else:
                    print('[SimpleNCAPLogger] no frames captured — check MUJOCO_GL=egl')
        except Exception as e:
            import traceback
            print(f'[SimpleNCAPLogger] video generation failed: {e}')
            traceback.print_exc()
        finally:
            vid_env.close()

    def finalize(self, agent, env, phase: int = 0):
        """End of training: save summary figure, final trajectory, video, and CSV export."""
        self._save_summary_figure()

        # Save TrainingLogger's own plots (episode reward curve etc.)
        if self._tl is not None:
            try:
                self._tl.save_metrics()
                self._tl.create_training_plots(save_plots=True)
                self._tl.create_summary_report()
            except Exception as e:
                print(f'[SimpleNCAPLogger] TrainingLogger finalize error: {e}')

        if _HAS_CCMN:
            try:
                log = self._rollout(self._make_adapter(agent), env, num_steps=1200)
                plot_ccmn_gait_trajectory(
                    log,
                    os.path.join(self.output_dir,
                                 f'{self.prefix}_final_trajectory_phase_phase{phase}.png')
                )
                save_log_to_csv(log, self.output_dir,
                                prefix=f'{self.prefix}_final_episode')
            except Exception as e:
                print(f'[SimpleNCAPLogger] finalize plots failed: {e}')

        # ── final video — always runs regardless of ccmn availability ─────
        try:
            self.generate_video(agent, step=0, num_steps=600, tag="final")
        except Exception as e:
            import traceback
            print(f'[SimpleNCAPLogger] final video failed: {e}')
            traceback.print_exc()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _make_adapter(self, agent) -> _NCAPAgentAdapter:
        return _NCAPAgentAdapter(
            tonic_agent=agent,
            ncap_model=agent.model,
            n_joints=self.n_links - 1,
        )

    def _rollout(self, adapter, env, num_steps: int = 1200):
        """
        Wraps env.step to give collect_ncap_episode_data a plain 4-tuple,
        restoring the original step afterwards.
        """
        orig = env.step

        def _plain(action):
            r = orig(action)
            if isinstance(r, tuple) and len(r) == 2 and isinstance(r[1], dict):
                obs, info = r
                return obs, float(info['rewards'][0]), bool(info['resets'][0]), {}
            return r

        env.step = _plain
        try:
            log = collect_ncap_episode_data(adapter, env, num_steps=num_steps)
        finally:
            env.step = orig
        return log

    def _save_summary_figure(self):
        """
        Two-panel training summary.

        Left panel  – uses plot_training_interval_rewards() from visualization.py
                      if available, otherwise falls back to inline matplotlib.
        Right panel – eval returns with std shading (always inline).
        """
        if not self._interval_rewards and not self._eval_returns:
            return

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle(f'Training Summary  –  {self.prefix}',
                     fontsize=12, fontweight='bold')

        # Left: interval rewards ──────────────────────────────────────────────
        ax0 = axes[0]
        if self._interval_rewards:
            if plot_training_interval_rewards is not None:
                # Save to a temp path then re-embed the axes content
                tmp = os.path.join(self.output_dir, '_tmp_interval_rewards.png')
                plot_training_interval_rewards(self._interval_rewards, tmp)
                # Read back and display in our subplot
                img = plt.imread(tmp)
                ax0.imshow(img)
                ax0.axis('off')
                try:
                    os.remove(tmp)
                except OSError:
                    pass
            else:
                # Inline fallback
                rewards = np.array(self._interval_rewards)
                ax0.plot(rewards, color=SWIM_COLOR, lw=0.8, alpha=0.5,
                         label='interval reward')
                if len(rewards) > 20:
                    k    = max(1, len(rewards) // 20)
                    smth = np.convolve(rewards, np.ones(k)/k, mode='valid')
                    ax0.plot(range(k-1, len(rewards)), smth,
                             color=CRAWL_COLOR, lw=2.0, label='smoothed')
                ax0.set_xlabel('Interval', fontsize=10)
                ax0.set_ylabel('Reward', fontsize=10)
                ax0.legend(fontsize=8)
        ax0.set_title('Step Rewards', fontsize=11, fontweight='bold')
        ax0.spines[['right', 'top']].set_visible(False)
        ax0.grid(True, alpha=0.2)

        # Right: eval returns ─────────────────────────────────────────────────
        ax1 = axes[1]
        if self._eval_returns:
            steps = np.array([r['step'] for r in self._eval_returns])
            means = np.array([r['mean'] for r in self._eval_returns])
            stds  = np.array([r['std']  for r in self._eval_returns])
            ax1.plot(steps, means, color=CRAWL_COLOR, lw=2.0,
                     marker='o', markersize=4, label='mean eval return')
            ax1.fill_between(steps, means - stds, means + stds,
                             color=CRAWL_COLOR, alpha=0.2)
        ax1.set_xlabel('Training step', fontsize=10)
        ax1.set_ylabel('Episode return', fontsize=10)
        ax1.set_title('Evaluation Returns (phase 0)', fontsize=11, fontweight='bold')
        ax1.legend(fontsize=8)
        ax1.spines[['right', 'top']].set_visible(False)
        ax1.grid(True, alpha=0.2)

        plt.tight_layout()
        out = os.path.join(self.output_dir, f'{self.prefix}_training_final.png')
        plt.savefig(out, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'✅ Training summary saved: {out}')


# ─────────────────────────────────────────────────────────────────────────────
# Prefix factory
# ─────────────────────────────────────────────────────────────────────────────

def make_prefix(n_links: int, algorithm: str = 'a2c',
                oscillator_period: int = 60,
                training_mode: str = 'simple') -> str:
    return (
        f'simple_ncap_{algorithm}_{n_links}links'
        f'_oscillator_period{oscillator_period}'
        f'_training_mode{training_mode}'
    )
