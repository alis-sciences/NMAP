#!/usr/bin/env python3
"""
ncap_analysis.py
────────────────
Generates the 8 analysis figures recommended for the Simple NCAP paper,
reusing ccmn_visualizations.py infrastructure wherever possible.

Figures produced
────────────────
  A  Phase portrait          — J1 angle vs J1 velocity (single limit cycle)
  B  Kymograph at checkpoints — body wave at 3 training snapshots
  C  Period distribution      — histogram of CPG period (single spike at 60)
  D  Algorithm kymograph      — side-by-side kymograph: PPO / DDPG / ES
  E  Weight trajectory        — 4 NCAP params vs timesteps for all algorithms
  F  Amplitude-frequency      — Berri et al. scatter, reused from ccmn_viz
  G  Wave propagation speed   — phase velocity J1→J5 vs checkpoint
  H  Weight convergence       — L2 distance to final weights vs timesteps

Progressive / substrate-aware figures (require --progressive flag):
  I  Substrate speed comparison   — mean forward speed water vs land per phase
  J  Phase portrait overlay       — water (blue) vs land (orange) limit cycles
  K  Kymograph split by substrate — body wave in water vs land side-by-side
  L  Reward per phase             — eval return curve split by training phase

Usage
─────
  # All figures from a single run
  python ncap_analysis.py \\
      --checkpoint_dir /path/to/outputs/ncap_swim_only_ppo_v3 \\
      --out_dir        /path/to/outputs/analysis \\
      --n_links 6

  # Multi-algorithm comparison (D, E, G, H)
  python ncap_analysis.py \\
      --checkpoint_dir /path/to/ppo_v3:/path/to/ddpg_v3:/path/to/es_v3 \\
      --labels PPO:DDPG:ES \\
      --colours "#2196F3:#FF9800:#4CAF50" \\
      --out_dir /path/to/outputs/analysis \\
      --n_links 6

  # Specific figures only
  python ncap_analysis.py --checkpoint_dir /path/to/ppo_v3 --figs A,B,C
"""

import argparse
import os
import sys
import json
import glob
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch

# ── colour palette matching paper ────────────────────────────────────────────
ALG_COLOURS = {'PPO': '#2196F3', 'DDPG': '#FF9800', 'ES': '#4CAF50',
               'A2C': '#E91E63'}
_DEFAULT_COLS = list(ALG_COLOURS.values())


# ─────────────────────────────────────────────────────────────────────────────
# Model + agent reconstruction helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_model(n_links: int, oscillator_period: int, algorithm: str):
    """Reconstruct _NCAPModel matching the training configuration."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from swimmer.training.simple_ncap_trainer import (
        SimpleNCAPSwimmer, _NCAPModel
    )
    n_joints = n_links - 1
    ncap = SimpleNCAPSwimmer(
        n_joints=n_joints,
        oscillator_period=oscillator_period,
        use_weight_sharing=True,
        use_weight_constraints=True,
        include_proprioception=True,
        include_head_oscillators=True,
    )
    model = _NCAPModel(ncap=ncap, algorithm=algorithm)
    return model


def _load_checkpoint(checkpoint_path: str, n_links: int,
                     oscillator_period: int, algorithm: str):
    """Load a checkpoint .pt file into a fresh model; return (model, step)."""
    import re as _re2
    step = 0
    base = os.path.basename(checkpoint_path)
    # Format D (curriculum): step encoded as _checkpoint_step_N in filename
    m = _re2.search(r'_checkpoint_step_(\d+)$', base)
    if m:
        step = int(m.group(1))
    else:
        # Format A/B/C (simple): _meta.json, then filename
        meta_path = checkpoint_path + '_meta.json'
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                step = json.load(f).get('step', 0)
        else:
            m2 = _re2.search(r'checkpoint_(\d+)$', base)
            if m2:
                step = int(m2.group(1))

    model = _build_model(n_links, oscillator_period, algorithm)

    loaded = False
    # ES saves as checkpoint_N_es_theta.npy (flat parameter vector)
    npy_path = checkpoint_path + '_es_theta.npy'
    if os.path.exists(npy_path):
        theta   = np.load(npy_path)
        shapes  = [p.data.shape for p in model.parameters()]
        sizes   = [p.data.numel() for p in model.parameters()]
        offset  = 0
        with torch.no_grad():
            for param, shape, size in zip(model.parameters(), shapes, sizes):
                chunk = theta[offset:offset + size].reshape(shape)
                param.data.copy_(torch.tensor(chunk, dtype=torch.float32))
                offset += size
        loaded = True

    if not loaded:
        for candidate in [checkpoint_path, checkpoint_path + '.pt']:
            if os.path.exists(candidate):
                raw = torch.load(candidate, map_location='cpu',
                                 weights_only=False)
                if isinstance(raw, dict):
                    state = (raw.get('model_state_dict')   # curriculum trainer
                             or raw.get('model')           # simple trainer
                             or raw)
                else:
                    state = raw

                # Diagnostic: count matching keys before any remapping
                model_keys  = set(model.state_dict().keys())
                ckpt_keys   = set(state.keys()) if isinstance(state, dict) else set()
                n_matched   = len(model_keys & ckpt_keys)

                if n_matched == 0 and isinstance(state, dict):
                    # Curriculum trainer saves CCMNSwimmerHRL weights flat
                    # (e.g. 'params.muscle_contra').  _NCAPModel wraps them as
                    # 'actor.ncap.params.muscle_contra'.  Remap by trying every
                    # prefix that would make checkpoint keys match model keys.
                    remapped = {}
                    for ck, cv in state.items():
                        for prefix in ('actor.ncap.', 'actor.', ''):
                            candidate_key = prefix + ck
                            if candidate_key in model_keys:
                                remapped[candidate_key] = cv
                                break
                    if remapped:
                        print(f'  [load] remapped {len(remapped)} keys from '
                              f'flat CCMN format → _NCAPModel format')
                        state = {**model.state_dict(), **remapped}  # fill rest with current
                    else:
                        print(f'  [load] ⚠️  ZERO keys matched after remapping — '
                              f'checkpoint may be incompatible')
                        print(f'         ckpt  keys (first 5): {list(ckpt_keys)[:5]}')
                        print(f'         model keys (first 5): {list(model_keys)[:5]}')
                else:
                    print(f'  [load] {os.path.basename(candidate)}  step={step}  '
                          f'keys matched {n_matched}/{len(model_keys)}')

                model.load_state_dict(state, strict=False)
                loaded = True
                break

    model.eval()
    return model, step


def _list_checkpoints(checkpoint_dir: str):
    """
    Return sorted list of checkpoint prefixes (without extension).

    Supports four formats:
      A  checkpoint_5000.pt                              simple trainer PPO/DDPG
      B  checkpoint_5000_es_theta.npy                   simple trainer ES
      C  checkpoint_5000                                 bare file
      D  {prefix}_checkpoint_step_5000.pt               curriculum trainer
         e.g. ccmn_hrl_ppo_6links_..._checkpoint_step_95000.pt
    """
    import re as _re
    prefixes = set()

    # Format A
    for f in glob.glob(os.path.join(checkpoint_dir, 'checkpoint_*.pt')):
        prefixes.add(f[:-3])

    # Format B
    for f in glob.glob(os.path.join(checkpoint_dir, 'checkpoint_*_es_theta.npy')):
        stem = os.path.basename(f).replace('_es_theta.npy', '')
        prefixes.add(os.path.join(checkpoint_dir, stem))

    # Format C
    for f in glob.glob(os.path.join(checkpoint_dir, 'checkpoint_*')):
        if not any(f.endswith(ext) for ext in ('.pt', '.json', '.npy')):
            prefixes.add(f)

    # Format D  —  *_checkpoint_step_N.pt  (curriculum trainer)
    for f in glob.glob(os.path.join(checkpoint_dir, '*_checkpoint_step_*.pt')):
        if not os.path.basename(f).startswith('checkpoint_'):
            prefixes.add(f[:-3])   # strip .pt

    def _extract_step(p):
        b = os.path.basename(p)
        m = _re.search(r'_checkpoint_step_(\d+)$', b)
        if m:
            return int(m.group(1))
        m = _re.search(r'checkpoint_(\d+)$', b)
        if m:
            return int(m.group(1))
        return 0

    return sorted(prefixes, key=_extract_step)


def _get_ncap_weights(model) -> dict:
    """Extract the 4 biological NCAP parameters as floats."""
    ncap = model.actor.ncap
    p = ncap.params
    return {
        'bneuron_prop':  float(p['bneuron_prop'].item()),
        'bneuron_osc':   float(p['bneuron_osc'].item()),
        'muscle_ipsi':   float(p['muscle_ipsi'].item()),
        'muscle_contra': float(p['muscle_contra'].item()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Rollout helper
# ─────────────────────────────────────────────────────────────────────────────

def _rollout_episode(model, n_links: int, num_steps: int = 1200,
                     progressive_env=None, training_progress: float = 0.0,
                     start_pos=None):
    """
    Run one episode and return a data dict:
        joint_pos   (T, n_joints)
        joint_vel   (T, n_joints)
        speeds      (T,)
        positions   (T, 2)
        substrate   (T,)  — 0=water, 1=land  (always 0 for water-only env)

    If progressive_env is given (NCAPProgressiveGymEnv instance) it is used
    instead of _VideoEnv, enabling land/water substrate tagging.
    """
    from swimmer.utils.simple_ncap_logger import _VideoAgent

    if progressive_env is not None:
        progressive_env.training_progress = training_progress
        env_obj   = progressive_env
        use_prog  = True
    else:
        from swimmer.utils.simple_ncap_logger import _VideoEnv
        env_obj  = _VideoEnv(n_links=n_links)
        use_prog = False

    agent = _VideoAgent(ncap_model=model, n_joints=n_links - 1)

    obs   = env_obj.reset()
    # Handle list wrapper from Tonic reset() or raw OrderedDict from gym reset()
    if isinstance(obs, list):
        obs = obs[0]

    # Teleport swimmer into a land zone if requested (must happen AFTER reset)
    if start_pos is not None:
        try:
            phys = (env_obj._dm_env.physics if use_prog
                    else env_obj._env.physics)
            # Try named qpos first, fall back to index
            try:
                phys.named.data.qpos['root'][0] = float(start_pos[0])
                phys.named.data.qpos['root'][1] = float(start_pos[1])
            except (KeyError, IndexError):
                phys.data.qpos[0] = float(start_pos[0])
                phys.data.qpos[1] = float(start_pos[1])
            phys.forward()
            # Re-read obs after teleport
            ts  = env_obj._dm_env.step(np.zeros(n_links - 1))
            obs = ts.observation
        except Exception as _e:
            pass  # silently continue from origin if teleport fails

    def _flatten_obs(o, n_links):
        """Flatten OrderedDict or numpy obs into a flat float32 array."""
        if isinstance(o, dict):
            joints   = np.asarray(o.get('joints',
                           np.zeros(n_links - 1)), dtype=np.float32).ravel()
            body_vel = np.asarray(o.get('body_velocities',
                           np.zeros(n_links * 3)), dtype=np.float32).ravel()
            return np.concatenate([joints, body_vel])
        if isinstance(o, (list, tuple)):
            o = o[0]
        return np.asarray(o, dtype=np.float32).ravel()

    obs = _flatten_obs(obs, n_links)
    joint_pos_list, speed_list, pos_list, substrate_list = [], [], [], []

    for _ in range(num_steps):
        n_joints = n_links - 1
        jp       = obs[:n_joints].copy()
        joint_pos_list.append(jp)

        # Physics readout
        try:
            phys = (env_obj._dm_env.physics if use_prog
                    else env_obj._env.physics)
            vel   = phys.named.data.cvel['head']
            speed = float(np.linalg.norm(vel[3:5]))
            pos   = phys.named.data.xpos['head'][:2].copy()
            # Substrate: check head position against task land zones
            # (viscosity only changes AFTER get_reward runs, so position is more reliable)
            substrate = 0
            try:
                head_xy = phys.named.data.xpos['head'][:2]
                task    = (env_obj._dm_env.task if use_prog
                           else env_obj._env.task)
                if hasattr(task, '_current_land_zones'):
                    for z in task._current_land_zones:
                        if np.linalg.norm(head_xy - z['center']) < z['radius']:
                            substrate = 1
                            break
                else:
                    # fallback: viscosity
                    substrate = 1 if float(phys.model.opt.viscosity) > 0.005 else 0
            except Exception:
                substrate = 0
        except Exception:
            speed = 0.0
            pos   = np.zeros(2)
            substrate = 0

        speed_list.append(speed)
        pos_list.append(pos)
        substrate_list.append(substrate)

        action = agent.test_step(obs)
        result = env_obj.step(action)
        # Both envs return (obs, reward, done, info) or Tonic (obs, infos)
        if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict) and 'rewards' in result[1]:
            obs, infos = result
            done = bool(infos['resets'][0])
        else:
            obs, _, done, _ = result
        obs = _flatten_obs(obs, n_links)
        if done:
            raw = env_obj.reset()
            if isinstance(raw, list): raw = raw[0]
            obs = _flatten_obs(raw, n_links)

    if not use_prog:
        env_obj.close()

    jp_arr = np.array(joint_pos_list)
    jv_arr = np.diff(jp_arr, axis=0, prepend=jp_arr[:1])
    return {
        'joint_pos':  jp_arr,
        'joint_vel':  jv_arr,
        'speeds':     np.array(speed_list),
        'positions':  np.array(pos_list),
        'substrate':  np.array(substrate_list, dtype=np.int8),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Figure A — Phase portrait
# ─────────────────────────────────────────────────────────────────────────────

def _period_mask(jp: np.ndarray, threshold_steps: int = 35):
    """
    Classify each timestep as swim (short period) or crawl (long period)
    by measuring the local half-cycle length from adjacent zero-crossings.
    Returns a bool array: True = crawl (period > threshold), False = swim.
    Falls back to all-False (all swim) if measurement fails.
    """
    mean_val = jp.mean()
    zc = np.where(np.diff(np.sign(jp - mean_val)))[0]
    period_at_step = np.zeros(len(jp), dtype=np.float32)
    for i in range(len(zc) - 1):
        half = zc[i + 1] - zc[i]
        period_at_step[zc[i]:zc[i + 1]] = half * 2   # full period estimate
    # Fill tails
    if len(zc) > 0:
        period_at_step[:zc[0]]   = period_at_step[zc[0]]  if len(zc) else 30
        period_at_step[zc[-1]:]  = period_at_step[zc[-1]] if len(zc) else 30
    return period_at_step > threshold_steps


def fig_A_phase_portrait(data: dict, save_path: str,
                         title: str = 'NCAP — Phase Portrait'):
    """
    J1 angle vs J1 angular velocity — split into separate panels per gait.

    When both substrates are present (progressive run) the figure has TWO
    panels side-by-side:
        Left  — Water substrate  (blue points)
        Right — Land substrate   (orange points)

    Each panel shows ONLY the points from that substrate, so the two limit
    cycles (fast/small swim orbit vs slow/large crawl orbit) are displayed
    separately.  This resolves the diffuse-cloud artefact caused by
    superimposing two distinct orbits of different sizes and timescales.

    For water-only runs a single panel is shown with the gait further split
    by estimated oscillator period (swim=blue, crawl=orange) using a zero-
    crossing period estimator with threshold=35 steps.
    """
    jp  = data['joint_pos'][:, 0]
    jv  = data['joint_vel'][:, 0]
    sub = data.get('substrate', np.zeros(len(jp), dtype=np.int8))

    has_water = (sub == 0).sum() > 20
    has_land  = (sub == 1).sum() > 20

    def _scatter_panel(ax, mask, jp, jv, colour, label, title_str):
        jp_s = jp[mask];  jv_s = jv[mask]
        if len(jp_s) == 0:
            ax.text(0.5, 0.5, f'No {label} steps', transform=ax.transAxes,
                    ha='center', va='center', color='grey')
            return
        ax.scatter(jp_s, jv_s, c=colour, s=3, alpha=0.55,
                   linewidths=0, rasterized=True)
        ax.set_title(title_str, fontsize=10, fontweight='bold', color=colour)
        ax.spines[['top', 'right']].set_visible(False)
        ax.axhline(0, color='grey', lw=0.5, ls=':')
        ax.axvline(0, color='grey', lw=0.5, ls=':')
        zc = int(np.sum(np.diff(np.sign(jp_s)) != 0) // 2)
        ax.text(0.97, 0.03, f'~{zc} cycles',
                transform=ax.transAxes, ha='right', va='bottom',
                fontsize=7, color='#555')

    if has_water and has_land:
        # Two-panel layout: left=water, right=land
        fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0), sharey=True)
        fig.suptitle(title, fontsize=11, fontweight='bold')
        _scatter_panel(axes[0], sub == 0, jp, jv, '#2196F3',
                       'Water', 'Water substrate')
        _scatter_panel(axes[1], sub == 1, jp, jv, '#FF9800',
                       'Land',  'Land substrate')
        for ax in axes:
            ax.set_xlabel('J1 angle  (rad)', fontsize=9)
        axes[0].set_ylabel('J1 angular velocity  (rad/step)', fontsize=9)
        plt.tight_layout()
    else:
        # Single-panel: split by estimated oscillator period
        crawl_mask = _period_mask(jp)
        fig, ax = plt.subplots(figsize=(4.5, 4.0))
        ax.set_title(title, fontsize=11, fontweight='bold', loc='left')
        for mask, col, lbl in [
            (~crawl_mask, '#2196F3', 'Swim (short period)'),
            ( crawl_mask, '#FF9800', 'Crawl (long period)'),
        ]:
            if mask.sum() > 5:
                ax.scatter(jp[mask], jv[mask], c=col, s=3, alpha=0.55,
                           linewidths=0, label=lbl, rasterized=True)
        ax.legend(fontsize=8, frameon=False, markerscale=4)
        ax.set_xlabel('J1 angle  (rad)', fontsize=10)
        ax.set_ylabel('J1 angular velocity  (rad/step)', fontsize=10)
        ax.spines[['top', 'right']].set_visible(False)
        ax.axhline(0, color='grey', lw=0.6, ls=':')
        ax.axvline(0, color='grey', lw=0.6, ls=':')
        zc = int(np.sum(np.diff(np.sign(jp)) != 0) // 2)
        ax.text(0.97, 0.03, f'~{zc} cycles',
                transform=ax.transAxes, ha='right', va='bottom',
                fontsize=8, color='#555')
        plt.tight_layout()

    _save(fig, save_path)


# ─────────────────────────────────────────────────────────────────────────────
# Figure B — Kymograph at checkpoints
# ─────────────────────────────────────────────────────────────────────────────

def fig_B_kymograph_checkpoints(checkpoint_dir: str, checkpoint_steps: list,
                                 n_links: int, oscillator_period: int,
                                 algorithm: str, save_path: str,
                                 num_steps: int = 400):
    """Body wave kymograph at 3 training snapshots side-by-side."""
    n_panels = len(checkpoint_steps)
    fig, axes = plt.subplots(1, n_panels, figsize=(4.0 * n_panels, 3.5),
                             sharey=True)
    if n_panels == 1:
        axes = [axes]
    fig.suptitle('Body Wave Kymograph at Training Checkpoints',
                 fontsize=12, fontweight='bold')

    # Build a step→path lookup from whatever filenames exist
    import re as _re
    all_ckpts = _list_checkpoints(checkpoint_dir)
    step_to_path = {}
    for p in all_ckpts:
        b = os.path.basename(p)
        for pat in [r'_checkpoint_step_(\d+)$', r'checkpoint_(\d+)$']:
            m = _re.search(pat, b)
            if m:
                step_to_path[int(m.group(1))] = p
                break

    for ax, step in zip(axes, checkpoint_steps):
        ckpt = step_to_path.get(step,
               os.path.join(checkpoint_dir, f'checkpoint_{step}'))
        model, _ = _load_checkpoint(ckpt, n_links, oscillator_period, algorithm)
        data      = _rollout_episode(model, n_links, num_steps=num_steps)
        jp        = data['joint_pos']   # (T, n_joints)
        n_joints  = jp.shape[1]

        # Normalise each joint column
        jp_norm = (jp - jp.mean(axis=0)) / (jp.std(axis=0) + 1e-8)

        im = ax.imshow(jp_norm.T, aspect='auto', origin='lower',
                       cmap='RdBu_r', vmin=-2, vmax=2,
                       extent=[0, num_steps, 0, n_joints])
        ax.set_xlabel('time  (steps)', fontsize=9)
        ax.set_title(f'step {step:,}', fontsize=10, fontweight='bold')
        ax.set_yticks(np.arange(n_joints) + 0.5)
        ax.set_yticklabels([f'J{i+1}' for i in range(n_joints)], fontsize=7)

    axes[0].set_ylabel('joint  (head → tail)', fontsize=9)
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02,
                 label='normalised angle')
    _save(fig, save_path)


# ─────────────────────────────────────────────────────────────────────────────
# Figure C — Period distribution
# ─────────────────────────────────────────────────────────────────────────────

def fig_C_period_distribution(data: dict, oscillator_period: int,
                               save_path: str,
                               title: str = 'NCAP — CPG Period Distribution'):
    """
    Histogram of effective oscillator period measured from J1 zero-crossings.
    NCAP produces a single spike; CCMN-HRL would show bimodal.
    """
    jp0 = data['joint_pos'][:, 0]
    # Detect half-cycles via sign changes
    zc = np.where(np.diff(np.sign(jp0 - jp0.mean())))[0]
    periods = []
    for i in range(0, len(zc) - 2, 2):
        periods.append(zc[i + 2] - zc[i])   # full cycle in steps

    fig, ax = plt.subplots(figsize=(5.0, 3.5))
    if periods:
        bins = np.arange(max(0, min(periods) - 5),
                         max(periods) + 10, 2)
        ax.hist(periods, bins=bins, color='#2196F3', edgecolor='white',
                linewidth=0.4, density=True)
    ax.axvline(oscillator_period, color='black', lw=2.0, ls='--',
               label=f'nominal period = {oscillator_period}')
    ax.set_xlabel('oscillator period  (steps)', fontsize=10)
    ax.set_ylabel('density', fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold', loc='left')
    ax.legend(fontsize=9)
    ax.spines[['top', 'right']].set_visible(False)

    # Annotate mean
    if periods:
        mu = float(np.mean(periods))
        ax.text(0.97, 0.95, f'mean = {mu:.1f} steps',
                transform=ax.transAxes, ha='right', va='top', fontsize=9)
    plt.tight_layout()
    _save(fig, save_path)


# ─────────────────────────────────────────────────────────────────────────────
# Figure D — Algorithm kymograph overlay
# ─────────────────────────────────────────────────────────────────────────────

def fig_D_algorithm_kymographs(runs: list, save_path: str,
                                num_steps: int = 400):
    """
    Side-by-side kymographs at the final checkpoint for each algorithm.
    Produces two figures:
        save_path              — water rollout (starting at origin)
        save_path → _land.png  — land rollout  (starting inside land zone)
    """
    n = len(runs)

    # Land start positions per phase (use Phase 1 island at [3,0])
    land_start = [3.0, 0.0]

    # Try to set up progressive env for land rollouts
    try:
        from swimmer.training.ncap_progressive_env import NCAPProgressiveGymEnv
        _prog_env_class = NCAPProgressiveGymEnv
    except ImportError:
        _prog_env_class = None

    for substrate, substrate_label, start_pos, cmap in [
        ('water', 'Water substrate', None,       'RdBu_r'),
        ('land',  'Land substrate',  land_start,  'PuOr'),
    ]:
        fig, axes = plt.subplots(1, n, figsize=(4.0 * n, 3.5), sharey=True)
        if n == 1:
            axes = [axes]
        fig.suptitle(f'Body Wave Kymograph — {substrate_label} '
                     f'(Algorithm Comparison, Final Checkpoint)',
                     fontsize=11, fontweight='bold')

        im = None
        for ax, run in zip(axes, runs):
            model, step = _load_checkpoint(
                run['checkpoint_path'], run['n_links'],
                run['oscillator_period'], run['algorithm']
            )

            # Use progressive env for land rollouts so viscosity is correct
            if substrate == 'land' and _prog_env_class is not None:
                prog_env = _prog_env_class(n_links=run['n_links'])
                data = _rollout_episode(model, run['n_links'],
                                        num_steps=num_steps,
                                        progressive_env=prog_env,
                                        training_progress=0.45,
                                        start_pos=start_pos)
            else:
                data = _rollout_episode(model, run['n_links'],
                                        num_steps=num_steps,
                                        start_pos=start_pos)

            jp       = data['joint_pos']
            n_joints = jp.shape[1]
            jp_norm  = (jp - jp.mean(0)) / (jp.std(0) + 1e-8)

            im = ax.imshow(jp_norm.T, aspect='auto', origin='lower',
                           cmap=cmap, vmin=-2, vmax=2,
                           extent=[0, num_steps, 0, n_joints])
            ax.set_title(f'{run["label"]}  (step {step:,})',
                         fontsize=10, fontweight='bold',
                         color=run.get('colour', '#333'))
            ax.set_xlabel('time  (steps)', fontsize=9)
            ax.set_yticks(np.arange(n_joints) + 0.5)
            ax.set_yticklabels([f'J{i+1}' for i in range(n_joints)], fontsize=7)

        axes[0].set_ylabel('joint  (head → tail)', fontsize=9)
        if im is not None:
            fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02,
                         label='normalised angle')

        out_path = save_path if substrate == 'water' else                    save_path.replace('.png', '_land.png')
        _save(fig, out_path)
    _save(fig, save_path)


# ─────────────────────────────────────────────────────────────────────────────
# Figure E — Weight trajectory across training
# ─────────────────────────────────────────────────────────────────────────────

def fig_E_weight_trajectory(runs: list, save_path: str):
    """
    4 NCAP biological parameters (bneuron_prop, bneuron_osc,
    muscle_ipsi, muscle_contra) vs timesteps for each algorithm.
    runs: same structure as fig_D.
    """
    param_names = ['bneuron_prop', 'bneuron_osc', 'muscle_ipsi', 'muscle_contra']
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=False)
    fig.suptitle('NCAP Biological Parameter Trajectories',
                 fontsize=12, fontweight='bold')
    axes_flat = axes.flatten()

    for run in runs:
        ckpts = _list_checkpoints(run['checkpoint_dir'])
        steps, weights = [], {p: [] for p in param_names}
        for ckpt in ckpts:
            try:
                model, step = _load_checkpoint(
                    ckpt, run['n_links'],
                    run['oscillator_period'], run['algorithm']
                )
                w = _get_ncap_weights(model)
                steps.append(step)
                for p in param_names:
                    weights[p].append(w[p])
            except Exception:
                continue

        colour = run.get('colour', '#333333')
        label  = run['label']
        steps  = np.array(steps)

        for ax, param in zip(axes_flat, param_names):
            vals = np.array(weights[param])
            ax.plot(steps, vals, color=colour, lw=1.8, label=label)
            ax.set_title(param, fontsize=10, fontweight='bold')
            ax.set_xlabel('timesteps', fontsize=9)
            ax.set_ylabel('weight value', fontsize=9)
            ax.spines[['top', 'right']].set_visible(False)
            ax.grid(True, alpha=0.2)
            ax.set_xscale('log')
            ax.xaxis.set_major_formatter(
                ticker.LogFormatterSciNotation(base=10, labelOnlyBase=True))

    # Single legend on first axis
    axes_flat[0].legend(fontsize=8, frameon=False)
    plt.tight_layout()
    _save(fig, save_path)


# ─────────────────────────────────────────────────────────────────────────────
# Figure F — Amplitude-frequency scatter (reuses ccmn_visualizations)
# ─────────────────────────────────────────────────────────────────────────────

def fig_F_amplitude_frequency(data: dict, save_path: str,
                               sim_hz: float = 30.0):
    """
    Amplitude-frequency scatter with Berri et al. reference ellipses.
    Reuses plot_ccmn_amplitude_frequency_scatter from ccmn_visualizations.
    """
    try:
        from swimmer.utils.ccmn_visualizations import (
            plot_ccmn_amplitude_frequency_scatter
        )
        # Build a minimal ccmn_log compatible dict
        log = {
            'joint_pos':   data['joint_pos'],
            'gait_labels': np.zeros(len(data['joint_pos']), dtype=np.float32),
            'speeds':      data['speeds'],
        }
        plot_ccmn_amplitude_frequency_scatter(log, save_path, sim_hz=sim_hz)
    except ImportError:
        # Fallback inline implementation
        _fig_F_fallback(data, save_path, sim_hz)


def _fig_F_fallback(data: dict, save_path: str, sim_hz: float = 30.0):
    from matplotlib.patches import Ellipse
    jp0 = data['joint_pos'][:, 0]
    zc  = np.where(np.diff(np.sign(jp0 - jp0.mean())))[0]
    freqs, amps = [], []
    for i in range(0, len(zc) - 2, 2):
        t0, t2 = zc[i], zc[i + 2]
        if t2 - t0 < 3:
            continue
        freqs.append(sim_hz / (t2 - t0))
        amps.append(jp0[t0:t2].max() - jp0[t0:t2].min())

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.scatter(freqs, amps, c='#2196F3', alpha=0.5, s=20, label='NCAP')
    # Berri et al. reference ellipses
    for cx, cy, w, h, col, lbl in [
            (2.0, 0.25, 1.2, 0.20, '#2196F3', 'Berri swim'),
            (0.5, 0.40, 0.6, 0.25, '#FF9800', 'Berri crawl')]:
        ax.add_patch(Ellipse((cx, cy), w, h, fill=False,
                             edgecolor=col, lw=1.8, ls='--', alpha=0.8,
                             label=lbl))
    ax.set_xlabel('undulation frequency  (Hz)', fontsize=10)
    ax.set_ylabel('joint amplitude  (rad)', fontsize=10)
    ax.set_title('Amplitude–Frequency  (Berri et al. 2009 ref.)',
                 fontsize=11, fontweight='bold', loc='left')
    ax.legend(fontsize=8, frameon=False)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    _save(fig, save_path)


# ─────────────────────────────────────────────────────────────────────────────
# Figure G — Wave propagation speed vs checkpoint
# ─────────────────────────────────────────────────────────────────────────────

def _wave_speed(joint_pos: np.ndarray, sim_hz: float = 30.0) -> float:
    """
    Estimate travelling wave phase velocity from J1 to J(last) using
    cross-correlation peak lag.  Returns speed in joints/second.
    """
    n_joints = joint_pos.shape[1]
    if n_joints < 2:
        return 0.0
    j0 = joint_pos[:, 0] - joint_pos[:, 0].mean()
    j1 = joint_pos[:, -1] - joint_pos[:, -1].mean()
    xcorr = np.correlate(j0, j1, mode='full')
    lags  = np.arange(-(len(j0) - 1), len(j0))
    lag   = lags[np.argmax(xcorr)]          # steps
    if lag == 0:
        return float('inf')
    speed = (n_joints - 1) * sim_hz / abs(lag)   # joints per second
    return float(speed)


def fig_G_wave_propagation(runs: list, save_path: str,
                            num_steps: int = 600, sim_hz: float = 30.0):
    """
    Phase velocity of the travelling body wave (J1→J_last) vs checkpoint step.
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    fig.suptitle('Body Wave Propagation Speed vs Training Step',
                 fontsize=11, fontweight='bold')

    for run in runs:
        ckpts  = _list_checkpoints(run['checkpoint_dir'])
        steps, speeds = [], []
        for ckpt in ckpts:
            try:
                model, step = _load_checkpoint(
                    ckpt, run['n_links'],
                    run['oscillator_period'], run['algorithm']
                )
                data  = _rollout_episode(model, run['n_links'],
                                          num_steps=num_steps)
                speed = _wave_speed(data['joint_pos'], sim_hz=sim_hz)
                steps.append(step)
                speeds.append(speed)
            except Exception:
                continue

        ax.plot(steps, speeds, color=run.get('colour', '#333'),
                lw=1.8, label=run['label'], marker='o', markersize=3)

    ax.set_xlabel('timesteps', fontsize=10)
    ax.set_ylabel('wave speed  (joints / s)', fontsize=10)
    ax.legend(fontsize=9, frameon=False)
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_xscale('log')
    ax.xaxis.set_major_formatter(
        ticker.LogFormatterSciNotation(base=10, labelOnlyBase=True))
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    _save(fig, save_path)


# ─────────────────────────────────────────────────────────────────────────────
# Figure H — Weight convergence (L2 distance to final weights)
# ─────────────────────────────────────────────────────────────────────────────

def fig_H_weight_convergence(runs: list, save_path: str):
    """
    L2 distance between current weights and final-checkpoint weights,
    as a function of training step.  Shows how quickly each algorithm
    converges the NCAP biological parameters.
    """
    param_names = ['bneuron_prop', 'bneuron_osc', 'muscle_ipsi', 'muscle_contra']

    fig, ax = plt.subplots(figsize=(7, 4))
    fig.suptitle('NCAP Weight Convergence — L2 Distance to Final Weights',
                 fontsize=11, fontweight='bold')

    for run in runs:
        ckpts = _list_checkpoints(run['checkpoint_dir'])
        if not ckpts:
            continue

        # Load final weights
        try:
            final_model, _ = _load_checkpoint(
                ckpts[-1], run['n_links'],
                run['oscillator_period'], run['algorithm']
            )
            final_w = np.array([_get_ncap_weights(final_model)[p]
                                 for p in param_names])
        except Exception:
            continue

        steps, dists = [], []
        for ckpt in ckpts:
            try:
                model, step = _load_checkpoint(
                    ckpt, run['n_links'],
                    run['oscillator_period'], run['algorithm']
                )
                w    = np.array([_get_ncap_weights(model)[p]
                                  for p in param_names])
                dist = float(np.linalg.norm(w - final_w))
                steps.append(step)
                dists.append(dist)
            except Exception:
                continue

        ax.plot(steps, dists, color=run.get('colour', '#333'),
                lw=1.8, label=run['label'], marker='o', markersize=3)

    ax.set_xlabel('timesteps', fontsize=10)
    ax.set_ylabel('L2 distance to final weights', fontsize=10)
    ax.legend(fontsize=9, frameon=False)
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_xscale('log')
    ax.xaxis.set_major_formatter(
        ticker.LogFormatterSciNotation(base=10, labelOnlyBase=True))
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    _save(fig, save_path)


# ─────────────────────────────────────────────────────────────────────────────
# Progressive / substrate-aware figures  I – L
# ─────────────────────────────────────────────────────────────────────────────

_WATER_C = '#2196F3'   # blue  — water
_LAND_C  = '#FF9800'   # orange — land
_PHASE_COLOURS = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63']
_PHASE_LABELS  = ['Phase 0\nPure water', 'Phase 1\nOne island',
                  'Phase 2\nTwo islands', 'Phase 3\nFour islands']


def _rollout_progressive(model, n_links, oscillator_period,
                         num_steps=1200, water_only_fallback=False):
    """
    Run rollouts at four training_progress values using the progressive env.
    Returns a list of 4 data dicts with substrate tags.

    If water_only_fallback=True (model trained without land), runs all 4
    phases in pure water so I-L figures still generate — they will show
    water-only substrate in every phase, making the baseline comparison clear.
    """
    if water_only_fallback:
        # No progressive env needed — run 4 identical water-only rollouts
        # with phase labels so figures render correctly
        print('  [info] water_only mode: running 4 water-only rollouts for I-L baseline')
        results = []
        for p in [0.15, 0.45, 0.70, 0.90]:
            data = _rollout_episode(model, n_links, num_steps=num_steps)
            data['phase_progress'] = p
            n_water = int((data['substrate'] == 0).sum())
            print(f'  [debug] progress={p:.2f}  water={n_water}  land=0  (water-only training)')
            results.append(data)
        return results

    try:
        from swimmer.training.ncap_progressive_env import NCAPProgressiveGymEnv
    except ImportError:
        try:
            from ncap_progressive_env import NCAPProgressiveGymEnv
        except ImportError:
            try:
                import importlib, sys
                spec = importlib.util.spec_from_file_location(
                    'ncap_progressive_env',
                    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 'swimmer', 'training', 'ncap_progressive_env.py'))
                if spec is None:
                    # try workspace root
                    spec = importlib.util.spec_from_file_location(
                        'ncap_progressive_env',
                        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     'ncap_progressive_env.py'))
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                NCAPProgressiveGymEnv = mod.NCAPProgressiveGymEnv
            except Exception as _e:
                print(f'  ⚠️  ncap_progressive_env import failed: {_e}')
                return None

    # mid-point of each phase + where to place the swimmer so it
    # starts inside a land zone (guarantees land segments in the rollout)
    # For each phase run TWO rollouts — one from water (origin) and one from
    # inside a land zone — then merge substrate tags so both media are present.
    phase_configs = [
        (0.15, None,         None),           # Phase 0: pure water only
        (0.45, None,         [3.0,  0.0]),    # Phase 1: water start + land start
        (0.70, None,         [-2.0, 0.0]),    # Phase 2
        (0.90, None,         [-3.0, 0.0]),    # Phase 3
    ]
    env = NCAPProgressiveGymEnv(n_links=n_links)
    results = []
    half = num_steps // 2
    for p, water_start, land_start in phase_configs:
        env.training_progress = p

        # Water rollout (start at origin or water_start)
        d_water = _rollout_episode(model, n_links, num_steps=half,
                                   progressive_env=env,
                                   training_progress=p,
                                   start_pos=water_start)

        # Land rollout (start inside zone); skip for pure-water phase
        if land_start is not None:
            d_land = _rollout_episode(model, n_links, num_steps=half,
                                      progressive_env=env,
                                      training_progress=p,
                                      start_pos=land_start)
            # Merge the two rollouts
            data = {
                'joint_pos': np.concatenate([d_water['joint_pos'],
                                             d_land['joint_pos']], axis=0),
                'joint_vel': np.concatenate([d_water['joint_vel'],
                                             d_land['joint_vel']], axis=0),
                'speeds':    np.concatenate([d_water['speeds'],
                                             d_land['speeds']]),
                'positions': np.concatenate([d_water['positions'],
                                             d_land['positions']], axis=0),
                'substrate': np.concatenate([d_water['substrate'],
                                             d_land['substrate']]),
            }
        else:
            data = d_water
        data['phase_progress'] = p
        n_land  = int(data['substrate'].sum())
        n_water = int((data['substrate'] == 0).sum())
        print(f'  [debug] progress={p:.2f}  water={n_water}  land={n_land}  '
              f'pos_range_x=[{data["positions"][:,0].min():.2f}, {data["positions"][:,0].max():.2f}]')
        results.append(data)
    return results


# ── Fig I — Substrate speed comparison ───────────────────────────────────────

def fig_I_substrate_speed(phase_data: list, save_path: str,
                          label: str = 'NCAP'):
    """
    Bar chart: mean forward speed in water vs land segments for each phase.
    Shows how the NCAP motor pattern degrades (or adapts) on land.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.suptitle(f'{label} — Forward Speed: Water vs Land by Phase',
                 fontsize=11, fontweight='bold')

    phases_with_land = [d for d in phase_data
                        if d['substrate'].sum() > 10]

    x      = np.arange(len(phases_with_land))
    width  = 0.35
    water_means, water_stds = [], []
    land_means,  land_stds  = [], []

    for d in phases_with_land:
        sub  = d['substrate']
        spd  = d['speeds']
        w    = spd[sub == 0]
        l    = spd[sub == 1]
        water_means.append(np.mean(w) if len(w) else 0.0)
        water_stds.append( np.std(w)  if len(w) else 0.0)
        land_means.append( np.mean(l) if len(l) else 0.0)
        land_stds.append(  np.std(l)  if len(l) else 0.0)

    ax.bar(x - width/2, water_means, width, yerr=water_stds,
           color=_WATER_C, label='Water', capsize=4, alpha=0.85)
    ax.bar(x + width/2, land_means,  width, yerr=land_stds,
           color=_LAND_C,  label='Land',  capsize=4, alpha=0.85)

    ax.set_xticks(x)
    labels = [_PHASE_LABELS[int(d['phase_progress'] * 4)]
              for d in phases_with_land]
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('forward speed  (m/s)', fontsize=10)
    ax.legend(fontsize=9, frameon=False)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', alpha=0.2)
    plt.tight_layout()
    _save(fig, save_path)


# ── Fig J — Phase portrait overlay: water vs land ────────────────────────────

def fig_J_substrate_phase_portrait(phase_data: list, save_path: str,
                                   label: str = 'NCAP'):
    """
    J1 angle vs velocity coloured by substrate (blue=water, orange=land).
    Reveals whether the NCAP limit cycle shifts shape on land.
    """
    phases_with_land = [d for d in phase_data
                        if d['substrate'].sum() > 10]
    n_phases = len(phases_with_land)
    if n_phases == 0:
        print('  ⚠️  Fig J skipped — no land segments found')
        return

    fig, axes = plt.subplots(1, n_phases,
                             figsize=(4.0 * n_phases, 4.0), sharey=True)
    if n_phases == 1:
        axes = [axes]
    fig.suptitle(f'{label} — Phase Portrait: Water (blue) vs Land (orange)',
                 fontsize=11, fontweight='bold')

    for ax, d in zip(axes, phases_with_land):
        jp  = d['joint_pos'][:, 0]
        jv  = d['joint_vel'][:, 0]
        sub = d['substrate']
        phase_idx = int(d['phase_progress'] * 4)

        for mask, col, lbl in [(sub == 0, _WATER_C, 'water'),
                                (sub == 1, _LAND_C,  'land')]:
            if mask.sum() > 5:
                ax.scatter(jp[mask], jv[mask], c=col, s=2,
                           alpha=0.6, linewidths=0, label=lbl,
                           rasterized=True)

        ax.set_title(_PHASE_LABELS[phase_idx], fontsize=9, fontweight='bold')
        ax.set_xlabel('J1 angle  (rad)', fontsize=8)
        ax.spines[['top', 'right']].set_visible(False)
        ax.axhline(0, color='grey', lw=0.4, ls=':')
        ax.axvline(0, color='grey', lw=0.4, ls=':')
        ax.legend(fontsize=7, frameon=False, markerscale=4)

    axes[0].set_ylabel('J1 angular velocity  (rad/step)', fontsize=9)
    plt.tight_layout()
    _save(fig, save_path)


# ── Fig K — Kymograph split by substrate ─────────────────────────────────────

def fig_K_substrate_kymograph(phase_data: list, save_path: str,
                               label: str = 'NCAP', seg_len: int = 200):
    """
    For each phase that has both water and land segments, extract the first
    seg_len consecutive water steps and the first seg_len consecutive land
    steps, then plot them side-by-side.  This shows directly whether the
    body wave shape differs between substrates.
    """
    phases_with_land = [d for d in phase_data
                        if d['substrate'].sum() > 20]
    if not phases_with_land:
        print('  ⚠️  Fig K skipped — no land segments found')
        return

    n_phases = len(phases_with_land)
    fig, axes = plt.subplots(n_phases, 2,
                             figsize=(8, 3.0 * n_phases),
                             sharey=True, sharex=True)
    if n_phases == 1:
        axes = axes.reshape(1, 2)
    fig.suptitle(f'{label} — Body Wave Kymograph: Water vs Land',
                 fontsize=11, fontweight='bold')

    for row, d in enumerate(phases_with_land):
        sub = d['substrate']
        jp  = d['joint_pos']
        n_joints = jp.shape[1]
        phase_idx = int(d['phase_progress'] * 4)

        for col, (medium, mask_val, colour, title) in enumerate([
            ('Water', 0, 'Blues',  'Water substrate'),
            ('Land',  1, 'Oranges','Land substrate'),
        ]):
            ax = axes[row, col]
            idx = np.where(sub == mask_val)[0]
            if len(idx) < seg_len:
                ax.text(0.5, 0.5, f'< {seg_len} steps on {medium}',
                        transform=ax.transAxes, ha='center', va='center',
                        fontsize=9, color='grey')
                ax.set_title(f'{_PHASE_LABELS[phase_idx]} — {title}',
                             fontsize=8)
                continue

            # Take first contiguous block of at least seg_len steps
            seg = jp[idx[:seg_len]]
            seg_norm = (seg - seg.mean(0)) / (seg.std(0) + 1e-8)

            ax.imshow(seg_norm.T, aspect='auto', origin='lower',
                      cmap=colour, vmin=-2, vmax=2,
                      extent=[0, seg_len, 0, n_joints])
            ax.set_title(f'{_PHASE_LABELS[phase_idx]}\n{title}',
                         fontsize=8, fontweight='bold')
            ax.set_yticks(np.arange(n_joints) + 0.5)
            ax.set_yticklabels([f'J{i+1}' for i in range(n_joints)],
                               fontsize=6)

        axes[row, 0].set_ylabel('joint', fontsize=8)

    for ax in axes[-1]:
        ax.set_xlabel('time  (steps)', fontsize=8)

    plt.tight_layout()
    _save(fig, save_path)


# ── Fig L — Reward per training phase ────────────────────────────────────────

def fig_L_reward_by_phase(runs: list, save_path: str,
                           num_steps: int = 600):
    """
    For each algorithm run, evaluate the final checkpoint in each of the 4
    progressive phases and plot a grouped bar chart of mean episode return.
    Quantifies how much reward the fixed NCAP motor pattern loses as land
    zones are introduced.
    """
    fig, ax = plt.subplots(figsize=(9, 4.5))
    fig.suptitle('NCAP Reward by Training Phase — Progressive Environment',
                 fontsize=11, fontweight='bold')

    try:
        from swimmer.training.ncap_progressive_env import NCAPProgressiveGymEnv
    except ImportError:
        print('  ⚠️  Fig L skipped — ncap_progressive_env not importable')
        _save(fig, save_path)
        return

    n_runs   = len(runs)
    n_phases = 4
    x        = np.arange(n_phases)
    width    = 0.7 / max(n_runs, 1)

    for i, run in enumerate(runs):
        ckpts = _list_checkpoints(run['checkpoint_dir'])
        if not ckpts:
            continue
        try:
            model, step = _load_checkpoint(
                ckpts[-1], run['n_links'],
                run['oscillator_period'], run['algorithm']
            )
        except Exception as e:
            print(f'  ⚠️  {run["label"]}: checkpoint load failed: {e}')
            continue

        env  = NCAPProgressiveGymEnv(n_links=run['n_links'])
        phase_returns = []

        for phase_idx in range(n_phases):
            progress = (phase_idx + 0.5) / 4.0
            env.training_progress = progress
            episode_rewards = []

            for _ in range(5):   # 5 eval episodes per phase
                obs    = env.reset()
                ep_ret = 0.0
                from swimmer.utils.simple_ncap_logger import _VideoAgent
                agent = _VideoAgent(ncap_model=model, n_joints=run['n_links'] - 1)
                def _flat(o):
                    if isinstance(o, dict):
                        j = np.asarray(o.get('joints', []), dtype=np.float32).ravel()
                        v = np.asarray(o.get('body_velocities', []), dtype=np.float32).ravel()
                        return np.concatenate([j, v])
                    return np.asarray(o, dtype=np.float32).ravel()

                for _ in range(num_steps):
                    obs              = _flat(obs)
                    action           = agent.test_step(obs)
                    obs, rew, done, _ = env.step(action)
                    ep_ret          += rew
                    if done:
                        break
                episode_rewards.append(ep_ret)

            phase_returns.append((np.mean(episode_rewards),
                                   np.std(episode_rewards)))

        means = [m for m, s in phase_returns]
        stds  = [s for m, s in phase_returns]
        offset = (i - n_runs / 2 + 0.5) * width
        ax.bar(x + offset, means, width, yerr=stds,
               color=run.get('colour', _DEFAULT_COLS[i % len(_DEFAULT_COLS)]),
               label=run['label'], capsize=4, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(_PHASE_LABELS, fontsize=8)
    ax.set_ylabel('episode return', fontsize=10)
    ax.legend(fontsize=9, frameon=False)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', alpha=0.2)
    plt.tight_layout()
    _save(fig, save_path)


# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────

def _save(fig, path: str):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'✅  {path}')


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='NCAP analysis figures A–H',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--checkpoint_dir', required=True,
                        help='Colon-separated list of checkpoint directories '
                             '(one per algorithm for multi-algo figures)')
    parser.add_argument('--labels', default='PPO',
                        help='Colon-separated algorithm labels (default: PPO)')
    parser.add_argument('--colours', default='',
                        help='Colon-separated hex colours (auto if empty)')
    parser.add_argument('--out_dir', default='outputs/ncap_analysis')
    parser.add_argument('--n_links', type=int, default=6)
    parser.add_argument('--oscillator_period', type=int, default=60)
    parser.add_argument('--num_rollout_steps', type=int, default=1200,
                        help='Steps per evaluation rollout (default 1200)')
    parser.add_argument('--kymograph_steps', type=str, default='',
                        help='Comma-separated training steps for Fig B '
                             '(default: first, mid, last checkpoint)')
    parser.add_argument('--progressive', action='store_true',
                        help='Run substrate-aware figures I-L '
                             '(requires ncap_progressive_env.py)')
    parser.add_argument('--figs', default='A,B,C,D,E,F,G,H',
                        help='Comma-separated figures to generate (default: all)')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    figs_wanted = {f.strip().upper() for f in args.figs.split(',')}

    # Parse multi-run config
    dirs    = args.checkpoint_dir.split(':')
    labels  = args.labels.split(':')
    colours_raw = args.colours.split(':') if args.colours else []
    while len(labels)  < len(dirs): labels.append(f'Run{len(labels)+1}')
    while len(colours_raw) < len(dirs):
        colours_raw.append(_DEFAULT_COLS[len(colours_raw) % len(_DEFAULT_COLS)])

    runs = []
    for d, lbl, col in zip(dirs, labels, colours_raw):
        col_resolved = ALG_COLOURS.get(lbl.upper(), col)
        runs.append({
            'checkpoint_dir':    d,
            'label':             lbl,
            'colour':            col_resolved,
            'n_links':           args.n_links,
            'oscillator_period': args.oscillator_period,
            'algorithm':         lbl.lower(),
        })

    # Use first run for single-run figures (A, B, C, F)
    r0       = runs[0]
    ckpts_r0 = _list_checkpoints(r0['checkpoint_dir'])

    if not ckpts_r0:
        print(f'❌  No checkpoints found in {r0["checkpoint_dir"]}')
        sys.exit(1)

    print(f'Found {len(ckpts_r0)} checkpoints in {r0["checkpoint_dir"]}')

    # ── load final model once for figures that need a single episode ──────
    final_model, final_step = _load_checkpoint(
        ckpts_r0[-1], r0['n_links'], r0['oscillator_period'], r0['algorithm']
    )

    # ── Fig A ─────────────────────────────────────────────────────────────
    if 'A' in figs_wanted:
        print('\n📊 Fig A — Phase portrait…')
        # For progressive runs: merge a water rollout + a land rollout so
        # both substrates appear in the portrait, colour-coded blue/orange.
        if args.progressive:
            try:
                from swimmer.training.ncap_progressive_env import NCAPProgressiveGymEnv
                _penv = NCAPProgressiveGymEnv(n_links=r0['n_links'])
                half  = args.num_rollout_steps // 2
                d_w = _rollout_episode(final_model, r0['n_links'],
                                       num_steps=half,
                                       progressive_env=_penv,
                                       training_progress=0.45,
                                       start_pos=None)
                d_l = _rollout_episode(final_model, r0['n_links'],
                                       num_steps=half,
                                       progressive_env=_penv,
                                       training_progress=0.45,
                                       start_pos=[3.0, 0.0])
                data = {k: np.concatenate([d_w[k], d_l[k]], axis=0)
                        for k in d_w}
            except Exception as _e:
                print(f'  ⚠️  progressive rollout failed ({_e}), falling back')
                data = _rollout_episode(final_model, r0['n_links'],
                                        num_steps=args.num_rollout_steps)
        else:
            data = _rollout_episode(final_model, r0['n_links'],
                                    num_steps=args.num_rollout_steps)
        fig_A_phase_portrait(
            data,
            os.path.join(args.out_dir, 'figA_phase_portrait.png'),
            title=f'{r0["label"]} — Phase Portrait (step {final_step:,})'
        )

    # ── Fig B ─────────────────────────────────────────────────────────────
    if 'B' in figs_wanted:
        print('\n📊 Fig B — Kymograph at checkpoints…')
        if args.kymograph_steps:
            ky_steps = [int(s) for s in args.kymograph_steps.split(',')]
        else:
            def _ckpt_step(p):
                import re as _re
                b = os.path.basename(p)
                for pat in [r'_checkpoint_step_(\d+)$', r'checkpoint_(\d+)$']:
                    m = _re.search(pat, b)
                    if m:
                        return int(m.group(1))
                return 0
            n = len(ckpts_r0)
            ky_steps = [
                _ckpt_step(ckpts_r0[0]),
                _ckpt_step(ckpts_r0[n // 2]),
                _ckpt_step(ckpts_r0[-1]),
            ]
        fig_B_kymograph_checkpoints(
            r0['checkpoint_dir'], ky_steps,
            r0['n_links'], r0['oscillator_period'], r0['algorithm'],
            os.path.join(args.out_dir, 'figB_kymograph_checkpoints.png'),
            num_steps=min(400, args.num_rollout_steps)
        )

    # ── Fig C ─────────────────────────────────────────────────────────────
    if 'C' in figs_wanted:
        print('\n📊 Fig C — Period distribution…')
        if 'A' not in figs_wanted:   # reuse data if A was already run
            data = _rollout_episode(final_model, r0['n_links'],
                                    num_steps=args.num_rollout_steps)
        fig_C_period_distribution(
            data, r0['oscillator_period'],
            os.path.join(args.out_dir, 'figC_period_distribution.png'),
            title=f'{r0["label"]} — CPG Period Distribution'
        )

    # ── Fig D ─────────────────────────────────────────────────────────────
    if 'D' in figs_wanted and len(runs) > 1:
        print('\n📊 Fig D — Algorithm kymograph overlay…')
        d_runs = []
        for run in runs:
            ckpts = _list_checkpoints(run['checkpoint_dir'])
            if ckpts:
                d_runs.append({**run, 'checkpoint_path': ckpts[-1]})
        if d_runs:
            fig_D_algorithm_kymographs(
                d_runs,
                os.path.join(args.out_dir, 'figD_algorithm_kymographs.png'),
                num_steps=min(400, args.num_rollout_steps)
            )
    elif 'D' in figs_wanted:
        print('⚠️  Fig D needs multiple --checkpoint_dir entries, skipping.')

    # ── Fig E ─────────────────────────────────────────────────────────────
    if 'E' in figs_wanted:
        print('\n📊 Fig E — Weight trajectories…')
        fig_E_weight_trajectory(
            runs,
            os.path.join(args.out_dir, 'figE_weight_trajectory.png')
        )

    # ── Fig F ─────────────────────────────────────────────────────────────
    if 'F' in figs_wanted:
        print('\n📊 Fig F — Amplitude-frequency scatter…')
        if 'A' not in figs_wanted and 'C' not in figs_wanted:
            data = _rollout_episode(final_model, r0['n_links'],
                                    num_steps=args.num_rollout_steps)
        fig_F_amplitude_frequency(
            data,
            os.path.join(args.out_dir, 'figF_amplitude_frequency.png')
        )

    # ── Fig G ─────────────────────────────────────────────────────────────
    if 'G' in figs_wanted:
        print('\n📊 Fig G — Wave propagation speed vs checkpoint…')
        fig_G_wave_propagation(
            runs,
            os.path.join(args.out_dir, 'figG_wave_propagation.png'),
            num_steps=min(600, args.num_rollout_steps)
        )

    # ── Fig H ─────────────────────────────────────────────────────────────
    if 'H' in figs_wanted:
        print('\n📊 Fig H — Weight convergence…')
        fig_H_weight_convergence(
            runs,
            os.path.join(args.out_dir, 'figH_weight_convergence.png')
        )

    # ── Figs I-L (progressive / substrate-aware) ────────────────────────────
    if args.progressive or any(f in figs_wanted for f in ('I','J','K','L')):
        print('\n📊 Progressive substrate figures I-L…')
        # water_only_fallback=True when model was trained without land zones.
        # Figures still generate but show water-only substrate in all phases,
        # providing a clean baseline for comparison with progressive runs.
        prog_data = _rollout_progressive(
            final_model, r0['n_links'], r0['oscillator_period'],
            num_steps=min(args.num_rollout_steps, 800),
            water_only_fallback=not args.progressive
        )
        if prog_data is None:
            print('  ⚠️  ncap_progressive_env not available — skipping I-L')
        else:
            if 'I' in figs_wanted or args.progressive:
                print('  Fig I — Substrate speed comparison…')
                fig_I_substrate_speed(
                    prog_data,
                    os.path.join(args.out_dir, 'figI_substrate_speed.png'),
                    label=r0['label']
                )
            if 'J' in figs_wanted or args.progressive:
                print('  Fig J — Phase portrait water vs land…')
                fig_J_substrate_phase_portrait(
                    prog_data,
                    os.path.join(args.out_dir, 'figJ_substrate_phase_portrait.png'),
                    label=r0['label']
                )
            if 'K' in figs_wanted or args.progressive:
                print('  Fig K — Kymograph water vs land…')
                fig_K_substrate_kymograph(
                    prog_data,
                    os.path.join(args.out_dir, 'figK_substrate_kymograph.png'),
                    label=r0['label']
                )
            if 'L' in figs_wanted or args.progressive:
                print('  Fig L — Reward by phase…')
                fig_L_reward_by_phase(
                    runs,
                    os.path.join(args.out_dir, 'figL_reward_by_phase.png'),
                    num_steps=min(args.num_rollout_steps, 600)
                )

    print(f'\n✅  All requested figures saved to: {args.out_dir}')


if __name__ == '__main__':
    main()
