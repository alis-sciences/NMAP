#!/usr/bin/env python3
"""
CCMN Diagnostic Utilities
==========================
Implements three standalone diagnostic tools that can be called before or
during curriculum training to verify that the CCMN internal machinery is
working correctly.  All functions are pure data + plotting — they do not
modify model weights.

Public API
----------
run_ccmn_sanity_check(agent, env, num_steps, label)
    → dict  (raw time-series log)

run_film_ablation_comparison(agent, env, ablated_agent, num_steps)
    → dict  (comparison metrics)

plot_ccmn_diagnostic_summary(log, save_dir)
    → None  (saves 3 PNG files answering the three diagnostic questions)
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings("ignore")

import torch


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _unwrap_ccmn(agent):
    """Return the raw CCMNSwimmer from an agent, unwrapping DataParallel."""
    model = agent.ncap_model
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def _step_ccmn_single(actual_model, obs_np, agent, t: int, device) -> dict:
    """
    Run one forward step through the unwrapped CCMNSwimmer and return a dict
    containing action, z_DA, z_5HT, gamma, beta, period.
    """
    n_joints = agent.n_joints
    env_start = agent.env_features_start

    jp = torch.tensor(obs_np[:n_joints], dtype=torch.float32,
                      device=device).unsqueeze(0)
    vis = float(obs_np[env_start]) if len(obs_np) > env_start else 0.0
    t_tensor = torch.tensor([t], dtype=torch.float32, device=device)

    with torch.no_grad():
        action = actual_model(jp, viscosity_norm=vis, timesteps=t_tensor)

    nm = actual_model.neuromod_state()
    return dict(
        action=action.squeeze(0).cpu().numpy(),
        z_DA=nm['z_DA'],
        z_5HT=nm['z_5HT'],
        gamma=np.array(nm['gamma'], dtype=np.float32),
        beta=np.array(nm['beta'],  dtype=np.float32),
        period=nm['period'],
        viscosity=vis,
    )


def _env_label(obs_np, agent) -> int:
    """Return 1 if the current observation indicates land, else 0."""
    idx = agent.env_features_start + 2
    return int(obs_np[idx] > 0.5) if len(obs_np) > idx else 0


# ─────────────────────────────────────────────────────────────────────────────
# 1.  SANITY CHECK
# ─────────────────────────────────────────────────────────────────────────────

def run_ccmn_sanity_check(agent, env, num_steps: int = 600,
                           label: str = 'sanity') -> dict:
    """
    Roll out *num_steps* steps and collect every CCMN internal signal.

    Returns a dict with keys:
        z_DA, z_5HT : (T,)    neuromodulatory context signals
        gamma        : (T, J)  FiLM gain per joint
        beta         : (T, J)  FiLM bias per joint
        periods      : (T,)    effective oscillator period
        env_labels   : (T,)    0=water, 1=land
        viscosity    : (T,)    normalised viscosity seen at each step
        speeds       : (T,)    approximate axial speed
        label        : str     name tag for this run
        period_bimodal : bool  True if period histogram shows two peaks
        z_DA_water_mean  : float
        z_DA_land_mean   : float
        z_DA_contrast    : float  |land_mean - water_mean|
        gamma_identity   : bool  True if all γ are within 0.05 of 1.0
    """
    actual = _unwrap_ccmn(agent)
    actual.reset()
    device = next(actual.parameters()).device

    z_DA_log, z_5HT_log, gamma_log, beta_log = [], [], [], []
    period_log, env_label_log, vis_log, speed_log = [], [], [], []

    obs = env.reset()

    for t in range(num_steps):
        if not isinstance(obs, np.ndarray):
            obs = np.asarray(obs, dtype=np.float32)

        step_data = _step_ccmn_single(actual, obs, agent, t, device)

        # Approximate speed from environment physics if available
        try:
            phys = env.env.env.physics
            vel  = phys.named.data.cvel['head']
            speed = float(np.linalg.norm(vel[3:5]))
        except Exception:
            speed = 0.0

        z_DA_log.append(step_data['z_DA'])
        z_5HT_log.append(step_data['z_5HT'])
        gamma_log.append(step_data['gamma'])
        beta_log.append(step_data['beta'])
        period_log.append(step_data['period'])
        env_label_log.append(_env_label(obs, agent))
        vis_log.append(step_data['viscosity'])
        speed_log.append(speed)

        # Advance environment using the agent's test_step
        action = agent.test_step(obs)
        obs, _, done, _ = env.step(action)
        if done:
            obs = env.reset()
            actual.reset()

    z_DA   = np.array(z_DA_log,   dtype=np.float32)
    z_5HT  = np.array(z_5HT_log,  dtype=np.float32)
    gamma  = np.array(gamma_log,  dtype=np.float32)   # (T, J)
    beta   = np.array(beta_log,   dtype=np.float32)
    periods = np.array(period_log, dtype=np.float32)
    env_lbl = np.array(env_label_log, dtype=np.int32)
    vis     = np.array(vis_log,   dtype=np.float32)
    speeds  = np.array(speed_log, dtype=np.float32)

    # ── Diagnostic metrics ───────────────────────────────────────────────────

    # 1. Does z_DA vary between water and land?
    water_mask = env_lbl == 0
    land_mask  = env_lbl == 1
    z_da_water = float(z_DA[water_mask].mean()) if water_mask.sum() > 0 else 0.0
    z_da_land  = float(z_DA[land_mask].mean())  if land_mask.sum()  > 0 else 0.0
    z_da_contrast = abs(z_da_land - z_da_water)

    # 2. Do oscillator periods show bimodal structure?
    #    Detect two peaks in the period histogram.
    bins = np.linspace(periods.min() - 1, periods.max() + 1, 30)
    hist, edges = np.histogram(periods, bins=bins, density=True)
    hist_smooth = gaussian_filter1d(hist.astype(float), sigma=1.5)
    peaks, _ = find_peaks(hist_smooth, height=hist_smooth.max() * 0.15,
                           distance=3)
    period_bimodal = len(peaks) >= 2

    # 3. Are γ values all near 1.0?
    gamma_identity = bool(np.abs(gamma - 1.0).max() < 0.05)

    return dict(
        z_DA=z_DA, z_5HT=z_5HT,
        gamma=gamma, beta=beta,
        periods=periods, env_labels=env_lbl,
        viscosity=vis, speeds=speeds,
        label=label,
        period_bimodal=period_bimodal,
        z_DA_water_mean=z_da_water,
        z_DA_land_mean=z_da_land,
        z_DA_contrast=z_da_contrast,
        gamma_identity=gamma_identity,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2.  FiLM ABLATION COMPARISON
# ─────────────────────────────────────────────────────────────────────────────

def run_film_ablation_comparison(agent, env,
                                  ablated_agent,
                                  num_steps: int = 800) -> dict:
    """
    Run both the full CCMN agent and the FiLM-ablated agent on the same
    environment seed and compare performance.

    Parameters
    ----------
    agent         : full CCMN agent (FiLM active)
    ablated_agent : CCMN agent with γ=1, β=0 frozen
    env           : single environment (will be reset for each agent)
    num_steps     : steps per evaluation episode

    Returns
    -------
    dict with keys:
        full_rewards, full_distances, full_speeds      : (T,) arrays
        ablated_rewards, ablated_distances, ablated_speeds : (T,)
        full_total_reward, ablated_total_reward        : float
        full_total_distance, ablated_total_distance    : float
        improvement_pct                                : float  (full vs ablated)
    """
    results = {}

    for tag, ag in [('full', agent), ('ablated', ablated_agent)]:
        actual = _unwrap_ccmn(ag)
        actual.reset()
        obs = env.reset()

        rewards, distances, speeds = [], [], []
        cum_reward = 0.0

        try:
            init_pos = env.env.env.physics.named.data.xpos['head'][:2].copy()
        except Exception:
            init_pos = np.zeros(2)

        for t in range(num_steps):
            if not isinstance(obs, np.ndarray):
                obs = np.asarray(obs, dtype=np.float32)

            action = ag.test_step(obs)
            obs, reward, done, _ = env.step(action)

            try:
                pos  = env.env.env.physics.named.data.xpos['head'][:2]
                vel  = env.env.env.physics.named.data.cvel['head']
                speed = float(np.linalg.norm(vel[3:5]))
                dist  = float(np.linalg.norm(pos - init_pos))
            except Exception:
                speed = 0.0
                dist  = 0.0

            cum_reward += float(reward)
            rewards.append(float(reward))
            distances.append(dist)
            speeds.append(speed)

            if done:
                obs = env.reset()
                actual.reset()
                try:
                    init_pos = env.env.env.physics.named.data.xpos['head'][:2].copy()
                except Exception:
                    init_pos = np.zeros(2)

        results[f'{tag}_rewards']       = np.array(rewards,   dtype=np.float32)
        results[f'{tag}_distances']     = np.array(distances, dtype=np.float32)
        results[f'{tag}_speeds']        = np.array(speeds,    dtype=np.float32)
        results[f'{tag}_total_reward']  = cum_reward
        results[f'{tag}_total_distance']= float(np.array(distances).max()
                                                 if distances else 0.0)

    full_d    = results['full_total_distance']
    ablated_d = results['ablated_total_distance']
    results['improvement_pct'] = (
        100.0 * (full_d - ablated_d) / (ablated_d + 1e-8)
    )
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 3.  DIAGNOSTIC PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def plot_ccmn_diagnostic_summary(log: dict, save_dir: str) -> None:
    """
    Save three focused diagnostic plots answering the three sanity questions:

    diagnostic_1_zDA_by_environment.png
        Does z_DA vary between water and land?
        → z_DA time-series colour-coded by env + box-plot comparison.

    diagnostic_2_period_bimodality.png
        Do oscillator periods show bimodal structure?
        → Period histogram with peak markers + KDE, split by environment.

    diagnostic_3_gamma_identity.png
        Are γ values all near 1.0 (expected for untrained model)?
        → γ distribution per joint + heatmap over time.
    """
    os.makedirs(save_dir, exist_ok=True)

    z_DA    = log['z_DA']
    gamma   = log['gamma']          # (T, J)
    periods = log['periods']
    env_lbl = log['env_labels']
    T, n_joints = gamma.shape
    time = np.arange(T)

    WATER_COLOR = '#2A5FA5'   # blue
    LAND_COLOR  = '#E07B2A'   # orange

    # ── Plot 1: z_DA vs environment ─────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 4),
                              gridspec_kw={'width_ratios': [3, 1]})

    ax0 = axes[0]
    # Coloured background strips (water/land)
    for t in range(T - 1):
        col = WATER_COLOR if env_lbl[t] == 0 else LAND_COLOR
        ax0.axvspan(t, t + 1, color=col, alpha=0.12, linewidth=0)
    ax0.plot(time, z_DA, color='black', lw=1.0)
    ax0.axhline(0, color='grey', lw=0.8, ls='--')
    ax0.set_xlim(0, T)
    ax0.set_ylim(-1.1, 1.1)
    ax0.set_xlabel('step', fontsize=10)
    ax0.set_ylabel('z_DA', fontsize=10)
    ax0.set_title(f'z_DA over time  –  {log.get("label", "")}', fontsize=11,
                  fontweight='bold')
    # Legend patches
    from matplotlib.patches import Patch
    ax0.legend(handles=[Patch(color=WATER_COLOR, alpha=0.5, label='Water'),
                         Patch(color=LAND_COLOR,  alpha=0.5, label='Land')],
               loc='upper right', fontsize=9)

    # Annotate contrast
    contrast = log.get('z_DA_contrast', 0.0)
    w_mean   = log.get('z_DA_water_mean', 0.0)
    l_mean   = log.get('z_DA_land_mean',  0.0)
    ax0.text(0.02, 0.05,
             f'Water z_DA: {w_mean:+.3f}\n'
             f'Land  z_DA: {l_mean:+.3f}\n'
             f'Contrast:   {contrast:.3f}',
             transform=ax0.transAxes, fontsize=9,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax1 = axes[1]
    water_mask = env_lbl == 0
    land_mask  = env_lbl == 1
    data_bp = [z_DA[water_mask] if water_mask.sum() > 0 else np.array([0.0]),
               z_DA[land_mask]  if land_mask.sum()  > 0 else np.array([0.0])]
    bp = ax1.boxplot(data_bp, patch_artist=True, widths=0.5,
                     medianprops=dict(color='black', lw=2))
    for patch, color in zip(bp['boxes'], [WATER_COLOR, LAND_COLOR]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax1.set_xticklabels(['Water', 'Land'], fontsize=10)
    ax1.set_ylabel('z_DA', fontsize=10)
    ax1.set_ylim(-1.1, 1.1)
    ax1.axhline(0, color='grey', lw=0.8, ls='--')
    ax1.set_title('z_DA distribution\nby environment', fontsize=10,
                  fontweight='bold')
    ax1.spines[['right', 'top']].set_visible(False)

    # Pass/fail annotation
    result_txt = (f"✅ PASS — contrast={contrast:.3f}"
                  if contrast > 0.05 else
                  f"⚠️  LOW — contrast={contrast:.3f} (<0.05)")
    fig.text(0.5, -0.02, result_txt, ha='center', fontsize=10,
             color='green' if contrast > 0.05 else 'darkorange')

    fig.suptitle('Diagnostic 1 — Does z_DA vary between water and land?',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    p1 = os.path.join(save_dir, 'diagnostic_1_zDA_by_environment.png')
    fig.savefig(p1, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'   📊 Diagnostic 1 saved: {p1}')

    # ── Plot 2: Period bimodality ────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax0 = axes[0]
    bins = np.linspace(max(0, periods.min() - 2), periods.max() + 2, 30)
    for mask, color, label in [(water_mask, WATER_COLOR, 'Water'),
                                (land_mask,  LAND_COLOR,  'Land')]:
        if mask.sum() > 0:
            ax0.hist(periods[mask], bins=bins, color=color, alpha=0.6,
                     label=label, density=True, edgecolor='none')
    # KDE overlay
    from scipy.stats import gaussian_kde
    if len(periods) > 5:
        kde_x = np.linspace(bins[0], bins[-1], 300)
        try:
            kde = gaussian_kde(periods.astype(float), bw_method=0.3)
            ax0.plot(kde_x, kde(kde_x), color='black', lw=2, label='KDE (all)')
        except Exception:
            pass

    # Mark biological targets
    ax0.axvline(15, color='navy',   lw=1.5, ls=':', alpha=0.8, label='Swim target (15)')
    ax0.axvline(60, color='sienna', lw=1.5, ls=':', alpha=0.8, label='Crawl target (60)')
    ax0.set_xlabel('oscillator period  (steps)', fontsize=10)
    ax0.set_ylabel('density', fontsize=10)
    ax0.set_title('Period histogram\n(bimodal = bistable switching)', fontsize=10,
                  fontweight='bold')
    ax0.legend(fontsize=8)
    ax0.spines[['right', 'top']].set_visible(False)

    ax1 = axes[1]
    ax1.plot(time, periods, color='black', lw=0.8)
    for t in range(T - 1):
        col = WATER_COLOR if env_lbl[t] == 0 else LAND_COLOR
        ax1.axvspan(t, t + 1, color=col, alpha=0.10, linewidth=0)
    ax1.axhline(15, color='navy',   lw=1.0, ls=':', alpha=0.7)
    ax1.axhline(60, color='sienna', lw=1.0, ls=':', alpha=0.7)
    ax1.set_xlim(0, T)
    ax1.set_xlabel('step', fontsize=10)
    ax1.set_ylabel('period  (steps)', fontsize=10)
    ax1.set_title('Period over time', fontsize=10, fontweight='bold')
    ax1.spines[['right', 'top']].set_visible(False)

    bimodal = log.get('period_bimodal', False)
    result_txt = ("✅ PASS — bimodal structure detected"
                  if bimodal else
                  "⚠️  FLAT — period distribution is unimodal "
                  "(bistable switching not yet learned)")
    fig.text(0.5, -0.02, result_txt, ha='center', fontsize=10,
             color='green' if bimodal else 'darkorange')

    fig.suptitle('Diagnostic 2 — Do oscillator periods show bimodal structure?',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    p2 = os.path.join(save_dir, 'diagnostic_2_period_bimodality.png')
    fig.savefig(p2, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'   📊 Diagnostic 2 saved: {p2}')

    # ── Plot 3: γ identity check ─────────────────────────────────────────────
    fig = plt.figure(figsize=(12, 5))
    gs  = gridspec.GridSpec(1, 2, width_ratios=[2, 3], wspace=0.3,
                             left=0.07, right=0.97, top=0.88, bottom=0.12)

    ax0 = fig.add_subplot(gs[0])
    # Box-plot per joint
    bp = ax0.boxplot([gamma[:, j] for j in range(n_joints)],
                     patch_artist=True, widths=0.5,
                     medianprops=dict(color='black', lw=2))
    for patch in bp['boxes']:
        patch.set_facecolor('#4292c6')
        patch.set_alpha(0.7)
    ax0.axhline(1.0, color='red', lw=1.5, ls='--', label='Identity (γ=1)')
    ax0.set_xlabel('joint index', fontsize=10)
    ax0.set_ylabel('γ  (FiLM gain)', fontsize=10)
    ax0.set_title('γ distribution per joint\n(red = identity)', fontsize=10,
                  fontweight='bold')
    ax0.legend(fontsize=9)
    ax0.spines[['right', 'top']].set_visible(False)

    ax1 = fig.add_subplot(gs[1])
    vmax = max(abs(gamma - 1.0).max(), 0.01)
    im = ax1.imshow((gamma - 1.0).T,
                    aspect='auto', origin='lower',
                    extent=[0, T, 0, n_joints],
                    cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    ax1.set_xlabel('step', fontsize=10)
    ax1.set_ylabel('joint index', fontsize=10)
    ax1.set_title('γ − 1 over time\n(blue=below 1, red=above 1)', fontsize=10,
                  fontweight='bold')
    cbar = fig.colorbar(im, ax=ax1, fraction=0.03, pad=0.02)
    cbar.set_label('γ − 1', fontsize=9)

    identity = log.get('gamma_identity', False)
    max_dev  = float(np.abs(gamma - 1.0).max())
    result_txt = (f"✅ IDENTITY — max |γ−1| = {max_dev:.4f} (<0.05, model untrained / ablated)"
                  if identity else
                  f"✅ ACTIVE  — max |γ−1| = {max_dev:.4f} (FiLM layer is modulating)")
    fig.text(0.5, -0.01, result_txt, ha='center', fontsize=10,
             color='steelblue' if identity else 'green')

    fig.suptitle('Diagnostic 3 — Are γ values near 1.0 (identity)?',
                 fontsize=12, fontweight='bold')
    p3 = os.path.join(save_dir, 'diagnostic_3_gamma_identity.png')
    fig.savefig(p3, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'   📊 Diagnostic 3 saved: {p3}')


# ─────────────────────────────────────────────────────────────────────────────
# 4.  FiLM ABLATION COMPARISON PLOT
# ─────────────────────────────────────────────────────────────────────────────

def plot_film_ablation_comparison(comparison: dict, save_dir: str) -> None:
    """
    Plot cumulative reward and distance for full-CCMN vs FiLM-ablated run.
    Saves ablation_comparison.png in save_dir.
    """
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    T = len(comparison['full_rewards'])
    time = np.arange(T)

    for ax, key, ylabel in [
            (axes[0], 'rewards',   'cumulative reward'),
            (axes[1], 'distances', 'distance from start  (m)')]:
        full_data    = comparison[f'full_{key}']
        ablated_data = comparison[f'ablated_{key}']
        if key == 'rewards':
            full_data    = np.cumsum(full_data)
            ablated_data = np.cumsum(ablated_data)
        ax.plot(time, full_data,    color='#2CA02C', lw=2.0, label='Full CCMN (FiLM active)')
        ax.plot(time, ablated_data, color='#D62728', lw=2.0, ls='--', label='Ablated (γ=1, β=0)')
        ax.set_xlabel('step', fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.legend(fontsize=9)
        ax.spines[['right', 'top']].set_visible(False)

    imp = comparison.get('improvement_pct', 0.0)
    fig.suptitle(
        f'FiLM Ablation Comparison  —  Full CCMN vs γ=1,β=0 frozen\n'
        f'Distance improvement of full CCMN: {imp:+.1f}%',
        fontsize=11, fontweight='bold'
    )
    plt.tight_layout()
    path = os.path.join(save_dir, 'ablation_comparison.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'   📊 Ablation comparison plot saved: {path}')
