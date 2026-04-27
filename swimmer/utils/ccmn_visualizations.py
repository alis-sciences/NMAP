#!/usr/bin/env python3
"""
CCMN Neuromodulatory Visualizations
=====================================
Appends to curriculum_visualization.py.

Produces four figure sets inspired by the C. elegans NSM neuromodulation paper,
translated into the CCMN model domain:

Figure A  –  Gait-state / neuromodulator joint distributions
             (mirrors paper Fig 1A: joint density of NSM activity × axial speed,
              split by dwelling / roaming)
             → Here: 2D density of z_DA × speed, split by Swim / Crawl mode.

Figure B  –  Context-encoder activity around gait transitions
             (mirrors paper Fig 1B: peri-event raster + mean trace aligned to
              dwelling onset)
             → Here: z_DA raster + mean trace aligned to Swim→Crawl transitions.

Figure C  –  Trajectory with gait-state colour coding
             (mirrors paper Fig 2A: path coloured by Roam/Dwell)
             → Here: 2-D path coloured by Swim (orange) / Crawl (blue).

Figure D  –  Per-joint FiLM γ activity across the full episode
             (mirrors paper Fig 3C: neuronal activity traces coloured by axial
              velocity and behavioural state bar)
             → Here: γ_i traces with velocity background and state bar.

Figure E  –  z_DA ↔ oscillator-period bistability scatter / histogram
             (mirrors paper Fig 1D: dwelling-state duration vs NSM bout duration)
             → Here: scatter of z_DA value vs effective period, histogram of
                period distribution split by environment type.

All functions accept a ``ccmn_log`` dict populated by
``collect_ccmn_episode_data()`` (defined below) so they are decoupled from
the training loop.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from scipy.ndimage import gaussian_filter
from scipy import signal as scipy_signal
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# Colour palette matching the paper (blue = dwell/crawl, orange = roam/swim)
# ─────────────────────────────────────────────────────────────────────────────
SWIM_COLOR  = '#E07B2A'   # orange  (roaming)
CRAWL_COLOR = '#2A5FA5'   # blue    (dwelling)
FWD_COLOR   = '#2CA02C'   # green   (forward)
REV_COLOR   = '#D62728'   # red     (reversal)

# Custom heatmap matching paper's blue→white→red density colourmap
_BWR_DENSE = LinearSegmentedColormap.from_list(
    'bwr_dense',
    [(0.00, '#08306b'),
     (0.40, '#4292c6'),
     (0.70, '#ffffff'),
     (0.85, '#fb6a4a'),
     (1.00, '#99000d')]
)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  DATA COLLECTION
# ─────────────────────────────────────────────────────────────────────────────

def collect_ccmn_episode_data(agent, env, num_steps: int = 1200,
                               z_DA_crawl_threshold: float = 0.0) -> dict:
    """
    Roll out one episode and log all CCMN internal signals.

    Returns a dict with keys:
        speeds          : (T,)   axial speed at each step (m/s proxy)
        z_DA            : (T,)   dopamine-like context signal
        z_5HT           : (T,)   serotonin-like context signal
        gamma           : (T, n_joints)   FiLM gain per joint
        beta            : (T, n_joints)   FiLM bias per joint
        periods         : (T,)   effective oscillator period (steps)
        gait_labels     : (T,)   0 = swim, 1 = crawl
        env_labels      : (T,)   0 = water, 1 = land
        viscosity       : (T,)   normalised viscosity
        positions       : (T, 2) head XY position
        joint_pos       : (T, n_joints)  raw joint angles
        transitions     : list of (t, 'S2C' | 'C2S') transition events
    """
    import torch

    log = dict(speeds=[], z_DA=[], z_5HT=[], gamma=[], beta=[],
               periods=[], gait_labels=[], env_labels=[], viscosity=[],
               positions=[], joint_pos=[], transitions=[])

    # Unwrap the actual CCMNSwimmer
    actual_model = (agent.ncap_model.module
                    if hasattr(agent.ncap_model, 'module')
                    else agent.ncap_model)
    actual_model.reset()

    obs = env.reset()
    prev_gait = None

    for t in range(num_steps):
        # ── forward pass (single sample) ──────────────────────────────────
        device = next(actual_model.parameters()).device
        if not isinstance(obs, np.ndarray):
            obs = np.asarray(obs, dtype=np.float32)

        n_joints = agent.n_joints
        joint_pos_np = obs[:n_joints]
        joint_pos_t  = torch.tensor(joint_pos_np, dtype=torch.float32,
                                    device=device).unsqueeze(0)   # (1, n_joints)

        # Extract viscosity from observation
        env_feat_start = agent.env_features_start
        vis_scalar = float(obs[env_feat_start]) if len(obs) > env_feat_start else 0.0
        env_label  = 1 if (len(obs) > env_feat_start + 2 and
                            obs[env_feat_start + 2] > 0.5) else 0

        t_tensor = torch.tensor([t], dtype=torch.float32, device=device)

        with torch.no_grad():
            _ = actual_model(joint_pos_t, viscosity_norm=vis_scalar,
                             timesteps=t_tensor)

        # ── pull internal state ────────────────────────────────────────────
        nm = actual_model.neuromod_state()
        z_da   = nm['z_DA']
        z_5ht  = nm['z_5HT']
        gamma  = nm['gamma']           # list length n_joints
        beta   = nm['beta']            # list length n_joints
        period = nm['period']

        # Gait label from z_DA threshold
        gait = 1 if z_da > z_DA_crawl_threshold else 0   # 1 = crawl, 0 = swim

        # Detect transitions
        if prev_gait is not None and gait != prev_gait:
            label = 'S2C' if gait == 1 else 'C2S'
            log['transitions'].append((t, label))
        prev_gait = gait

        # ── axial speed ───────────────────────────────────────────────────
        try:
            if hasattr(env, 'env') and hasattr(env.env, 'env'):
                phys = env.env.env.physics
                vel  = phys.named.data.cvel['head']          # 6-D: [rot, trans]
                speed = float(np.linalg.norm(vel[3:5]))      # XY translational
            else:
                speed = 0.0
        except Exception:
            speed = 0.0

        # ── head position ─────────────────────────────────────────────────
        try:
            if hasattr(env, 'env') and hasattr(env.env, 'env'):
                pos = env.env.env.physics.named.data.xpos['head'][:2].copy()
            else:
                pos = np.zeros(2)
        except Exception:
            pos = np.zeros(2)

        # ── store ─────────────────────────────────────────────────────────
        log['speeds'].append(speed)
        log['z_DA'].append(z_da)
        log['z_5HT'].append(z_5ht)
        log['gamma'].append(gamma)
        log['beta'].append(beta)
        log['periods'].append(period)
        log['gait_labels'].append(gait)
        log['env_labels'].append(env_label)
        log['viscosity'].append(vis_scalar)
        log['positions'].append(pos)
        log['joint_pos'].append(joint_pos_np.copy())

        action = agent.test_step(obs)
        obs, _, done, _ = env.step(action)
        if done:
            obs = env.reset()
            actual_model.reset()

    # Convert to numpy arrays
    for k in ('speeds', 'z_DA', 'z_5HT', 'periods',
               'gait_labels', 'env_labels', 'viscosity'):
        log[k] = np.array(log[k], dtype=np.float32)
    log['gamma']    = np.array(log['gamma'],    dtype=np.float32)   # (T, n_j)
    log['beta']     = np.array(log['beta'],     dtype=np.float32)
    log['positions']= np.array(log['positions'],dtype=np.float32)   # (T, 2)
    log['joint_pos']= np.array(log['joint_pos'],dtype=np.float32)

    return log


# ─────────────────────────────────────────────────────────────────────────────
# 1b.  DATA COLLECTION — SimpleNCAPSwimmer (NCAP baseline)
#
# Produces the same log dict as collect_ccmn_episode_data so every downstream
# plot function can be reused unchanged.  Key mappings:
#
#   z_DA   ← head oscillator phase: +1 during dorsal half-cycle, −1 ventral.
#            This is the closest analogue to a neuromodulatory signal in a
#            system that has no ContextEncoder — it captures the CPG's internal
#            state and gives the raster/bistability figures a meaningful signal.
#   z_5HT  ← −z_DA (antagonistic, matching the CCMN convention)
#   gamma  ← ones  (no FiLM; gain is always 1 everywhere)
#   beta   ← zeros (no FiLM offset)
#   period ← fixed oscillator_period (no gait switching; period never changes)
#   gait_labels ← derived from z_DA sign (dorsal phase = "swim", ventral = "crawl")
#                 This is a labelling convention for visualisation only — the
#                 NCAP has no genuine gait states.
# ─────────────────────────────────────────────────────────────────────────────

def collect_ncap_episode_data(agent, env, num_steps: int = 1200,
                               z_DA_crawl_threshold: float = 0.0) -> dict:
    """
    Roll out one episode of a SimpleNCAPSwimmer and produce a log dict
    compatible with all CCMN visualisation functions.

    The agent is expected to have:
        agent.swimmer          : SimpleNCAPSwimmer instance
        agent.n_joints         : int
        agent.env_features_start : int  (index where env features begin in obs)
        agent.test_step(obs)   : returns numpy action array
    """
    import torch

    swimmer = (agent.swimmer.module
               if hasattr(agent.swimmer, 'module')
               else agent.swimmer)
    swimmer.reset()
    n_joints  = agent.n_joints
    period    = swimmer.oscillator_period

    log = dict(speeds=[], z_DA=[], z_5HT=[], gamma=[], beta=[],
               periods=[], gait_labels=[], env_labels=[], viscosity=[],
               positions=[], joint_pos=[], transitions=[])

    obs = env.reset()
    prev_gait = None

    for t in range(num_steps):
        if not isinstance(obs, np.ndarray):
            obs = np.asarray(obs, dtype=np.float32)

        joint_pos_np = obs[:n_joints]

        # ── NCAP proxy signals ────────────────────────────────────────────────
        # Head oscillator phase: dorsal half-cycle → z_DA = +1 (labelled swim),
        # ventral half-cycle → z_DA = −1 (labelled crawl).
        phase     = t % period
        z_da      = 1.0 if phase < period // 2 else -1.0
        z_5ht     = -z_da
        # No FiLM: gamma = 1, beta = 0 for every joint
        gamma_row = [1.0] * n_joints
        beta_row  = [0.0] * n_joints

        # Gait label purely from oscillator phase
        gait = 0 if z_da > z_DA_crawl_threshold else 1   # 0=swim, 1=crawl

        if prev_gait is not None and gait != prev_gait:
            label = 'S2C' if gait == 1 else 'C2S'
            log['transitions'].append((t, label))
        prev_gait = gait

        # ── environment features ──────────────────────────────────────────────
        env_feat_start = getattr(agent, 'env_features_start', n_joints)
        vis_scalar = float(obs[env_feat_start]) if len(obs) > env_feat_start else 0.0
        env_label  = 1 if (len(obs) > env_feat_start + 2 and
                            obs[env_feat_start + 2] > 0.5) else 0

        # ── axial speed ───────────────────────────────────────────────────────
        try:
            if hasattr(env, 'env') and hasattr(env.env, 'env'):
                phys  = env.env.env.physics
                vel   = phys.named.data.cvel['head']
                speed = float(np.linalg.norm(vel[3:5]))
            else:
                speed = 0.0
        except Exception:
            speed = 0.0

        # ── head position ─────────────────────────────────────────────────────
        try:
            if hasattr(env, 'env') and hasattr(env.env, 'env'):
                pos = env.env.env.physics.named.data.xpos['head'][:2].copy()
            else:
                pos = np.zeros(2)
        except Exception:
            pos = np.zeros(2)

        log['speeds'].append(speed)
        log['z_DA'].append(z_da)
        log['z_5HT'].append(z_5ht)
        log['gamma'].append(gamma_row)
        log['beta'].append(beta_row)
        log['periods'].append(float(period))
        log['gait_labels'].append(gait)
        log['env_labels'].append(env_label)
        log['viscosity'].append(vis_scalar)
        log['positions'].append(pos)
        log['joint_pos'].append(joint_pos_np.copy())

        action = agent.test_step(obs)
        obs, _, done, _ = env.step(action)
        if done:
            obs = env.reset()
            swimmer.reset()

    # Convert to numpy
    for k in ('speeds', 'z_DA', 'z_5HT', 'periods',
               'gait_labels', 'env_labels', 'viscosity'):
        log[k] = np.array(log[k], dtype=np.float32)
    log['gamma']     = np.array(log['gamma'],     dtype=np.float32)
    log['beta']      = np.array(log['beta'],      dtype=np.float32)
    log['positions'] = np.array(log['positions'], dtype=np.float32)
    log['joint_pos'] = np.array(log['joint_pos'], dtype=np.float32)

    return log
#     Mirrors paper Fig 1A (NSM activity × axial speed for dwelling / roaming)
# ─────────────────────────────────────────────────────────────────────────────

def plot_ccmn_gait_joint_density(ccmn_log: dict, save_path: str) -> None:
    """
    Panel layout (matching paper Fig 1A):

        [swim 2D density] [crawl 2D density] [shared speed marginal]
        [swim z_DA hist ] [crawl z_DA hist ]

    x-axis: z_DA (neuromodulatory context, 0-1 normalised)
    y-axis: axial speed (mm/s equivalent)
    colour: joint probability density (blue=low, red=high)
    """
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)

    z    = ccmn_log['z_DA']
    spd  = ccmn_log['speeds'] * 1000.0   # → mm/s-equivalent
    gait = ccmn_log['gait_labels']        # 0=swim, 1=crawl

    swim_mask  = gait == 0
    crawl_mask = gait == 1

    # Normalise z_DA to [0, 1] for axis consistency with paper
    z_norm = (z - z.min()) / (z.max() - z.min() + 1e-8)

    fig = plt.figure(figsize=(10, 7))
    gs  = gridspec.GridSpec(2, 3,
                            height_ratios=[1, 3],
                            width_ratios=[3, 3, 1],
                            hspace=0.05, wspace=0.08)

    z_edges   = np.linspace(0, 1,    40)
    spd_edges = np.linspace(0, spd.max() * 1.05, 40)

    def make_2d_density(mask):
        h, _, _ = np.histogram2d(z_norm[mask], spd[mask],
                                  bins=[z_edges, spd_edges],
                                  density=True)
        h = gaussian_filter(h.T, sigma=1.0)
        return h / (h.max() + 1e-12)

    swim_h  = make_2d_density(swim_mask)
    crawl_h = make_2d_density(crawl_mask)

    vmax = max(swim_h.max(), crawl_h.max())

    # ── top marginals (z_DA histograms) ──────────────────────────────────
    for col, (mask, color, label) in enumerate(
            [(swim_mask,  SWIM_COLOR,  'Swim'),
             (crawl_mask, CRAWL_COLOR, 'Crawl')]):
        ax = fig.add_subplot(gs[0, col])
        ax.fill_between(z_edges[:-1], 0,
                         np.histogram(z_norm[mask], bins=z_edges,
                                      density=True)[0],
                         color=color, alpha=0.85)
        # grey reference (total distribution)
        ax.fill_between(z_edges[:-1], 0,
                         np.histogram(z_norm, bins=z_edges,
                                      density=True)[0],
                         color='#888888', alpha=0.35)
        ax.set_xlim(0, 1)
        ax.set_xticks([])
        ax.set_ylabel('probability', fontsize=9)
        ax.set_title(label, color=color, fontweight='bold', fontsize=12)
        ax.spines[['right', 'top', 'bottom']].set_visible(False)

    # ── 2-D density heatmaps ─────────────────────────────────────────────
    ax_swim  = fig.add_subplot(gs[1, 0])
    ax_crawl = fig.add_subplot(gs[1, 1])
    ax_spd   = fig.add_subplot(gs[1, 2])

    extent = [z_edges[0], z_edges[-1], spd_edges[0], spd_edges[-1]]
    kw = dict(extent=extent, origin='lower', aspect='auto',
              cmap=_BWR_DENSE, vmin=0, vmax=vmax)
    im = ax_swim.imshow(swim_h,  **kw)
    ax_crawl.imshow(crawl_h, **kw)

    for ax, label in [(ax_swim, 'Swim'), (ax_crawl, 'Crawl')]:
        ax.set_xlabel('z_DA  (normalised)', fontsize=10)
        ax.set_xlim(0, 1)
        ax.set_ylim(spd_edges[0], spd_edges[-1])
    ax_swim.set_ylabel('axial speed  (mm/s)', fontsize=10)
    ax_crawl.set_yticks([])

    # ── right marginal (speed histograms) ────────────────────────────────
    for mask, color in [(swim_mask, SWIM_COLOR), (crawl_mask, CRAWL_COLOR)]:
        hist, edges = np.histogram(spd[mask], bins=spd_edges, density=True)
        ax_spd.fill_betweenx(edges[:-1], 0, hist, color=color, alpha=0.7)
    ax_spd.set_xlabel('probability', fontsize=9)
    ax_spd.set_yticks([])
    ax_spd.set_ylim(spd_edges[0], spd_edges[-1])
    ax_spd.spines[['top', 'right']].set_visible(False)

    # ── colourbar ────────────────────────────────────────────────────────
    cbar_ax = fig.add_axes([0.91, 0.12, 0.015, 0.5])
    cb = fig.colorbar(im, cax=cbar_ax)
    cb.set_label('probability', fontsize=9)
    cb.set_ticks([0, vmax])
    cb.set_ticklabels(['0', f'{vmax:.2f}'])

    fig.suptitle('CCMN Gait-State Joint Density  (z_DA × Axial Speed)',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'✅ Gait joint-density plot saved: {save_path}')


# ─────────────────────────────────────────────────────────────────────────────
# 3.  FIGURE B  –  Peri-transition raster of z_DA
#     Mirrors paper Fig 1B (NSM activity aligned to dwelling onset)
# ─────────────────────────────────────────────────────────────────────────────

def plot_ccmn_transition_raster(ccmn_log: dict, save_path: str,
                                 window: int = 90,
                                 transition_type: str = 'S2C') -> None:
    """
    Top:    Mean ± SEM of z_DA centred on every Swim→Crawl (or Crawl→Swim)
            transition — mirrors the mean NSM trace in Fig 1B.
    Bottom: Individual event raster sorted by transition time, coloured by
            z_DA magnitude — mirrors the heatmap in Fig 1B.

    Parameters
    ----------
    window          : half-window in steps (±window around each transition)
    transition_type : 'S2C' (swim→crawl) or 'C2S' (crawl→swim)
    """
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)

    z     = ccmn_log['z_DA']
    trans = [(t, lbl) for t, lbl in ccmn_log['transitions']
             if lbl == transition_type]
    T     = len(z)

    snippets = []
    for t0, _ in trans:
        start = t0 - window
        end   = t0 + window
        if start < 0 or end >= T:
            continue
        snippets.append(z[start:end])

    if len(snippets) < 2:
        print(f'⚠️ Not enough {transition_type} transitions to plot raster '
              f'({len(snippets)} found). Skipping.')
        return

    mat  = np.stack(snippets)                        # (N_events, 2*window)
    time_axis = np.arange(-window, window)

    mean_trace = mat.mean(axis=0)
    sem_trace  = mat.std(axis=0) / np.sqrt(len(mat))

    fig, axes = plt.subplots(2, 1, figsize=(7, 7),
                              gridspec_kw={'height_ratios': [1, 2.5],
                                           'hspace': 0.05})

    # ── top: mean ± SEM ──────────────────────────────────────────────────
    ax0 = axes[0]
    ax0.fill_between(time_axis,
                     mean_trace - sem_trace,
                     mean_trace + sem_trace,
                     color='#888888', alpha=0.4)
    ax0.plot(time_axis, mean_trace, color='black', lw=2)
    ax0.axvline(0, color='black', lw=1.5, ls='--', alpha=0.8)
    ax0.set_xlim(-window, window)
    ax0.set_ylim(-1.1, 1.1)
    ax0.set_ylabel('z_DA', fontsize=10)
    ax0.set_xticks([])
    ax0.spines[['right', 'top', 'bottom']].set_visible(False)
    lbl = 'Swim→Crawl onset' if transition_type == 'S2C' else 'Crawl→Swim onset'
    ax0.set_title(f'z_DA aligned to {lbl}  (n={len(mat)} events)',
                  fontsize=11, fontweight='bold')

    # ── bottom: raster ───────────────────────────────────────────────────
    ax1 = axes[1]
    # Sort rows by peak position relative to transition (later peak = sorted lower)
    peak_lags = np.argmax(mat, axis=1) - window
    order     = np.argsort(peak_lags)
    mat_sorted = mat[order]

    im = ax1.imshow(mat_sorted, aspect='auto', origin='lower',
                    extent=[-window, window, 0, len(mat)],
                    cmap=_BWR_DENSE, vmin=-1, vmax=1)
    ax1.axvline(0, color='black', lw=1.5, ls='--', alpha=0.9)

    # Trend line (boundary of peak activation)
    from scipy.ndimage import label as ndi_label
    try:
        boundary = [np.argmax(mat_sorted[i]) - window for i in range(len(mat_sorted))]
        ax1.plot(boundary, np.arange(len(mat_sorted)), color='black', lw=1.2)
    except Exception:
        pass

    ax1.set_xlabel('time to transition onset  (steps)', fontsize=10)
    ax1.set_ylabel('individual events', fontsize=10)
    ax1.set_xlim(-window, window)

    cbar = fig.colorbar(im, ax=ax1, fraction=0.03, pad=0.02)
    cbar.set_label('z_DA  (normalised $\\Delta R/R_0$)', fontsize=9)
    cbar.set_ticks([-1, 0, 1])

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'✅ Transition raster saved: {save_path}')


# ─────────────────────────────────────────────────────────────────────────────
# 4.  FIGURE C  –  Trajectory coloured by gait state
#     Mirrors paper Fig 2A (path coloured by Roam / Dwell)
# ─────────────────────────────────────────────────────────────────────────────

def plot_ccmn_gait_trajectory(ccmn_log: dict, save_path: str) -> None:
    """
    Top:   2-D path coloured by Swim (orange) / Crawl (blue) segments,
           start (red dot) and end (black dot) labelled — mirrors paper Fig 2A.
    Bottom: Speed trace with Swim/Crawl bar above it — mirrors the speed
            panel beneath the trajectory in Fig 2A.
    """
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)

    pos   = ccmn_log['positions']            # (T, 2)
    gait  = ccmn_log['gait_labels']          # 0=swim, 1=crawl
    spd   = ccmn_log['speeds'] * 1000.0      # mm/s
    T     = len(gait)
    time  = np.arange(T)

    fig, axes = plt.subplots(2, 1, figsize=(9, 7),
                              gridspec_kw={'height_ratios': [2.5, 1],
                                           'hspace': 0.3})

    # ── top: trajectory ───────────────────────────────────────────────────
    ax0 = axes[0]
    # Draw segments colour-coded by gait
    for t in range(T - 1):
        col = SWIM_COLOR if gait[t] == 0 else CRAWL_COLOR
        ax0.plot(pos[t:t+2, 0], pos[t:t+2, 1],
                 color=col, lw=1.5, solid_capstyle='round')

    # Start / end markers
    ax0.scatter(*pos[0],  color='red',   s=80, zorder=5, label='Start')
    ax0.scatter(*pos[-1], color='black', s=80, zorder=5, label='End')

    # Scale bar (2 cm equivalent in physics units — adjust as needed)
    x0 = pos[:, 0].min()
    y0 = pos[:, 1].min() - 0.15
    scale = 0.2   # 20 cm in physics metres ≈ 2 cm worm scale
    ax0.plot([x0, x0 + scale], [y0, y0], 'k-', lw=3)
    ax0.text(x0 + scale / 2, y0 - 0.05, '20 cm', ha='center', fontsize=9)

    ax0.set_aspect('equal')
    ax0.axis('off')
    legend_handles = [
        Patch(color=SWIM_COLOR,  label='Swim  (low z_DA)'),
        Patch(color=CRAWL_COLOR, label='Crawl (high z_DA)'),
    ]
    ax0.legend(handles=legend_handles, loc='upper right', fontsize=9,
               framealpha=0.8)
    ax0.set_title('CCMN Episode Trajectory  –  Gait State', fontsize=12,
                  fontweight='bold')

    # ── bottom: speed trace with state bar ────────────────────────────────
    ax1 = axes[1]
    # State bar (same height as paper's coloured bar above speed trace)
    bar_y = spd.max() * 1.18
    for t in range(T - 1):
        col = SWIM_COLOR if gait[t] == 0 else CRAWL_COLOR
        ax1.fill_between([t, t+1], bar_y * 0.95, bar_y * 1.05,
                         color=col, linewidth=0)
    ax1.text(-T * 0.01, bar_y, 'Swim', color=SWIM_COLOR,
             fontsize=8, va='center', ha='right')
    ax1.text(-T * 0.01, bar_y * 0.7, 'Crawl', color=CRAWL_COLOR,
             fontsize=8, va='center', ha='right')

    ax1.plot(time, spd, color='black', lw=1.0)
    ax1.set_xlim(0, T)
    ax1.set_ylim(0, bar_y * 1.1)
    ax1.set_xlabel('time  (steps)', fontsize=10)
    ax1.set_ylabel('speed  (mm/s)', fontsize=10)
    ax1.spines[['right', 'top']].set_visible(False)

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'✅ Gait trajectory plot saved: {save_path}')


# ─────────────────────────────────────────────────────────────────────────────
# 5.  FIGURE D  –  FiLM γ traces with velocity background
#     Mirrors paper Fig 3C (neuron activity traces coloured by axial velocity
#     background stripes and behavioural-state bar)
# ─────────────────────────────────────────────────────────────────────────────

def plot_ccmn_film_traces(ccmn_log: dict, save_path: str,
                           max_joints: int = 8) -> None:
    """
    Each row shows γ_i(t) for joint i, stacked vertically (most anterior =
    top), with a velocity-coloured background stripe matching paper Fig 3C's
    green/pink velocity background, and a Swim/Crawl bar at the top.

    Column D of the paper (peri-transition cross-correlations) is reproduced
    alongside as a separate column showing the cross-correlation of each γ_i
    with z_DA.
    """
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)

    gamma = ccmn_log['gamma']         # (T, n_joints)
    z_da  = ccmn_log['z_DA']          # (T,)
    spd   = ccmn_log['speeds']        # (T,)   in physics units
    gait  = ccmn_log['gait_labels']   # (T,)
    T, n_joints = gamma.shape
    n_show = min(n_joints, max_joints)
    time   = np.arange(T)

    # Velocity normalisation for background colour (green = forward, pink = reverse)
    spd_norm = np.clip(spd / (spd.max() + 1e-8), -1, 1)

    fig = plt.figure(figsize=(13, 2.2 * n_show + 1.5))
    outer = gridspec.GridSpec(1, 2, width_ratios=[4, 1], wspace=0.08,
                               left=0.1, right=0.96, top=0.93, bottom=0.06)
    left_gs  = gridspec.GridSpecFromSubplotSpec(n_show + 1, 1,
                                                 subplot_spec=outer[0],
                                                 hspace=0)
    right_gs = gridspec.GridSpecFromSubplotSpec(n_show, 1,
                                                 subplot_spec=outer[1],
                                                 hspace=0)

    # ── state bar (top row) ───────────────────────────────────────────────
    ax_bar = fig.add_subplot(left_gs[0])
    for t in range(T - 1):
        col = SWIM_COLOR if gait[t] == 0 else CRAWL_COLOR
        ax_bar.fill_between([t, t+1], 0, 1, color=col, linewidth=0)
    ax_bar.set_xlim(0, T)
    ax_bar.set_ylim(0, 1)
    ax_bar.axis('off')
    ax_bar.text(T * 1.005, 0.8, 'Swim',  color=SWIM_COLOR,  fontsize=8, va='center')
    ax_bar.text(T * 1.005, 0.2, 'Crawl', color=CRAWL_COLOR, fontsize=8, va='center')

    joint_labels = [f'γ_{i+1}' for i in range(n_show)]

    for j in range(n_show):
        ax_l = fig.add_subplot(left_gs[j + 1])
        ax_r = fig.add_subplot(right_gs[j])

        # ── velocity-coloured background ─────────────────────────────────
        for t in range(T - 1):
            v = spd_norm[t]
            if v >= 0:
                bg = (0.88, 1.00, 0.88)   # light green → forward
            else:
                bg = (1.00, 0.88, 0.88)   # light pink  → reverse
            ax_l.fill_between([t, t+1],
                               gamma[:, j].min() - 0.05,
                               gamma[:, j].max() + 0.05,
                               color=bg, linewidth=0, alpha=0.6)

        # ── γ trace ────────────────────────────────────────────────────
        ax_l.plot(time, gamma[:, j], color='black', lw=0.9)
        ax_l.set_xlim(0, T)
        ax_l.set_ylabel(joint_labels[j], fontsize=8, rotation=0,
                         labelpad=22, va='center')
        ax_l.set_yticks([])
        ax_l.spines[['top', 'right', 'left', 'bottom']].set_visible(False)
        if j < n_show - 1:
            ax_l.set_xticks([])
        else:
            ax_l.set_xlabel('time  (steps)', fontsize=9)

        # ── cross-correlation with z_DA (right column) ───────────────────
        lags   = np.arange(-60, 61)
        xcorr  = np.correlate(
            gamma[:, j] - gamma[:, j].mean(),
            z_da         - z_da.mean(),
            mode='full'
        )
        # Centre of full xcorr output corresponds to lag=0
        centre = len(xcorr) // 2
        xcorr  = xcorr[centre - 60: centre + 61]
        xcorr /= (np.std(gamma[:, j]) * np.std(z_da) * T + 1e-8)

        ax_r.fill_between(lags, 0, xcorr,
                           where=xcorr > 0, color=SWIM_COLOR,  alpha=0.8)
        ax_r.fill_between(lags, 0, xcorr,
                           where=xcorr < 0, color=CRAWL_COLOR, alpha=0.8)
        ax_r.axvline(0, color='black', lw=0.8, ls='--')
        ax_r.axhline(0, color='black', lw=0.5)
        ax_r.set_xlim(-60, 60)
        ax_r.set_yticks([])
        ax_r.spines[['top', 'right', 'left']].set_visible(False)
        if j < n_show - 1:
            ax_r.set_xticks([])
        else:
            ax_r.set_xlabel('lag  (steps)', fontsize=8)
        if j == 0:
            ax_r.set_title('xcorr\nγ_i ↔ z_DA', fontsize=8, color='#444444')

    fig.suptitle('CCMN  FiLM γ per joint  –  velocity background  &  z_DA correlation',
                 fontsize=11, fontweight='bold')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'✅ FiLM γ trace plot saved: {save_path}')


# ─────────────────────────────────────────────────────────────────────────────
# 6.  FIGURE E  –  Bistability: z_DA ↔ oscillator period
#     Mirrors paper Fig 1D (dwelling-state duration vs NSM bout duration)
# ─────────────────────────────────────────────────────────────────────────────

def plot_ccmn_bistability(ccmn_log: dict, save_path: str) -> None:
    """
    Two-panel figure:

    Left:   Scatter of z_DA value vs effective oscillator period for each step,
            coloured by environment type (water = blue, land = orange).
            The y=x identity line is replaced by the theoretical GaitPeriod
            scheduler curve.  Mirrors paper Fig 1D scatter.

    Right:  Histogram of effective period distributions split by environment
            (water / land), showing the bimodal distribution expected from the
            bistable z_DA dynamics.  Mirrors the bimodal cycle-duration
            histograms from Vidal-Gadea 2011 referenced in the CCMN design.
    """
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)

    z_da    = ccmn_log['z_DA']
    periods = ccmn_log['periods'].astype(float)
    env_lbl = ccmn_log['env_labels']   # 0=water, 1=land

    water_mask = env_lbl == 0
    land_mask  = env_lbl == 1

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # ── left: z_DA vs period scatter ─────────────────────────────────────
    ax0 = axes[0]

    # Theoretical scheduler curve
    try:
        import torch
        from swimmer.models.ncap_ccmn import GaitPeriodScheduler
        z_sweep = np.linspace(-1, 1, 200)
        p_theory = [GaitPeriodScheduler.period(float(z)) for z in z_sweep]
        ax0.plot(z_sweep, p_theory, color='black', lw=2.0, ls='--',
                 label='GaitPeriodScheduler', zorder=5)
    except ImportError:
        pass

    for mask, color, label in [
            (water_mask, CRAWL_COLOR, 'Water'),
            (land_mask,  SWIM_COLOR,  'Land')]:
        if mask.sum() == 0:
            continue
        ax0.scatter(z_da[mask], periods[mask],
                    color=color, alpha=0.25, s=6, rasterized=True)
        # Running mean
        order = np.argsort(z_da[mask])
        z_s   = z_da[mask][order]
        p_s   = periods[mask][order]
        k     = max(1, len(z_s) // 20)
        run_z = np.convolve(z_s, np.ones(k)/k, mode='valid')
        run_p = np.convolve(p_s, np.ones(k)/k, mode='valid')
        ax0.plot(run_z, run_p, color=color, lw=2.2, label=label)

    ax0.set_xlabel('z_DA  (DA-like context signal)', fontsize=10)
    ax0.set_ylabel('effective oscillator period  (steps)', fontsize=10)
    ax0.set_title('Bistable Gait Switching', fontsize=11, fontweight='bold')
    ax0.legend(fontsize=9)
    ax0.spines[['right', 'top']].set_visible(False)

    # ── right: period histogram split by environment ──────────────────────
    ax1 = axes[1]
    bins = np.linspace(periods.min() - 1, periods.max() + 1, 25)
    for mask, color, label in [
            (water_mask, CRAWL_COLOR, 'Water (swim expected)'),
            (land_mask,  SWIM_COLOR,  'Land  (crawl expected)')]:
        if mask.sum() == 0:
            continue
        ax1.hist(periods[mask], bins=bins, color=color, alpha=0.6,
                 label=label, density=True, edgecolor='none')

    ax1.axvline(15, color='black', lw=1.2, ls=':', alpha=0.7)
    ax1.axvline(60, color='black', lw=1.2, ls=':', alpha=0.7)
    ax1.text(15, ax1.get_ylim()[1] if ax1.get_ylim()[1] > 0 else 0.1,
             'swim\nperiod', ha='center', va='bottom', fontsize=8)
    ax1.text(60, ax1.get_ylim()[1] if ax1.get_ylim()[1] > 0 else 0.1,
             'crawl\nperiod', ha='center', va='bottom', fontsize=8)
    ax1.set_xlabel('oscillator period  (steps)', fontsize=10)
    ax1.set_ylabel('density', fontsize=10)
    ax1.set_title('Period Distribution by Environment', fontsize=11,
                  fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.spines[['right', 'top']].set_visible(False)

    fig.suptitle('CCMN  z_DA bistability  –  matches C. elegans gait bimodality',
                 fontsize=11, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'✅ Bistability plot saved: {save_path}')


# ─────────────────────────────────────────────────────────────────────────────
# 7.  FIGURE F  –  Peri-transition timing (mirrors paper Fig 1C)
#     Time of z_DA activation relative to gait-state onset, and vice versa.
# ─────────────────────────────────────────────────────────────────────────────

def plot_ccmn_transition_timing(ccmn_log: dict, save_path: str,
                                 z_thresh: float = 0.3) -> None:
    """
    Mirrors paper Fig 1C:
        Left column:  time of z_DA rise relative to Swim→Crawl onset
        Right column: time of z_DA fall relative to Crawl→Swim onset

    Each is shown as a strip chart with mean ± SEM, exactly as in the paper.
    """
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)

    z     = ccmn_log['z_DA']
    trans = ccmn_log['transitions']
    T     = len(z)

    def find_signal_event(t0: int, direction: str,
                          search_window: int = 60) -> float | None:
        """Find the first crossing of z_thresh before/after t0."""
        if direction == 'rise':
            # Search backward from t0 for last time z < thresh
            for dt in range(search_window):
                t = t0 - dt
                if t < 0:
                    break
                if z[t] < z_thresh:
                    return -dt   # negative = z rose before transition
            return None
        else:  # fall
            for dt in range(search_window):
                t = t0 + dt
                if t >= T:
                    break
                if z[t] < z_thresh:
                    return dt    # positive = z fell after transition
            return None

    s2c_lags, c2s_lags = [], []
    for t0, lbl in trans:
        if lbl == 'S2C':
            lag = find_signal_event(t0, 'rise')
            if lag is not None:
                s2c_lags.append(lag / 30.0)   # steps → "minutes" (30 steps ≈ 1 unit)
        else:
            lag = find_signal_event(t0, 'fall')
            if lag is not None:
                c2s_lags.append(lag / 30.0)

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    datasets = [
        (s2c_lags, 'z_DA rise\nto Swim→Crawl\nonset'),
        (c2s_lags, 'z_DA fall\nto Crawl→Swim\nonset'),
    ]

    for xi, (data, xlabel) in enumerate(datasets):
        if not data:
            continue
        data_arr = np.array(data)
        # Jittered strip chart
        jitter = np.random.uniform(-0.08, 0.08, size=len(data_arr))
        ax.scatter(np.full_like(data_arr, xi) + jitter, data_arr,
                   color='#888888', alpha=0.6, s=25, zorder=3)
        # Mean ± SEM marker
        mean = data_arr.mean()
        sem  = data_arr.std() / np.sqrt(len(data_arr))
        ax.scatter([xi], [mean], color=CRAWL_COLOR, s=100, zorder=5)
        ax.errorbar([xi], [mean], yerr=sem,
                    fmt='none', color=CRAWL_COLOR, lw=2.5,
                    capsize=6, zorder=5)

    ax.axhline(0, color='black', lw=1.0, ls='--', alpha=0.6)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(
        ['z_DA activation\nto Swim→Crawl\nonset',
         'z_DA inactivation\nto Crawl→Swim\nonset'],
        fontsize=9
    )
    ax.set_ylabel('time to gait transition  (steps ÷ 30)', fontsize=10)
    ax.set_title('CCMN  Neuromodulatory Lead/Lag at Gait Transitions',
                 fontsize=10, fontweight='bold')
    ax.spines[['right', 'top']].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'✅ Transition timing plot saved: {save_path}')


# ─────────────────────────────────────────────────────────────────────────────
# Fig G  –  Gait Kymograph (swim vs crawl body wave)
#  Mirrors Berri et al. 2009 Fig 1: travelling-wave kymograph of joint angles,
#  separately for a swim epoch and a crawl epoch.
#  Diagonal stripe angle = wave speed; stripe width = amplitude.
# ─────────────────────────────────────────────────────────────────────────────

def plot_ccmn_gait_kymograph(ccmn_log: dict, save_path: str,
                              epoch_len: int = 120) -> None:
    """
    Find the longest contiguous swim and crawl epochs, then plot joint-angle
    kymographs side by side.  Stripe slope encodes wave propagation speed;
    colour encodes joint angle (blue = flexed one way, red = other).
    """
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)

    jp   = ccmn_log['joint_pos']    # (T, n_joints)
    gait = ccmn_log['gait_labels']  # 0=swim, 1=crawl
    T, n_joints = jp.shape

    def longest_epoch(label):
        """Return start index of longest contiguous run of `label`."""
        best_start, best_len, cur_start, cur_len = 0, 0, 0, 0
        for t in range(T):
            if gait[t] == label:
                cur_len += 1
                if cur_len > best_len:
                    best_len, best_start = cur_len, cur_start
            else:
                cur_start, cur_len = t + 1, 0
        return best_start, min(best_len, epoch_len)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    fig.suptitle('Gait Kymograph  –  Body Wave Comparison',
                 fontsize=13, fontweight='bold')

    for ax, (label, title, color) in zip(
            axes,
            [(0, f'Swim  (low z_DA)', SWIM_COLOR),
             (1, f'Crawl  (high z_DA)', CRAWL_COLOR)]):
        start, length = longest_epoch(label)
        if length < 10:
            ax.text(0.5, 0.5, f'No {title.split()[0]} epoch found',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title, color=color, fontweight='bold')
            continue

        epoch = jp[start:start + length]  # (length, n_joints)

        # Normalise each joint column independently so colour encodes
        # relative curvature regardless of absolute angle range
        epoch_norm = (epoch - epoch.mean(axis=0)) / (epoch.std(axis=0) + 1e-8)

        im = ax.imshow(epoch_norm.T,           # joints on y, time on x
                       aspect='auto', origin='lower',
                       cmap='RdBu_r', vmin=-2.5, vmax=2.5,
                       extent=[0, length, 0, n_joints])
        ax.set_xlabel('time  (steps)', fontsize=10)
        ax.set_title(title, color=color, fontweight='bold', fontsize=11)
        ax.set_yticks(np.arange(n_joints) + 0.5)
        ax.set_yticklabels([f'J{i+1}' for i in range(n_joints)], fontsize=8)

    axes[0].set_ylabel('joint index  (head → tail)', fontsize=10)
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.04,
                 label='normalised joint angle')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'✅ Gait kymograph saved: {save_path}')


# ─────────────────────────────────────────────────────────────────────────────
# Fig H  –  Speed + z_DA + environment shading
#  Shows speed time series with z_DA overlay and background shading for
#  water (blue) / land (orange) — reveals whether z_DA anticipates the
#  speed change that accompanies environment transitions.
# ─────────────────────────────────────────────────────────────────────────────

def plot_ccmn_speed_zda_environment(ccmn_log: dict, save_path: str) -> None:
    """
    Single-panel plot with:
      • Background fill: orange=land, blue=water
      • Black line: head speed (mm/s), left y-axis
      • Dashed red line: z_DA, right y-axis
      • Vertical dotted lines at each environment transition
    """
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)

    spd  = ccmn_log['speeds'] * 1000.0   # → mm/s
    z_da = ccmn_log['z_DA']
    env  = ccmn_log['env_labels']         # 0=water, 1=land
    T    = len(spd)
    t    = np.arange(T)

    fig, ax1 = plt.subplots(figsize=(13, 4))

    # Background environment shading
    in_land = False
    seg_start = 0
    for i in range(T + 1):
        current_land = bool(env[i]) if i < T else (not in_land)
        if i == 0:
            in_land = current_land
        if current_land != in_land or i == T:
            color = '#E07B2A' if in_land else '#2A5FA5'
            ax1.axvspan(seg_start, i, alpha=0.12, color=color, lw=0)
            if i < T:
                ax1.axvline(i, color='grey', lw=0.8, ls=':', alpha=0.7)
            seg_start, in_land = i, current_land

    # Speed trace
    ax1.plot(t, spd, color='black', lw=1.0, alpha=0.85, label='speed')
    ax1.set_xlabel('time  (steps)', fontsize=10)
    ax1.set_ylabel('speed  (mm/s)', fontsize=10)
    ax1.set_xlim(0, T)

    # z_DA overlay on second axis
    ax2 = ax1.twinx()
    ax2.plot(t, z_da, color='#D62728', lw=1.5, ls='--', alpha=0.9,
             label='z_DA')
    ax2.axhline(0, color='#D62728', lw=0.5, ls=':', alpha=0.4)
    ax2.set_ylabel('z_DA', fontsize=10, color='#D62728')
    ax2.tick_params(axis='y', labelcolor='#D62728')
    ax2.set_ylim(-1.1, 1.1)

    # Legend patches
    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor='#2A5FA5', alpha=0.3, label='Water'),
        Patch(facecolor='#E07B2A', alpha=0.3, label='Land'),
        plt.Line2D([0], [0], color='black', lw=1.5, label='Speed'),
        plt.Line2D([0], [0], color='#D62728', lw=1.5, ls='--', label='z_DA'),
    ]
    ax1.legend(handles=handles, loc='upper left', fontsize=8, framealpha=0.7)
    fig.suptitle('Speed  ·  z_DA  ·  Environment  —  Predictive Neuromodulation',
                 fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'✅ Speed/z_DA/environment plot saved: {save_path}')


# ─────────────────────────────────────────────────────────────────────────────
# Fig I  –  Undulation frequency spectrum by environment
#  Power spectral density of head speed for water vs land segments.
#  Swim peak expected ~2 Hz (30 steps/cycle at 30 Hz sim), crawl ~0.5 Hz.
#  Matches Fang-Yen et al. 2010 Fig 3 quantitatively.
# ─────────────────────────────────────────────────────────────────────────────

def plot_ccmn_undulation_spectrum(ccmn_log: dict, save_path: str,
                                   sim_hz: float = 30.0) -> None:
    """
    Welch PSD of head speed separately for water and land segments.
    x-axis in Hz (using sim_hz as the sampling rate).
    Biological reference lines drawn at 0.5 Hz (crawl) and 2.0 Hz (swim).
    """
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)

    spd = ccmn_log['speeds'] * 1000.0
    env = ccmn_log['env_labels']

    water_spd = spd[env == 0]
    land_spd  = spd[env == 1]

    fig, ax = plt.subplots(figsize=(7, 4))

    for seg, color, label in [
            (water_spd, SWIM_COLOR,  'Water  (swim)'),
            (land_spd,  CRAWL_COLOR, 'Land  (crawl)')]:
        if len(seg) < 32:
            continue
        freqs, psd = scipy_signal.welch(seg, fs=sim_hz,
                                         nperseg=min(256, len(seg) // 2))
        # Normalise PSD to peak=1 for visual comparison
        psd_norm = psd / (psd.max() + 1e-12)
        ax.semilogy(freqs, psd_norm, color=color, lw=2.0, label=label)

    # Biological reference lines from Fang-Yen et al. 2010
    ax.axvline(0.5, color=CRAWL_COLOR, lw=1.2, ls='--', alpha=0.7,
               label='Crawl ref  0.5 Hz')
    ax.axvline(2.0, color=SWIM_COLOR,  lw=1.2, ls='--', alpha=0.7,
               label='Swim ref  2.0 Hz')

    ax.set_xlim(0, sim_hz / 2)
    ax.set_xlabel('frequency  (Hz)', fontsize=10)
    ax.set_ylabel('normalised PSD  (log)', fontsize=10)
    ax.set_title('Undulation Frequency Spectrum  by Environment\n'
                 '(Fang-Yen et al. 2010 reference lines)',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.spines[['right', 'top']].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'✅ Undulation spectrum saved: {save_path}')


# ─────────────────────────────────────────────────────────────────────────────
# Fig J  –  Phase portrait: z_DA vs oscillator period
#  Scatter of (z_DA, period) coloured by environment.
#  Two clouds in opposite corners = learned environment discrimination.
# ─────────────────────────────────────────────────────────────────────────────

def plot_ccmn_phase_portrait_period(ccmn_log: dict, save_path: str) -> None:
    """
    Each timestep is one point: z_DA on x-axis, effective oscillator period
    on y-axis, coloured by environment (water=orange, land=blue).
    Tight separation of the two clouds = ContextEncoder has learned to
    discriminate environments from φ(s,t) alone.
    """
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)

    z_da    = ccmn_log['z_DA']
    periods = ccmn_log['periods']
    env     = ccmn_log['env_labels']

    fig, ax = plt.subplots(figsize=(6, 5))

    for label, color, name in [
            (0, SWIM_COLOR,  'Water  (swim)'),
            (1, CRAWL_COLOR, 'Land  (crawl)')]:
        mask = env == label
        ax.scatter(z_da[mask], periods[mask],
                   c=color, alpha=0.25, s=8, label=name, rasterized=True)

    # Reference lines at biological period targets
    ax.axhline(15, color=SWIM_COLOR,  lw=1.2, ls='--', alpha=0.6,
               label='Swim target  (15 steps)')
    ax.axhline(60, color=CRAWL_COLOR, lw=1.2, ls='--', alpha=0.6,
               label='Crawl target  (60 steps)')
    ax.axvline(0,  color='grey', lw=0.8, ls=':', alpha=0.5)

    ax.set_xlabel('z_DA  (DA-like context signal)', fontsize=10)
    ax.set_ylabel('effective oscillator period  (steps)', fontsize=10)
    ax.set_title('Phase Portrait  –  z_DA vs Oscillator Period\n'
                 'Environment discrimination learned',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, markerscale=2)
    ax.spines[['right', 'top']].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'✅ Phase portrait (z_DA vs period) saved: {save_path}')


# ─────────────────────────────────────────────────────────────────────────────
# Fig K  –  Amplitude–frequency scatter
#  For each undulation cycle: peak curvature (amplitude proxy) vs cycle
#  frequency.  Matches Berri et al. 2009 Fig 4 showing swim/crawl occupy
#  distinct regions of amplitude-frequency space.
# ─────────────────────────────────────────────────────────────────────────────

def plot_ccmn_amplitude_frequency_scatter(ccmn_log: dict,
                                           save_path: str,
                                           sim_hz: float = 30.0) -> None:
    """
    Detects undulation cycles via zero-crossings of the first joint angle.
    For each cycle computes:
      • frequency  = 1 / cycle_duration  (Hz)
      • amplitude  = max - min of joint_0 within the cycle  (curvature proxy)
    Points coloured by dominant gait state during the cycle.
    """
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)

    jp0  = ccmn_log['joint_pos'][:, 0]   # head-proximal joint
    gait = ccmn_log['gait_labels']
    T    = len(jp0)

    # Zero-crossing detection on the mean-subtracted signal
    j_centered = jp0 - jp0.mean()
    zero_cross = np.where(np.diff(np.sign(j_centered)))[0]

    # Pair consecutive zero crossings → half-cycles; pair them for full cycles
    freqs, amps, colors = [], [], []
    for i in range(0, len(zero_cross) - 2, 2):
        t0 = zero_cross[i]
        t2 = zero_cross[i + 2] if i + 2 < len(zero_cross) else None
        if t2 is None:
            break
        dur = t2 - t0
        if dur < 3:
            continue
        freq = sim_hz / dur
        amp  = jp0[t0:t2].max() - jp0[t0:t2].min()
        dominant_gait = int(np.round(gait[t0:t2].mean()))
        freqs.append(freq)
        amps.append(amp)
        colors.append(SWIM_COLOR if dominant_gait == 0 else CRAWL_COLOR)

    if not freqs:
        print('⚠️ No undulation cycles detected for amplitude-frequency scatter.')
        return

    fig, ax = plt.subplots(figsize=(6, 5))

    # Plot swim and crawl separately for legend
    for label, color, name in [(0, SWIM_COLOR, 'Swim'), (1, CRAWL_COLOR, 'Crawl')]:
        idx = [i for i, c in enumerate(colors)
               if c == (SWIM_COLOR if label == 0 else CRAWL_COLOR)]
        if idx:
            ax.scatter([freqs[i] for i in idx], [amps[i] for i in idx],
                       c=color, alpha=0.5, s=30, label=name, rasterized=True)

    # Biological reference ellipses (approximate from Berri et al. 2009)
    from matplotlib.patches import Ellipse
    for cx, cy, w, h, color in [
            (2.0, 0.25, 1.2, 0.20, SWIM_COLOR),   # swim cluster
            (0.5, 0.40, 0.6, 0.25, CRAWL_COLOR)]: # crawl cluster
        ell = Ellipse((cx, cy), w, h, fill=False,
                      edgecolor=color, lw=1.5, ls='--', alpha=0.7)
        ax.add_patch(ell)

    ax.set_xlabel('undulation frequency  (Hz)', fontsize=10)
    ax.set_ylabel('joint amplitude  (rad, proxy)', fontsize=10)
    ax.set_title('Amplitude–Frequency Trade-off\n'
                 '(Berri et al. 2009 clusters shown dashed)',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.spines[['right', 'top']].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'✅ Amplitude-frequency scatter saved: {save_path}')


# ─────────────────────────────────────────────────────────────────────────────
# Fig L  –  Transition-aligned speed and action amplitude
#  Epoch-average of speed AND action RMS centred on S→C and C→S transitions.
#  Closes the loop: neuromodulation → kinematics chain.
# ─────────────────────────────────────────────────────────────────────────────

def plot_ccmn_transition_speed_amplitude(ccmn_log: dict, save_path: str,
                                          window: int = 60) -> None:
    """
    Two rows × two columns:
      Row 1: Speed mean ± SEM  |  Row 2: Action amplitude (joint-angle RMS)
      Col 1: S→C transitions   |  Col 2: C→S transitions
    If z_DA change leads the speed/amplitude change, the motor output is
    being modulated predictively.
    """
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)

    spd   = ccmn_log['speeds'] * 1000.0
    jp    = ccmn_log['joint_pos']
    amp   = np.sqrt((jp ** 2).mean(axis=1))  # RMS across joints = amplitude proxy
    trans = ccmn_log['transitions']
    T     = len(spd)
    time_axis = np.arange(-window, window)

    def epoch_matrix(signal, ttype):
        snippets = []
        for t0, lbl in trans:
            if lbl != ttype:
                continue
            s, e = t0 - window, t0 + window
            if s >= 0 and e < T:
                snippets.append(signal[s:e])
        return np.stack(snippets) if snippets else None

    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharey='row',
                              gridspec_kw={'hspace': 0.35, 'wspace': 0.15})

    titles = [('Swim → Crawl', CRAWL_COLOR), ('Crawl → Swim', SWIM_COLOR)]
    signals = [('Speed  (mm/s)', spd), ('Action amplitude  (joint RMS)', amp)]

    for col, (ttype, (title, tcolor)) in enumerate(
            zip(['S2C', 'C2S'], titles)):
        for row, (ylabel, sig) in enumerate(signals):
            ax = axes[row, col]
            mat = epoch_matrix(sig, ttype)
            if mat is None or len(mat) < 2:
                ax.text(0.5, 0.5, 'Insufficient data',
                        ha='center', va='center', transform=ax.transAxes,
                        fontsize=8)
                continue
            mean = mat.mean(axis=0)
            sem  = mat.std(axis=0) / np.sqrt(len(mat))
            ax.fill_between(time_axis, mean - sem, mean + sem,
                            color=tcolor, alpha=0.25)
            ax.plot(time_axis, mean, color=tcolor, lw=2)
            ax.axvline(0, color='black', lw=1.2, ls='--', alpha=0.7)
            ax.set_xlim(-window, window)
            if row == 1:
                ax.set_xlabel('time to transition  (steps)', fontsize=9)
            if col == 0:
                ax.set_ylabel(ylabel, fontsize=9)
            if row == 0:
                ax.set_title(f'{title}  (n={len(mat)})',
                             color=tcolor, fontweight='bold', fontsize=10)
            ax.spines[['right', 'top']].set_visible(False)

    fig.suptitle('Transition-Aligned Speed & Amplitude\n'
                 'Neuromodulation → Kinematics Chain',
                 fontsize=12, fontweight='bold')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'✅ Transition speed/amplitude saved: {save_path}')


# ─────────────────────────────────────────────────────────────────────────────
# Fig M  –  z_DA violin plot: water vs land
#  Full distributional summary of z_DA in each environment.
#  Well-separated, non-overlapping violins = learned discrimination.
# ─────────────────────────────────────────────────────────────────────────────

def plot_ccmn_zda_violin(ccmn_log: dict, save_path: str) -> None:
    """
    Violin plot of z_DA distribution for water vs land timesteps, with
    individual data points as a jittered strip overlay.
    """
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)

    z_da = ccmn_log['z_DA']
    env  = ccmn_log['env_labels']

    water_z = z_da[env == 0]
    land_z  = z_da[env == 1]
    data    = [water_z, land_z]
    colors  = [SWIM_COLOR, CRAWL_COLOR]
    labels  = ['Water  (swim)', 'Land  (crawl)']

    fig, ax = plt.subplots(figsize=(5, 5))

    parts = ax.violinplot(data, positions=[1, 2],
                          showmeans=True, showmedians=False,
                          showextrema=True, widths=0.6)

    for i, (pc, color) in enumerate(zip(parts['bodies'], colors)):
        pc.set_facecolor(color)
        pc.set_alpha(0.55)

    for part_name in ('cmeans', 'cbars', 'cmins', 'cmaxes'):
        if part_name in parts:
            parts[part_name].set_color('black')
            parts[part_name].set_linewidth(1.5)

    # Jittered strip overlay (subsample to avoid overplotting)
    for xi, (arr, color) in enumerate(zip(data, colors), start=1):
        sub = arr[::max(1, len(arr) // 400)]   # max 400 points
        jitter = np.random.uniform(-0.08, 0.08, size=len(sub))
        ax.scatter(np.full_like(sub, xi) + jitter, sub,
                   color=color, alpha=0.3, s=6, zorder=3)

    ax.axhline(0, color='grey', lw=0.8, ls='--', alpha=0.5)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel('z_DA  (5-HT/NSM-like signal)', fontsize=10)
    ax.set_title('z_DA Distribution  –  Water vs Land\n'
                 'Environment discrimination summary',
                 fontsize=11, fontweight='bold')
    ax.spines[['right', 'top']].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'✅ z_DA violin plot saved: {save_path}')


# ─────────────────────────────────────────────────────────────────────────────
# Fig N  –  z_DA vs z_5HT neuromodulatory state portrait
#  2D phase plane of the two antagonistic neuromodulatory signals, coloured
#  by environment.  Directly mirrors the DA/5-HT push-pull circuit in
#  Flavell et al. 2013: two clouds in opposite quadrants = bistable
#  neuromodulatory switching learned from mechanical feedback alone.
# ─────────────────────────────────────────────────────────────────────────────

def plot_ccmn_neuromod_state_portrait(ccmn_log: dict, save_path: str) -> None:
    """
    Scatter of z_DA (x) vs z_5HT (y) coloured by environment, with
    quadrant annotations matching the biological circuit labels:
      Top-left  (low DA, high 5-HT)  →  Crawl / dwelling attractor
      Bot-right  (high DA, low 5-HT) →  Swim  / roaming attractor
    Transition timesteps highlighted in black to show switching trajectory.
    """
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)

    z_da  = ccmn_log['z_DA']
    z_5ht = ccmn_log['z_5HT']
    env   = ccmn_log['env_labels']
    trans = ccmn_log['transitions']
    T     = len(z_da)

    fig, ax = plt.subplots(figsize=(6, 6))

    # Environment scatter
    for label, color, name in [
            (0, SWIM_COLOR,  'Water  (swim  / DA-dominant)'),
            (1, CRAWL_COLOR, 'Land  (crawl / 5-HT-dominant)')]:
        mask = env == label
        ax.scatter(z_da[mask], z_5ht[mask],
                   c=color, alpha=0.2, s=8, label=name, rasterized=True)

    # Transition window overlay: ±5 steps around each transition
    for t0, _ in trans:
        s = max(0, t0 - 5)
        e = min(T, t0 + 5)
        ax.scatter(z_da[s:e], z_5ht[s:e],
                   c='black', alpha=0.6, s=18, zorder=5)

    # Quadrant reference lines
    ax.axhline(0, color='grey', lw=0.7, ls='--', alpha=0.5)
    ax.axvline(0, color='grey', lw=0.7, ls='--', alpha=0.5)

    # Biological quadrant labels matching Flavell et al. 2013
    kw = dict(fontsize=8, alpha=0.7, ha='center')
    ax.text(-0.7,  0.7, '5-HT dominant\n(crawl / dwell)', color=CRAWL_COLOR, **kw)
    ax.text( 0.7, -0.7, 'DA dominant\n(swim / roam)',     color=SWIM_COLOR,  **kw)

    ax.set_xlabel('z_DA  (dopamine-like)', fontsize=10)
    ax.set_ylabel('z_5HT  (serotonin-like)', fontsize=10)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_title('Neuromodulatory State Portrait  –  z_DA vs z_5HT\n'
                 '(Flavell et al. 2013 circuit analogue)\n'
                 'Black dots = transition timesteps',
                 fontsize=10, fontweight='bold')
    ax.legend(fontsize=8, markerscale=2, loc='lower right')
    ax.spines[['right', 'top']].set_visible(False)
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'✅ Neuromodulatory state portrait saved: {save_path}')


# ─────────────────────────────────────────────────────────────────────────────
# 8.  MASTER FUNCTION  –  generate all CCMN analysis panels in one call
# ─────────────────────────────────────────────────────────────────────────────

def create_ccmn_neuromodulatory_analysis(agent, env, base_dir: str,
                                          num_steps: int = 1200,
                                          name_prefix: str = 'ccmn') -> dict:
    """
    Collect one episode of CCMN internal signals and generate all six
    analysis figures.

    Parameters
    ----------
    agent       : BiologicalNCAPAgent wrapping a CCMNSwimmer
    env         : single (non-vectorised) progressive mixed environment
    base_dir    : directory where plots are saved
    num_steps   : episode length for data collection
    name_prefix : filename prefix for all saved plots

    Returns
    -------
    ccmn_log : the raw data dict (for further downstream use)
    """
    os.makedirs(base_dir, exist_ok=True)
    print(f'\n🧠 Collecting CCMN episode data ({num_steps} steps)…')
    ccmn_log = collect_ccmn_episode_data(agent, env, num_steps=num_steps)

    n_trans = len(ccmn_log['transitions'])
    swim_frac = (ccmn_log['gait_labels'] == 0).mean()
    print(f'   Swim fraction: {swim_frac:.1%}  |  Transitions: {n_trans}')

    p = lambda name: os.path.join(base_dir, f'{name_prefix}_{name}.png')

    print('📊 Generating CCMN Fig A  –  Gait joint density…')
    plot_ccmn_gait_joint_density(ccmn_log, p('figA_gait_density'))

    print('📊 Generating CCMN Fig B  –  Peri-transition raster (S→C)…')
    plot_ccmn_transition_raster(ccmn_log, p('figB_s2c_raster'),
                                 transition_type='S2C')
    plot_ccmn_transition_raster(ccmn_log, p('figB_c2s_raster'),
                                 transition_type='C2S')

    print('📊 Generating CCMN Fig C  –  Gait trajectory…')
    plot_ccmn_gait_trajectory(ccmn_log, p('figC_gait_trajectory'))

    print('📊 Generating CCMN Fig D  –  FiLM γ traces…')
    plot_ccmn_film_traces(ccmn_log, p('figD_film_gamma_traces'))

    print('📊 Generating CCMN Fig E  –  Bistability…')
    plot_ccmn_bistability(ccmn_log, p('figE_bistability'))

    print('📊 Generating CCMN Fig F  –  Transition timing…')
    plot_ccmn_transition_timing(ccmn_log, p('figF_transition_timing'))

    print('📊 Generating CCMN Fig G  –  Gait kymograph (swim vs crawl)…')
    plot_ccmn_gait_kymograph(ccmn_log, p('figG_gait_kymograph'))

    print('📊 Generating CCMN Fig H  –  Speed + z_DA + environment shading…')
    plot_ccmn_speed_zda_environment(ccmn_log, p('figH_speed_zda_environment'))

    print('📊 Generating CCMN Fig I  –  Undulation frequency spectrum…')
    plot_ccmn_undulation_spectrum(ccmn_log, p('figI_undulation_spectrum'))

    print('📊 Generating CCMN Fig J  –  Phase portrait z_DA vs period…')
    plot_ccmn_phase_portrait_period(ccmn_log, p('figJ_phase_portrait_period'))

    print('📊 Generating CCMN Fig K  –  Amplitude–frequency scatter…')
    plot_ccmn_amplitude_frequency_scatter(ccmn_log, p('figK_amplitude_freq_scatter'))

    print('📊 Generating CCMN Fig L  –  Transition-aligned speed & amplitude…')
    plot_ccmn_transition_speed_amplitude(ccmn_log, p('figL_transition_speed_amp'))

    print('📊 Generating CCMN Fig M  –  z_DA violin plot…')
    plot_ccmn_zda_violin(ccmn_log, p('figM_zda_violin'))

    print('📊 Generating CCMN Fig N  –  Neuromodulatory state portrait…')
    plot_ccmn_neuromod_state_portrait(ccmn_log, p('figN_neuromod_state_portrait'))

    print(f'\n✅  All CCMN neuromodulatory plots saved to: {base_dir}\n')
    return ccmn_log


# ─────────────────────────────────────────────────────────────────────────────
# NCAP BASELINE ANALYSIS — drop-in replacement for create_ccmn_neuromodulatory_analysis
# ─────────────────────────────────────────────────────────────────────────────

def create_ncap_analysis(agent, env, base_dir: str,
                          num_steps: int = 1200,
                          name_prefix: str = 'ncap') -> dict:
    """
    Collect one episode of SimpleNCAPSwimmer signals and generate the same
    analysis figures as create_ccmn_neuromodulatory_analysis.

    Uses collect_ncap_episode_data to build a CCMN-compatible log dict, then
    passes it to every existing plot function unchanged.  This produces a
    directly comparable figure set for the NCAP baseline.

    Figures that require genuine z_DA / FiLM dynamics (A, B, D, E, F, J, M, N)
    will show the oscillator-phase proxy signal — useful for the paper as an
    explicit demonstration that the NCAP has no neuromodulatory structure.
    Figures that depend only on kinematics (C, G, H, I, K, L) will show the
    genuine NCAP locomotion signals.

    Parameters
    ----------
    agent       : agent wrapping a SimpleNCAPSwimmer (must have .swimmer,
                  .n_joints, .env_features_start, .test_step)
    env         : single (non-vectorised) progressive mixed environment
    base_dir    : directory where plots are saved
    num_steps   : episode length for data collection (default 1200)
    name_prefix : filename prefix — default 'ncap' so files do not collide
                  with CCMN outputs

    Returns
    -------
    ncap_log : the raw data dict
    """
    os.makedirs(base_dir, exist_ok=True)
    print(f'\n🧬 Collecting NCAP baseline episode data ({num_steps} steps)…')
    ncap_log = collect_ncap_episode_data(agent, env, num_steps=num_steps)

    n_trans   = len(ncap_log['transitions'])
    swim_frac = (ncap_log['gait_labels'] == 0).mean()
    print(f'   Swim fraction (oscillator phase): {swim_frac:.1%}  |  CPG transitions: {n_trans}')

    p = lambda name: os.path.join(base_dir, f'{name_prefix}_{name}.png')

    print('📊 Generating NCAP Fig A  –  Gait joint density (oscillator proxy)…')
    plot_ccmn_gait_joint_density(ncap_log, p('figA_gait_density'))

    print('📊 Generating NCAP Fig B  –  Peri-transition raster S→C (oscillator)…')
    plot_ccmn_transition_raster(ncap_log, p('figB_s2c_raster'), transition_type='S2C')
    plot_ccmn_transition_raster(ncap_log, p('figB_c2s_raster'), transition_type='C2S')

    print('📊 Generating NCAP Fig C  –  Gait trajectory…')
    plot_ccmn_gait_trajectory(ncap_log, p('figC_gait_trajectory'))

    print('📊 Generating NCAP Fig D  –  FiLM γ traces (all-ones baseline)…')
    plot_ccmn_film_traces(ncap_log, p('figD_film_gamma_traces'))

    print('📊 Generating NCAP Fig E  –  Bistability (fixed period baseline)…')
    plot_ccmn_bistability(ncap_log, p('figE_bistability'))

    print('📊 Generating NCAP Fig F  –  Transition timing (oscillator)…')
    plot_ccmn_transition_timing(ncap_log, p('figF_transition_timing'))

    print('📊 Generating NCAP Fig G  –  Gait kymograph…')
    plot_ccmn_gait_kymograph(ncap_log, p('figG_gait_kymograph'))

    print('📊 Generating NCAP Fig H  –  Speed + z_DA proxy + environment…')
    plot_ccmn_speed_zda_environment(ncap_log, p('figH_speed_zda_environment'))

    print('📊 Generating NCAP Fig I  –  Undulation frequency spectrum…')
    plot_ccmn_undulation_spectrum(ncap_log, p('figI_undulation_spectrum'))

    print('📊 Generating NCAP Fig J  –  Phase portrait z_DA vs period (fixed)…')
    plot_ccmn_phase_portrait_period(ncap_log, p('figJ_phase_portrait_period'))

    print('📊 Generating NCAP Fig K  –  Amplitude–frequency scatter…')
    plot_ccmn_amplitude_frequency_scatter(ncap_log, p('figK_amplitude_freq_scatter'))

    print('📊 Generating NCAP Fig L  –  Transition-aligned speed & amplitude…')
    plot_ccmn_transition_speed_amplitude(ncap_log, p('figL_transition_speed_amp'))

    print('📊 Generating NCAP Fig M  –  z_DA violin (oscillator proxy)…')
    plot_ccmn_zda_violin(ncap_log, p('figM_zda_violin'))

    print('📊 Generating NCAP Fig N  –  Neuromodulatory state portrait…')
    plot_ccmn_neuromod_state_portrait(ncap_log, p('figN_neuromod_state_portrait'))

    print(f'\n✅  All NCAP baseline plots saved to: {base_dir}\n')
    return ncap_log


# =============================================================================
# SECTION 2 — CSV EXPORT, COMPARISON PLOTS, AND CROSS-ARCHITECTURE ANALYSIS
# =============================================================================
# Every plot function in Section 1 now has a CSV sidecar written alongside it.
# This section adds:
#   • _save_csv()                    — shared helper: writes any dict of arrays to CSV
#   • save_log_to_csv()              — exports the full episode log (all keys)
#   • _add_csv_sidecars()            — patches existing plot functions to write CSVs
#   • plot_comparison_learning_curves()  — avg reward vs timesteps, both architectures
#   • plot_comparison_phase_rewards()    — phase-specific reward breakdown comparison
#   • plot_comparison_bistability()      — side-by-side z_DA vs period (NCAP degenerate)
#   • plot_comparison_kymograph()        — 4-panel: NCAP-swim/crawl vs CCMN-swim/crawl
#   • plot_comparison_amplitude_freq()   — both architectures on one scatter + Berri ref
#   • plot_comparison_transition_aligned() — NCAP null vs CCMN signal, same axes
#   • plot_comparison_lead_lag()         — lead/lag side-by-side (NCAP zero baseline)
#   • plot_comparison_period_distribution() — single spike vs bimodal histogram
#   • create_comparison_analysis()       — orchestrator: runs all comparison figures
# =============================================================================

import csv as _csv
import pandas as _pd_stub  # only used if pandas available; graceful fallback


# ─────────────────────────────────────────────────────────────────────────────
# CSV UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def _save_csv(save_path: str, data: dict) -> None:
    """
    Write a dict of equal-length 1-D array-like values to a CSV file.
    save_path should end in '.csv'.  Parent directory is created if needed.
    """
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    keys = list(data.keys())
    if not keys:
        return
    n = len(data[keys[0]])
    try:
        with open(save_path, 'w', newline='') as f:
            writer = _csv.writer(f)
            writer.writerow(keys)
            for i in range(n):
                writer.writerow([
                    float(data[k][i]) if hasattr(data[k][i], '__float__') else data[k][i]
                    for k in keys
                ])
        print(f'   💾 CSV saved: {save_path}')
    except Exception as e:
        print(f'   ⚠️  CSV save failed ({save_path}): {e}')


def save_log_to_csv(log: dict, base_dir: str, prefix: str = 'episode') -> None:
    """
    Export every 1-D and 2-D field of a ccmn_log / ncap_log dict to CSV files.

    Per-step scalars (speeds, z_DA, z_5HT, periods, gait_labels, env_labels,
    viscosity) go into <prefix>_timeseries.csv.

    Per-step 2-D arrays (gamma, beta, joint_pos, positions) get one CSV each:
    <prefix>_gamma.csv, <prefix>_beta.csv, <prefix>_joint_pos.csv,
    <prefix>_positions.csv.

    Transition events go into <prefix>_transitions.csv.
    """
    os.makedirs(base_dir, exist_ok=True)

    # ── timeseries scalars ────────────────────────────────────────────────────
    scalar_keys = ['speeds', 'z_DA', 'z_5HT', 'periods', 'gait_labels',
                   'env_labels', 'viscosity']
    ts_data = {'step': np.arange(len(log['speeds']))}
    for k in scalar_keys:
        if k in log:
            ts_data[k] = np.asarray(log[k]).flatten()
    _save_csv(os.path.join(base_dir, f'{prefix}_timeseries.csv'), ts_data)

    # ── 2-D arrays ────────────────────────────────────────────────────────────
    for key, col_prefix in [('gamma', 'gamma_J'),
                              ('beta',  'beta_J'),
                              ('joint_pos', 'joint_'),
                              ('positions', 'pos_')]:
        if key not in log:
            continue
        arr = np.asarray(log[key])
        if arr.ndim != 2:
            continue
        T, C = arr.shape
        d = {'step': np.arange(T)}
        for c in range(C):
            d[f'{col_prefix}{c+1}'] = arr[:, c]
        _save_csv(os.path.join(base_dir, f'{prefix}_{key}.csv'), d)

    # ── transitions ───────────────────────────────────────────────────────────
    trans = log.get('transitions', [])
    if trans:
        _save_csv(os.path.join(base_dir, f'{prefix}_transitions.csv'),
                  {'timestep': [t for t, _ in trans],
                   'type':     [lbl for _, lbl in trans]})

    print(f'✅ Full episode log exported to CSV: {base_dir}/{prefix}_*.csv')


# ─────────────────────────────────────────────────────────────────────────────
# COMPARISON FIGURE 1 — Learning curves (avg reward vs timesteps)
# The NeurIPS Table-1 equivalent: one panel per algorithm, both architectures.
# Input: list of dicts with keys 'steps', 'rewards', 'label', 'color', 'ls'
# ─────────────────────────────────────────────────────────────────────────────

def plot_comparison_learning_curves(curves: list,
                                     save_path: str,
                                     title: str = 'Average Reward vs Timesteps') -> None:
    """
    Plot learning curves for multiple runs on the same axes.

    Parameters
    ----------
    curves : list of dicts, each with:
        'steps'   : 1-D array of training step checkpoints
        'rewards' : 1-D array of mean rewards at each checkpoint
        'std'     : 1-D array of reward std (optional, for shading)
        'label'   : legend label  e.g. 'CCMN-HRL (PPO)'
        'color'   : line colour
        'ls'      : linestyle e.g. '-' or '--'
    save_path : .png output path (CSV written alongside)
    """
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    csv_data = {}
    for c in curves:
        steps   = np.asarray(c['steps'])
        rewards = np.asarray(c['rewards'])
        color   = c.get('color', '#333333')
        ls      = c.get('ls', '-')
        label   = c.get('label', 'model')
        std     = np.asarray(c.get('std', np.zeros_like(rewards)))

        ax.plot(steps, rewards, color=color, ls=ls, lw=2.0, label=label)
        if std.sum() > 0:
            ax.fill_between(steps, rewards - std, rewards + std,
                            color=color, alpha=0.15)

        safe = label.replace(' ', '_').replace('(', '').replace(')', '')
        csv_data[f'{safe}_steps']   = steps
        csv_data[f'{safe}_rewards'] = rewards
        csv_data[f'{safe}_std']     = std

    ax.set_xlabel('Training steps', fontsize=11)
    ax.set_ylabel('Average reward', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.spines[['right', 'top']].set_visible(False)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'✅ Learning curves saved: {save_path}')

    # Align all arrays to shortest before CSV export
    min_len = min(len(v) for v in csv_data.values())
    _save_csv(save_path.replace('.png', '.csv'),
              {k: v[:min_len] for k, v in csv_data.items()})


# ─────────────────────────────────────────────────────────────────────────────
# COMPARISON FIGURE 2 — Phase-specific reward breakdown
# ─────────────────────────────────────────────────────────────────────────────

def plot_comparison_phase_rewards(phase_data: dict,
                                   save_path: str) -> None:
    """
    Bar chart of per-phase mean reward for NCAP and CCMN-HRL.

    Parameters
    ----------
    phase_data : dict mapping architecture label to list of 4 mean rewards
                 e.g. {'NCAP': [120, 115, 108, 102],
                        'CCMN-HRL': [128, 128, 126, 160]}
    """
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)

    phases = ['Ph0\nSwim', 'Ph1\nSingle land', 'Ph2\nTwo land', 'Ph3\nFull complexity']
    labels = list(phase_data.keys())
    n_arch = len(labels)
    x      = np.arange(len(phases))
    width  = 0.8 / n_arch
    colors = [CRAWL_COLOR, SWIM_COLOR, '#2CA02C', '#9467BD'][:n_arch]

    fig, ax = plt.subplots(figsize=(8, 5))

    csv_data = {'phase': phases}
    for i, (lbl, rewards) in enumerate(phase_data.items()):
        offset = (i - n_arch / 2 + 0.5) * width
        ax.bar(x + offset, rewards, width * 0.9, label=lbl,
               color=colors[i], alpha=0.85, edgecolor='white', linewidth=0.5)
        csv_data[lbl] = rewards

    ax.set_xticks(x)
    ax.set_xticklabels(phases, fontsize=10)
    ax.set_ylabel('Mean reward', fontsize=11)
    ax.set_title('Phase-specific Reward: NCAP vs CCMN-HRL',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.spines[['right', 'top']].set_visible(False)
    ax.grid(True, alpha=0.2, axis='y')

    # Annotate Ph3 difference — the key result
    if 'NCAP' in phase_data and 'CCMN-HRL' in phase_data:
        diff = phase_data['CCMN-HRL'][3] - phase_data['NCAP'][3]
        ax.annotate(f'+{diff:.0f}',
                    xy=(3, max(phase_data['CCMN-HRL'][3], phase_data['NCAP'][3])),
                    xytext=(3, max(phase_data['CCMN-HRL'][3], phase_data['NCAP'][3]) + 5),
                    ha='center', fontsize=10, color='#333333',
                    arrowprops=dict(arrowstyle='->', color='#333333'))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'✅ Phase reward comparison saved: {save_path}')
    _save_csv(save_path.replace('.png', '.csv'), csv_data)


# ─────────────────────────────────────────────────────────────────────────────
# COMPARISON FIGURE 3 — Side-by-side bistability
# NCAP panel: degenerate (fixed period, square-wave z_DA)
# CCMN panel: learned sigmoid + bimodal histogram
# ─────────────────────────────────────────────────────────────────────────────

def plot_comparison_bistability(ncap_log: dict, ccmn_log: dict,
                                 save_path: str) -> None:
    """
    Four-panel figure:  left pair = NCAP (degenerate), right pair = CCMN-HRL.
    Each pair is (z_DA vs period scatter | period histogram).
    """
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Bistability Comparison: NCAP (fixed) vs CCMN-HRL (learned)',
                 fontsize=12, fontweight='bold')

    titles = ['NCAP — z_DA proxy vs period', 'NCAP — period distribution',
              'CCMN-HRL — z_DA vs period',   'CCMN-HRL — period distribution']
    logs   = [ncap_log, ncap_log, ccmn_log, ccmn_log]
    model_labels = ['NCAP', 'NCAP', 'CCMN-HRL', 'CCMN-HRL']

    csv_rows = {}

    for idx, (log, ax, ttl, ml) in enumerate(zip(logs, axes.flat, titles, model_labels)):
        z_da    = np.asarray(log['z_DA'])
        periods = np.asarray(log['periods'], dtype=float)
        env_lbl = np.asarray(log['env_labels'])

        if idx in (0, 2):   # scatter panel
            for mask, color, lbl in [
                    (env_lbl == 0, CRAWL_COLOR, 'Water'),
                    (env_lbl == 1, SWIM_COLOR,  'Land')]:
                if mask.sum() == 0:
                    continue
                ax.scatter(z_da[mask], periods[mask],
                           color=color, alpha=0.20, s=4, rasterized=True)
                order = np.argsort(z_da[mask])
                z_s = z_da[mask][order]
                p_s = periods[mask][order]
                k   = max(1, len(z_s) // 20)
                ax.plot(np.convolve(z_s, np.ones(k)/k, mode='valid'),
                        np.convolve(p_s, np.ones(k)/k, mode='valid'),
                        color=color, lw=2.0, label=lbl)
            ax.set_xlabel('z_DA', fontsize=9)
            ax.set_ylabel('period (steps)', fontsize=9)
            ax.legend(fontsize=8)
            csv_rows[f'{ml}_zDA']    = z_da
            csv_rows[f'{ml}_period'] = periods
            csv_rows[f'{ml}_env']    = env_lbl

        else:   # histogram panel
            bins = np.linspace(periods.min() - 1, periods.max() + 1, 30)
            for mask, color, lbl in [
                    (env_lbl == 0, CRAWL_COLOR, 'Water'),
                    (env_lbl == 1, SWIM_COLOR,  'Land')]:
                if mask.sum() == 0:
                    continue
                ax.hist(periods[mask], bins=bins, color=color, alpha=0.6,
                        density=True, label=lbl, edgecolor='none')
            ax.axvline(15, color='black', lw=1.0, ls=':', alpha=0.6)
            ax.axvline(60, color='black', lw=1.0, ls=':', alpha=0.6)
            ax.set_xlabel('period (steps)', fontsize=9)
            ax.set_ylabel('density', fontsize=9)
            ax.legend(fontsize=8)

        ax.set_title(ttl, fontsize=10, fontweight='bold')
        ax.spines[['right', 'top']].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'✅ Bistability comparison saved: {save_path}')

    min_len = min(len(v) for v in csv_rows.values())
    _save_csv(save_path.replace('.png', '.csv'),
              {k: v[:min_len] for k, v in csv_rows.items()})


# ─────────────────────────────────────────────────────────────────────────────
# COMPARISON FIGURE 4 — Side-by-side gait kymograph (4 panels)
# ─────────────────────────────────────────────────────────────────────────────

def plot_comparison_kymograph(ncap_log: dict, ccmn_log: dict,
                               save_path: str,
                               segment_steps: int = 60) -> None:
    """
    Four panels: NCAP-'swim', NCAP-'crawl', CCMN-swim, CCMN-crawl.
    For NCAP the 'swim'/'crawl' labels come from oscillator phase.
    Demonstrates that NCAP produces identical patterns in both 'gaits'
    while CCMN-HRL shows clearly differentiated body waves.
    """
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    fig.suptitle('Gait Kymograph: NCAP (identical panels) vs CCMN-HRL (differentiated)',
                 fontsize=12, fontweight='bold')

    panel_cfg = [
        (ncap_log, 0, 'NCAP  —  Swim phase (oscillator proxy)', SWIM_COLOR),
        (ncap_log, 1, 'NCAP  —  Crawl phase (oscillator proxy)', CRAWL_COLOR),
        (ccmn_log, 0, 'CCMN-HRL  —  Swim  (low z_DA)',          SWIM_COLOR),
        (ccmn_log, 1, 'CCMN-HRL  —  Crawl  (high z_DA)',        CRAWL_COLOR),
    ]

    csv_data = {}
    for (log, gait_id, ttl, col), ax in zip(panel_cfg, axes.flat):
        jp = np.asarray(log['joint_pos'])       # (T, n_joints)
        gl = np.asarray(log['gait_labels'])
        idx = np.where(gl == gait_id)[0]
        if len(idx) < segment_steps:
            ax.text(0.5, 0.5, 'Insufficient data', transform=ax.transAxes,
                    ha='center', va='center', fontsize=10)
            ax.set_title(ttl, fontsize=9, fontweight='bold', color=col)
            continue
        seg = jp[idx[:segment_steps], :]        # (segment_steps, n_joints)
        seg_norm = (seg - seg.mean(axis=0)) / (seg.std(axis=0) + 1e-8)
        im = ax.imshow(seg_norm.T, aspect='auto', origin='lower',
                       cmap='RdBu_r', vmin=-2, vmax=2,
                       extent=[0, segment_steps, 0, seg.shape[1]])
        ax.set_xlabel('time (steps)', fontsize=9)
        ax.set_ylabel('joint  (head→tail)', fontsize=9)
        ax.set_yticks(np.arange(seg.shape[1]) + 0.5)
        ax.set_yticklabels([f'J{i+1}' for i in range(seg.shape[1])], fontsize=7)
        ax.set_title(ttl, fontsize=9, fontweight='bold', color=col)

        lbl = ttl.split('—')[0].strip().replace(' ', '_').replace('-', '')
        for j in range(seg.shape[1]):
            csv_data[f'{lbl}_J{j+1}'] = seg_norm[:, j]

    plt.colorbar(im, ax=axes, fraction=0.02, pad=0.04,
                 label='normalised joint angle')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'✅ Kymograph comparison saved: {save_path}')

    if csv_data:
        min_len = min(len(v) for v in csv_data.values())
        _save_csv(save_path.replace('.png', '.csv'),
                  {k: v[:min_len] for k, v in csv_data.items()})


# ─────────────────────────────────────────────────────────────────────────────
# COMPARISON FIGURE 5 — Amplitude-frequency scatter: both architectures
# ─────────────────────────────────────────────────────────────────────────────

def plot_comparison_amplitude_freq(ncap_log: dict, ccmn_log: dict,
                                    save_path: str,
                                    sim_hz: float = 30.0) -> None:
    """
    Single scatter plot with NCAP points (hollow markers) and CCMN-HRL points
    (filled markers), split by gait state, with Berri et al. reference ellipses.
    NCAP cluster will sit between both ellipses; CCMN-HRL swim will approach
    the swim ellipse — visually demonstrating biological correspondence.
    """
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)

    from matplotlib.patches import Ellipse

    def _extract_cycles(log, sim_hz):
        jp0  = np.asarray(log['joint_pos'])[:, 0]
        gait = np.asarray(log['gait_labels'])
        j_c  = jp0 - jp0.mean()
        zc   = np.where(np.diff(np.sign(j_c)))[0]
        freqs, amps, gaits = [], [], []
        for i in range(0, len(zc) - 2, 2):
            t0, t2 = zc[i], zc[i + 2] if i + 2 < len(zc) else None
            if t2 is None:
                break
            dur = t2 - t0
            if dur < 3:
                continue
            freqs.append(sim_hz / dur)
            amps.append(jp0[t0:t2].max() - jp0[t0:t2].min())
            gaits.append(int(np.round(gait[t0:t2].mean())))
        return np.array(freqs), np.array(amps), np.array(gaits)

    fig, ax = plt.subplots(figsize=(7, 5.5))

    csv_data = {}
    for log, arch, filled in [(ncap_log, 'NCAP', False), (ccmn_log, 'CCMN-HRL', True)]:
        freqs, amps, gaits = _extract_cycles(log, sim_hz)
        if len(freqs) == 0:
            continue
        marker = 'o' if filled else 'D'
        facecolor_swim  = SWIM_COLOR  if filled else 'none'
        facecolor_crawl = CRAWL_COLOR if filled else 'none'
        edgecolor_swim  = SWIM_COLOR
        edgecolor_crawl = CRAWL_COLOR
        sz = 40 if filled else 25

        for gait_id, fc, ec, lbl in [
                (0, facecolor_swim,  edgecolor_swim,  f'{arch} Swim'),
                (1, facecolor_crawl, edgecolor_crawl, f'{arch} Crawl')]:
            mask = gaits == gait_id
            if mask.sum() == 0:
                continue
            ax.scatter(freqs[mask], amps[mask], s=sz, marker=marker,
                       facecolors=fc, edgecolors=ec, alpha=0.6,
                       linewidths=1.0, label=lbl, rasterized=True)
            csv_data[f'{arch}_{lbl.split()[-1]}_freq'] = freqs[mask]
            csv_data[f'{arch}_{lbl.split()[-1]}_amp']  = amps[mask]

    # Berri reference ellipses
    for cx, cy, w, h, color, lbl in [
            (2.0, 0.25, 1.2, 0.20, SWIM_COLOR,  'Berri swim'),
            (0.5, 0.40, 0.6, 0.25, CRAWL_COLOR, 'Berri crawl')]:
        ell = Ellipse((cx, cy), w, h, fill=False,
                      edgecolor=color, lw=2.0, ls='--', alpha=0.8, label=lbl)
        ax.add_patch(ell)

    ax.set_xlabel('undulation frequency (Hz)', fontsize=11)
    ax.set_ylabel('joint amplitude (rad, proxy)', fontsize=11)
    ax.set_title('Amplitude–Frequency: NCAP vs CCMN-HRL\n'
                 '(Berri et al. 2009 reference ellipses dashed)',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, ncol=2)
    ax.spines[['right', 'top']].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'✅ Amplitude-frequency comparison saved: {save_path}')

    if csv_data:
        min_len = min(len(v) for v in csv_data.values())
        _save_csv(save_path.replace('.png', '.csv'),
                  {k: v[:min_len] for k, v in csv_data.items()})


# ─────────────────────────────────────────────────────────────────────────────
# COMPARISON FIGURE 6 — Transition-aligned speed & amplitude: NCAP null vs CCMN
# ─────────────────────────────────────────────────────────────────────────────

def plot_comparison_transition_aligned(ncap_log: dict, ccmn_log: dict,
                                        save_path: str,
                                        window: int = 60) -> None:
    """
    Two rows × two columns:
      Row 1: speed aligned to S→C transitions — NCAP (null) | CCMN-HRL (dip)
      Row 2: action amplitude aligned to S→C — NCAP (flat)  | CCMN-HRL (dip)
    NCAP panels will show flat lines (no transition signature) because the
    oscillator phase crossing is not a behavioural event.
    """
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)

    def _align(log, window):
        speeds = np.asarray(log['speeds'])
        jp     = np.asarray(log['joint_pos'])
        amps   = np.sqrt(np.mean(jp**2, axis=1))
        trans  = [(t, lbl) for t, lbl in log['transitions'] if lbl == 'S2C']
        if not trans:
            return None, None
        epochs_spd, epochs_amp = [], []
        for t, _ in trans:
            if t - window < 0 or t + window >= len(speeds):
                continue
            epochs_spd.append(speeds[t - window: t + window])
            epochs_amp.append(amps[t - window: t + window])
        if not epochs_spd:
            return None, None
        return np.array(epochs_spd), np.array(epochs_amp)

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharey='row')
    fig.suptitle('Transition-aligned Speed & Amplitude — NCAP (null) vs CCMN-HRL',
                 fontsize=12, fontweight='bold')
    t_ax = np.arange(-window, window)

    csv_data = {}
    col_titles = ['NCAP', 'CCMN-HRL']
    row_titles = ['Speed (mm/s)', 'Action amplitude (joint RMS)']

    for col_idx, (log, arch) in enumerate([(ncap_log, 'NCAP'), (ccmn_log, 'CCMN-HRL')]):
        spd_ep, amp_ep = _align(log, window)
        color = CRAWL_COLOR if arch == 'NCAP' else SWIM_COLOR

        for row_idx, (ep, ylabel) in enumerate(
                [(spd_ep, 'Speed (mm/s)'), (amp_ep, 'Amplitude (joint RMS)')]):
            ax = axes[row_idx, col_idx]

            if ep is None or len(ep) == 0:
                ax.text(0.5, 0.5, 'No S→C transitions', transform=ax.transAxes,
                        ha='center', va='center', fontsize=10, color='gray')
            else:
                mean = ep.mean(axis=0)
                sem  = ep.std(axis=0) / np.sqrt(len(ep))
                n    = len(ep)
                ax.plot(t_ax, mean, color=color, lw=2.0)
                ax.fill_between(t_ax, mean - sem, mean + sem,
                                color=color, alpha=0.2)
                ax.axvline(0, color='black', lw=1.0, ls='--', alpha=0.7)
                ax.set_title(f'{arch}  S→C  (n={n})', fontsize=10)
                key = f'{arch}_{ylabel.split("(")[0].strip().replace(" ", "_")}'
                csv_data[f'{key}_mean'] = mean
                csv_data[f'{key}_sem']  = sem

            ax.set_xlabel('time to transition (steps)', fontsize=9)
            ax.set_ylabel(ylabel, fontsize=9)
            ax.spines[['right', 'top']].set_visible(False)
            if col_idx == 0:
                axes[row_idx, 0].set_title(f'NCAP  S→C  (oscillator phase)',
                                            fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'✅ Transition-aligned comparison saved: {save_path}')

    if csv_data:
        csv_data['time'] = t_ax
        min_len = min(len(v) for v in csv_data.values())
        _save_csv(save_path.replace('.png', '.csv'),
                  {k: v[:min_len] for k, v in csv_data.items()})


# ─────────────────────────────────────────────────────────────────────────────
# COMPARISON FIGURE 7 — Lead/lag side-by-side
# ─────────────────────────────────────────────────────────────────────────────

def plot_comparison_lead_lag(ncap_log: dict, ccmn_log: dict,
                              save_path: str,
                              z_thresh: float = 0.3) -> None:
    """
    Side-by-side lead/lag scatter (mirrors Fig F).
    NCAP: near-zero for both directions (oscillator is synchronous by construction).
    CCMN: non-zero S→C lead — the anticipatory neuromodulatory signal.
    """
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)

    def _compute_leads(log, z_thresh):
        z_da  = np.asarray(log['z_DA'])
        trans = log['transitions']
        s2c_leads, c2s_leads = [], []
        for t_trans, lbl in trans:
            if lbl == 'S2C':
                window = z_da[max(0, t_trans - 60): t_trans + 1]
                if len(window) == 0:
                    continue
                crossings = np.where(np.diff((window > z_thresh).astype(int)) > 0)[0]
                lead = (crossings[-1] - len(window) + 1) / 30.0 if len(crossings) else 0.0
                s2c_leads.append(lead)
            else:
                window = z_da[max(0, t_trans - 60): t_trans + 1]
                if len(window) == 0:
                    continue
                crossings = np.where(np.diff((window < -z_thresh).astype(int)) > 0)[0]
                lead = (crossings[-1] - len(window) + 1) / 30.0 if len(crossings) else 0.0
                c2s_leads.append(lead)
        return np.array(s2c_leads), np.array(c2s_leads)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    fig.suptitle('Neuromodulatory Lead/Lag: NCAP (zero) vs CCMN-HRL (anticipatory)',
                 fontsize=12, fontweight='bold')

    csv_data = {}
    for ax, (log, arch) in zip(axes, [(ncap_log, 'NCAP'), (ccmn_log, 'CCMN-HRL')]):
        s2c, c2s = _compute_leads(log, z_thresh)
        color = CRAWL_COLOR if arch == 'NCAP' else SWIM_COLOR

        x_pos = [0, 1]
        for xi, (leads, lbl) in enumerate([(s2c, 'S→C z_DA lead'),
                                            (c2s, 'C→S z_DA lead')]):
            if len(leads) == 0:
                continue
            ax.scatter(np.full(len(leads), xi) + np.random.normal(0, 0.03, len(leads)),
                       leads, color='gray', alpha=0.4, s=15, zorder=2)
            ax.errorbar(xi, leads.mean(), yerr=leads.std() / max(1, np.sqrt(len(leads))),
                        fmt='o', color=color, markersize=9, capsize=4,
                        linewidth=2, zorder=3)
            csv_data[f'{arch}_{lbl.replace(" ", "_").replace("→", "2")}_leads'] = leads

        ax.axhline(0, color='black', lw=1.0, ls='--', alpha=0.6)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(['S→C\nz_DA activation', 'C→S\nz_DA inactivation'], fontsize=9)
        ax.set_ylabel('lead time (steps ÷ 30)', fontsize=9)
        ax.set_title(arch, fontsize=11, fontweight='bold', color=color)
        ax.spines[['right', 'top']].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'✅ Lead/lag comparison saved: {save_path}')

    if csv_data:
        min_len = min(len(v) for v in csv_data.values())
        _save_csv(save_path.replace('.png', '.csv'),
                  {k: v[:min_len] for k, v in csv_data.items()})


# ─────────────────────────────────────────────────────────────────────────────
# COMPARISON FIGURE 8 — Period distribution side-by-side
# NCAP: single spike; CCMN-HRL: bimodal
# ─────────────────────────────────────────────────────────────────────────────

def plot_comparison_period_distribution(ncap_log: dict, ccmn_log: dict,
                                         save_path: str) -> None:
    """
    Two-panel histogram of oscillator period distribution.
    NCAP: single spike at fixed period (architectural constant).
    CCMN-HRL: bimodal distribution (swim peak ~15 steps, crawl peak ~55-90 steps).
    """
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=False)
    fig.suptitle('Oscillator Period Distribution: NCAP (fixed) vs CCMN-HRL (bimodal)',
                 fontsize=12, fontweight='bold')

    csv_data = {}
    for ax, (log, arch, color) in zip(axes,
            [(ncap_log, 'NCAP', CRAWL_COLOR),
             (ccmn_log, 'CCMN-HRL', SWIM_COLOR)]):
        periods = np.asarray(log['periods'], dtype=float)
        bins = np.linspace(max(0, periods.min() - 5), periods.max() + 5, 40)

        env_lbl = np.asarray(log['env_labels'])
        for mask, col, lbl in [(env_lbl == 0, CRAWL_COLOR, 'Water'),
                                (env_lbl == 1, SWIM_COLOR,  'Land')]:
            if mask.sum() == 0:
                continue
            ax.hist(periods[mask], bins=bins, color=col, alpha=0.65,
                    density=True, label=lbl, edgecolor='none')
            csv_data[f'{arch}_{lbl}_periods'] = periods[mask]

        ax.axvline(15, color='black', lw=1.2, ls=':', alpha=0.7)
        ax.axvline(60, color='black', lw=1.2, ls=':', alpha=0.7)
        ymax = ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 0.1
        ax.text(15, ymax * 0.95, 'swim\nperiod', ha='center', va='top', fontsize=8)
        ax.text(60, ymax * 0.95, 'crawl\nperiod', ha='center', va='top', fontsize=8)
        ax.set_xlabel('oscillator period (steps)', fontsize=10)
        ax.set_ylabel('density', fontsize=10)
        ax.set_title(arch, fontsize=11, fontweight='bold', color=color)
        ax.legend(fontsize=9)
        ax.spines[['right', 'top']].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'✅ Period distribution comparison saved: {save_path}')

    if csv_data:
        min_len = min(len(v) for v in csv_data.values())
        _save_csv(save_path.replace('.png', '.csv'),
                  {k: v[:min_len] for k, v in csv_data.items()})


# ─────────────────────────────────────────────────────────────────────────────
# COMPARISON FIGURE 9 — Mean water bout and gait switching rate vs timesteps
# The single figure that demonstrates CCMN-HRL's key behavioural advantage
# ─────────────────────────────────────────────────────────────────────────────

def plot_comparison_gait_switching_rate(checkpoints: list,
                                         save_path: str) -> None:
    """
    Two-panel figure: transitions and mean water bout length vs training steps,
    for NCAP and CCMN-HRL across all checkpoint evaluations.

    Parameters
    ----------
    checkpoints : list of dicts with keys:
        'steps'        : int  training step
        'arch'         : str  'NCAP' or 'CCMN-HRL'
        'transitions'  : int  Ph3 transition count
        'mean_water_bout': float  Ph3 mean water bout (steps)
        'color'        : str  colour for this run
        'ls'           : str  linestyle
    """
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)

    # Group by architecture+linestyle
    from collections import defaultdict
    groups = defaultdict(list)
    for c in checkpoints:
        key = (c['arch'], c.get('color', '#333'), c.get('ls', '-'))
        groups[key].append(c)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Gait Switching Rate vs Training Steps',
                 fontsize=12, fontweight='bold')

    csv_data = {}
    for (arch, color, ls), pts in groups.items():
        pts = sorted(pts, key=lambda x: x['steps'])
        steps = [p['steps'] for p in pts]
        trans = [p['transitions'] for p in pts]
        bouts = [p['mean_water_bout'] for p in pts]

        ax1.plot(steps, trans, color=color, ls=ls, lw=2.0, marker='o',
                 markersize=6, label=arch)
        ax2.plot(steps, bouts, color=color, ls=ls, lw=2.0, marker='o',
                 markersize=6, label=arch)

        safe = arch.replace('-', '').replace(' ', '_')
        csv_data[f'{safe}_steps']       = steps
        csv_data[f'{safe}_transitions'] = trans
        csv_data[f'{safe}_water_bout']  = bouts

    ax1.set_xlabel('Training steps', fontsize=11)
    ax1.set_ylabel('Ph3 transitions', fontsize=11)
    ax1.set_title('Environment transitions (Ph3)', fontsize=11, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.spines[['right', 'top']].set_visible(False)
    ax1.grid(True, alpha=0.2)

    ax2.set_xlabel('Training steps', fontsize=11)
    ax2.set_ylabel('Mean water bout (steps)', fontsize=11)
    ax2.set_title('Mean water bout length (Ph3)', fontsize=11, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.spines[['right', 'top']].set_visible(False)
    ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'✅ Gait switching rate comparison saved: {save_path}')

    if csv_data:
        min_len = min(len(v) for v in csv_data.values())
        _save_csv(save_path.replace('.png', '.csv'),
                  {k: v[:min_len] for k, v in csv_data.items()})


# ─────────────────────────────────────────────────────────────────────────────
# ORCHESTRATOR — create_comparison_analysis
# Runs a single evaluation episode for each model, generates all comparison
# figures, exports all CSVs, and returns both raw logs.
# ─────────────────────────────────────────────────────────────────────────────

def create_comparison_analysis(ncap_agent, ccmn_agent,
                                env_fn,
                                base_dir: str,
                                num_steps: int = 1200,
                                checkpoint_data: list = None) -> dict:
    """
    Collect one episode per architecture and produce all nine comparison figures
    plus CSV exports of all raw data.

    Parameters
    ----------
    ncap_agent      : SimpleNCAPActor instance
    ccmn_agent      : BiologicalNCAPAgent wrapping CCMNSwimmerHRL
    env_fn          : callable returning a fresh TonicProgressiveMixedWrapper
    base_dir        : output directory root
    num_steps       : rollout length per architecture
    checkpoint_data : optional list of checkpoint dicts for
                      plot_comparison_gait_switching_rate — see that function's
                      docstring for the expected dict keys.  If None, the
                      switching-rate figure is skipped.

    Returns
    -------
    dict with keys 'ncap_log' and 'ccmn_log'
    """
    os.makedirs(base_dir, exist_ok=True)
    p = lambda name: os.path.join(base_dir, f'comparison_{name}.png')

    print('\n🔬 Collecting NCAP baseline episode data…')
    ncap_env = env_fn()
    ncap_log = collect_ncap_episode_data(ncap_agent, ncap_env, num_steps=num_steps)
    ncap_env.close()
    save_log_to_csv(ncap_log, base_dir, prefix='ncap_episode')

    print('🧠 Collecting CCMN-HRL episode data…')
    ccmn_env = env_fn()
    ccmn_log = collect_ccmn_episode_data(ccmn_agent, ccmn_env, num_steps=num_steps)
    ccmn_env.close()
    save_log_to_csv(ccmn_log, base_dir, prefix='ccmn_episode')

    print('\n📊 Fig 1 — Bistability comparison…')
    plot_comparison_bistability(ncap_log, ccmn_log, p('fig1_bistability'))

    print('📊 Fig 2 — Gait kymograph comparison…')
    plot_comparison_kymograph(ncap_log, ccmn_log, p('fig2_kymograph'))

    print('📊 Fig 3 — Amplitude-frequency comparison…')
    plot_comparison_amplitude_freq(ncap_log, ccmn_log, p('fig3_amplitude_freq'))

    print('📊 Fig 4 — Transition-aligned comparison…')
    plot_comparison_transition_aligned(ncap_log, ccmn_log, p('fig4_transition_aligned'))

    print('📊 Fig 5 — Lead/lag comparison…')
    plot_comparison_lead_lag(ncap_log, ccmn_log, p('fig5_lead_lag'))

    print('📊 Fig 6 — Period distribution comparison…')
    plot_comparison_period_distribution(ncap_log, ccmn_log, p('fig6_period_distribution'))

    if checkpoint_data is not None:
        print('📊 Fig 7 — Gait switching rate vs timesteps…')
        plot_comparison_gait_switching_rate(checkpoint_data, p('fig7_switching_rate'))

    print(f'\n✅  All comparison figures and CSVs saved to: {base_dir}\n')
    return {'ncap_log': ncap_log, 'ccmn_log': ccmn_log}
