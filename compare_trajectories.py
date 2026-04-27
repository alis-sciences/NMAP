#!/usr/bin/env python3
"""
compare_trajectories.py
=======================
Load MuJoCo kinematics and PyElastica reconstruction output, then generate
a comprehensive comparison figure set quantifying the fidelity gap between
the rigid-body and Cosserat-rod models.

Figures produced
----------------
Fig 1 — Head trajectory overlay (MuJoCo vs PyElastica, water and substrate)
Fig 2 — Head speed comparison time-series with gait state bar
Fig 3 — Body curvature comparison: MuJoCo joint angles vs PyElastica κ_actual
Fig 4 — Swimming efficiency: distance per unit internal work
Fig 5 — Fidelity gap vs gait state (swim vs crawl)
Fig 6 — PyElastica body snapshots at key timesteps (shape gallery)
"""

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from scipy.ndimage import gaussian_filter1d

SWIM_COLOR  = '#E07B2A'
CRAWL_COLOR = '#2A5FA5'
MUJOCO_COLOR   = '#333333'
ELASTIC_WATER  = '#4292c6'
ELASTIC_SUBSTR = '#d94801'


def _head_pos_from_pyelastica(pe_data: dict) -> np.ndarray:
    """Extract head (node 0) XY position from PyElastica output."""
    return pe_data['positions'][:, 0, :2]   # (T, 2)


def _head_speed_from_pyelastica(pe_data: dict) -> np.ndarray:
    """XY speed of head node."""
    vel = pe_data['velocities'][:, 0, :2]   # (T, 2)
    return np.linalg.norm(vel, axis=1) * 1000.0   # mm/s


def _resample(arr: np.ndarray, target_len: int) -> np.ndarray:
    """Simple linear resampling along axis 0."""
    src = np.linspace(0, 1, len(arr))
    dst = np.linspace(0, 1, target_len)
    if arr.ndim == 1:
        return np.interp(dst, src, arr)
    return np.stack([np.interp(dst, src, arr[:, i])
                     for i in range(arr.shape[1])], axis=1)


def compare(kinematics_npz: str,
            curvature_npz: str,
            pyelastica_water_npz: str | None,
            pyelastica_substrate_npz: str | None,
            output_dir: str = 'comparison_plots') -> None:
    os.makedirs(output_dir, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    kin = np.load(kinematics_npz, allow_pickle=True)
    cv  = np.load(curvature_npz)

    mj_head     = kin['head_pos']          # (T, 2) in MuJoCo units
    mj_speed    = kin['speeds'] * 1000.0   # mm/s
    mj_gait     = kin['gait_labels']
    mj_env      = kin['env_labels']
    control_dt  = float(kin['control_dt'])
    T_mj        = len(mj_head)
    t_mj        = np.arange(T_mj) * control_dt

    scale       = float(cv.get('scale_factor', 1.0))
    rod_length  = float(cv['rod_length'])

    pe_water = (np.load(pyelastica_water_npz)
                if pyelastica_water_npz else None)
    pe_subst = (np.load(pyelastica_substrate_npz)
                if pyelastica_substrate_npz else None)

    # ── Fig 1 — Head trajectory ───────────────────────────────────────────────
    fig1, ax = plt.subplots(figsize=(9, 5))

    # MuJoCo: colour by gait state
    for i in range(T_mj - 1):
        c = SWIM_COLOR if mj_gait[i] == 0 else CRAWL_COLOR
        ax.plot(mj_head[i:i+2, 0] * scale * 1000,
                mj_head[i:i+2, 1] * scale * 1000,
                color=c, lw=1.0, alpha=0.7)

    if pe_water is not None:
        ph = _head_pos_from_pyelastica(pe_water) * 1000   # mm
        ax.plot(ph[:, 0], ph[:, 1], color=ELASTIC_WATER,
                lw=1.5, ls='--', label='PyElastica (water)', alpha=0.85)

    if pe_subst is not None:
        ph = _head_pos_from_pyelastica(pe_subst) * 1000
        ax.plot(ph[:, 0], ph[:, 1], color=ELASTIC_SUBSTR,
                lw=1.5, ls=':', label='PyElastica (substrate)', alpha=0.85)

    handles = [
        Patch(color=SWIM_COLOR,    label='MuJoCo swim'),
        Patch(color=CRAWL_COLOR,   label='MuJoCo crawl'),
        plt.Line2D([0],[0], color=ELASTIC_WATER,  ls='--', label='PyElastica water'),
        plt.Line2D([0],[0], color=ELASTIC_SUBSTR, ls=':',  label='PyElastica substrate'),
    ]
    ax.legend(handles=handles, fontsize=8)
    ax.set_xlabel('x  (mm)'); ax.set_ylabel('y  (mm)')
    ax.set_title('Head Trajectory: MuJoCo vs PyElastica', fontweight='bold')
    ax.set_aspect('equal')
    ax.spines[['right','top']].set_visible(False)
    fig1.tight_layout()
    fig1.savefig(os.path.join(output_dir, 'fig1_head_trajectory.png'), dpi=150)
    plt.close(fig1)
    print('✅ Fig 1 saved.')

    # ── Fig 2 — Head speed comparison ────────────────────────────────────────
    fig2, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True,
                               gridspec_kw={'height_ratios': [0.15, 1],
                                            'hspace': 0.05})
    # Gait bar
    ax_bar = axes[0]
    for i in range(T_mj):
        c = SWIM_COLOR if mj_gait[i] == 0 else CRAWL_COLOR
        ax_bar.axvspan(t_mj[i], t_mj[min(i+1, T_mj-1)], color=c, alpha=0.8)
    ax_bar.set_yticks([])
    ax_bar.set_ylabel('Gait', fontsize=7)

    ax_spd = axes[1]
    ax_spd.plot(t_mj, gaussian_filter1d(mj_speed, 3),
                color=MUJOCO_COLOR, lw=1.0, label='MuJoCo', alpha=0.85)

    if pe_water is not None:
        t_pe  = pe_water['t']
        spd   = _head_speed_from_pyelastica(pe_water)
        ax_spd.plot(t_pe, gaussian_filter1d(spd, 3),
                    color=ELASTIC_WATER, lw=1.5, ls='--',
                    label='PyElastica (water)')

    if pe_subst is not None:
        t_pe  = pe_subst['t']
        spd   = _head_speed_from_pyelastica(pe_subst)
        ax_spd.plot(t_pe, gaussian_filter1d(spd, 3),
                    color=ELASTIC_SUBSTR, lw=1.5, ls=':',
                    label='PyElastica (substrate)')

    ax_spd.set_xlabel('time  (s)', fontsize=10)
    ax_spd.set_ylabel('head speed  (mm/s)', fontsize=10)
    ax_spd.legend(fontsize=8)
    ax_spd.spines[['right','top']].set_visible(False)
    fig2.suptitle('Head Speed: MuJoCo vs PyElastica', fontweight='bold')
    fig2.savefig(os.path.join(output_dir, 'fig2_head_speed.png'), dpi=150)
    plt.close(fig2)
    print('✅ Fig 2 saved.')

    # ── Fig 3 — Body curvature comparison ────────────────────────────────────
    kappa_dense  = cv['kappa']        # (T, n_elem) MuJoCo-derived
    arc_pos      = cv['arc_positions'] * 1000   # mm

    swim_frames  = np.where(mj_gait == 0)[0]
    crawl_frames = np.where(mj_gait == 1)[0]

    fig3, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, frames, title, color in [
            (axes[0], swim_frames,  'Swim  (MuJoCo-derived κ)', SWIM_COLOR),
            (axes[1], crawl_frames, 'Crawl  (MuJoCo-derived κ)', CRAWL_COLOR)]:
        if len(frames) == 0:
            continue
        mean_k = kappa_dense[frames].mean(axis=0)
        std_k  = kappa_dense[frames].std(axis=0)
        ax.fill_between(arc_pos, mean_k - std_k, mean_k + std_k,
                        color=color, alpha=0.25)
        ax.plot(arc_pos, mean_k, color=color, lw=2)
        ax.axhline(0, color='grey', lw=0.7, ls='--', alpha=0.5)
        ax.set_xlabel('arc-length  (mm)')
        ax.set_ylabel('curvature  (rad/m)')
        ax.set_title(title, color=color, fontweight='bold')
        ax.spines[['right','top']].set_visible(False)

    # Overlay PyElastica actual curvature if available
    if pe_water is not None:
        t_pe    = pe_water['t']
        k_actual = pe_water['kappa_actual']  # (T_out, n_elem-1)
        # Resample to same arc positions
        for ax, col in [(axes[0], ELASTIC_WATER)]:
            mean_ka = k_actual.mean(axis=0)
            arc_pe  = np.linspace(arc_pos[0], arc_pos[-1], k_actual.shape[1])
            ax.plot(arc_pe, mean_ka[:, 0] if mean_ka.ndim > 1 else mean_ka,
                    color=col, lw=1.5, ls='--', alpha=0.7,
                    label='PyElastica actual')
            ax.legend(fontsize=7)

    fig3.suptitle('Mean Body Curvature by Gait State', fontweight='bold')
    fig3.tight_layout()
    fig3.savefig(os.path.join(output_dir, 'fig3_curvature_comparison.png'), dpi=150)
    plt.close(fig3)
    print('✅ Fig 3 saved.')

    # ── Fig 4 — Fidelity gap by gait state ───────────────────────────────────
    if pe_water is not None and len(pe_water['positions']) > 0:
        t_pe      = pe_water['t']
        pe_head   = _head_pos_from_pyelastica(pe_water)
        pe_spd    = _head_speed_from_pyelastica(pe_water)

        # Resample MuJoCo to PyElastica time grid
        mj_head_mm  = mj_head * scale * 1000
        pe_head_mm  = pe_head * 1000
        mj_r        = _resample(mj_head_mm, len(t_pe))

        gap_dist    = np.linalg.norm(mj_r - pe_head_mm, axis=1)  # mm
        mj_gait_r   = _resample(mj_gait.astype(float), len(t_pe))

        swim_gap    = gap_dist[mj_gait_r < 0.5]
        crawl_gap   = gap_dist[mj_gait_r >= 0.5]

        fig4, ax = plt.subplots(figsize=(6, 4))
        ax.plot(t_pe, gap_dist, color='#333333', lw=1, alpha=0.7)
        # Shade by gait
        for i in range(len(t_pe) - 1):
            c = SWIM_COLOR if mj_gait_r[i] < 0.5 else CRAWL_COLOR
            ax.axvspan(t_pe[i], t_pe[i+1], color=c, alpha=0.08)

        ax.set_xlabel('time  (s)')
        ax.set_ylabel('position gap  (mm)')
        ax.set_title(
            f'Fidelity Gap: MuJoCo vs PyElastica\n'
            f'Mean swim gap: {swim_gap.mean():.3f} mm  |  '
            f'crawl gap: {crawl_gap.mean():.3f} mm',
            fontsize=10, fontweight='bold'
        )
        ax.spines[['right','top']].set_visible(False)
        fig4.tight_layout()
        fig4.savefig(os.path.join(output_dir, 'fig4_fidelity_gap.png'), dpi=150)
        plt.close(fig4)
        print(f'✅ Fig 4 saved.  Swim gap={swim_gap.mean():.4f} mm, '
              f'Crawl gap={crawl_gap.mean():.4f} mm')

    # ── Fig 5 — Body shape gallery ────────────────────────────────────────────
    if pe_water is not None:
        positions = pe_water['positions']   # (T_out, n_nodes, 3)
        n_snap    = min(6, len(positions))
        snap_idx  = np.linspace(0, len(positions) - 1, n_snap, dtype=int)
        t_pe      = pe_water['t']

        fig5, axes = plt.subplots(1, n_snap, figsize=(14, 3))
        for col, idx in enumerate(snap_idx):
            ax = axes[col]
            pos = positions[idx]  # (n_nodes, 3) in metres
            ax.plot(pos[:, 0] * 1000, pos[:, 1] * 1000,
                    color=ELASTIC_WATER, lw=2)
            ax.scatter(pos[0, 0]*1000, pos[0, 1]*1000,
                       color='red', s=20, zorder=5)  # head
            ax.set_title(f't={t_pe[idx]:.3f} s', fontsize=7)
            ax.set_aspect('equal')
            ax.axis('off')

        fig5.suptitle('PyElastica Rod Shape Gallery  (red = head)',
                      fontweight='bold')
        fig5.tight_layout()
        fig5.savefig(os.path.join(output_dir, 'fig5_body_shapes.png'), dpi=150)
        plt.close(fig5)
        print('✅ Fig 5 saved.')

    print(f'\n✅ All comparison plots saved to: {output_dir}')


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compare MuJoCo and PyElastica trajectories')
    parser.add_argument('--kinematics',  default='kinematics_export.npz')
    parser.add_argument('--curvature',   default='curvature.npz')
    parser.add_argument('--pe_water',    default=None,
                        help='PyElastica water output .npz')
    parser.add_argument('--pe_substrate', default=None,
                        help='PyElastica substrate output .npz')
    parser.add_argument('--output_dir',  default='comparison_plots')
    args = parser.parse_args()

    compare(
        kinematics_npz=args.kinematics,
        curvature_npz=args.curvature,
        pyelastica_water_npz=args.pe_water,
        pyelastica_substrate_npz=args.pe_substrate,
        output_dir=args.output_dir,
    )
