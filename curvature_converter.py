#!/usr/bin/env python3
"""
curvature_converter.py
======================
Convert MuJoCo 6-link rigid-body joint angles to distributed curvature
κ(s, t) along the PyElastica Cosserat rod arc-length.

The mapping:
    MuJoCo joint angle θᵢ (rad) at arc-length position sᵢ
        ↓  divide by segment length
    Local curvature κᵢ = θᵢ / L_seg  (rad/m)
        ↓  cubic spline interpolation
    Dense curvature κ(s, t) at n_elem PyElastica element positions

Also extracts segment geometry from the MJCF XML and computes the
arc-length positions of each MuJoCo joint.

Output .npz:
    kappa           : (T, n_elem)   curvature at each rod element (rad/m)
    kappa_dot       : (T, n_elem)   curvature rate (rad/m/s)
    arc_positions   : (n_elem,)     arc-length position of each element (m)
    joint_arc_pos   : (n_joints,)   arc-length of each MuJoCo joint (m)
    rod_length      : scalar        total rod length (m)
    L_seg           : scalar        individual segment length (m)
    control_dt      : scalar        timestep (s)
    n_joints        : scalar
    n_elem          : scalar
    t               : (T,)          time axis (s)
"""

import argparse
import re
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d


# ── C. elegans biological scaling ─────────────────────────────────────────────
# The MuJoCo swimmer is dimensionless by default.  We scale to C. elegans
# biology (Fang-Yen et al. 2010) so that PyElastica uses physically meaningful
# SI units.
_WORM_LENGTH_M    = 1.0e-3    # 1 mm body length
_WORM_RADIUS_M    = 4.0e-5    # 40 µm mean radius
_ELASTIC_MODULUS  = 1.0e4     # 10 kPa (Fang-Yen et al. 2010)
_SHEAR_MODULUS    = 3.3e3     # ≈ E/3 (nearly incompressible)
_DENSITY_KG_M3    = 1000.0    # matched to water


def _parse_segment_length_from_mjcf(mjcf_xml: str, n_links: int) -> float:
    """
    Parse the half-length of each capsule body from the MJCF XML.
    Returns the full length of one segment in MuJoCo units.
    Falls back to L_total / n_links if parsing fails.
    """
    # dm_control swimmer XML uses geom fromto="0 0 0  L 0 0" per link
    pattern = re.compile(r'fromto\s*=\s*"([^"]+)"')
    matches = pattern.findall(mjcf_xml)
    lengths = []
    for m in matches:
        nums = [float(x) for x in m.split()]
        if len(nums) == 6:
            dx = nums[3] - nums[0]
            dy = nums[4] - nums[1]
            dz = nums[5] - nums[2]
            L  = float(np.sqrt(dx**2 + dy**2 + dz**2))
            if L > 1e-6:
                lengths.append(L)
    if lengths:
        return float(np.median(lengths))
    # Fallback: assume uniform segments spanning a normalised unit length
    return 1.0 / n_links


def convert_to_curvature(export_npz_path: str,
                          output_path: str = 'curvature.npz',
                          n_elem: int = 60,
                          smooth_sigma: float = 1.0) -> str:
    """
    Load the kinematic export and produce the curvature time-series.

    Parameters
    ----------
    export_npz_path : path to output of export_kinematics.py
    output_path     : where to save the curvature .npz
    n_elem          : number of PyElastica rod elements (should be ≥ 50
                      for convergence at this body length)
    smooth_sigma    : Gaussian smoothing sigma in timesteps applied to
                      the raw curvature before upsampling (reduces
                      high-frequency noise from rigid-body discretisation)
    """
    print(f'📂 Loading kinematics: {export_npz_path}')
    data = np.load(export_npz_path, allow_pickle=True)

    joint_angles = data['joint_angles']   # (T, n_joints)
    joint_vels   = data['joint_velocities']
    control_dt   = float(data['control_dt'])
    n_links      = int(data['n_links'])
    n_joints     = n_links - 1
    T            = joint_angles.shape[0]

    mjcf_xml     = str(data['mjcf_xml'])

    # ── Segment geometry ─────────────────────────────────────────────────────
    L_seg_mujoco = _parse_segment_length_from_mjcf(mjcf_xml, n_links)
    print(f'   MuJoCo segment length: {L_seg_mujoco:.4f} (model units)')

    # Scale MuJoCo segment length to biological C. elegans dimensions
    # Total MuJoCo rod = n_links * L_seg_mujoco → maps to _WORM_LENGTH_M
    scale_factor = _WORM_LENGTH_M / (n_links * L_seg_mujoco)
    L_seg_m      = L_seg_mujoco * scale_factor   # segment length in metres
    rod_length   = _WORM_LENGTH_M                # = n_links * L_seg_m
    print(f'   Biological rod length : {rod_length*1000:.3f} mm')
    print(f'   Segment length        : {L_seg_m*1e6:.1f} µm')

    # Arc-length positions of MuJoCo joints (centred within each inter-segment gap)
    # Joint i sits between segment i and segment i+1
    joint_arc_pos = np.array(
        [(i + 1) * L_seg_m for i in range(n_joints)], dtype=np.float64
    )

    # Dense arc-length grid for PyElastica elements (element centres)
    arc_positions = np.linspace(
        L_seg_m / 2, rod_length - L_seg_m / 2, n_elem, dtype=np.float64
    )

    # ── Convert joint angles to local curvature ───────────────────────────────
    # κᵢ = θᵢ / L_seg  (small-angle RFT approximation, valid for undulation)
    # Sign convention: positive κ = dorsal bend (matches MuJoCo qpos convention)
    kappa_at_joints = joint_angles / L_seg_m      # (T, n_joints), rad/m

    # Optional temporal smoothing to remove rigid-body discretisation artefacts
    if smooth_sigma > 0:
        kappa_at_joints = gaussian_filter1d(
            kappa_at_joints, sigma=smooth_sigma, axis=0
        )

    # ── Upsample to dense rod via cubic spline ────────────────────────────────
    print(f'   Upsampling {n_joints} joints → {n_elem} rod elements…')
    kappa_dense = np.zeros((T, n_elem), dtype=np.float32)

    for t in range(T):
        # Add boundary conditions: curvature = 0 at head and tail
        # (free-end boundary condition for C. elegans)
        s_pts = np.concatenate([[0.0], joint_arc_pos, [rod_length]])
        k_pts = np.concatenate([[0.0], kappa_at_joints[t], [0.0]])
        cs    = CubicSpline(s_pts, k_pts, bc_type='not-a-knot')
        kappa_dense[t] = cs(arc_positions).astype(np.float32)

    # ── Curvature rate via finite differences ─────────────────────────────────
    kappa_dot = np.gradient(kappa_dense, control_dt, axis=0).astype(np.float32)

    # ── Time axis ─────────────────────────────────────────────────────────────
    t_axis = np.arange(T, dtype=np.float32) * control_dt

    # ── Save ──────────────────────────────────────────────────────────────────
    np.savez(
        output_path,
        kappa=kappa_dense,
        kappa_dot=kappa_dot,
        arc_positions=arc_positions.astype(np.float32),
        joint_arc_pos=joint_arc_pos.astype(np.float32),
        rod_length=np.float32(rod_length),
        L_seg=np.float32(L_seg_m),
        control_dt=np.float32(control_dt),
        n_joints=np.int32(n_joints),
        n_elem=np.int32(n_elem),
        t=t_axis,
        # Biological parameters for PyElastica rod construction
        worm_radius=np.float32(_WORM_RADIUS_M),
        elastic_modulus=np.float32(_ELASTIC_MODULUS),
        shear_modulus=np.float32(_SHEAR_MODULUS),
        density=np.float32(_DENSITY_KG_M3),
        scale_factor=np.float32(scale_factor),
    )

    print(f'✅ Curvature saved: {output_path}')
    print(f'   kappa shape : {kappa_dense.shape}')
    print(f'   kappa range : [{kappa_dense.min():.1f}, {kappa_dense.max():.1f}] rad/m')
    print(f'   Equivalent undulation wavelength: '
          f'{2*np.pi / (np.abs(kappa_dense).mean() + 1e-6)*1000:.2f} mm')
    return output_path


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Convert MuJoCo joint angles to Cosserat curvature')
    parser.add_argument('--input', default='kinematics_export.npz',
                        help='Output of export_kinematics.py')
    parser.add_argument('--output', default='curvature.npz')
    parser.add_argument('--n_elem', type=int, default=60,
                        help='Number of PyElastica rod elements')
    parser.add_argument('--smooth_sigma', type=float, default=1.0,
                        help='Gaussian smoothing in timesteps (0 = off)')
    args = parser.parse_args()

    convert_to_curvature(
        export_npz_path=args.input,
        output_path=args.output,
        n_elem=args.n_elem,
        smooth_sigma=args.smooth_sigma,
    )
