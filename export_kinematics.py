#!/usr/bin/env python3
"""
export_kinematics.py
====================
Load a trained CCMN checkpoint and roll out one evaluation episode,
recording all kinematic and neuromodulatory signals needed for
PyElastica reconstruction.

Saves a structured .npz file with:
    joint_angles   : (T, n_joints)   dorsoventral joint angles qpos[3:]
    joint_velocities:(T, n_joints)   joint velocities qvel[3:]
    segment_xpos   : (T, n_seg, 3)   3-D position of every segment body
    segment_xvel   : (T, n_seg, 3)   3-D velocity of every segment body
    head_pos       : (T, 2)          XY head position
    speeds         : (T,)            axial head speed (m/s)
    z_DA           : (T,)            neuromodulatory context
    z_5HT          : (T,)            antagonist context signal
    gait_labels    : (T,)            0=swim, 1=crawl
    env_labels     : (T,)            0=water, 1=land
    viscosity      : (T,)            local viscosity
    gamma          : (T, n_joints)   FiLM gain per joint
    control_dt     : scalar          MuJoCo control timestep (s)
    n_links        : scalar          number of rigid links
    segment_names  : list of str     body names in head→tail order
    mjcf_xml       : str             full MJCF model XML for geometry extraction
"""

import os
import sys
import argparse
import numpy as np
import torch

# ── path setup ────────────────────────────────────────────────────────────────
# Run from the workspace root: python export_kinematics.py --checkpoint ...
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _get_physics(env):
    """Traverse wrapper layers to reach the dm_control physics object."""
    if hasattr(env, 'env') and hasattr(env.env, 'env'):
        return env.env.env.physics
    if hasattr(env, 'env') and hasattr(env.env, 'physics'):
        return env.env.physics
    if hasattr(env, 'physics'):
        return env.physics
    return None


def export_kinematics(checkpoint_path: str,
                      n_links: int = 6,
                      num_steps: int = 1800,
                      output_path: str = 'kinematics_export.npz',
                      training_progress: float = 0.7,
                      oscillator_period: int = 60,
                      anisotropic_drag_mode: str = 'proxy',
                      anisotropic_drag_ratio: float = 12.0,
                      anisotropic_drag_gain: float = 0.04,
                      device_str: str = 'cpu'):
    """
    Roll out the trained CCMN model for num_steps and export all kinematics.
    training_progress=0.7 places the environment in Phase 3 (two land zones)
    so both swim and crawl gait epochs are captured.
    """
    device = torch.device(device_str)

    # ── Load model ────────────────────────────────────────────────────────────
    print(f'📂 Loading checkpoint: {checkpoint_path}')
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Extract model state dict — handle both full checkpoint and bare state dict
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
    else:
        state_dict = ckpt

    # Instantiate CCMNSwimmer
    from swimmer.models.ncap_ccmn import CCMNSwimmer
    n_joints = n_links - 1
    model = CCMNSwimmer(
        n_joints=n_joints,
        oscillator_period=oscillator_period,
        context_hidden=16,
        use_weight_sharing=True,
        use_weight_constraints=True,
        include_proprioception=True,
        include_head_oscillators=True,
        use_viscosity_input=False,
        use_amplitude_scaling=True,
    ).to(device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.reset()
    print(f'✅ Model loaded: {sum(p.numel() for p in model.parameters())} parameters')

    # ── Build environment ─────────────────────────────────────────────────────
    from swimmer.environments.progressive_mixed_env import TonicProgressiveMixedWrapper
    env = TonicProgressiveMixedWrapper(
        n_links=n_links,
        time_feature=True,
        expose_environment_observation=True,
        expose_viscosity_observation=True,
        anisotropic_drag_mode=anisotropic_drag_mode,
        anisotropic_drag_ratio=anisotropic_drag_ratio,
        anisotropic_drag_gain=anisotropic_drag_gain,
        anisotropic_drag_land_only=False,
        use_metabolic_bonus=True,
    )
    
    # With this:
    if hasattr(env, 'set_manual_progress'):
        env.set_manual_progress(training_progress)
    elif hasattr(env, 'env') and hasattr(env.env, 'set_training_progress'):
        env.env.set_training_progress(training_progress)
    elif hasattr(env, 'env') and hasattr(env.env, 'training_progress'):
        env.env.training_progress = training_progress
    
    obs = env.reset()

    # ── Discover segment body names in head→tail order ────────────────────────
    phys = _get_physics(env)
    all_names = list(phys.named.data.xpos.axes.row.names)
    segment_names = [
        n for n in all_names
        if any(k in n for k in ('head', 'seg', 'link', 'torso', 'body', 'mid'))
        and 'world' not in n
    ]
    n_seg = len(segment_names)
    print(f'   Segment bodies detected: {segment_names}')

    # ── Export MJCF XML for geometry extraction ───────────────────────────────
    from dm_control.suite import swimmer as dm_swimmer
    mjcf_xml, _ = dm_swimmer.get_model_and_assets(n_links)
    control_dt = float(dm_swimmer._CONTROL_TIMESTEP)
    print(f'   Control timestep: {control_dt} s  ({1/control_dt:.1f} Hz)')

    # ── Storage buffers ───────────────────────────────────────────────────────
    buf = dict(
        joint_angles=[],
        joint_velocities=[],
        segment_xpos=[],       # (T, n_seg, 3)
        segment_xvel=[],       # (T, n_seg, 3)
        head_pos=[],
        speeds=[],
        z_DA=[],
        z_5HT=[],
        gait_labels=[],
        env_labels=[],
        viscosity=[],
        gamma=[],
    )

    env_feat_start = n_joints + (n_joints * 3 + 3)  # after joints + body_vel

    print(f'🎬 Rolling out {num_steps} steps…')
    for t in range(num_steps):
        if t % 200 == 0:
            print(f'   step {t}/{num_steps}', end='\r')

        obs_np = np.asarray(obs, dtype=np.float32)

        # Forward pass
        jp_t = torch.tensor(obs_np[:n_joints], dtype=torch.float32,
                            device=device).unsqueeze(0)
        vis_scalar = float(obs_np[env_feat_start]) if len(obs_np) > env_feat_start else 0.0
        env_label  = 1 if (len(obs_np) > env_feat_start + 2 and
                            obs_np[env_feat_start + 2] > 0.5) else 0
        t_tensor = torch.tensor([t], dtype=torch.float32, device=device)

        with torch.no_grad():
            _ = model(jp_t, viscosity_norm=vis_scalar, timesteps=t_tensor)

        nm = model.neuromod_state()
        z_da = nm['z_DA']
        gait = 1 if z_da > 0.0 else 0

        # ── Physics data ───────────────────────────────────────────────────
        phys = _get_physics(env)

        # Joint angles and velocities (exclude root qpos[0:3])
        try:
            qpos_full = phys.data.qpos.copy()
            qvel_full = phys.data.qvel.copy()
            joint_angles = qpos_full[3:3 + n_joints]
            joint_vels   = qvel_full[3:3 + n_joints]
        except Exception:
            joint_angles = obs_np[:n_joints].copy()
            joint_vels   = np.zeros(n_joints)

        # All segment 3-D positions and velocities
        seg_xpos = np.zeros((n_seg, 3), dtype=np.float32)
        seg_xvel = np.zeros((n_seg, 3), dtype=np.float32)
        for si, sname in enumerate(segment_names):
            try:
                seg_xpos[si] = phys.named.data.xpos[sname].copy()
                # cvel layout: [ang_vel(3), lin_vel(3)]
                cv = phys.named.data.cvel[sname]
                seg_xvel[si] = cv[3:6]
            except Exception:
                pass

        # Head position and speed
        try:
            head_pos = phys.named.data.xpos['head'][:2].copy()
            cv_head  = phys.named.data.cvel['head']
            speed    = float(np.linalg.norm(cv_head[3:5]))
        except Exception:
            head_pos = np.zeros(2)
            speed    = 0.0

        # ── Store ─────────────────────────────────────────────────────────
        buf['joint_angles'].append(joint_angles.astype(np.float32))
        buf['joint_velocities'].append(joint_vels.astype(np.float32))
        buf['segment_xpos'].append(seg_xpos)
        buf['segment_xvel'].append(seg_xvel)
        buf['head_pos'].append(head_pos.astype(np.float32))
        buf['speeds'].append(speed)
        buf['z_DA'].append(z_da)
        buf['z_5HT'].append(nm['z_5HT'])
        buf['gait_labels'].append(gait)
        buf['env_labels'].append(env_label)
        buf['viscosity'].append(vis_scalar)
        buf['gamma'].append(nm['gamma'])

        action = model(jp_t, viscosity_norm=vis_scalar,
                       timesteps=t_tensor).detach().cpu().numpy().flatten()
        obs, _, done, _ = env.step(action)
        if done:
            obs = env.reset()
            model.reset()

    print(f'\n✅ Rollout complete.')

    # ── Convert and save ──────────────────────────────────────────────────────
    arrays = {k: np.array(v, dtype=np.float32) for k, v in buf.items()}
    arrays['segment_xpos'] = np.stack(buf['segment_xpos']).astype(np.float32)
    arrays['segment_xvel'] = np.stack(buf['segment_xvel']).astype(np.float32)

    np.savez(
        output_path,
        **arrays,
        control_dt=np.float32(control_dt),
        n_links=np.int32(n_links),
        segment_names=np.array(segment_names),
        mjcf_xml=np.array(mjcf_xml),
    )

    swim_frac = (arrays['gait_labels'] == 0).mean()
    n_trans = sum(1 for i in range(1, len(arrays['gait_labels']))
                  if arrays['gait_labels'][i] != arrays['gait_labels'][i-1])

    print(f'💾 Saved: {output_path}')
    print(f'   Shape summary:')
    print(f'     joint_angles    : {arrays["joint_angles"].shape}')
    print(f'     segment_xpos    : {arrays["segment_xpos"].shape}')
    print(f'     z_DA range      : [{arrays["z_DA"].min():.3f}, {arrays["z_DA"].max():.3f}]')
    print(f'     Swim fraction   : {swim_frac:.1%}')
    print(f'     Gait transitions: {n_trans}')
    return output_path


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export MuJoCo kinematics for PyElastica')
    parser.add_argument('--checkpoint', required=True,
                        help='Path to trained .pt checkpoint')
    parser.add_argument('--n_links', type=int, default=6)
    parser.add_argument('--num_steps', type=int, default=1800,
                        help='Episode length to export (default: 1800)')
    parser.add_argument('--output', default='kinematics_export.npz')
    parser.add_argument('--training_progress', type=float, default=0.7,
                        help='Curriculum progress 0-1 (0.7 = Phase 3, mixed env)')
    parser.add_argument('--oscillator_period', type=int, default=60)
    parser.add_argument('--anisotropic_drag_mode', default='proxy')
    parser.add_argument('--anisotropic_drag_ratio', type=float, default=12.0)
    parser.add_argument('--anisotropic_drag_gain', type=float, default=0.04)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    export_kinematics(
        checkpoint_path=args.checkpoint,
        n_links=args.n_links,
        num_steps=args.num_steps,
        output_path=args.output,
        training_progress=args.training_progress,
        oscillator_period=args.oscillator_period,
        anisotropic_drag_mode=args.anisotropic_drag_mode,
        anisotropic_drag_ratio=args.anisotropic_drag_ratio,
        anisotropic_drag_gain=args.anisotropic_drag_gain,
        device_str=args.device,
    )
