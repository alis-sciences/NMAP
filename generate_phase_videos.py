#!/usr/bin/env python3
"""
generate_phase_videos.py
========================
Standalone script that loads a trained CCMN-HRL (or any supported model)
checkpoint and renders one video per training phase (0–3), clearly showing
the different gaits on water and land.

Each video shows:
  • Phase label and training progress in the top-left corner
  • Land zones as semi-transparent brown disks with labels
  • Swimmer indicator colour-coded by substrate:
      Cyan  = swimming (water)
      Orange = crawling (land)
  • Minimap with swimmer trail in the top-right corner
  • Gait indicator (SWIM / CRAWL) updated each step
  • z_DA context signal bar (CCMN only) showing neuromodulation level

Produces per-phase MP4 + GIF preview + contact-sheet PNG, plus a
combined side-by-side comparison video across all four phases.

Usage
-----
  # Basic (uses final checkpoint automatically)
  python generate_phase_videos.py \\
      --checkpoint_dir /workspace/.../outputs/ccmn_hrl_ppo_3M/curriculum_training/checkpoints/ccmn_hrl \\
      --out_dir /workspace/videos/

  # Specify a particular checkpoint step
  python generate_phase_videos.py \\
      --checkpoint_dir /workspace/.../checkpoints/ccmn_hrl \\
      --checkpoint_step 2950000 \\
      --out_dir /workspace/videos/

  # Override model type and number of links
  python generate_phase_videos.py \\
      --checkpoint_dir /workspace/.../checkpoints/ccmn_hrl \\
      --model_type ccmn_hrl --n_links 6 \\
      --out_dir /workspace/videos/

  # Shorter videos for quick preview
  python generate_phase_videos.py \\
      --checkpoint_dir /workspace/.../checkpoints/ccmn_hrl \\
      --num_steps 500 --out_dir /workspace/videos/

Requirements
------------
  opencv-python (pip install opencv-python-headless)
  imageio[ffmpeg]
  torch, numpy, matplotlib
  dm_control
  The swimmer package must be importable (run from project root or set PYTHONPATH)
"""

import argparse
import os
import sys
import re
import glob
import json
import numpy as np
import torch
import imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False
    print("⚠️  OpenCV not available — install with: pip install opencv-python-headless")
    print("   Videos will be saved without HUD overlays.\n")

# ─────────────────────────────────────────────────────────────────────────────
# Phase configuration
# ─────────────────────────────────────────────────────────────────────────────

PHASES = [
    {
        'idx':      0,
        'name':     'Phase 0 — Pure Water',
        'progress': 0.15,      # mid-point of 0–30%
        'start_pos': None,     # origin (water)
        'colour':   (0, 200, 255),   # cyan
    },
    {
        'idx':      1,
        'name':     'Phase 1 — One Island',
        'progress': 0.45,      # mid-point of 30–60%
        'start_pos': [3.0, 0.0],     # inside the single island
        'colour':   (50, 180, 50),
    },
    {
        'idx':      2,
        'name':     'Phase 2 — Two Islands',
        'progress': 0.70,      # mid-point of 60–80%
        'start_pos': [-2.0, 0.0],    # inside left island
        'colour':   (255, 150, 0),
    },
    {
        'idx':      3,
        'name':     'Phase 3 — Four Islands',
        'progress': 0.90,      # mid-point of 80–100%
        'start_pos': [-3.0, 0.0],    # inside one of the four islands
        'colour':   (200, 0, 200),
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def list_checkpoints(checkpoint_dir: str):
    """Return sorted list of (step, path) from the checkpoint directory."""
    results = []

    # Format D: curriculum trainer  {prefix}_checkpoint_step_N.pt
    for f in glob.glob(os.path.join(checkpoint_dir, '*_checkpoint_step_*.pt')):
        m = re.search(r'_checkpoint_step_(\d+)\.pt$', f)
        if m:
            results.append((int(m.group(1)), f[:-3]))   # strip .pt

    # Format A: simple trainer  checkpoint_N.pt
    for f in glob.glob(os.path.join(checkpoint_dir, 'checkpoint_*.pt')):
        m = re.search(r'checkpoint_(\d+)\.pt$', f)
        if m and '_step_' not in os.path.basename(f):
            results.append((int(m.group(1)), f[:-3]))

    # Format B: ES  checkpoint_N_es_theta.npy
    for f in glob.glob(os.path.join(checkpoint_dir, '*_es_theta.npy')):
        m = re.search(r'checkpoint_(\d+)_es_theta\.npy$', f)
        if m:
            results.append((int(m.group(1)),
                             f.replace('_es_theta.npy', '')))

    results.sort(key=lambda x: x[0])
    return results


def load_checkpoint(checkpoint_path: str, n_links: int,
                    oscillator_period: int, model_type: str,
                    device: str = 'cpu'):
    """
    Load a checkpoint into the appropriate model class.
    Returns (model, step).
    """
    # ── Step from filename ────────────────────────────────────────────────────
    step = 0
    base = os.path.basename(checkpoint_path)
    m = re.search(r'_checkpoint_step_(\d+)$', base)
    if m:
        step = int(m.group(1))
    else:
        meta = checkpoint_path + '_meta.json'
        if os.path.exists(meta):
            with open(meta) as f:
                step = json.load(f).get('step', 0)
        else:
            m2 = re.search(r'checkpoint_(\d+)$', base)
            if m2:
                step = int(m2.group(1))

    # ── Build model ───────────────────────────────────────────────────────────
    n_joints = n_links - 1
    model = _build_model(model_type, n_joints, oscillator_period, device)

    # ── Load weights ──────────────────────────────────────────────────────────
    loaded = False

    # ES flat numpy vector
    npy = checkpoint_path + '_es_theta.npy'
    if os.path.exists(npy):
        theta  = np.load(npy)
        shapes = [p.data.shape for p in model.parameters()]
        sizes  = [p.data.numel() for p in model.parameters()]
        offset = 0
        with torch.no_grad():
            for param, shape, size in zip(model.parameters(), shapes, sizes):
                param.data.copy_(
                    torch.tensor(theta[offset:offset+size].reshape(shape),
                                 dtype=torch.float32))
                offset += size
        loaded = True
        print(f"  [load] ES npy  step={step:,}")

    if not loaded:
        # Determine target device from model so state dict goes directly there.
        # Using map_location='cpu' then load_state_dict copies CPU tensors into
        # the model — overwriting CUDA parameters with CPU ones when the model
        # lives on GPU.
        model_device = str(next(model.parameters()).device)
        for cand in [checkpoint_path, checkpoint_path + '.pt']:
            if os.path.exists(cand):
                raw = torch.load(cand, map_location=model_device, weights_only=False)
                if isinstance(raw, dict):
                    state = (raw.get('model_state_dict')
                             or raw.get('model')
                             or raw)
                else:
                    state = raw

                model_keys = set(model.state_dict().keys())
                ckpt_keys  = set(state.keys()) if isinstance(state, dict) else set()
                matched    = model_keys & ckpt_keys

                if len(matched) == 0 and isinstance(state, dict):
                    # Curriculum trainer saves flat keys — remap to model structure
                    remapped = {}
                    for ck, cv in state.items():
                        for prefix in ('actor.ncap.', 'actor.', ''):
                            candidate = prefix + ck
                            if candidate in model_keys:
                                remapped[candidate] = cv
                                break
                    if remapped:
                        print(f"  [load] remapped {len(remapped)} keys from flat CCMN format")
                        state = {**model.state_dict(), **remapped}

                model.load_state_dict(state, strict=False)
                loaded = True
                print(f"  [load] {os.path.basename(cand)}  step={step:,}  "
                      f"matched {len(matched)}/{len(model_keys)} keys")
                break

    model.eval()
    return model, step


def _build_model(model_type: str, n_joints: int,
                 oscillator_period: int, device: str):
    """Build the correct model class."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    if model_type == 'ccmn_hrl':
        from swimmer.models.ncap_ccmn_hrl import CCMNSwimmerHRL
        model = CCMNSwimmerHRL(
            n_joints=n_joints,
            oscillator_period=oscillator_period,
            context_hidden=16,
            use_weight_sharing=True,
            use_weight_constraints=True,
            include_proprioception=True,
            include_head_oscillators=True,
            use_viscosity_input=False,
            use_amplitude_scaling=True,
        )
    elif model_type in ('ccmn', 'ccmn_hrl_legacy'):
        from swimmer.models.ncap_ccmn_hrl import CCMNSwimmerHRL
        model = CCMNSwimmerHRL(
            n_joints=n_joints,
            oscillator_period=oscillator_period,
        )
    elif model_type == 'simple_ncap':
        from swimmer.models.simple_ncap import SimpleNCAPSwimmer
        model = SimpleNCAPSwimmer(
            n_joints=n_joints,
            oscillator_period=oscillator_period,
            use_weight_sharing=True,
        )
    elif model_type == 'biological_ncap':
        from swimmer.models.biological_ncap import BiologicalNCAPSwimmer
        model = BiologicalNCAPSwimmer(
            n_joints=n_joints,
            oscillator_period=oscillator_period,
        )
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    return model.to(device)


# ─────────────────────────────────────────────────────────────────────────────
# Environment helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_env(n_links: int, training_progress: float,
              expose_env_obs: bool = True,
              use_ext_env: bool = False):
    """
    Build a single (non-vectorised) environment at the requested phase.
    Returns a Tonic-style wrapper whose step() uses the plain 4-tuple format.
    """
    if use_ext_env:
        from swimmer.environments.progressive_ext_env import (
            TonicProgressiveMixedWrapperExt as WrapperClass
        )
    else:
        from swimmer.environments.progressive_mixed_env import (
            TonicProgressiveMixedWrapper as WrapperClass
        )

    env = WrapperClass(
        n_links=n_links,
        time_feature=True,
        desired_speed=0.15,
        expose_environment_observation=expose_env_obs,
        expose_viscosity_observation=True,
    )
    # Set phase
    if hasattr(env, 'env') and hasattr(env.env, 'env') and hasattr(env.env.env, '_task'):
        env.env.env._task._training_progress = training_progress
        if hasattr(env.env.env._task, '_update_land_zones'):
            env.env.env._task._update_land_zones()
        elif hasattr(env.env.env._task, '_current_land_zones'):
            env.env.env._task._current_land_zones = \
                env.env.env._task._get_land_zones() \
                if hasattr(env.env.env._task, '_get_land_zones') else []
    return env


def teleport_swimmer(env, pos_xy):
    """Move the swimmer's root to pos_xy immediately after reset."""
    if pos_xy is None:
        return
    try:
        phys = env.env.env.physics
        try:
            phys.named.data.qpos['root'][0] = float(pos_xy[0])
            phys.named.data.qpos['root'][1] = float(pos_xy[1])
        except (KeyError, IndexError):
            phys.data.qpos[0] = float(pos_xy[0])
            phys.data.qpos[1] = float(pos_xy[1])
        phys.forward()
    except Exception as e:
        print(f"  ⚠️  teleport failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Agent wrapper
# ─────────────────────────────────────────────────────────────────────────────

class VideoAgent:
    """
    Minimal agent adapter that calls the model's forward() with a single
    flat numpy observation and returns a numpy action.  Works for all
    supported model types.
    """

    def __init__(self, model, n_joints: int, model_type: str,
                 env_features_start: int, device: str = 'cpu'):
        self.model            = model
        self.n_joints         = n_joints
        self.model_type       = model_type
        self.env_features_start = env_features_start
        self.device           = device
        self._step_count      = 0
        self._last_z_da       = 0.0   # for HUD display

    def test_step(self, obs):
        """obs: flat numpy array (obs_dim,)"""
        if isinstance(obs, (list, tuple)):
            obs = obs[0] if len(obs) == 1 else np.array(obs, dtype=np.float32)
        obs = np.asarray(obs, dtype=np.float32).ravel()

        device = self.device
        device = str(next(self.model.parameters()).device)  # always follow model
        self.device = device   # keep in sync
        jp = torch.as_tensor(
            obs[:self.n_joints], dtype=torch.float32, device=device
        ).unsqueeze(0)
        t  = torch.tensor([float(self._step_count)],
                           dtype=torch.float32, device=device)

        vis_norm = 0.0
        if obs.shape[0] > self.env_features_start:
            vis_norm = float(obs[self.env_features_start])

        with torch.no_grad():
            if self.model_type in ('ccmn_hrl', 'ccmn'):
                out = self.model(jp, viscosity_norm=vis_norm, timesteps=t,
                                 worker_id=0)
                # Try to capture z_DA for HUD
                try:
                    if hasattr(self.model, '_last_z') and self.model._last_z is not None:
                        z = self.model._last_z
                        self._last_z_da = float(z[0, 0].item()
                                                if z.dim() > 1 else z[0].item())
                except Exception:
                    pass
            else:
                out = self.model(jp, timesteps=t)

        self._step_count += 1
        return out.squeeze(0).cpu().numpy()

    def reset(self):
        self._step_count = 0
        self._last_z_da  = 0.0
        if hasattr(self.model, 'reset'):
            self.model.reset()


# ─────────────────────────────────────────────────────────────────────────────
# HUD annotation
# ─────────────────────────────────────────────────────────────────────────────

def annotate_frame(frame, step, phase_info, substrate, z_da=None,
                   position_history=None):
    """
    Draw all HUD elements onto the frame using OpenCV.
    Falls back silently if cv2 is not available.
    """
    if not _HAS_CV2:
        return frame

    f = frame.copy()
    h, w = f.shape[:2]

    # ── Land zones ────────────────────────────────────────────────────────────
    land_zones = phase_info.get('land_zones', [])
    x_range, y_range = 12.0, 8.0
    for i, z in enumerate(land_zones):
        cx = int((z['center'][0] + 6.0) / x_range * w)
        cy = int(h - (z['center'][1] + 4.0) / y_range * h)
        sr = max(15, int(z['radius'] / x_range * w))
        if 0 <= cx < w and 0 <= cy < h:
            overlay = f.copy()
            cv2.circle(overlay, (cx, cy), sr, (101, 67, 33), -1)
            f = cv2.addWeighted(f, 0.75, overlay, 0.25, 0)
            cv2.circle(f, (cx, cy), sr + 3, (139, 69, 19), 3)
            cv2.circle(f, (cx, cy), sr + 3, (255, 255, 255), 1)
            lbl = f"Land {i+1}"
            lx = max(5, cx - cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)[0][0]//2)
            ly = max(25, cy - sr - 12)
            cv2.rectangle(f, (lx-4, ly-18), (lx + 80, ly+4), (0, 0, 0), -1)
            cv2.putText(f, lbl, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (255, 255, 255), 2)

    # ── Swimmer dot ───────────────────────────────────────────────────────────
    swim_pos = phase_info.get('swimmer_pos')
    if swim_pos is not None:
        sx = int((swim_pos[0] + 6.0) / x_range * w)
        sy = int(h - (swim_pos[1] + 4.0) / y_range * h)
        if 0 <= sx < w and 0 <= sy < h:
            # Orange on land, cyan on water
            col = (0, 165, 255) if substrate == 'Land' else (0, 255, 255)
            pulse = int(12 + 5 * abs(np.sin(step * 0.2)))
            cv2.circle(f, (sx, sy), pulse, col, 3)
            cv2.circle(f, (sx, sy), 8, (255, 255, 255), -1)
            cv2.circle(f, (sx, sy), 8, col, 2)

    # ── Minimap with trail ────────────────────────────────────────────────────
    if land_zones and position_history:
        mm_size, mm_margin = 120, 10
        mm_x = w - mm_size - mm_margin
        mm_y = mm_margin
        minimap = np.zeros((mm_size, mm_size, 3), dtype=np.uint8)
        minimap[:, :, 0] = 40  # dark blue
        env_bounds = 8.0

        def to_mm(pos):
            mx = int((pos[0] + env_bounds/2) / env_bounds * mm_size)
            my = int((1 - (pos[1] + env_bounds/2) / env_bounds) * mm_size)
            return (max(0, min(mm_size-1, mx)), max(0, min(mm_size-1, my)))

        for z in land_zones:
            mc = to_mm(z['center'])
            mr = max(3, int(z['radius'] / env_bounds * mm_size))
            cv2.circle(minimap, mc, mr, (101, 67, 33), -1)
            cv2.circle(minimap, mc, mr, (139, 69, 19), 1)

        if len(position_history) > 1:
            pts = [to_mm(p) for p in position_history]
            for i in range(1, len(pts)):
                intensity = int(255 * i / len(pts))
                cv2.line(minimap, pts[i-1], pts[i], (0, intensity, 0), 1)

        if position_history:
            cv2.circle(minimap, to_mm(position_history[-1]), 3, (0, 0, 255), -1)

        cv2.rectangle(minimap, (0, 0), (mm_size-1, mm_size-1), (255, 255, 255), 2)
        f[mm_y:mm_y+mm_size, mm_x:mm_x+mm_size] = minimap
        cv2.rectangle(f, (mm_x, mm_y-18), (mm_x+mm_size, mm_y), (0, 0, 0), -1)
        cv2.putText(f, "Map", (mm_x+4, mm_y-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # ── Phase banner ──────────────────────────────────────────────────────────
    banner = phase_info.get('name', '')
    cv2.rectangle(f, (5, 5), (500, 38), (0, 0, 0), -1)
    cv2.putText(f, banner, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, (255, 255, 255), 2)

    # ── Substrate indicator ───────────────────────────────────────────────────
    sub_col   = (0, 165, 255) if substrate == 'Land' else (0, 255, 255)
    gait_text = 'CRAWL' if substrate == 'Land' else 'SWIM'
    cv2.rectangle(f, (5, 45), (180, 75), (0, 0, 0), -1)
    cv2.putText(f, f"Gait: {gait_text}", (10, 67),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, sub_col, 2)

    # ── z_DA bar (CCMN only) ──────────────────────────────────────────────────
    if z_da is not None:
        bar_x, bar_y, bar_w, bar_h = 5, 82, 200, 18
        cv2.rectangle(f, (bar_x, bar_y), (bar_x+bar_w, bar_y+bar_h),
                      (40, 40, 40), -1)
        # Map z_DA (-1..1) to bar width
        norm = (np.clip(float(z_da), -1.0, 1.0) + 1.0) / 2.0
        fill = int(norm * bar_w)
        # Colour: blue (swim) → orange (crawl) gradient
        r = int(255 * norm)
        g = int(100 * (1 - norm))
        b = int(255 * (1 - norm))
        cv2.rectangle(f, (bar_x, bar_y), (bar_x+fill, bar_y+bar_h),
                      (b, g, r), -1)
        cv2.rectangle(f, (bar_x, bar_y), (bar_x+bar_w, bar_y+bar_h),
                      (200, 200, 200), 1)
        cv2.putText(f, f"z_DA={z_da:.2f}", (bar_x+2, bar_y-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)

    # ── Step counter ──────────────────────────────────────────────────────────
    cv2.putText(f, f"step {step}", (w - 110, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    return f


# ─────────────────────────────────────────────────────────────────────────────
# Core rollout + video writer
# ─────────────────────────────────────────────────────────────────────────────

def _get_physics(env):
    """Navigate env wrappers to reach the dm_control Physics object."""
    for attr_path in [
        lambda e: e.env.env.physics,
        lambda e: e.env.physics,
        lambda e: e.physics,
    ]:
        try:
            return attr_path(env)
        except AttributeError:
            continue
    return None


def _get_task(env):
    """Navigate env wrappers to reach the dm_control task."""
    for attr_path in [
        lambda e: e.env.env._task,
        lambda e: e.env._task,
        lambda e: e._task,
    ]:
        try:
            return attr_path(env)
        except AttributeError:
            continue
    return None


def _detect_substrate(phys, task):
    """Return 'Land' or 'Water' based on head position vs land zones."""
    try:
        head_pos = phys.named.data.xpos['head'][:2]
        zones = getattr(task, '_current_land_zones', []) or []
        for z in zones:
            if np.linalg.norm(head_pos - z['center']) < z['radius']:
                return 'Land', head_pos
        return 'Water', head_pos
    except Exception:
        return 'Water', np.zeros(2)


def rollout_and_record(agent, env, phase_info: dict,
                       num_steps: int = 800,
                       fps: int = 30,
                       start_pos=None) -> list:
    """
    Run one episode and collect annotated frames.
    Returns list of uint8 numpy arrays.
    """
    agent.reset()
    obs = env.reset()
    if isinstance(obs, (list, tuple)):
        obs = obs[0]
    obs = np.asarray(obs, dtype=np.float32).ravel()

    # Teleport into land zone if requested
    if start_pos is not None:
        teleport_swimmer(env, start_pos)
        # Re-read obs after teleport via a dummy step
        try:
            step_result = env.step(np.zeros(agent.n_joints))
            if isinstance(step_result, tuple) and len(step_result) == 2:
                raw_obs = step_result[0]
            else:
                raw_obs = step_result[0]
            if isinstance(raw_obs, (list, tuple)):
                raw_obs = raw_obs[0]
            obs = np.asarray(raw_obs, dtype=np.float32).ravel()
        except Exception:
            pass

    phys              = _get_physics(env)
    task              = _get_task(env)
    position_history  = []
    frames            = []
    substrate_counts  = {'Water': 0, 'Land': 0}

    for step in range(num_steps):
        # Render
        try:
            frame = env.render(mode='rgb_array')
        except Exception as e:
            print(f"  ⚠️  render error at step {step}: {e}")
            break

        if frame is None:
            continue

        frame = np.asarray(frame, dtype=np.uint8)

        # Substrate detection
        substrate, head_pos = _detect_substrate(phys, task)
        substrate_counts[substrate] += 1
        position_history.append(head_pos.copy())
        if len(position_history) > 60:
            position_history = position_history[-60:]

        # Build per-frame phase_info with live data
        live_info = dict(phase_info)
        live_info['land_zones']  = (getattr(task, '_current_land_zones', []) or [])
        live_info['swimmer_pos'] = head_pos

        # Get z_DA from agent (CCMN only)
        z_da = agent._last_z_da if hasattr(agent, '_last_z_da') else None

        frame = annotate_frame(frame, step, live_info, substrate,
                               z_da=z_da,
                               position_history=position_history)
        frames.append(frame)

        # Step
        action = agent.test_step(obs)
        try:
            result = env.step(action)
            if isinstance(result, tuple) and len(result) == 2 and \
                    isinstance(result[1], dict) and 'rewards' in result[1]:
                raw_obs, infos = result
                done = bool(infos['resets'][0])
            else:
                raw_obs, _, done, _ = result
            if isinstance(raw_obs, (list, tuple)):
                raw_obs = raw_obs[0]
            obs = np.asarray(raw_obs, dtype=np.float32).ravel()
            if done:
                obs = np.asarray(env.reset(), dtype=np.float32).ravel()
                if obs.ndim > 1:
                    obs = obs[0]
        except Exception as e:
            print(f"  ⚠️  step error at step {step}: {e}")
            break

    water_pct = 100 * substrate_counts['Water'] / max(1, sum(substrate_counts.values()))
    land_pct  = 100 * substrate_counts['Land']  / max(1, sum(substrate_counts.values()))
    print(f"  📊 {len(frames)} frames  |  "
          f"Water {water_pct:.0f}%  Land {land_pct:.0f}%")
    return frames


def save_video(frames: list, path: str, fps: int = 30):
    """Save frames as MP4 + GIF preview + contact sheet."""
    if not frames:
        print(f"  ⚠️  no frames to save for {path}")
        return
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)

    # MP4
    try:
        writer = imageio.get_writer(path, format='FFMPEG', mode='I',
                                    fps=fps, codec='libx264',
                                    pixelformat='yuv420p', macro_block_size=16)
        for f in frames:
            writer.append_data(f)
        writer.close()
        print(f"  🎬 MP4 saved: {path}")
    except Exception as e:
        print(f"  ⚠️  MP4 writer failed ({e}), trying fallback…")
        try:
            imageio.mimsave(path, frames, fps=fps)
            print(f"  🎬 MP4 (fallback) saved: {path}")
        except Exception as e2:
            print(f"  ❌ Could not save MP4: {e2}")

    # GIF preview
    try:
        gif_path = path.replace('.mp4', '_preview.gif')
        n = min(80, len(frames))
        indices = np.linspace(0, len(frames)-1, n, dtype=int)
        imageio.mimsave(gif_path, [frames[i] for i in indices], fps=min(fps, 10))
        print(f"  🖼️  GIF preview: {gif_path}")
    except Exception as e:
        print(f"  ⚠️  GIF failed: {e}")

    # Contact sheet
    try:
        sheet_path = path.replace('.mp4', '_contact.png')
        n = min(12, len(frames))
        indices = np.linspace(0, len(frames)-1, n, dtype=int)
        cols, rows = 4, int(np.ceil(n / 4))
        fig, axes = plt.subplots(rows, cols, figsize=(cols*3.2, rows*2.4))
        axes = np.atleast_1d(axes).reshape(rows, cols)
        for ax in axes.flat:
            ax.axis('off')
        for ax, idx in zip(axes.flat, indices):
            ax.imshow(frames[idx])
            ax.set_title(f'f{idx}', fontsize=7)
            ax.axis('off')
        fig.tight_layout()
        fig.savefig(sheet_path, dpi=120, bbox_inches='tight')
        plt.close(fig)
        print(f"  🖼️  Contact sheet: {sheet_path}")
    except Exception as e:
        print(f"  ⚠️  Contact sheet failed: {e}")


def save_combined_video(phase_frames: dict, path: str, fps: int = 30):
    """
    Stitch all four phase videos into a 2×2 grid side-by-side comparison.
    Falls back to vertical stack if cv2 is not available.
    """
    # Make all clips the same length
    max_len = max(len(v) for v in phase_frames.values() if v)
    if max_len == 0:
        return

    ordered = [phase_frames.get(i, []) for i in range(4)]
    # Pad shorter clips by repeating last frame
    padded = []
    for frames in ordered:
        if not frames:
            frames = [np.zeros_like(list(phase_frames.values())[0][0])]
        while len(frames) < max_len:
            frames.append(frames[-1])
        padded.append(frames[:max_len])

    combined = []
    for t in range(max_len):
        if _HAS_CV2:
            # 2×2 grid
            h, w = padded[0][t].shape[:2]
            row0 = cv2.hconcat([padded[0][t], padded[1][t]])
            row1 = cv2.hconcat([padded[2][t], padded[3][t]])
            grid = cv2.vconcat([row0, row1])
        else:
            # Vertical stack fallback
            grid = np.concatenate([p[t] for p in padded], axis=0)
        combined.append(grid)

    save_video(combined, path, fps=fps)
    print(f"  🎬 Combined 2×2 video: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Generate per-phase locomotion videos from a trained model.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--checkpoint_dir', required=True,
                        help='Directory containing checkpoint files')
    parser.add_argument('--checkpoint_step', type=int, default=None,
                        help='Specific step to load (default: latest checkpoint)')
    parser.add_argument('--out_dir', default='outputs/phase_videos',
                        help='Output directory for videos (default: outputs/phase_videos)')
    parser.add_argument('--model_type', default='ccmn_hrl',
                        choices=['ccmn_hrl', 'ccmn', 'simple_ncap',
                                 'biological_ncap', 'enhanced_ncap'],
                        help='Model architecture (default: ccmn_hrl)')
    parser.add_argument('--n_links', type=int, default=6,
                        help='Number of swimmer links (default: 6)')
    parser.add_argument('--oscillator_period', type=int, default=60,
                        help='CPG oscillator period (default: 60)')
    parser.add_argument('--num_steps', type=int, default=800,
                        help='Simulation steps per video (default: 800)')
    parser.add_argument('--fps', type=int, default=30,
                        help='Video frame rate (default: 30)')
    parser.add_argument('--device', default='cpu',
                        help='Torch device (default: cpu)')
    parser.add_argument('--use_ext_env', action='store_true',
                        help='Use progressive_ext_env instead of progressive_mixed_env')
    parser.add_argument('--phases', default='0,1,2,3',
                        help='Comma-separated list of phases to render (default: 0,1,2,3)')
    parser.add_argument('--no_land_start', action='store_true',
                        help='Do not teleport swimmer into land zone — start at origin')
    parser.add_argument('--no_combined', action='store_true',
                        help='Skip combined 2×2 grid video')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    phases_to_render = [int(p.strip()) for p in args.phases.split(',')]

    # ── Find checkpoint ────────────────────────────────────────────────────────
    print(f"\n🔍 Scanning checkpoints in: {args.checkpoint_dir}")
    ckpts = list_checkpoints(args.checkpoint_dir)
    if not ckpts:
        print("❌ No checkpoints found. Check --checkpoint_dir.")
        sys.exit(1)

    if args.checkpoint_step is not None:
        match = [(s, p) for s, p in ckpts if s == args.checkpoint_step]
        if not match:
            print(f"❌ Checkpoint step {args.checkpoint_step} not found. "
                  f"Available: {[s for s, _ in ckpts[:5]]} ...")
            sys.exit(1)
        step, ckpt_path = match[0]
    else:
        step, ckpt_path = ckpts[-1]   # latest

    print(f"📦 Loading checkpoint: step {step:,}  ({ckpt_path})")

    # ── Load model ─────────────────────────────────────────────────────────────
    model, _ = load_checkpoint(ckpt_path, args.n_links,
                                args.oscillator_period,
                                args.model_type, args.device)
    n_joints = args.n_links - 1

    # Build a dummy env to measure obs_dim / env_features_start
    _tmp_env = build_env(args.n_links, 0.15, use_ext_env=args.use_ext_env)
    obs_dim  = _tmp_env.observation_space.shape[0]
    body_vel_size    = args.n_links * 3 + 3
    env_features_start = n_joints + body_vel_size
    _tmp_env.close() if hasattr(_tmp_env, 'close') else None
    del _tmp_env

    # Auto-detect device from where the model actually lives.
    # The checkpoint may have been saved on cuda; _build_model honours
    # --device but load_checkpoint uses map_location='cpu' then the
    # model stays on cpu unless CCMNSwimmerHRL.__init__ moves it itself.
    actual_device = str(next(model.parameters()).device)
    if actual_device != args.device:
        print(f"  ℹ️  model is on {actual_device} (--device was {args.device}) "
              f"— using {actual_device}")
    agent = VideoAgent(model, n_joints, args.model_type,
                       env_features_start, actual_device)

    print(f"\n🎬 Generating videos for phases: {phases_to_render}")
    print(f"   Steps per video : {args.num_steps}")
    print(f"   Output directory: {args.out_dir}\n")

    phase_frames = {}

    for ph in PHASES:
        if ph['idx'] not in phases_to_render:
            continue

        print(f"{'='*60}")
        print(f"▶  {ph['name']}  (progress={ph['progress']:.0%})")
        print(f"{'='*60}")

        env = build_env(args.n_links, ph['progress'],
                        use_ext_env=args.use_ext_env)

        start_pos = None if args.no_land_start else ph['start_pos']
        frames = rollout_and_record(
            agent, env,
            phase_info=ph,
            num_steps=args.num_steps,
            fps=args.fps,
            start_pos=start_pos,
        )
        phase_frames[ph['idx']] = frames

        vid_name = (f"phase{ph['idx']}_"
                    f"{ph['name'].split('—')[1].strip().replace(' ', '_').lower()}"
                    f"_step{step}.mp4")
        save_video(frames, os.path.join(args.out_dir, vid_name), fps=args.fps)

        env.close() if hasattr(env, 'close') else None
        print()

    # ── Combined video ─────────────────────────────────────────────────────────
    if not args.no_combined and len(phase_frames) == 4:
        print("🎞️  Creating combined 2×2 comparison video…")
        combined_path = os.path.join(
            args.out_dir, f"all_phases_combined_step{step}.mp4"
        )
        save_combined_video(phase_frames, combined_path, fps=args.fps)

    print(f"\n✅ Done. All videos saved to: {args.out_dir}")


if __name__ == '__main__':
    main()
