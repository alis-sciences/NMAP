# NMAP — Neuro-Modulating Architecture Priors

**Context-Dependent Reconfiguration of Locomotory Circuits**

NMAP is a research codebase that models how neuromodulatory signals (inspired by dopaminergic pathways) can dynamically reconfigure a locomotory controller to adapt between aquatic and terrestrial environments. A multi-link swimmer agent is trained with reinforcement learning to switch gaits (swimming vs. crawling) as it crosses water–land boundaries.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Training Modes](#training-modes)
  - [Evaluation](#evaluation)
  - [Key Arguments](#key-arguments)
- [Model Types](#model-types)
- [Environment & Reward Shaping](#environment--reward-shaping)
- [Ablations](#ablations)
- [Scripts Reference](#scripts-reference)
- [Citation](#citation)

---

## Overview

Animals such as mudskippers and salamanders seamlessly switch locomotory strategies as they move between water and land. NMAP investigates whether a neural architecture equipped with neuromodulatory priors can reproduce this behaviour in a simulated multi-link swimmer (MuJoCo via `dm-control`).

The central component is the **CCMN (Context-Conditioned Motor Network)**, which uses a GRU-based `ContextEncoder` to infer the current medium (water or land) from proprioceptive signals alone, then emits a neuromodulatory signal `z_DA` that gates motor output via **FiLM** (Feature-wise Linear Modulation) layers.

---

## Architecture

```
Proprioception (s, t)
        │
        ▼
 ContextEncoder (GRU)
        │  z_DA ∈ [0, 1]   ← dopamine-like neuromodulatory signal
        ▼
  FiLM Layer (γ, β)         ← scales & shifts motor network activations
        │
        ▼
  Motor Policy (PPO / A2C)
        │
        ▼
   Joint Torques → Swimmer
```

| Component | Role |
|---|---|
| `ContextEncoder` | GRU that infers medium from joint-velocity RMS; produces `z_DA` |
| `FiLM Layer` | Applies `z_DA`-conditioned affine transform to motor features |
| `Amplitude Scaling` | Scales oscillator amplitude with `z_DA` to modulate gait strength |
| `Bayesian Belief` | Maintains a running belief `B(t)` over medium for reward shaping |

---

## Repository Structure

```
NMAP/
├── main.py                        # Entry point: CLI for training & evaluation
├── micro_publication_main.py      # Minimal reproduction script (paper baseline)
├── compare_trajectories.py        # Compare agent trajectories across checkpoints
├── curvature_converter.py         # Convert joint angles → body curvature
├── diag_model.py                  # Diagnostic utilities for model inspection
├── export_kinematics.py           # Export kinematic data to CSV/NPZ
├── generate_phase_videos.py       # Render phase-space videos per gait
├── ncap_analysis.py               # Post-hoc analysis of NCAP experiment logs
├── pyelastica_reconstruction.py   # Reconstruct body shape via PyElastica
├── ncap_5m.sh                     # Bash script: 5M-step curriculum run
├── setup_env.sh                   # Environment setup helper
├── environment.yml                # Conda environment specification
├── requirements.txt               # Pip dependencies
├── swimmer/                       # Core library
│   └── training/
│       ├── __init__.py            # Exports ImprovedNCAPTrainer
│       ├── curriculum_trainer.py  # CurriculumNCAPTrainer (main training loop)
│       └── simple_biological_trainer.py
├── tests/                         # Unit & integration tests
└── references/                    # Paper PDFs / supplementary material
```

---

## Installation

### Prerequisites

- **Python 3.12**
- **Conda** (recommended) or pip
- **CUDA 12.1** (for GPU training; CPU-only also works)
- **MuJoCo** (installed automatically via `dm-control`)

### Conda (recommended)

```bash
git clone https://github.com/alis-sciences/NMAP.git
cd NMAP
conda env create -f environment.yml
conda activate swimmer_ncap
```

### Pip

```bash
pip install -r requirements.txt
```

### Environment setup script

```bash
bash setup_env.sh
```

---

## Usage

All experiments are launched through `main.py`.

```bash
python main.py --mode <mode> [options]
```

### Training Modes

| Mode | Description |
|---|---|
| `train_improved` | Stable NCAP training (default) |
| `train_biological` | Biologically-faithful oscillator training |
| `train_curriculum` | Progressive curriculum: swimming → swimming+crawling |
| `evaluate` | Evaluate a saved model (requires `--load_model`) |
| `evaluate_curriculum` | Evaluate a curriculum checkpoint (requires `--resume_checkpoint`) |

#### Quick start — curriculum training

```bash
python main.py \
  --mode train_curriculum \
  --model_type ccmn \
  --algorithm ppo \
  --n_links 6 \
  --training_steps 5000000 \
  --output_dir outputs/ccmn_run1
```

#### Reproduce paper baseline (swimming only)

```bash
python main.py --mode train_curriculum --swim_only --model_type enhanced_ncap
```

#### 5-million-step run (shell script)

```bash
bash ncap_5m.sh
```

---

### Evaluation

```bash
# Evaluate an improved trainer checkpoint
python main.py --mode evaluate --load_model outputs/my_run/checkpoint.pt

# Evaluate a curriculum checkpoint with videos
python main.py \
  --mode evaluate_curriculum \
  --resume_checkpoint outputs/my_run/checkpoint.pt \
  --eval_episodes 10 \
  --eval_video_steps 600
```

---

### Key Arguments

| Argument | Default | Description |
|---|---|---|
| `--n_links` | `6` | Number of swimmer body links |
| `--model_type` | `enhanced_ncap` | Architecture variant (see [Model Types](#model-types)) |
| `--algorithm` | `ppo` | RL algorithm: `ppo`, `a2c`, `ddpg`, `es` |
| `--training_steps` | `1000000` | Total environment steps |
| `--num_workers` | `8` | Parallel environment workers |
| `--output_dir` | `outputs` | Root directory for checkpoints, videos, logs |
| `--training_mode` | `progressive` | Curriculum schedule: `progressive`, `forced_alternating`, `land_target_forced` |
| `--oscillator_period` | `60` | Base oscillator period (biological modes) |
| `--swim_only` | `False` | Pin to Phase 0 (pure swimming baseline) |

---

## Model Types

| `--model_type` | Description |
|---|---|
| `enhanced_ncap` | Default: NCAP with stability improvements |
| `ccmn` | Context-Conditioned Motor Network (GRU + FiLM neuromodulation) |
| `ccmn_hrl` | CCMN with hierarchical RL |
| `biological_ncap` | Biologically constrained oscillator network |
| `simple_ncap` | Minimal NCAP for ablations |

---

## Environment & Reward Shaping

The swimmer operates in a mixed water–land arena built on `dm-control`. Reward shaping options include:

- **Transition bonus** (`--transition_bonus`, default `5.0`): sparse reward at water/land boundary crossings.
- **Mismatch penalty** (`--mismatch_penalty_weight`, default `0.3`): penalises swimming kinematics on land or crawling in water.
- **Water velocity scaling** (`--water_velocity_scale`, default `6.0`): rebalances velocity reward between media.
- **Sustained swim bonus** (`--sustained_swim_bonus_per_step`): per-step bonus for sustained aquatic bouts (activates after `--sustained_swim_min_bout` consecutive water steps).
- **Bayesian reward** (`--bayes_reward_weight`): gait-appropriateness reward derived from running medium belief `B(t)`.
- **Transition cooldown** (`--transition_cooldown_steps`, default `100`): suppresses repeated transition bonuses to prevent reward gaming.

### Curriculum phases

1. **Phase 0** — Pure swimming (optional baseline with `--swim_only`).
2. **Phase 1** — Introduces land zones; `--ccmn_phase1_land_fraction` controls early land exposure.
3. **Phase 2+** — Full mixed-media curriculum.

---

## Ablations

The following flags disable specific components to isolate their contribution:

| Flag | Effect |
|---|---|
| `--hide_environment_observation` | Zeros out privileged medium observation |
| `--hide_viscosity_observation` | Zeros out viscosity channel |
| `--film_ablation` | Freezes FiLM to identity (γ=1, β=0) |
| `--ablate_amplitude_scaling` | Disables `z_DA`-dependent amplitude scaling |
| `--ablate_metabolic_bonus` | Disables metabolic efficiency reward |
| `--ccmn_no_sequential_training` | Reverts to shuffled mini-batches for GRU training |

---

## Scripts Reference

| Script | Purpose |
|---|---|
| `compare_trajectories.py` | Side-by-side trajectory comparison across checkpoints or seeds |
| `curvature_converter.py` | Convert joint angle time-series to body curvature |
| `diag_model.py` | Inspect model weights, `z_DA` distributions, FiLM parameters |
| `export_kinematics.py` | Export joint angles, positions, velocities to CSV/NPZ |
| `generate_phase_videos.py` | Render videos colour-coded by gait phase |
| `ncap_analysis.py` | Aggregate training logs, plot learning curves |
| `pyelastica_reconstruction.py` | Physics-based body shape reconstruction via PyElastica |

---

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{karalasingham2025nmap,
  title        = {Neuro-Modulating Architecture Priors for Context Dependent
                  Reconfiguration of Locomotory Circuits},
  author       = {Karalasingham, Sagthitharan and
                  Shirodkar, Tejas Pradeep and
                  Fathollahi, Amin and
                  Ramaswamy, Srikanth},
  year         = {2025},
  note         = {Karalasingham, Shirodkar, and Fathollahi: Neuromatch Academy,
                  Neuromatch, Inc. Ramaswamy: Neural Circuits Laboratory,
                  Faculty of Medical Sciences, Newcastle University, UK},
  howpublished = {\url{https://github.com/alis-sciences/NMAP}}
}
```

---

*Maintained by [alis-sciences](https://github.com/alis-sciences).*
