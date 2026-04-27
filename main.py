#!/usr/bin/env python3
"""
Improved Mixed Environment Swimmer with Training Integration
Main script for running the improved mixed environment swimmer with proper RL training.
"""

import argparse
import matplotlib
matplotlib.use('Agg')  # Force non-interactive backend to avoid tkinter threading issues
from swimmer.training import ImprovedNCAPTrainer

def main():
    parser = argparse.ArgumentParser(description='Improved Mixed Environment Swimmer')
    parser.add_argument('--mode', choices=['train_improved', 'train_biological', 'train_curriculum', 'evaluate', 'evaluate_curriculum'], default='train_improved',
                       help='Mode to run: train_improved (stable NCAP), train_biological (preserve biology), train_curriculum (progressive swim+crawl), evaluate, or evaluate_curriculum (eval only)')
    parser.add_argument('--model', choices=['ncap', 'mlp'], default='ncap',
                       help='Model type: ncap or mlp')
    parser.add_argument('--model_type', choices=['biological_ncap', 'enhanced_ncap', 'ccmn', 'ccmn_hrl', 'simple_ncap'], default='enhanced_ncap',
                       help='NCAP sub-type for biological models')
    parser.add_argument('--algorithm', choices=['ppo', 'a2c', 'es', 'ddpg'], default='ppo',
                       help='RL algorithm to use')
    parser.add_argument('--n_links', type=int, default=6,
                       help='Number of links in the swimmer')
    parser.add_argument('--training_steps', type=int, default=1000000,
                       help='Number of training steps')
    parser.add_argument('--save_steps', type=int, default=50000,
                       help='Steps between model saves')
    parser.add_argument('--log_episodes', type=int, default=10,
                       help='Episodes between progress logs')
    parser.add_argument('--load_model', type=str, default=None,
                       help='Path to load a trained model')
    parser.add_argument('--resume_checkpoint', type=str, default=None,
                       help='Path to checkpoint to resume curriculum training from')
    parser.add_argument('--eval_episodes', type=int, default=10,
                       help='Number of episodes per phase for evaluation')
    parser.add_argument('--eval_video_steps', type=int, default=600,
                       help='Number of steps per video for evaluation')
    parser.add_argument('--oscillator_period', type=int, default=60,
                       help='Base period for biological oscillators')
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='Custom name for the experiment')
    parser.add_argument('--use-locomotion-only-early-training', type=bool, default=True,
                       help='Use locomotion-only mode for early training')
    parser.add_argument('--num_workers', type=int, default=8,
                       help='Number of parallel environment workers')
    parser.add_argument('--use_multi_gpu', action='store_true', default=True,
                       help='Use multiple GPUs if available')
    parser.add_argument('--hide_environment_observation', action='store_true',
                       help='Zero out privileged water/land observation channels for ablations')
    parser.add_argument('--hide_viscosity_observation', action='store_true',
                       help='Zero out viscosity observation channel for ablations')
    parser.add_argument('--anisotropic_drag_mode', choices=['off', 'on', 'proxy'], default='off',
                       help='Anisotropic drag mode. "on"/"proxy" enable the lateral-drag proxy; "off" disables it.')
    parser.add_argument('--anisotropic_drag_ratio', type=float, default=10.0,
                       help='Normal-to-tangential drag ratio for the anisotropic proxy')
    parser.add_argument('--anisotropic_drag_gain', type=float, default=0.02,
                       help='Overall strength of the anisotropic drag proxy')
    parser.add_argument('--anisotropic_drag_all_media', action='store_true',
                       help='Apply anisotropic drag proxy in both water and land instead of land only')

    # ── CCMN-specific options ──────────────────────────────────────────────────
    parser.add_argument('--ccmn_phase1_land_fraction', type=float, default=0.15,
                       help='[CCMN] Fraction of Phase-1 steps that expose a land zone '
                            'to give the ContextEncoder early contrast signal (0=off, default=0.15)')
    parser.add_argument('--ccmn_no_sequential_training', action='store_true',
                       help='[CCMN] Disable sequential full-episode GRU training '
                            '(reverts to shuffled mini-batches — for ablation only)')
    parser.add_argument('--film_ablation', action='store_true',
                       help='[CCMN] Freeze FiLM layer (γ=1, β=0) to ablate the '
                            'neuromodulatory contribution and compare against full CCMN')
    parser.add_argument('--run_sanity_check', action='store_true',
                       help='[CCMN] Run a diagnostic episode before training starts '
                            'to verify z_DA variation, period bimodality, and γ identity')
    parser.add_argument('--sanity_check_steps', type=int, default=600,
                       help='[CCMN] Steps per sanity-check episode (default: 600)')
    parser.add_argument('--bistability_regulariser', action='store_true',
                       help='[CCMN] Add auxiliary loss penalising intermediate z_DA '
                            'values to encourage bistable gait switching')
    parser.add_argument('--bistability_lambda', type=float, default=0.01,
                       help='[CCMN] Weight of the bistability regularisation loss (default: 0.01)')
    parser.add_argument('--ccmn_use_viscosity_input', action='store_true',
                       help='[CCMN] Pass viscosity directly to ContextEncoder GRU '
                            '(legacy/ablation only; default is bio-faithful φ(s,t) inference)')
    parser.add_argument('--ablate_amplitude_scaling', action='store_true',
                       help='[CCMN] Disable z_DA-dependent amplitude scaling (ablation: fixed amplitude=1.0)')
    parser.add_argument('--ablate_metabolic_bonus', action='store_true',
                       help='[CCMN] Disable metabolic efficiency reward shaping (ablation)')
    parser.add_argument('--use_ext_env', action='store_true', default=False,
                       help='Use progressive_ext_env: composes dm_control base velocity reward '
                            'with switching components, producing episode returns in the '
                            '200-1000 range matching the NCAP paper for direct comparison.')
    parser.add_argument('--output_dir', type=str, default='outputs',
                       help='Root directory for all training outputs (checkpoints, plots, videos, summaries)')

    # ── Reward shaping options ─────────────────────────────────────────────────
    parser.add_argument('--transition_bonus', type=float, default=5.0,
                       help='Sparse reward added each time the agent crosses a water/land boundary (default: 5.0)')
    parser.add_argument('--mismatch_penalty_weight', type=float, default=0.3,
                       help='Weight of the medium-mismatch penalty (swim kinematics on land or crawl in water; default: 0.3)')
    parser.add_argument('--water_velocity_scale', type=float, default=6.0,
                       help='Velocity reward scale factor in water to rebalance against land (default: 6.0 ≈ land/water speed ratio)')
    parser.add_argument('--training_mode',
                       choices=['progressive', 'forced_alternating', 'land_target_forced'],
                       default='progressive',
                       help=(
                           'Training environment mode. '
                           '"progressive": original curriculum (default). '
                           '"forced_alternating" [Option 1]: episode is divided into alternating '
                           'water/land segments of --forced_segment_steps each, teleporting the '
                           'swimmer at each boundary — guarantees ContextEncoder sees both media. '
                           '"land_target_forced" [Option 2]: first target each phase is placed deep '
                           'inside a land zone so the agent must cross the boundary to earn any '
                           'target-completion reward.'
                       ))
    parser.add_argument('--forced_segment_steps', type=int, default=200,
                       help='[forced_alternating only] Steps per water/land segment before forced medium switch (default: 200)')
    parser.add_argument('--sustained_swim_bonus_per_step', type=float, default=0.05,
                       help='Per-step bonus awarded when water bout exceeds --sustained_swim_min_bout (default: 0.05). '
                            'Replaces the flat transition bonus for sustained aquatic locomotion.')
    parser.add_argument('--sustained_swim_min_bout', type=int, default=50,
                       help='Minimum consecutive water steps before the sustained swim bonus activates (default: 50)')
    parser.add_argument('--transition_cooldown_steps', type=int, default=100,
                       help='Steps after a water/land crossing during which the transition bonus is suppressed, '
                            'preventing reward gaming via rapid repeated crossings (default: 100)')

    # ── Bayesian reward options ────────────────────────────────────────────────
    parser.add_argument('--bayes_likelihood_sharpness', type=float, default=4.0,
                       help='Sharpness of the sigmoid likelihood used to update the Bayesian '
                            'medium belief B(t) from joint-velocity RMS (default: 4.0)')
    parser.add_argument('--bayes_reward_weight', type=float, default=0.3,
                       help='Scale of the gait-appropriateness and certainty-gain Bayesian reward '
                            'terms. 0.0 disables the Bayesian reward entirely (default: 0.3)')
    parser.add_argument('--bayes_epistemic_penalty_weight', type=float, default=0.2,
                       help='Scale of the entropy penalty discouraging uncertain medium belief '
                            '(indeterminate z_DA state) (default: 0.2)')

    # ── BOriginal NCAP ────────────────────────────────────────────────
    parser.add_argument('--swim_only', action='store_true', default=False,
                       help='Pin training to Phase 0 (pure swimming) for the entire run. '
                            'Reproduces the original NCAP paper single-environment baseline.')

    args = parser.parse_args()
    
    # train_simple and legacy modes removed
    if args.mode == 'train_improved':
        trainer = ImprovedNCAPTrainer(
            n_links=args.n_links,
            training_steps=args.training_steps,
            save_steps=args.save_steps,
            log_episodes=args.log_episodes
        )
        trainer.train()
    elif args.mode == 'train_biological':
        from swimmer.training.simple_biological_trainer import SimpleBiologicalTrainer
        trainer = SimpleBiologicalTrainer(
            n_links=args.n_links,
            training_steps=args.training_steps,
            save_steps=args.save_steps,
            log_episodes=args.log_episodes
        )
        trainer.train()
    elif args.mode == 'train_curriculum':
        print("🎓 Starting curriculum training for swimming and crawling...")
        from swimmer.training.curriculum_trainer import CurriculumNCAPTrainer
        trainer = CurriculumNCAPTrainer(
            n_links=args.n_links,
            learning_rate=3e-5,  # Conservative learning rate for long training
            training_steps=args.training_steps,
            save_steps=args.save_steps,
            log_episodes=args.log_episodes,
            resume_from_checkpoint=args.resume_checkpoint,
            model_type=args.model_type,
            algorithm=args.algorithm,
            num_workers=args.num_workers,
            use_multi_gpu=args.use_multi_gpu,
            oscillator_period=args.oscillator_period,
            use_locomotion_only_early_training=args.use_locomotion_only_early_training,
            expose_environment_observation=not args.hide_environment_observation,
            expose_viscosity_observation=not args.hide_viscosity_observation,
            anisotropic_drag_mode=args.anisotropic_drag_mode,
            anisotropic_drag_ratio=args.anisotropic_drag_ratio,
            anisotropic_drag_gain=args.anisotropic_drag_gain,
            anisotropic_drag_land_only=not args.anisotropic_drag_all_media,
            ccmn_phase1_land_fraction=args.ccmn_phase1_land_fraction,
            ccmn_sequential_training=not args.ccmn_no_sequential_training,
            film_ablation=args.film_ablation,
            run_sanity_check=args.run_sanity_check,
            sanity_check_steps=args.sanity_check_steps,
            bistability_regulariser=args.bistability_regulariser,
            bistability_lambda=args.bistability_lambda,
            ccmn_use_viscosity_input=args.ccmn_use_viscosity_input,
            use_amplitude_scaling=not args.ablate_amplitude_scaling,
            use_metabolic_bonus=not args.ablate_metabolic_bonus,
            transition_bonus=args.transition_bonus,
            mismatch_penalty_weight=args.mismatch_penalty_weight,
            water_velocity_scale=args.water_velocity_scale,
            training_mode=args.training_mode,
            forced_segment_steps=args.forced_segment_steps,
            sustained_swim_bonus_per_step=args.sustained_swim_bonus_per_step,
            sustained_swim_min_bout=args.sustained_swim_min_bout,
            transition_cooldown_steps=args.transition_cooldown_steps,
            bayes_likelihood_sharpness=args.bayes_likelihood_sharpness,
            bayes_reward_weight=args.bayes_reward_weight,
            bayes_epistemic_penalty_weight=args.bayes_epistemic_penalty_weight,
            use_ext_env=args.use_ext_env,
            swim_only=args.swim_only,
            output_dir=args.output_dir,
        )
        trainer.train()
    elif args.mode == 'evaluate_curriculum':
        if args.resume_checkpoint is None:
            print("⛔  --resume_checkpoint is required for curriculum evaluation"); return
        
        print("📊 Starting curriculum evaluation from checkpoint...")
        from swimmer.training.curriculum_trainer import CurriculumNCAPTrainer
        trainer = CurriculumNCAPTrainer(
            n_links=args.n_links,
            training_steps=0,  # No training
            resume_from_checkpoint=args.resume_checkpoint,
            model_type=args.model_type,
            algorithm=args.algorithm,
            use_locomotion_only_early_training=args.use_locomotion_only_early_training,
            expose_environment_observation=not args.hide_environment_observation,
            expose_viscosity_observation=not args.hide_viscosity_observation,
            anisotropic_drag_mode=args.anisotropic_drag_mode,
            anisotropic_drag_ratio=args.anisotropic_drag_ratio,
            anisotropic_drag_gain=args.anisotropic_drag_gain,
            anisotropic_drag_land_only=not args.anisotropic_drag_all_media,
            output_dir=args.output_dir,
        )
        trainer.evaluate_only(
            eval_episodes=args.eval_episodes,
            video_steps=args.eval_video_steps
        )
    elif args.mode == 'evaluate':
        if args.load_model is None:
            print("⛔  --load_model is required for evaluation"); return

        trainer = ImprovedNCAPTrainer(n_links=args.n_links)
        trainer.load_tonic_model(args.load_model)
        trainer.evaluate_mixed_environment(max_frames=1800)
        
if __name__ == "__main__":
    main() 
