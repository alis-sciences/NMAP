#!/bin/bash

python main.py   --mode train_curriculum --model_type simple_ncap   --algorithm ppo --training_steps 5000000 --save_steps 500000   --num_workers 8 --oscillator_period 60   --training_mode land_target_forced   --transition_bonus 0.0 --mismatch_penalty_weight 0.5 --water_velocity_scale 10.0   --sustained_swim_bonus_per_step 0.05 --sustained_swim_min_bout 50   --transition_cooldown_steps 100   --output_dir /workspace/nma_nai_hrl/outputs/simple_ncap_ppo_5m/   --n_links 6


python main.py   --mode train_curriculum --model_type simple_ncap   --algorithm es --training_steps 5000000 --save_steps 500000   --num_workers 8 --oscillator_period 60   --training_mode land_target_forced   --transition_bonus 0.0 --mismatch_penalty_weight 0.5 --water_velocity_scale 10.0   --sustained_swim_bonus_per_step 0.05 --sustained_swim_min_bout 50   --transition_cooldown_steps 100   --output_dir /workspace/nma_nai_hrl/outputs/simple_ncap_es_5m/   --n_links 6

python main.py   --mode train_curriculum --model_type simple_ncap   --algorithm ddpg --training_steps 5000000 --save_steps 500000   --num_workers 8 --oscillator_period 60   --training_mode land_target_forced   --transition_bonus 0.0 --mismatch_penalty_weight 0.5 --water_velocity_scale 10.0   --sustained_swim_bonus_per_step 0.05 --sustained_swim_min_bout 50   --transition_cooldown_steps 100   --output_dir /workspace/nma_nai_hrl/outputs/simple_ncap_ddpg_5m/   --n_links 6

