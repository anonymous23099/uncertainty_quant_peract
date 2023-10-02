#!/bin/bash

# Define the list of tasks
# tasks=(open_drawer push_buttons slide_block_to_color_target put_money_in_safe put_groceries_in_cupboard)

# export PYTHONPATH=/home/bobwu/UQ/peract_headless/YARR/
# export PYTHONPATH=/home/bobwu/UQ/peract_headless


# TASKS=(open_drawer push_buttons slide_block_to_color_target put_money_in_safe put_groceries_in_cupboard)

# for TASK in "${TASKS[@]}"; do
#     CUDA_VISIBLE_DEVICES=1 python peract_reliability/eval.py \
#         rlbench.tasks=[$TASK] \
#         rlbench.task_name='multi' \
#         rlbench.demo_path=$PERACT_ROOT/data/val \
#         framework.gpu=0 \
#         framework.logdir=$PERACT_ROOT/ckpts/ \
#         framework.start_seed=0 \
#         framework.eval_envs=1 \
#         framework.eval_from_eps_number=0 \
#         framework.eval_episodes=10 \
#         framework.csv_logging=True \
#         framework.tensorboard_logging=True \
#         framework.eval_type='last' \
#         rlbench.headless=True
# done

# tasks=(open_drawer)
# tasks=(push_buttons)
# tasks=(put_money_in_safe)
# tasks=(place_wine_at_rack_location)
# tasks=(push_buttons put_money_in_safe place_wine_at_rack_location open_drawer)


tasks=(
    change_channel 
    close_jar 
    insert_onto_square_peg 
    light_bulb_in 
    meat_off_grill 
    place_cups 
    place_shape_in_shape_sorter 
    put_groceries_in_cupboard 
    put_item_in_drawer 
    reach_and_drag 
    stack_blocks 
    stack_cups 
    turn_tap 
    set_clock_to_time 
    put_rubbish_in_color_bin 
    slide_block_to_color_target 
    sweep_to_dustpan_of_size
    push_buttons 
    put_money_in_safe 
    place_wine_at_rack_location 
    open_drawer
)
# tasks=(put_money_in_safe place_wine_at_rack_location)
export PYTHONPATH=/home/bobwu/UQ/peract_headless/YARR/
export PYTHONPATH=/home/bobwu/UQ/peract_headless

#tau = 2, 3, 4 
# for t in "${tasks[@]}"; do
#     for tau_value in 5; do
#         CUDA_VISIBLE_DEVICES=3 python peract_reliability/eval.py \
#             rlbench.tasks=[$t] \
#             rlbench.task_name='multi' \
#             rlbench.demo_path=$PERACT_ROOT/data/val \
#             framework.gpu=0 \
#             framework.logdir=$PERACT_ROOT/ckpts/ \
#             framework.start_seed=0 \
#             framework.eval_envs=1 \
#             framework.eval_from_eps_number=0 \
#             framework.eval_episodes=25 \
#             framework.csv_logging=True \
#             framework.tensorboard_logging=True \
#             framework.eval_type='last' \
#             rlbench.headless=True \
#             risk.tau=$tau_value \
#             risk.trans_conf_thresh=1.0e-6 \
#             risk.rot_conf_thresh=0.014 \
#             risk.search_size=20 \
#             risk.search_step=2
#     done
# done

for t in "${tasks[@]}"; do
    for tau_value in 5; do
        CUDA_VISIBLE_DEVICES=2 python peract_reliability/eval.py \
            rlbench.tasks=[$t] \
            rlbench.task_name='multi' \
            rlbench.demo_path=$PERACT_ROOT/data/extended \
            framework.gpu=0 \
            framework.logdir=$PERACT_ROOT/ckpts/ \
            framework.start_seed=0 \
            framework.eval_envs=1 \
            framework.eval_from_eps_number=0 \
            framework.eval_episodes=100 \
            framework.csv_logging=True \
            framework.tensorboard_logging=True \
            framework.eval_type='last' \
            rlbench.headless=True \
            risk.tau=$tau_value \
            risk.trans_conf_thresh=1.0e-6 \
            risk.rot_conf_thresh=0.014 \
            risk.search_size=20 \
            risk.search_step=2 \
            risk.log_dir="/home/bobwu/shared/safe_action_conf100_set/base_${t}/" \
            risk.enabled=True
    done
done
