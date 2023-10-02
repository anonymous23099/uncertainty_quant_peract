#!/bin/bash

# Define the list of tasks
# tasks=(open_drawer push_buttons slide_block_to_color_target put_money_in_safe put_groceries_in_cupboard)
# tasks=(push_buttons) # temp = 5.5
# tasks=(put_money_in_safe)

# [close_jar,insert_onto_square_peg,light_bulb_in,meat_off_grill,open_drawer,place_cups,place_shape_in_shape_sorter,push_buttons,put_groceries_in_cupboard,put_item_in_drawer,put_money_in_safe,reach_and_drag,stack_blocks,stack_cups,turn_tap,place_wine_at_rack_location,slide_block_to_color_target,sweep_to_dustpan_of_size] \

export PYTHONPATH=/home/bobwu/UQ/peract_headless/YARR/
export PYTHONPATH=/home/bobwu/UQ/peract_headless

# tasks=(close_jar insert_onto_square_peg light_bulb_in meat_off_grill open_drawer place_cups place_shape_in_shape_sorter push_buttons put_groceries_in_cupboard put_item_in_drawer put_money_in_safe reach_and_drag stack_blocks stack_cups turn_tap place_wine_at_rack_location slide_block_to_color_target sweep_to_dustpan_of_size)
# tasks=(light_bulb_in meat_off_grill open_drawer place_cups place_shape_in_shape_sorter push_buttons put_groceries_in_cupboard put_item_in_drawer put_money_in_safe reach_and_drag stack_blocks stack_cups turn_tap place_wine_at_rack_location slide_block_to_color_target sweep_to_dustpan_of_size)
# tasks=(light_bulb_in meat_off_grill open_drawer place_cups place_shape_in_shape_sorter push_buttons put_groceries_in_cupboard put_item_in_drawer put_money_in_safe reach_and_drag stack_blocks stack_cups turn_tap place_wine_at_rack_location slide_block_to_color_target sweep_to_dustpan_of_size close_jar insert_onto_square_peg)
# temps=(2.655 3.015 5.236 3.321 2.597 4.204 4.726 4.356 5.973 5.376 2.575 3.049 6.136 5.059 4.072 6.86 2.984 2.74)

# for t in "${tasks[@]}"; do
#     CUDA_VISIBLE_DEVICES=3 python peract_reliability/train.py \
#         method=PERACT_BC \
#         rlbench.tasks=[$t] \
#         rlbench.task_name='multi' \
#         rlbench.cameras=[front,left_shoulder,right_shoulder,wrist] \
#         rlbench.demos=25 \
#         rlbench.demo_path=$PERACT_ROOT/data/val \
#         replay.batch_size=1 \
#         replay.path=/tmp/replay \
#         replay.max_parallel_processes=2 \
#         method.voxel_sizes=[100] \
#         method.voxel_patch_size=5 \
#         method.voxel_patch_stride=5 \
#         method.num_latents=2048 \
#         method.transform_augmentation.apply_se3=True \
#         method.transform_augmentation.aug_rpy=[0.0,0.0,45.0] \
#         method.pos_encoding_with_lang=False \
#         framework.training_iterations=600000 \
#         framework.num_weights_to_keep=60 \
#         framework.start_seed=0 \
#         framework.log_freq=1000 \
#         framework.save_freq=1000 \
#         framework.logdir=$PERACT_ROOT/ckpts/ \
#         framework.csv_logging=True \
#         framework.tensorboard_logging=True \
#         ddp.num_devices=1 \
#         temperature.temperature_training=False \
#         temperature.temperature_use_hard_temp=True \
#         temperature.temperature_hard_temp=3

# done

# task and temperature for 18 tasks
# tasks=(light_bulb_in meat_off_grill open_drawer place_cups place_shape_in_shape_sorter push_buttons put_groceries_in_cupboard put_item_in_drawer put_money_in_safe reach_and_drag stack_blocks stack_cups turn_tap place_wine_at_rack_location slide_block_to_color_target sweep_to_dustpan_of_size close_jar insert_onto_square_peg)
# temps=(2.655 3.015 5.236 3.321 2.597 4.204 4.726 4.356 5.973 5.376 2.575 3.049 6.136 5.059 4.072 6.86 2.984 2.74)

# for i in "${!tasks[@]}"; do
#     t=${tasks[$i]}
#     temp=${temps[$i]}
#     CUDA_VISIBLE_DEVICES=3 python peract_reliability/train.py \
#         method=PERACT_BC \
#         rlbench.tasks=[$t] \
#         rlbench.task_name='multi' \
#         rlbench.cameras=[front,left_shoulder,right_shoulder,wrist] \
#         rlbench.demos=25 \
#         rlbench.demo_path=$PERACT_ROOT/data/val \
#         replay.batch_size=1 \
#         replay.path=/tmp/replay \
#         replay.max_parallel_processes=2 \
#         method.voxel_sizes=[100] \
#         method.voxel_patch_size=5 \
#         method.voxel_patch_stride=5 \
#         method.num_latents=2048 \
#         method.transform_augmentation.apply_se3=True \
#         method.transform_augmentation.aug_rpy=[0.0,0.0,45.0] \
#         method.pos_encoding_with_lang=False \
#         framework.training_iterations=600000 \
#         framework.num_weights_to_keep=60 \
#         framework.start_seed=0 \
#         framework.log_freq=1000 \
#         framework.save_freq=1000 \
#         framework.logdir=$PERACT_ROOT/ckpts/ \
#         framework.csv_logging=True \
#         framework.tensorboard_logging=True \
#         ddp.num_devices=1 \
#         temperature.temperature_training=False \
#         temperature.temperature_use_hard_temp=True \
#         temperature.temperature_hard_temp=$temp
# done



tasks=(set_clock_to_time put_rubbish_in_color_bin change_channel pick_and_lift play_jenga)
temps=(7.391 3.487 2.981 4.221 7.43)

for i in "${!tasks[@]}"; do
    t=${tasks[$i]}
    temp=${temps[$i]}
    CUDA_VISIBLE_DEVICES=3 python peract_reliability/train.py \
        method=PERACT_BC \
        rlbench.tasks=[$t] \
        rlbench.task_name='multi' \
        rlbench.cameras=[front,left_shoulder,right_shoulder,wrist] \
        rlbench.demos=100 \
        rlbench.demo_path=$PERACT_ROOT/data/zeroshot_eval \
        replay.batch_size=1 \
        replay.path=/tmp/replay \
        replay.max_parallel_processes=2 \
        method.voxel_sizes=[100] \
        method.voxel_patch_size=5 \
        method.voxel_patch_stride=5 \
        method.num_latents=2048 \
        method.transform_augmentation.apply_se3=True \
        method.transform_augmentation.aug_rpy=[0.0,0.0,45.0] \
        method.pos_encoding_with_lang=False \
        framework.training_iterations=600000 \
        framework.num_weights_to_keep=60 \
        framework.start_seed=0 \
        framework.log_freq=10000000 \
        framework.save_freq=10000000 \
        framework.logdir=$PERACT_ROOT/ckpts/ \
        framework.csv_logging=True \
        framework.tensorboard_logging=True \
        ddp.num_devices=1 \
        temperature.temperature_training=False \
        temperature.temperature_use_hard_temp=True \
        temperature.temperature_hard_temp=$temp
done
