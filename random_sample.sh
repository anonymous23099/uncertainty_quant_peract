#!/bin/bash

# Define the list of tasks
# tasks=(open_drawer push_buttons slide_block_to_color_target put_money_in_safe put_groceries_in_cupboard)
# tasks=(push_buttons) # temp = 5.5
tasks=(put_money_in_safe)

export PYTHONPATH=/home/bobwu/UQ/peract_headless/YARR/
export PYTHONPATH=/home/bobwu/UQ/peract_headless


# Loop over each task
for t in "${tasks[@]}"; do
    CUDA_VISIBLE_DEVICES=0 python peract_reliability/train.py \
        method=PERACT_BC \
        rlbench.tasks=[$t] \
        rlbench.task_name='multi' \
        rlbench.cameras=[front,left_shoulder,right_shoulder,wrist] \
        rlbench.demos=10 \
        rlbench.demo_path=$PERACT_ROOT/data/val \
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
        framework.log_freq=1000 \
        framework.save_freq=10000 \
        framework.logdir=$PERACT_ROOT/ckpts/ \
        framework.csv_logging=True \
        framework.tensorboard_logging=True \
        framework.load_existing_weights=True \
        ddp.num_devices=1
done
