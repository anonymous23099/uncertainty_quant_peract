#!/bin/bash

# Generate 100 eps test data for 21 tasks 
# tasks=(
#   "change_channel"
#   "close_jar"
#   "insert_onto_square_peg"
#   "light_bulb_in"
#   "meat_off_grill"
#   "place_cups"
#   "place_shape_in_shape_sorter"
#   "put_groceries_in_cupboard"
#   "put_item_in_drawer"
#   "reach_and_drag"
#   "stack_blocks"
#   "stack_cups"
#   "turn_tap"
#   "set_clock_to_time"
#   "put_rubbish_in_color_bin"
#   "slide_block_to_color_target"
#   "sweep_to_dustpan_of_size"
# )

# for task in "${tasks[@]}"; do
#   python RLBench/tools/dataset_generator.py --tasks="$task" \
#                                             --save_path="$PERACT_ROOT/data/extended" \
#                                             --image_size=128,128 \
#                                             --renderer=opengl \
#                                             --episodes_per_task=100 \
#                                             --processes=1 \
#                                             --all_variations=True
# done


# # Generate 25 eps calibration data for 5 out of distribution tasks 
# tasks=(set_clock_to_time) # put_rubbish_in_color_bin change_channel pick_and_lift play_jenga)

# for task in "${tasks[@]}"; do
#   python RLBench/tools/dataset_generator.py --tasks="$task" \
#                                             --save_path="$PERACT_ROOT/data/zeroshot_calib" \
#                                             --image_size=128,128 \
#                                             --renderer=opengl \
#                                             --episodes_per_task=25 \
#                                             --processes=1 \
#                                             --all_variations=True
# done

# Generate 100 eps test data for 5 out of distribution tasks 
tasks=(set_clock_to_time put_rubbish_in_color_bin change_channel pick_and_lift play_jenga)

for task in "${tasks[@]}"; do
  python RLBench/tools/dataset_generator.py --tasks="$task" \
                                            --save_path="$PERACT_ROOT/data/zeroshot_eval" \
                                            --image_size=128,128 \
                                            --renderer=opengl \
                                            --episodes_per_task=100 \
                                            --processes=1 \
                                            --all_variations=True
done