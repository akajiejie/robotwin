#!/bin/bash
# bash process_data_depth.sh ${task_name} ${task_config} ${expert_data_num}
# Example: bash process_data_depth.sh beat_block_hammer demo_randomized 50
# bash process_data_depth.sh click_bell demo_randomized 50
# bash process_data_depth.sh click_alarmclock demo_randomized 50
# bash process_data_depth.sh adjust_bottle demo_randomized 50
# bash process_data_depth.sh adjust_bottle demo_clean 50
# bash process_data_depth.sh place_burger_fries demo_randomized 50
# bash process_data_depth.sh place_empty_cup demo_randomized 50
# bash process_data_depth.sh handover_mic demo_randomized 50
# bash process_data_depth.sh open_laptop demo_randomized 50
# bash process_data_depth.sh move_can_pot demo_randomized 50
# bash process_data_depth.sh move_can_pot demo_clean 50
# bash process_data_depth.sh put_object_cabinet demo_randomized 50
# bash process_data_depth.sh lift_pot demo_randomized 50
# bash process_data_depth.sh lift_pot demo_clean 50
# bash process_data_depth.sh pick_diverse_bottles demo_randomized 50
# bash process_data_depth.sh pick_dual_bottles demo_randomized 50
# bash process_data_depth.sh hanging_mug demo_randomized 50
# bash process_data_depth.sh place_can_basket demo_randomized 50
# bash process_data_depth.sh place_cans_plasticbox demo_randomized 50
# bash process_data_depth.sh place_dual_shoes demo_randomized 50
# bash process_data_depth.sh place_object_basket demo_randomized 50
# bash process_data_depth.sh place_shoe demo_randomized 50
# bash process_data_depth.sh place_phone_stand demo_randomized 50

task_name=${1}
task_config=${2}
expert_data_num=${3}

python scripts/process_data_depth.py $task_name $task_config $expert_data_num