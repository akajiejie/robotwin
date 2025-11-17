#!/bin/bash
# bash process_data_unified.sh ${task_name} ${task_config} ${expert_data_num}
# Example: bash process_data_unified.sh beat_block_hammer demo_randomized 50
# bash process_data_unified.sh handover_mic demo_randomized 50
# bash process_data_unified.sh open_laptop demo_randomized 50
# bash process_data_unified.sh move_can_pot demo_randomized 50
# bash process_data_unified.sh put_object_cabinet demo_randomized 50
# bash process_data_unified.sh lift_pot demo_randomized 50
# bash process_data_unified.sh pick_diverse_bottles demo_randomized 50
# bash process_data_unified.sh pick_dual_bottles demo_randomized 50
# bash process_data_unified.sh hanging_mug demo_randomized 50
# bash process_data_unified.sh place_can_basket demo_randomized 50
# bash process_data_unified.sh place_cans_plasticbox demo_randomized 50
# bash process_data_unified.sh place_dual_shoes demo_randomized 50
# bash process_data_unified.sh place_object_basket demo_randomized 50
# bash process_data_unified.sh place_shoe demo_randomized 50
# bash process_data_unified.sh place_phone_stand demo_randomized 50

task_name=${1}
task_config=${2}
expert_data_num=${3}

python scripts/process_data_unified.py $task_name $task_config $expert_data_num