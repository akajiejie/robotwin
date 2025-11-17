#!/bin/bash

# bash train_eval.sh lift_pot robot_unified_maniflow_pointmap demo_randomized 50 0905_MMDiT-G_Mask_Attn_R3M_224_Img_0.1_PointMap_w_color_0.3_h1_bs64_epo300 0 2 # trained 0, eval
# bash train_eval.sh lift_pot robot_unified_maniflow_pointmap demo_randomized 50 0905_MMDiT-G_Mask_Attn_R3M_224_Img_0.1_PointMap_w_color_0.8_h1_bs64_epo300 0 2 # trained 0, eval
# bash train_eval.sh lift_pot robot_unified_maniflow_pointmap demo_randomized 50 0905_MMDiT-G_Mask_Attn_R3M_224_Img_0.1_PointMap_w_color_shard_enc_0.3_h1_epo300 0 2 # trained 0, eval

# bash train_eval.sh lift_pot robot_unified_maniflow_pointmap demo_randomized 50 0905_MMDiT-G_Mask_Attn_Clip_224_Img_0.1_PointMap_R3M_0.3_h1_bs64_epo300 0 2 # trained 0
# bash train_eval.sh lift_pot robot_unified_maniflow_pointmap demo_randomized 50 0905_MMDiT-G_Mask_Attn_Clip_224_Img_0.1_PointMap_R3M_0.8_h1_bs64_epo300 0 2 # trained 0, eval
# bash train_eval.sh lift_pot robot_unified_maniflow_pointmap demo_randomized 50 0905_MMDiT-G_Mask_Attn_Clip_224_Img_0.1_PointMap_Clip_0.3_h1_bs64_epo300 0 2 # trained 0
# bash train_eval.sh lift_pot robot_unified_maniflow_pointmap demo_randomized 50 0905_MMDiT-G_Mask_Attn_Clip_224_Img_0.1_PointMap_Clip_0.8_h1_bs64_epo300 0 2 # trained 0, eval

# bash train_eval.sh pick_dual_bottles robot_unified_maniflow_pointmap demo_randomized 50 0905_MMDiT-G_Mask_Attn_RGB_only_R3M_224_h1_bs64_gpus1_demo50_epo300 0 2 # trained 0
# bash train_eval.sh pick_dual_bottles robot_unified_maniflow_pointmap demo_randomized 50 0905_MMDiT-G_Mask_Attn_R3M_224_Img_0.1_PointMap_no_color_0.1_h1_bs64_epo300 0 2
# bash train_eval.sh pick_dual_bottles robot_unified_maniflow_pointmap demo_randomized 50 0905_MMDiT-G_Mask_Attn_R3M_224_Img_0.1_PointMap_no_color_0.3_h1_bs64_epo300 0 2 # trained 0
# bash train_eval.sh pick_dual_bottles robot_unified_maniflow_pointmap demo_randomized 50 0905_MMDiT-G_Mask_Attn_R3M_224_Img_0.1_PointMap_no_color_0.5_h1_bs64_epo300 0 2
# bash train_eval.sh pick_dual_bottles robot_unified_maniflow_pointmap demo_randomized 50 0905_MMDiT-G_Mask_Attn_R3M_224_Img_0.1_PointMap_no_color_0.8_h1_bs64_epo300 0 2 # trained 0

# bash train_eval.sh pick_dual_bottles robot_unified_maniflow_pointmap demo_randomized 50 0906_MMDiT-G_Mask_Attn_RGB_only_Clip_224_h1_bs64_gpus1_demo50_epo300 0 2 # trained 0, eval
# bash train_eval.sh beat_block_hammer robot_unified_maniflow_pointmap demo_randomized 50 0906_MMDiT-G_Mask_Attn_RGB_only_R3M_224_h1_bs64_gpus1_demo50_epo300 0 2 # trained 0, eval
# bash train_eval.sh beat_block_hammer robot_unified_maniflow_pointmap demo_randomized 50 0906_MMDiT-G_Mask_Attn_RGB_only_Clip_224_h1_bs64_gpus1_demo50_epo300 0 1 # trained 0, eval
# bash train_eval.sh beat_block_hammer robot_unified_maniflow_pointmap demo_randomized 50 0906_MMDiT-G_Mask_Attn_R3M_224_Img_0.1_PointMap_no_color_0.3_h1_bs64_epo300 0 2 # trained 0, eval
# bash train_eval.sh beat_block_hammer robot_unified_maniflow_pointmap demo_randomized 50 0906_MMDiT-G_Mask_Attn_R3M_224_Img_0.1_PointMap_no_color_0.5_h1_bs64_epo300 0 2
# bash train_eval.sh beat_block_hammer robot_unified_maniflow_pointmap demo_randomized 50 0906_MMDiT-G_Mask_Attn_R3M_224_Img_0.1_PointMap_no_color_0.8_h1_bs64_epo300 0 2 # trained 0, eval
# bash train_eval.sh beat_block_hammer robot_unified_maniflow_pointmap demo_randomized 50 0907_MMDiT-G_Mask_Attn_R3M_224_Img_0.1_PointMap_no_color_0.3_h1_bs64_epo300 0 2 # trained 0

# bash train_eval.sh click_bell robot_unified_maniflow_pointmap demo_randomized 50 0907_MMDiT-G_Mask_Attn_RGB_only_R3M_224_h1_bs64_gpus1_demo50_epo300 0 2 # trained 0, eval
# bash train_eval.sh click_bell robot_unified_maniflow_pointmap demo_randomized 50 0907_MMDiT-G_Mask_Attn_RGB_only_Clip_224_h1_bs64_gpus1_demo50_epo300 0 1 # trained 0, eval
# bash train_eval.sh click_bell robot_unified_maniflow_pointmap demo_randomized 50 0907_MMDiT-G_Mask_Attn_R3M_224_Img_0.1_PointMap_no_color_0.3_h1_bs64_epo300 0 2 # trained 0
# bash train_eval.sh click_bell robot_unified_maniflow_pointmap demo_randomized 50 0907_MMDiT-G_Mask_Attn_Clip_224_Img_0.1_PointMap_R3M_no_color_0.3_h1_epo300 0 2
# bash train_eval.sh click_bell robot_unified_maniflow_pointmap demo_randomized 50 0907_MMDiT-G_Mask_Attn_R3M_224_Img_0.1_PointMap_no_color_0.5_h1_bs64_epo300 0 2
# bash train_eval.sh click_bell robot_unified_maniflow_pointmap demo_randomized 50 0907_MMDiT-G_Mask_Attn_R3M_224_Img_0.1_PointMap_no_color_0.8_h1_bs64_epo300 0 2
# bash train_eval.sh click_bell robot_unified_maniflow_pointmap demo_randomized 50 0910_MMDiT-G_Mask_Attn_R3M_224_Img_0.1_PointMap_CLIP_no_color_0.3_bs64 0 2 # trained 0
# bash train_eval.sh click_bell robot_unified_maniflow_pointmap demo_randomized 50 0910_MMDiT-G_Mask_Attn_R3M_224_Img_0.1_PointMap_CLIP_w_color_shared_0.3_bs64 0 2

# bash train_eval.sh adjust_bottle robot_unified_maniflow_pointmap demo_randomized 50 0908_MMDiT-G_Mask_Attn_RGB_only_R3M_224_h1_bs128_gpus1_demo50_epo300 0 2 # trained 0, eval
# bash train_eval.sh adjust_bottle robot_unified_maniflow_pointmap demo_randomized 50 0908_MMDiT-G_Mask_Attn_RGB_only_Clip_224_h1_bs128_gpus1_demo50_epo300 0 1 # trained 0
# bash train_eval.sh adjust_bottle robot_unified_maniflow_pointmap demo_randomized 50 0908_MMDiT-G_Mask_Attn_R3M_224_Img_0.1_PointMap_no_color_0.3_h1_bs64_epo300 0 2 # trained 0, eval
# bash train_eval.sh adjust_bottle robot_unified_maniflow_pointmap demo_randomized 50 0908_MMDiT-G_Mask_Attn_Clip_224_Img_0.1_PointMap_R3M_no_color_0.3_h1_epo300 0 2
# bash train_eval.sh adjust_bottle robot_unified_maniflow_pointmap demo_randomized 50 0908_MMDiT-G_Mask_Attn_R3M_224_Img_0.1_PointMap_no_color_0.5_h1_bs64_epo300 0 2
# bash train_eval.sh adjust_bottle robot_unified_maniflow_pointmap demo_randomized 50 0908_MMDiT-G_Mask_Attn_R3M_224_Img_0.1_PointMap_no_color_0.8_h1_bs64_epo300 0 2

# bash train_eval.sh place_empty_cup robot_unified_maniflow_pointmap demo_randomized 50 0909_MMDiT-G_Mask_Attn_RGB_only_R3M_224_h1_bs128_gpus1_demo50_epo300 0 2 # trained 0, eval
# bash train_eval.sh place_empty_cup robot_unified_maniflow_pointmap demo_randomized 50 0909_MMDiT-G_Mask_Attn_RGB_only_Clip_224_h1_bs128_gpus1_demo50_epo300 0 1 # trained 0, eval
# bash train_eval.sh place_empty_cup robot_unified_maniflow_pointmap demo_randomized 50 0909_MMDiT-G_Mask_Attn_R3M_224_Img_0.1_PointMap_no_color_0.3_h1_bs128_epo300 0 2 # trained 0, eval
# bash train_eval.sh place_empty_cup robot_unified_maniflow_pointmap demo_randomized 50 0909_MMDiT-G_Mask_Attn_Clip_224_Img_0.1_PointMap_R3M_no_color_0.3_h1_epo300 0 2
# bash train_eval.sh place_empty_cup robot_unified_maniflow_pointmap demo_randomized 50 0909_MMDiT-G_Mask_Attn_R3M_224_Img_0.1_PointMap_no_color_0.5_h1_bs64_epo300 0 2
# bash train_eval.sh place_empty_cup robot_unified_maniflow_pointmap demo_randomized 50 0909_MMDiT-G_Mask_Attn_R3M_224_Img_0.1_PointMap_no_color_0.8_h1_bs64_epo300 0 2
# bash train_eval.sh place_empty_cup robot_unified_maniflow_pointmap demo_randomized 50 0910_MMDiT-G_Mask_Attn_R3M_224_Img_0.1_PointMap_CLIP_no_color_0.3_bs64 0 2
# bash train_eval.sh place_empty_cup robot_unified_maniflow_pointmap demo_randomized 50 0910_MMDiT-G_Mask_Attn_R3M_224_Img_0.1_PointMap_CLIP_w_color_shared_0.3_bs64 0 2

# bash train_eval.sh click_alarmclock robot_unified_maniflow_pointmap demo_randomized 50 0909_MMDiT-G_Mask_Attn_RGB_only_R3M_224_h1_bs128_gpus1_demo50_epo300 0 2 # trained 0
# bash train_eval.sh click_alarmclock robot_unified_maniflow_pointmap demo_randomized 50 0909_MMDiT-G_Mask_Attn_RGB_only_Clip_224_h1_bs128_gpus1_demo50_epo300 0 1 # trained 0
# bash train_eval.sh click_alarmclock robot_unified_maniflow_pointmap demo_randomized 50 0909_MMDiT-G_Mask_Attn_R3M_224_Img_0.1_PointMap_no_color_0.3_h1_bs128_epo300 0 2 # trained 0
# bash train_eval.sh click_alarmclock robot_unified_maniflow_pointmap demo_randomized 50 0909_MMDiT-G_Mask_Attn_Clip_224_Img_0.1_PointMap_R3M_no_color_0.3_h1_epo300 0 2
# bash train_eval.sh click_alarmclock robot_unified_maniflow_pointmap demo_randomized 50 0909_MMDiT-G_Mask_Attn_R3M_224_Img_0.1_PointMap_no_color_0.5_h1_bs64_epo300 0 2
# bash train_eval.sh click_alarmclock robot_unified_maniflow_pointmap demo_randomized 50 0909_MMDiT-G_Mask_Attn_R3M_224_Img_0.1_PointMap_no_color_0.8_h1_bs64_epo300 0 2

# bash train_eval.sh lift_pot robot_unified_maniflow_pointmap demo_clean 50 0909_MMDiT-G_Mask_Attn_RGB_only_R3M_224_h1_bs128_gpus1_demo50_epo300 0 2 # trained 0
# bash train_eval.sh lift_pot robot_unified_maniflow_pointmap demo_clean 50 0909_MMDiT-G_Mask_Attn_RGB_only_Clip_224_h1_bs128_gpus1_demo50_epo300 0 1 # trained 0
# bash train_eval.sh lift_pot robot_unified_maniflow_pointmap demo_clean 50 0910_MMDiT-G_Mask_Attn_R3M_224_Img_0.1_PointMap_no_color_0.3_h1_bs128_epo300 0 2 # trained 0
# bash train_eval.sh lift_pot robot_unified_maniflow_pointmap demo_clean 50 0909_MMDiT-G_Mask_Attn_Clip_224_Img_0.1_PointMap_R3M_no_color_0.3_h1_epo300 0 2
# bash train_eval.sh lift_pot robot_unified_maniflow_pointmap demo_clean 50 0909_MMDiT-G_Mask_Attn_R3M_224_Img_0.1_PointMap_no_color_0.5_h1_bs64_epo300 0 2
# bash train_eval.sh lift_pot robot_unified_maniflow_pointmap demo_clean 50 0909_MMDiT-G_Mask_Attn_R3M_224_Img_0.1_PointMap_no_color_0.8_h1_bs64_epo300 0 2

# bash train_eval.sh move_can_pot robot_unified_maniflow_pointmap demo_randomized 50 0909_MMDiT-G_Mask_Attn_RGB_only_R3M_224_h1_bs128_gpus1_demo50_epo300 0 2 # trained 0, eval
# bash train_eval.sh move_can_pot robot_unified_maniflow_pointmap demo_randomized 50 0909_MMDiT-G_Mask_Attn_RGB_only_Clip_224_h1_bs128_gpus1_demo50_epo300 0 1 # trained 0
# bash train_eval.sh move_can_pot robot_unified_maniflow_pointmap demo_randomized 50 0909_MMDiT-G_Mask_Attn_R3M_224_Img_0.1_PointMap_no_color_0.3_h1_bs128_epo300 0 2 # trained 0
# bash train_eval.sh move_can_pot robot_unified_maniflow_pointmap demo_randomized 50 0909_MMDiT-G_Mask_Attn_Clip_224_Img_0.1_PointMap_R3M_no_color_0.3_h1_epo300 0 2
# bash train_eval.sh move_can_pot robot_unified_maniflow_pointmap demo_randomized 50 0909_MMDiT-G_Mask_Attn_R3M_224_Img_0.1_PointMap_no_color_0.5_h1_bs64_epo300 0 2
# bash train_eval.sh move_can_pot robot_unified_maniflow_pointmap demo_randomized 50 0909_MMDiT-G_Mask_Attn_R3M_224_Img_0.1_PointMap_no_color_0.8_h1_bs64_epo300 0 2

# bash train_eval.sh adjust_bottle robot_unified_maniflow_pointmap demo_clean 50 0910_MMDiT-G_Mask_Attn_RGB_only_R3M_224_h1_bs128_gpus1_demo50_epo300 0 2 # trained 0, eval
# bash train_eval.sh adjust_bottle robot_unified_maniflow_pointmap demo_clean 50 0910_MMDiT-G_Mask_Attn_RGB_only_Clip_224_h1_bs128_gpus1_demo50_epo300 0 1
# bash train_eval.sh adjust_bottle robot_unified_maniflow_pointmap demo_clean 50 0910_MMDiT-G_Mask_Attn_R3M_224_Img_0.1_PointMap_no_color_0.3_h1_bs128_epo300 0 2 # trained 0
# bash train_eval.sh adjust_bottle robot_unified_maniflow_pointmap demo_clean 50 0910_2_MMDiT-G_Mask_Attn_R3M_224_Img_0.1_PointMap_no_color_0.3_h1_bs256_epo300 0 2 # trained 0

# bash train_eval.sh place_burger_fries robot_unified_maniflow_pointmap demo_randomized 50 0910_MMDiT-G_Mask_Attn_RGB_only_R3M_224_h1_bs128_gpus1_demo50_epo300 0 2 # trained 0, eval
# bash train_eval.sh place_burger_fries robot_unified_maniflow_pointmap demo_randomized 50 0910_MMDiT-G_Mask_Attn_RGB_only_Clip_224_h1_bs128_gpus1_demo50_epo300 0 1
# bash train_eval.sh place_burger_fries robot_unified_maniflow_pointmap demo_randomized 50 0910_MMDiT-G_Mask_Attn_R3M_224_Img_0.1_PointMap_no_color_0.3_h1_bs128_epo300 0 2

################# new ################

# bash train_eval.sh adjust_bottle robot_unified_maniflow_pointmap demo_clean 200 0910_MMDiT-G_R3M_224_Img_0.1_PointMap_no_color_0.3_h1_bs256_demo200_epo300 0 2 # trained 0, local
# bash train_eval.sh adjust_bottle robot_unified_maniflow_pointmap demo_clean 200 0912_MMDiT-G_R3M_224_Img_0.1_PointMap_no_color_0.3_h1_bs256_demo200_epo500 0 2 # trained 0, eval
# bash train_eval.sh adjust_bottle robot_unified_maniflow_pointmap demo_clean 100 0912_MMDiT-G_R3M_224_Img_0.1_PointMap_no_color_0.3_h1_bs256_demo100_epo500 0 2 # trained 0, eval
# bash train_eval.sh adjust_bottle robot_unified_maniflow_pointmap demo_clean 300 0912_MMDiT-G_R3M_224_Img_0.1_PointMap_no_color_0.3_h1_bs512_demo300_epo500 0 2 # trained 0, eval
# bash train_eval.sh adjust_bottle robot_unified_maniflow_pointmap demo_clean 400 0912_MMDiT-G_R3M_224_Img_0.1_PointMap_no_color_0.3_h1_bs512_demo400_gpu2_epo600 0 2_3, eval
# bash train_eval.sh adjust_bottle robot_unified_maniflow_pointmap demo_clean 500 0912_MMDiT-G_R3M_224_Img_0.1_PointMap_no_color_0.3_h1_bs512_demo500_gpu4_epo600 0 4_5_6_7, eval
# bash train_eval.sh move_can_pot robot_unified_maniflow_pointmap demo_clean 50 0911_MMDiT-G_Mask_Attn_R3M_224_Img_0.1_PointMap_no_color_0.3_h1_bs128_epo300 0 2 # trained 0, eval
# bash train_eval.sh lift_pot robot_unified_maniflow_pointmap demo_clean 50 0911_MMDiT-G_R3M_224_Img_0.1_PointMap_no_color_0.3_Que64_h1_bs128_epo300 0 2 # trained 0, eval
# bash train_eval.sh adjust_bottle robot_unified_maniflow_pointmap demo_clean 50 0911_MMDiT-G_R3M_224_Img_0.1_PointMap_no_color_0.3_Que64_h1_bs128_epo300 0 2 # trained 0, eval
# bash train_eval.sh adjust_bottle robot_unified_maniflow_pointmap demo_clean 50 0911_MMDiT-G_PointMap_Only_no_color_h1_bs128_epo300 0 2 # trained 0, eval

# bash train_eval.sh move_can_pot robot_unified_maniflow_pointmap demo_randomized 50 0912_Img_R3M_0.1_PointMap_R3M_no_color_add_que64_0.3_h1_bs64_epo300 0 2 # trained 0, eval
# bash train_eval.sh move_can_pot robot_unified_maniflow_pointmap demo_randomized 50 0912_Img_R3M_0.1_PointMap_CLIP_no_color_0.3_h1_bs64_epo300 0 2 # trained 0, eval
# bash train_eval.sh move_can_pot robot_unified_maniflow_pointmap demo_randomized 50 0912_Img_CLIP_0.1_PointMap_CLIP_no_color_0.3_h1_bs64_epo300 0 2 # trained 0, eval
# bash train_eval.sh move_can_pot robot_unified_maniflow_pointmap demo_randomized 50 0912_Img_R3M_0.1_PointMap_CLIP_no_color_add_que64_0.3_h1_bs128_epo300 0 2 # trained 0, eval
# bash train_eval.sh move_can_pot robot_unified_maniflow_pointmap demo_randomized 50 0912_Img_CLIP_0.1_PointMap_CLIP_no_color_add_que64_0.3_h1_bs64_epo300 0 2 # trained 0, eval

# bash train_eval.sh lift_pot robot_unified_maniflow_pointmap demo_randomized 50 0913_Img_R3M_0.1_PointMap_R3M_no_color_add_que64_0.3_h1_bs128_epo300 0 2 # trained 0
# bash train_eval.sh adjust_bottle robot_unified_maniflow_pointmap demo_randomized 50 0913_Img_R3M_0.1_PointMap_R3M_no_color_add_que64_0.3_h1_bs128_epo300 0 2 # trained 0
# bash train_eval.sh move_can_pot robot_unified_maniflow_pointmap demo_clean 50 0913_C_Img_R3M_0.1_PointMap_R3M_no_color_add_que64_0.3_h1_bs128_epo300 0 2 # trained 0
# bash train_eval.sh lift_pot robot_unified_maniflow_pointmap demo_clean 50 0913_C_Img_R3M_0.1_PointMap_R3M_no_color_add_que64_0.3_h1_bs128_epo300 0 2 # trained 0
# bash train_eval.sh adjust_bottle robot_unified_maniflow_pointmap demo_clean 50 0913_C_Img_R3M_0.1_PointMap_R3M_no_color_add_que64_0.3_h1_bs128_epo300 0 2 # trained 0
# bash train_eval.sh move_can_pot robot_unified_maniflow_pointmap demo_randomized 50 0913_Img_R3M_0.1_PointMap_CLIP_no_color_replace_que256_0.3_h1_bs128_epo300 0 2 # trained 0
# bash train_eval.sh lift_pot robot_unified_maniflow_pointmap demo_randomized 50 0913_Img_R3M_0.1_PointMap_CLIP_no_color_replace_que256_0.3_h1_bs128_epo300 0 2 # trained 0
# bash train_eval.sh adjust_bottle robot_unified_maniflow_pointmap demo_randomized 50 0913_Img_R3M_0.1_PointMap_CLIP_no_color_replace_que256_0.3_h1_bs128_epo300 0 2 # trained 0
# bash train_eval.sh move_can_pot robot_unified_maniflow_pointmap demo_clean 50 0913_C_Img_R3M_0.1_PointMap_CLIP_no_color_replace_que256_0.3_h1_bs128_epo300 0 2 # trained 0
# bash train_eval.sh lift_pot robot_unified_maniflow_pointmap demo_clean 50 0913_C_Img_R3M_0.1_PointMap_CLIP_no_color_replace_que256_0.3_h1_bs128_epo300 0 2 # trained 0
# bash train_eval.sh adjust_bottle robot_unified_maniflow_pointmap demo_clean 50 0913_C_Img_R3M_0.1_PointMap_CLIP_no_color_replace_que256_0.3_h1_bs128_epo300 0 2 # trained 0

# bash train_eval.sh lift_pot robot_unified_maniflow_pointmap demo_clean 50 0913_PointMap_Only_R3M_no_color_h1_bs128_epo300 0 2 # trained 0, eval
# bash train_eval.sh adjust_bottle robot_unified_maniflow_pointmap demo_randomized 50 0913_PointMap_Only_R3M_no_color_h1_bs128_epo300 0 2 # trained 0, eval
# bash train_eval.sh adjust_bottle robot_unified_maniflow_pointmap demo_clean 50 0913_C_PointMap_Only_R3M_no_color_h1_bs128_epo300 0 2 # trained 0, eval
# bash train_eval.sh move_can_pot robot_unified_maniflow_pointmap demo_randomized 50 0913_PointMap_Only_R3M_no_color_h1_bs128_epo300 0 2 # trained 0, eval
# bash train_eval.sh move_can_pot robot_unified_maniflow_pointmap demo_clean 50 0913_C_PointMap_Only_R3M_no_color_h1_bs128_epo300 0 2 # trained 0

# bash train_eval.sh lift_pot robot_unified_maniflow_pointmap demo_clean 50 0914_PointMap_Only_R3M_w_color_h1_bs128_epo300 0 2 # trained 0
# bash train_eval.sh lift_pot robot_unified_maniflow_pointmap demo_randomized 50 0914_PointMap_Only_R3M_w_color_h1_bs128_epo300 0 2 # trained 0
# bash train_eval.sh adjust_bottle robot_unified_maniflow_pointmap demo_clean 50 0914_PointMap_Only_R3M_w_color_h1_bs128_epo300 0 2 # trained 0
# bash train_eval.sh adjust_bottle robot_unified_maniflow_pointmap demo_randomized 50 0914_PointMap_Only_R3M_w_color_h1_bs128_epo300 0 2 # trained 0
# bash train_eval.sh move_can_pot robot_unified_maniflow_pointmap demo_clean 50 0914_PointMap_Only_R3M_w_color_h1_bs128_epo300 0 2 # trained 0
# bash train_eval.sh move_can_pot robot_unified_maniflow_pointmap demo_randomized 50 0914_PointMap_Only_R3M_w_color_h1_bs128_epo300 0 2 # trained 0

# bash train_eval.sh lift_pot robot_unified_maniflow_pointmap demo_clean 50 0914_Image_Only_R3M_w_color_h1_bs128_epo300 0 2 # trained 0
# bash train_eval.sh adjust_bottle robot_unified_maniflow_pointmap demo_clean 50 0914_C_Image_Only_R3M_w_color_h1_bs128_epo300 0 2 # trained 0
# bash train_eval.sh adjust_bottle robot_unified_maniflow_pointmap demo_randomized 50 0914_Image_Only_R3M_w_color_h1_bs128_epo300 0 2 # trained 0
# bash train_eval.sh move_can_pot robot_unified_maniflow_pointmap demo_clean 50 0914_C_Image_Only_R3M_w_color_h1_bs128_epo300 0 2 # trained 0
# bash train_eval.sh move_can_pot robot_unified_maniflow_pointmap demo_randomized 50 0914_Image_Only_R3M_w_color_h1_bs128_epo300 0 2 # trained 0

# bash train_eval.sh lift_pot robot_unified_maniflow_pointmap demo_clean 50 0914_Co_Img_PointMap_R3M_no_color_h1_bs128_epo300 0 2 # trained 0
# bash train_eval.sh lift_pot robot_unified_maniflow_pointmap demo_randomized 50 0914_Co_Img_PointMap_R3M_no_color_h1_bs128_epo300 0 2 # trained 0
# bash train_eval.sh adjust_bottle robot_unified_maniflow_pointmap demo_clean 50 0914_Co_Img_PointMap_R3M_no_color_h1_bs128_epo300 0 2 # trained 0
# bash train_eval.sh adjust_bottle robot_unified_maniflow_pointmap demo_randomized 50 0914_Co_Img_PointMap_R3M_no_color_h1_bs128_epo300 0 2 # trained 0
# bash train_eval.sh move_can_pot robot_unified_maniflow_pointmap demo_clean 50 0914_Co_Img_PointMap_R3M_no_color_h1_bs128_epo300 0 2 # trained 0
# bash train_eval.sh move_can_pot robot_unified_maniflow_pointmap demo_randomized 50 0914_Co_Img_PointMap_R3M_no_color_h1_bs128_epo300 0 2 # trained 0

# bash train_eval_ddp.sh lift_pot robot_maniflow demo_randomized 50 debug_ddp 0 0_1 # trained 0
# bash train_eval_ddp.sh lift_pot robot_unified_maniflow demo_randomized 50 debug_img 0 2_3 # trained 0
# bash train_eval_ddp.sh lift_pot robot_unified_maniflow_pointmap demo_randomized 50 debug_img 0 2_3 # trained 0

train=false
eval=true
policy_name=ManiFlow
task_name=${1}
alg_name=${2} # alg_name is the algorithm name, e.g., robot_maniflow
task_config=${3}
expert_data_num=${4}
addition_info=${5}
seed=${6}
gpu_id=${7} # Example: "2_4_5"

eval_seed=0 # seed for evaluation, can be changed to 1, 2, etc.


if [ "$train" = true ]; then
    echo "Training is enabled."
    if [ ! -d "./data/${task_name}-${task_config}-${expert_data_num}.zarr" ]; then
        bash process_data.sh ${task_name} ${task_config} ${expert_data_num}
    fi
    bash scripts/train_policy_ddp.sh ${alg_name} ${task_name} ${task_config} ${expert_data_num} ${addition_info} ${seed} ${gpu_id}
else
    echo "Training is disabled."
fi



# if eval is false, skip evaluation
if [ "$eval" = false ]; then
    echo "Evaluation is disabled."
    exit 0
else
    echo "Evaluation is enabled."
    echo "Evaluating policy with task: ${task_name}, config: ${task_config}, expert data num: ${expert_data_num}, seed: ${seed}, gpu id: ${gpu_id}"
fi

# take first gpu id if multiple are provided
if [[ $gpu_id == *"_"* ]]; then
    gpu_id=$(echo $gpu_id | cut -d'_' -f1)
fi
echo -e "\033[33mgpu id (to use) for eval: ${gpu_id}\033[0m"

# Evaluate the trained policy
export CUDA_VISIBLE_DEVICES=${gpu_id}
export HYDRA_FULL_ERROR=1
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

cd ../.. # move to root

ckpt_setting=${task_config} # setting for evaluation, can be changed to demo_clean

task_config=demo_clean # setting for evaluation, can be changed to demo_clean

PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy.py --config policy/$policy_name/deploy_policy.yml \
    --overrides \
    --config_name ${alg_name} \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --ckpt_setting ${ckpt_setting} \
    --expert_data_num ${expert_data_num} \
    --seed ${eval_seed} \
    --training_seed ${seed} \
    --policy_name ${policy_name} \
    --addition_info ${addition_info} \
    --alg_name ${alg_name} \


# task_config=demo_randomized # setting for evaluation, can be changed to demo_clean

# PYTHONWARNINGS=ignore::UserWarning \
# python script/eval_policy.py --config policy/$policy_name/deploy_policy.yml \
#     --overrides \
#     --config_name ${alg_name} \
#     --task_name ${task_name} \
#     --task_config ${task_config} \
#     --ckpt_setting ${ckpt_setting} \
#     --expert_data_num ${expert_data_num} \
#     --seed ${eval_seed} \
#     --training_seed ${seed} \
#     --policy_name ${policy_name} \
#     --addition_info ${addition_info} \
#     --alg_name ${alg_name} \
