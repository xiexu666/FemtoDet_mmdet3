# ./tools/dist_train.sh configs/yolox/yolox_s_8xb8-300e_coco_cls20_mean.py 8
# ./tools/dist_train.sh configs/yolox/yolox_s_8xb8-300e_coco_femoto_mean.py 8
# ./tools/dist_train.sh configs/yolox/yolox_s_8xb8-300e_coco_femoto_mean_dc.py 8
# ./tools/dist_train.sh configs/yolox/yolox_s_8xb8-300e_coco_femoto_mean_dc.py 8
# ./tools/dist_train.sh configs/yolox/yolox_s_8xb8-300e_coco_femoto_mean_dc_last0.py 8
# ./tools/dist_train.sh configs/yolox/yolox_s_8xb8-300e_coco_femoto_mean_dc_stage1.py 8
# ./tools/dist_train.sh configs/yolox/yolox_s_8xb8-300e_coco_femoto_mean_dc_last0_changeaffine.py 8
# ./tools/dist_train.sh configs/yolox/yolox_s_8xb8-300e_coco_femoto_mean_dc_last0_changeaffine_normal.py 8
# ./tools/dist_train.sh configs/yolox/yolox_s_8xb8-300e_coco_femoto_mean_dc_last0_changeaffine_normal_cosin.py 8
# CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29501 ./tools/dist_train.sh configs/yolox/yolox_s_8xb8-300e_coco_femoto_mean_dc_last0_changeaffine_normal_cosin.py 4
# CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29501 ./tools/dist_train.sh configs/yolox/yolox_s_8xb8-300e_coco_femoto_mean_dc_last0_changeaffine_normal_cosin_stage1.py 4
# CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29501 ./tools/dist_train.sh configs/yolox/yolox_s_8xb8-300e_coco_femoto_mean_dc_last0_changeaffine_normal_cosin2.py 4

# CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29501 ./tools/dist_train.sh configs/yolox/yolox_s_8xb8-300e_coco_femoto_mean_dc_last0_changeaffine_normal_cosin_stage1_lr.py 4
# CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29501 ./tools/dist_train.sh configs/yolox/yolox_s_8xb8-300e_coco_femoto_mean_dc_last0_changeaffine_normal_cosin_stage2.py 4 
# CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29501 ./tools/dist_train.sh configs/yolox/yolox_s_8xb8-300e_coco_femoto_mean_dc_last0_changeaffine_normal_cosin_stage_xx.py 4
# CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29501 ./tools/dist_train.sh configs/yolox/yolox_s_8xb8-300e_coco_femoto_mean_dc_last0_changeaffine_normal_cosin_stage2.py 4
# CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29501 ./tools/dist_train.sh configs/yolox/yolox_s_8xb8-300e_coco_femoto_mean_dc_last0_changeaffine_normal_cosin_stage_xx_2.py 4
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29501 ./tools/dist_train.sh configs/yolox_femtodet/femto_stage0.py 4
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29501 ./tools/dist_train.sh configs/yolox_femtodet/femto_stage1.py 4
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29501 ./tools/dist_train.sh configs/yolox_femtodet/femto_stage2.py 4
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29501 ./tools/dist_train.sh configs/yolox_femtodet/femto_stage3.py 4
python /home/xiexu/dance_burn.py