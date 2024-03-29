CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29501 ./tools/dist_train.sh configs/yolox_femtodet/femto_stage0.py 4
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29501 ./tools/dist_train.sh configs/yolox_femtodet/femto_stage1.py 4
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29501 ./tools/dist_train.sh configs/yolox_femtodet/femto_stage2.py 4
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29501 ./tools/dist_train.sh configs/yolox_femtodet/femto_stage3.py 4