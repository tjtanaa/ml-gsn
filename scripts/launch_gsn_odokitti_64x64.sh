# CUDA_VISIBLE_DEVICES=4,5,6,7 python train_gsn.py \
# --base_config 'configs/models/gsn_odokitti_corrected_config_0_crop.yaml' \
# --log_dir 'test_odokitti_Kcorrected_logs_0_crop' \
# --eval_freq 5 \
# data_config.dataset='odokittiKCorrected' \
# data_config.data_dir='data/odokitti_Kcorrected'

CUDA_VISIBLE_DEVICES=0,1 python train_gsn.py \
--base_config 'configs/models/gsn_odokitti_config_6_crop.yaml' \
--log_dir 'test_odokitti_logs_6_crop' \
--eval_freq 5 \
data_config.dataset='odokitti' \
data_config.data_dir='data/odokitti'