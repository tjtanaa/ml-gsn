CUDA_VISIBLE_DEVICES=0,1,2,3 python train_gsn.py \
--base_config 'configs/models/gsn_odokitti_config_2.yaml' \
--log_dir 'test_odokitti_logs_2' \
--eval_freq 5 \
data_config.dataset='odokitti' \
data_config.data_dir='data/odokitti'