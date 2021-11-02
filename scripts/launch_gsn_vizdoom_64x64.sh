CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train_gsn.py \
--base_config 'configs/models/gsn_vizdoom_config.yaml' \
--log_dir 'logs' \
data_config.dataset='vizdoom' \
data_config.data_dir='data/vizdoom_data_iss'