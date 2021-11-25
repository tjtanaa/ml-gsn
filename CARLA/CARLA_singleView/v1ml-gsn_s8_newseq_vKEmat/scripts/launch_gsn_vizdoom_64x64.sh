CUDA_VISIBLE_DEVICES=4,5 python train_gsn.py \
--base_config 'configs/models/gsn_vizdoom_config.yaml' \
--log_dir 'logs' \
data_config.dataset='vizdoom' \
data_config.data_dir='data/new_Town02_left25resize512'