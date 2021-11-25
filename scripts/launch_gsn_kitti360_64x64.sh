CUDA_VISIBLE_DEVICES=8,9 python train_gsn.py \
--base_config 'configs/models/gsn_kitti360_config.yaml' \
--log_dir 'logs' \
--eval_freq 5 \
data_config.dataset='kitti360' \
data_config.data_dir='data/kitti360'