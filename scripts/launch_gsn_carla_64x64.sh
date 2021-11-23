CUDA_VISIBLE_DEVICES=0,1,2,3 python train_gsn.py \
--base_config 'configs/models/gsn_carla_config_1.yaml' \
--log_dir 'test_carla_logs_1' \
--eval_freq 5 \
data_config.dataset='carla' \
data_config.data_dir='data/carla_sequence_7_tiff_64x64'