CUDA_VISIBLE_DEVICES=0,1,2,3 python train_gsn.py \
--base_config 'configs/models/gsn_carla_config_2.yaml' \
--log_dir 'test_carla_logs_2_slow' \
--eval_freq 5 \
data_config.dataset='carla' \
data_config.data_dir='data/carla_sequence_8_tiff_slow_64x64'
# data_config.data_dir='data/carla_sequence_7_tiff_64x64'