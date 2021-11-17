CUDA_VISIBLE_DEVICES=0,1,2,3 python train_gsn.py \
--base_config 'configs/models/gsn_odokitti_config_128.yaml' \
--log_dir 'test_odokitti_128_logs_multi_seq_0_40' \
data_config.dataset='odokitti' \
data_config.data_dir='data/odokitti_multi_seq'