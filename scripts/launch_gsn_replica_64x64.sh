CUDA_VISIBLE_DEVICES=4,5,6,7 python train_gsn.py \
--base_config 'configs/models/gsn_replica_config.yaml' \
--log_dir 'test_replica_logs' \
data_config.dataset='replica_all' \
data_config.data_dir='data/replica_all'