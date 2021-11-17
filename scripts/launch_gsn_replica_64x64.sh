CUDA_VISIBLE_DEVICES=4 python train_gsn.py \
--base_config 'configs/models/gsn_replica_config.yaml' \
--log_dir 'test_replica_logs2' \
data_config.dataset='replica_all' \
data_config.data_dir='data/replica_all'