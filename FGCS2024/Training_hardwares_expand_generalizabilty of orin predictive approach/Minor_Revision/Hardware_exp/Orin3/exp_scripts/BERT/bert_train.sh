#!/bin/bash


prefetch_factor=2
no_workers=(4)
external_device="nvme0n1p1"
batch_size=16

python3 /home/dream-orin3/kedar/bert/Bert_train.py  "/home/dream-orin3/kedar/bert" "$no_workers" "$prefetch_factor" "$external_device" "$batch_size" > "train_log_file_pm_14"

mkdir Train_logs
mv mn_nw* Train_logs
mv log_file_* Train_logs
