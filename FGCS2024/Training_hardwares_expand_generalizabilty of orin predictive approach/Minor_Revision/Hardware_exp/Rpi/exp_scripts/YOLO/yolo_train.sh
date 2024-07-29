#!/bin/bash

prefetch_factor=2
no_workers=(4)
external_device="mmcblk0p2"
batch_size=16

python3 /home/pi5/kedar/Yolo/Yolo_Train.py "/home/pi5/kedar/Yolo/" "$no_workers" "$prefetch_factor" "$external_device" "$batch_size" > "train_log_file_pm_14"

mkdir Train_logs
mv mn_nw* Train_logs
mv log_file_* Train_logs