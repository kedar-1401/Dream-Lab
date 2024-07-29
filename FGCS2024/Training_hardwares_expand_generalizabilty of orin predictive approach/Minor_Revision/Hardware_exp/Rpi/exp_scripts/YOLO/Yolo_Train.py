from sympy import Q
from ultralytics import YOLO
import torch
import torch.nn as nn
import torch.optim as optim
import time
import pandas as pd
import os
import logging
formatter = logging.Formatter('%(message)s')
# from jtop import jtop
import sys
import csv
import multiprocessing

os.environ['WANDB_DISABLED'] = 'true'

# Initialize batch count variable
batch_count = 0
DEVICE = str(torch.device("cpu"))

def setup_logger(name, log_file, level=logging.INFO):
    """To setup as many loggers as you want"""
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger

def train(batch_size,dataset_outer_folder, file_prefix):

    path = dataset_outer_folder+"coco25.yaml"
    # Initialize an empty list for each column
    start_times = []
    stop_times = []
    epochs = []

    current_epoch = {'value': -1}
    # Custom exception to stop training
    class StopTrainingException(Exception):
        pass

    def on_train_epoch_start(trainer):
        current_epoch['value'] += 1

    def on_train_batch_start(trainer):
        start_times.append(time.time())
        epochs.append(current_epoch['value'])

    def on_train_batch_end(trainer):
        global batch_count
        stop_times.append(time.time())
        batch_count += 1
        
        # Check if we've reached 50 batches
        if batch_count > 999999999:
            raise StopTrainingException("Training completed after 10 batches")

    model = YOLO('yolov8n.pt')
    model.add_callback("on_train_epoch_start", on_train_epoch_start)    
    model.add_callback("on_train_batch_start", on_train_batch_start)
    model.add_callback("on_train_batch_end", on_train_batch_end)

    try:
        model.train(data=path, epochs=1, imgsz=320, batch=batch_size, workers=0, val=False, save=False, amp=False, device=DEVICE, pretrained=False)
    except StopTrainingException as e:
        print(f"Training stopped: {e}")

    # Create a DataFrame using the lists and calculate Minibatch_time
    df = pd.DataFrame({'epoch': epochs,'Start_time': start_times, 'Stop_time': stop_times})
    df['epochtime_ms'] = (df['Stop_time'] - df['Start_time'])*1000
    df['log_time'] = df['Stop_time'] - reference_time
    print(df)
    df.to_csv(file_prefix+"_epoch_stats.csv" , index=False)

def main(dataset_outer_folder, num_workers, prefetch_factor, reference_time, batch_size, file_prefix):
    print('Reference Time is', str(reference_time))
    # print('No. of testing batches = ' + str(len(train_loader)))
    print('No of workers is ' + str(num_workers))
    print('Prefetch factor is ' + str(prefetch_factor))
    print('Batch size is ', str(batch_size))
    train(batch_size, dataset_outer_folder, file_prefix)

if __name__ == "__main__":
    dataset_outer_folder = sys.argv[1]
    num_workers = int(sys.argv[2])
    prefetch_factor = int(sys.argv[3])
    external_device = sys.argv[4]
    batch_size = 16
    pass

    file_prefix = 'mn_'+'nw' + str(num_workers) + '_pf'+str(prefetch_factor)

    logger_fetch = setup_logger("logger_fetch", file_prefix + "_fetch.csv")

    logger_compute = setup_logger("logger_compute", file_prefix + "_compute.csv")
    logger_e2e = setup_logger("logger_e2e", file_prefix + "_epoch_stats.csv")

    logger_fetch.info('epoch,batch_idx,fetchtime,fetchtime_ms,log_time')
    logger_compute.info('epoch,batch_idx,computetime,computetime_ms,log_time')
    logger_e2e.info('epoch,time,loss,accuracy,epochtime_ms,log_time')
    

    reference_time = time.time()
    try:
        main(dataset_outer_folder, num_workers, prefetch_factor, reference_time, batch_size, file_prefix)
    except ():
        print("hit an exception")
        print()

