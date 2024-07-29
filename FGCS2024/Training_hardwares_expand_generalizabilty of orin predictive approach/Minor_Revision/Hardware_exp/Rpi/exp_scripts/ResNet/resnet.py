import torch
from PIL import Image
from torchvision.datasets import ImageNet, ImageFolder
import time
from torchvision import transforms
import multiprocessing
import logging
import sys
import csv
import pandas as pd
import os
import numpy as np
from torchvision import models

formatter = logging.Formatter('%(message)s')
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def setup_logger(name, log_file, level=logging.INFO):
    """To setup as many loggers as you want"""
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger


def train(model, train_loader, epoch_fname, fetch_fname, compute_fname, vmtouch_fname, dataset_outer_folder, reference_time, epochs):
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    start_time = time.time()
    epoch_count = 0

    img_idx = batch_size - 1
    e2e_first_batch = 0
    dataloader_start_time = time.time()
    stabilization_time = 15
    num_epochs = 1
    for epoch in range(num_epochs):
        print('Epoch: ' + str(epoch) + ' Begins')
        if torch.cuda.is_available():
            start1 = torch.cuda.Event(enable_timing=True)
            end1 = torch.cuda.Event(enable_timing=True)
            start2 = torch.cuda.Event(enable_timing=True)
            end2 = torch.cuda.Event(enable_timing=True)
            start3 = torch.cuda.Event(enable_timing=True)
            end3 = torch.cuda.Event(enable_timing=True)
            start1.record()
            start2.record()
        else:
            start1 = start2 = start3 = time.time()

        start_time = time.time()
        batch_count = 0
        for batch_idx, (images, labels) in enumerate(train_loader):
            print("Current time :", time.time())
            if time.time() - start_time > stabilization_time:
                batch_count += 1
            if batch_count == 999999999:
                break

            print('Batch index = ' + str(batch_idx))
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            if torch.cuda.is_available():
                end2.record()
                start3.record()
            else:
                end2 = start3 = time.time()

            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()

            if torch.cuda.is_available():
                end3.record()
                torch.cuda.synchronize()
                end1.record()
                torch.cuda.synchronize()
                fetch_time = start2.elapsed_time(end2)
                compute_time = start3.elapsed_time(end3)
                epoch_time = start1.elapsed_time(end1)
            else:
                end3 = end1 = time.time()
                fetch_time = end2 - start2
                compute_time = end3 - start3
                epoch_time = end1 - start1

            fetch_fname.info(f"{epoch},{batch_idx},null,{fetch_time},{time.time() - reference_time}")
            compute_fname.info(f"{epoch},{batch_idx},null,{compute_time},{time.time() - reference_time}")
            epoch_fname.info(f"{epoch},null,: ,null,{epoch_time},{time.time() - reference_time}")

            if torch.cuda.is_available():
                start1.record()
                start2.record()
            else:
                start1 = start2 = time.time()

            img_idx += batch_size
        print('Epoch: ' + str(epoch) + ' Ends')
        epoch_count += 1

def main(dataset_outer_folder, no_workers, pf_factor, epoch_fname, fetch_fname, compute_fname, vmtouch_fname, reference_time, batch_size):
    data_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    model = models.resnet18()
    train_data = ImageFolder(root=dataset_outer_folder + "/ILSVRC/val", transform=data_transforms)
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        num_workers=no_workers
    )
    print('Reference Time is', str(reference_time))
    print('No. of testing batches = ' + str(len(train_loader)))
    print('No of workers is ' + str(no_workers))
    print('Prefetch factor is ' + str(pf_factor))
    print('Batch size is ', str(batch_size))
    model.to(DEVICE)
    train(model, train_loader, epoch_fname, fetch_fname, compute_fname, vmtouch_fname, dataset_outer_folder, reference_time, epochs=3)

if __name__ == "__main__":
    dataset_outer_folder = "/media/ssd/"
    num_workers = int(sys.argv[2])
    prefetch_factor = int(sys.argv[3])
    external_device = sys.argv[4]
    batch_size = int(sys.argv[5])
    file_prefix = 'mn_' + 'nw' + str(num_workers) + '_pf' + str(prefetch_factor)

    logger_fetch = setup_logger("logger_fetch", file_prefix + "_fetch.csv")
    logger_compute = setup_logger("logger_compute", file_prefix + "_compute.csv")
    logger_e2e = setup_logger("logger_e2e", file_prefix + "_epoch_stats.csv")

    logger_fetch.info('epoch,batch_idx,fetchtime,fetchtime_ms,log_time')
    logger_compute.info('epoch,batch_idx,computetime,computetime_ms,log_time')
    logger_e2e.info('epoch,time,loss,accuracy,epochtime_ms,log_time')

    reference_time = time.time()
    try:
        main(dataset_outer_folder, num_workers, prefetch_factor, logger_e2e, logger_fetch, logger_compute, None, reference_time, batch_size)
    except Exception as e:
        print("hit an exception")
        print(e)
