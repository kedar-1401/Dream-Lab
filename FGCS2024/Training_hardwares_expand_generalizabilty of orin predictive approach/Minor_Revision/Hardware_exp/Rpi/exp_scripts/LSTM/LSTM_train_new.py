from curses import noecho
import torch
import torch.nn as nn
import torch.optim as optim
#from sklearn.metrics import top_k_accuracy_score
import math
from transformers import BertTokenizer, BertForQuestionAnswering
import json
from tqdm import tqdm
import time
import os
import logging
formatter = logging.Formatter('%(message)s')
# from jtop import jtop
import sys
import csv
import multiprocessing
from torch.utils.data import DataLoader, TensorDataset, RandomSampler
from torchvision import transforms
import pandas as pd

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import json
def setup_logger(name, log_file, level=logging.INFO):
    """To setup as many loggers as you want"""
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger

def main(dataset_outer_folder, no_workers, pf_factor, epoch_fname, fetch_fname, compute_fname, vmtouch_fname, reference_time):

    class LSTM_B_Model(nn.Module):
        def __init__(
            self,
            vocab_size,
            embedding_dim,
            hidden_dim,
            num_layers,
            dropout_rate,
            tie_weights,
        ):
            super().__init__()
            self.num_layers = num_layers
            self.hidden_dim = hidden_dim
            self.embedding_dim = embedding_dim

            self.embedding = nn.Embedding(vocab_size, embedding_dim)
            self.lstm = nn.LSTM(
                embedding_dim,
                hidden_dim,
                num_layers=num_layers,
                dropout=dropout_rate,
                batch_first=True,
            )
            self.dropout = nn.Dropout(dropout_rate)
            self.fc = nn.Linear(hidden_dim, vocab_size)

            if tie_weights:
                assert embedding_dim == hidden_dim, "cannot tie, check dims"
                self.embedding.weight = self.fc.weight
            self.init_weights()

        def forward(self, src, hidden):
            embedding = self.dropout(self.embedding(src))
            output, hidden = self.lstm(embedding, hidden)
            output = self.dropout(output)
            prediction = self.fc(output)
            return prediction, hidden

        def init_weights(self):
            init_range_emb = 0.1
            init_range_other = 1 / math.sqrt(self.hidden_dim)
            self.embedding.weight.data.uniform_(-init_range_emb, init_range_emb)
            self.fc.weight.data.uniform_(-init_range_other, init_range_other)
            self.fc.bias.data.zero_()
            for i in range(self.num_layers):
                self.lstm.all_weights[i][0] = torch.FloatTensor(
                    self.embedding_dim, self.hidden_dim
                ).uniform_(-init_range_other, init_range_other)
                self.lstm.all_weights[i][1] = torch.FloatTensor(
                    self.hidden_dim, self.hidden_dim
                ).uniform_(-init_range_other, init_range_other)

        def init_hidden(self, batch_size, device):
            hidden = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
            cell = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
            return hidden, cell

        def detach_hidden(self, hidden):
            hidden, cell = hidden
            hidden = hidden.detach()
            cell = cell.detach()
            return hidden, cell   

    class CustomDataLoader:
        def __init__(self) -> None:
            pass

        def get_train_test_dataset_loaders(
            self, batch_size=1, dataset_path=None, args: dict = None
        ):
            train_data = torch.load(
                    "WIKITEXT-29423-16/partition_0/train_partition_0.pth"
            )
            
            return train_data#, test_data
           
    train_data = CustomDataLoader().get_train_test_dataset_loaders(
        args={"num_samples": 798}
    )

    vocab_size = 29423
    embedding_dim = 256 # 400 in the paper
    hidden_dim = 256  # 1150 in the paper
    num_layers = 2  # 3 in the paper
    dropout_rate = 0.2
    tie_weights = True
    seq_length = 8
    batch_size = 16
    clip = 0.5
    lr = 5e-5

    model = LSTM_B_Model(vocab_size, embedding_dim, hidden_dim, num_layers, dropout_rate, tie_weights)
    model.to(DEVICE)
    train(model, train_data,epoch_fname, fetch_fname, compute_fname, vmtouch_fname, dataset_outer_folder, reference_time, seq_length, batch_size, clip, lr, epochs=1)

def train(net, dataloader, epoch_fname, fetch_fname, compute_fname, vmtouch_fname, dataset_outer_folder, reference_time, seq_length, batch_size, clip, lr, epochs):

    def get_batch(data, seq_len, idx):
        src = data[:, idx : idx + seq_len]
        target = data[:, idx + 1 : idx + seq_len + 1]
        return src, target

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)
    num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    
    num_batches = dataloader.shape[-1]
    data = dataloader[:, : num_batches - (num_batches - 1) % seq_length]
    num_batches = data.shape[-1]
    net.train()
    vmtouch_dir = dataset_outer_folder

    e2e_first_batch = 0
    dataloader_start_time = time.time()
    stabilization_time = 20

    for e in tqdm(range(epochs), desc="Training", leave=False):        
        
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
        
        print("Start time :", start_time)
        batch_count = 0    
        for idx in tqdm(range(0, num_batches - 1, seq_length), desc="Mini-batch", leave=False):   
            if batch_count == 0:
                pass
            else:
                if batch_count == 1:
                    batch_start_time = time.time()                    
                if batch_start_time is not None and time.time() - batch_start_time > stabilization_time and batch_count > 999999999:
                    break
            batch_count += 1
            # print('Batch index = '+ str(batch_idx))
            hidden = net.init_hidden(batch_size, DEVICE)
            optimizer.zero_grad()
            hidden = net.detach_hidden(hidden)
            src, target = get_batch(data, seq_length, idx)
            src, target = src.to(DEVICE), target.to(DEVICE)
            
            
            if torch.cuda.is_available():
                end2.record()
                start3.record()
            else:
                end2 = start3 = time.time()
            
            batch_size = src.shape[0]

            prediction, hidden = net(src, hidden)
            prediction = prediction.reshape(batch_size * seq_length, -1)
            target = target.reshape(-1)
            loss = criterion(prediction, target)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), clip)
            optimizer.step()

            # epoch_loss += loss.item() * seq_len
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

            
            fetch_fname.info(f"{e},{idx},null,{fetch_time},{time.time() - reference_time}")
            compute_fname.info(f"{e},{idx},null,{compute_time},{time.time() - reference_time}")
            epoch_fname.info(f"{e},null,: ,null,{epoch_time},{time.time() - reference_time}")

            if torch.cuda.is_available():
                start1.record()
                start2.record()
            else:
                start1 = start2 = time.time()



if __name__ == "__main__":

    dataset_outer_folder=sys.argv[1]
    num_workers=int(sys.argv[2])
    prefetch_factor=int(sys.argv[3])
    external_device = sys.argv[4]
    batch_size = sys.argv[5]
    
    file_prefix='mn_'+'nw'+ str(num_workers) + '_pf'+str(prefetch_factor)

    logger_fetch = setup_logger("logger_fetch", file_prefix + "_fetch.csv")
    logger_compute = setup_logger("logger_compute", file_prefix + "_compute.csv")
    logger_e2e = setup_logger("logger_e2e", file_prefix + "_epoch_stats.csv")

    
    logger_fetch.info('epoch,batch_idx,fetchtime,fetchtime_ms,log_time')
    logger_compute.info('epoch,batch_idx,computetime,computetime_ms,log_time')
    logger_e2e.info('epoch,time,loss,accuracy,epochtime_ms,log_time')


    reference_time = time.time()

    try:
        main(dataset_outer_folder, num_workers, prefetch_factor, logger_e2e, logger_fetch, logger_compute, None, reference_time)
    except():
        print("hit an exception")
    