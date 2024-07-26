import os
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import seaborn as sns
import random
import csv


def file_exists(filepath):
    return os.path.isfile(filepath)

def is_numeric_value(val):
    try:
        float(val)
        return True
    except ValueError:
        return False

def populate_data_time(sampled_powermodes, path, filename, offset_dict):
    cpu_cores_multipler = 1
    cpu_frq_divider = 1
    gpu_frq_divider = 1
    mem_frq_divider = 1

    all_data = []

    for powermode in sampled_powermodes:
        # Get the offset before scaling
        offset = offset_dict.get(powermode, -1)
        file = path + "/" + "pm_" + powermode + "/" + filename
        cores = int(powermode.split("_")[0]) * cpu_cores_multipler
        cpu = int(powermode.split("_")[1]) / cpu_frq_divider
        gpu = int(powermode.split("_")[2]) / gpu_frq_divider
        mem = int(powermode.split("_")[3]) / mem_frq_divider

        temp_df = pd.read_csv(file, header=None)
        # temp_df = temp_df[temp_df[4].replace('.', '', 1).astype(str).str.isnumeric()]
        temp_df = temp_df[temp_df[4].apply(is_numeric_value)]
        # print(temp_df.head())

        if offset == -1:
            start = len(temp_df) - 40
            end = len(temp_df)
        elif offset == 0:
            start = offset + 1
            end = start + 40
        else:
            start = offset
            end = start + 40

        diff = end-start
        if len(temp_df[start:end]) != 40:
            print("Diff :", diff)
            print("Start :",start)
            print("End:",end)
            print("Offset: ",offset)
            print("Powermode :",powermode)
            print("Length :",len(temp_df))

        temp_df = temp_df.iloc[start:end]
        temp_df = temp_df.iloc[:,4]
        temp_df = temp_df.to_frame() 
        # print(temp_df.head())
        temp_df['Cores'] = cores
        temp_df['CPU_frequency'] = cpu
        temp_df['GPU_frequency'] = gpu
        temp_df['Memory_frequency'] = mem
        temp_df.columns = ['Minibatch_time', 'Cores', 'CPU_frequency', 'GPU_frequency', 'Memory_frequency']
        # print(temp_df.head())

        all_data.append(temp_df)

    master_df = pd.concat(all_data, ignore_index=True)
    return master_df

def populate_data_power(sampled_powermodes, path, filename, tg_filename, offset_dict, start_dict, end_dict):

    rows = []
    for powermode in sampled_powermodes:
        offset = offset_dict.get(powermode, -1)
        start_time = start_dict.get(powermode, -1)
        end_time = end_dict.get(powermode, -1)
        file = path + "/" + "pm_" + powermode + "/" + filename
        tg_file = path + "/" + "pm_" + powermode + "/" + tg_filename

        tg_df = pd.read_csv(tg_file)
        filtered_df = tg_df[(tg_df['log_time'] >= start_time) & (tg_df['log_time'] <= end_time)]
        power_list = filtered_df['power cur'].astype(float).dropna().tolist()

        required_length = 12
        if len(power_list) < 12:
            repeats_required = -(-required_length // len(power_list))  
            power_list = (power_list * repeats_required)[:required_length]
        power_list = power_list[:12]  # If more than 12, truncate to the first 12 samples

        # Split the powermode into its components
        cores, cpu, gpu, mem = powermode.split("_")

        for sample in power_list:
            rows.append({
                'cores': cores,
                'cpu': cpu,
                'gpu': gpu,
                'mem': mem,
                'power_sample': sample
            })

    minibatch_power_df = pd.DataFrame(rows)
    # minibatch_power_df.to_csv("val_power.csv", index=False)
    return minibatch_power_df


def extract_offsets_from_csv(csv_file):
    df = pd.read_csv(csv_file) 
    powermode_offsets = {}
    start_times = {}
    end_times = {}
    # Iterate through the DataFrame and populate the dictionary
    for _, row in df.iterrows():
        powermode = f"{int(row['cores'])}_{int(row['cpu'])}_{int(row['gpu'])}_{int(row['mem'])}"
        offset = int(row['skip_index'])
        start_time = float(row['start_time'])
        end_time = float(row['end_time'])
        powermode_offsets[powermode] = offset
        start_times[powermode] = start_time
        end_times[powermode] = end_time
    return powermode_offsets, start_times, end_times

def generate_powermodes():
    cpu_core_vals=[2, 4, 6, 8, 10, 12] #6 possible values
    gpu_frequencies=[1300500000, 1236750000, 1134750000, 1032750000, 930750000, 828750000, 726750000, 624750000, 522750000, 420750000, 318750000, 216750000, 114750000] #in Hz, 13 possible values
    cpu_frequencies=[268800, 422400, 576000, 729600, 883200, 1036800, 1190400, 1344000, 1497600, 1651200, 1804800, 1958400, 2112000, 2201600] #in kHz, 14 possible values
    mem_frequencies=[204000000, 2133000000, 3199000000, 665600000] #in Hz, 4 possible values
    #get combinations of all 4 as powermode Ex.2_1300500000_268800_204000000
    all_powermodes=[] #6*13*14*4=4368 possible values
    for cpu_core in cpu_core_vals:
        for cpu_frequency in cpu_frequencies:
            for gpu_frequency in gpu_frequencies:
                for mem_frequency in mem_frequencies:
                    all_powermodes.append(str(cpu_core)+"_"+str(cpu_frequency)+"_"+str(gpu_frequency)+"_"+str(mem_frequency))
    return all_powermodes

def neural_network(input_dim):
    model = Sequential()
    model.add(Dense(256, input_dim=input_dim, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='linear'))

    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

def NN_run_time(all_powermodes, path, filename, offsets):

    if not os.path.exists("val.csv"):
        val_time = time.time()
        val_data = []
        for powermode in all_powermodes:
            val_data.append(populate_data_time([powermode], path, filename, offsets))
        df = pd.concat(val_data, ignore_index=True)
        df.to_csv('val.csv', index=False)


    t_time = time.time()
    df = pd.read_csv("val.csv")
    x = df.drop('Minibatch_time', axis=1)
    y = df['Minibatch_time']

    # 80:10:10
    trainX, tempX, trainY, tempY = train_test_split(x, y, test_size=0.2)
    valX, testX, valY, testY = train_test_split(tempX, tempY, test_size=0.5)
    # trainX, testX, trainY, testY = train_test_split(x, y, test_size = 0.05) 
    # Standardize features by removing the mean and scaling to unit variance
    sc = StandardScaler()
    scaler = sc.fit(trainX)
    trainX_scaled = scaler.transform(trainX)
    valX_scaled = scaler.transform(valX)
    testX_scaled = scaler.transform(testX)

    # Creating and training the model
    model = neural_network(trainX_scaled.shape[1])
    checkpoint = ModelCheckpoint('NN_time.h5', save_best_only=True, monitor='val_loss', mode='min', verbose=1)
    t_time = time.time()
    history = model.fit(
        trainX_scaled, trainY, 
        epochs=100, 
        batch_size=32, 
        validation_data=(valX_scaled, valY), 
        verbose=1, 
        callbacks=[checkpoint]
    )

    # Evaluating the model
    y_pred_val = model.predict(valX_scaled).flatten()
    y_pred_test = model.predict(testX_scaled).flatten()

    print("Training Time:", time.time() - t_time)
    print('Validation Mean Absolute Percentage Error:', metrics.mean_absolute_percentage_error(y_pred_val, valY))
    print('Test Mean Absolute Percentage Error:', metrics.mean_absolute_percentage_error(y_pred_test, testY))

def plot_loss_curve(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.show()

def NN_run_power(all_powermodes, path, filename, tg_filename, offsets, start_times, end_times):

    #Need to run power_modelling_violin.py to generate val_power.csv first.
    if not os.path.exists("val_power.csv"):
        val_data = []
        for powermode in all_powermodes:
            val_data.append(populate_data_power([powermode], path, filename, tg_filename, offsets, start_times, end_times))
        df = pd.concat(val_data, ignore_index=True)
        df.to_csv('val_power.csv', index=False)

    t_time = time.time()
    df = pd.read_csv("val_power.csv")
    x = df.drop('power_sample', axis=1)
    y = df['power_sample']

    # 80:10:10
    trainX, tempX, trainY, tempY = train_test_split(x, y, test_size=0.2)
    valX, testX, valY, testY = train_test_split(tempX, tempY, test_size=0.5)
    # trainX, testX, trainY, testY = train_test_split(x, y, test_size = 0.05) 
    # Standardize features by removing the mean and scaling to unit variance
    sc = StandardScaler()
    scaler = sc.fit(trainX)
    trainX_scaled = scaler.transform(trainX)
    valX_scaled = scaler.transform(valX)
    testX_scaled = scaler.transform(testX)

    # Creating and training the model
    model = neural_network(trainX_scaled.shape[1])
    checkpoint = ModelCheckpoint('NN_power.h5', save_best_only=True, monitor='val_loss', mode='min', verbose=1)
    t_time = time.time()
    history = model.fit(
        trainX_scaled, trainY, 
        epochs=100, 
        batch_size=32, 
        validation_data=(valX_scaled, valY), 
        verbose=1, 
        callbacks=[checkpoint]
    )
    plot_loss_curve(history)
    # Evaluating the model
    y_pred_val = model.predict(valX_scaled).flatten()
    y_pred_test = model.predict(testX_scaled).flatten()

    print("Training Time:", time.time() - t_time)
    print('Validation Mean Absolute Percentage Error:', metrics.mean_absolute_percentage_error(y_pred_val, valY))
    print('Test Mean Absolute Percentage Error:', metrics.mean_absolute_percentage_error(y_pred_test, testY))

path="resnet_v3_runs_merged"
filename="mn_nw4_pf2_epoch_stats.csv"
tg_filename="mn_nw4_pf2_tegrastats.csv"
offset_file = "minibatch_index_list_resnet_new.csv"
offsets, start_times, end_times = extract_offsets_from_csv(offset_file)

all_powermodes = generate_powermodes()
# NN_run_time(all_powermodes, path, filename, offsets)
NN_run_power(all_powermodes, path, filename, tg_filename, offsets, start_times, end_times)