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


def populate_data(sampled_powermodes, path, filename, tg_filename, offset_dict, start_dict, end_dict):

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

def calculate_data_collection_time(sampled_powermodes, path, filename, offset_dict):
    pm_count = 0
    total_time = 0  # across all sampled power modes

    for powermode in sampled_powermodes:
        pm_count += 1
        file = path + "/" + "pm_" + powermode + "/" + filename
        # print("File path: ",file)
        
        # Get the offset before scaling
        offset = offset_dict.get(powermode, -1)

        # Store rows temporarily
        rows = []
        with open(file) as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            
            for row in reader:
                if is_numeric_value(row[4]):  # Check if the value is numeric
                    rows.append(row)
                
            # Determine the start index based on the offset
            if offset == -1:
                start = len(rows) - 40
                end = len(rows)
            elif offset == 0:
                start = offset + 1
                end = start + 40
            else:
                start = offset
                end = start + 40

            
            # Iterate through rows of interest to calculate sum
            for i in range(start, end):
                total_time += float(rows[i][4])

    total_time = total_time / 1000  # convert to seconds
    return total_time

def neural_network(input_dim):
    model = Sequential()
    model.add(Dense(256, input_dim=input_dim, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='linear'))

    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model  # Only return the model

def NN_run_power(all_powermodes, sampled_powermodes, sample_count_list, runs, path, filename, tg_filename, offsets, start_times, end_times):

    list_stats=[]

    if not os.path.exists("val_power.csv"):
        val_data = []
        for powermode in all_powermodes:
            val_data.append(populate_data([powermode], path, filename, tg_filename, offsets, start_times, end_times))
        df = pd.concat(val_data, ignore_index=True)
        df.to_csv('val_power.csv', index=False)

    count = 0
    for sample_count in sample_count_list:
        print(sample_count)
        for run in range(runs):                
                print("Run number :",run)
                dl_start = time.time()
                sampled_powermode = sampled_powermodes[count]
                count+=1
                # print("Length of power mode: ",len(sampled_powermodes))
                # sampled_powermodes=random.sample(all_powermodes, sample_count) #sample 25% of 4368 powermodes = 1092 powermodes for training and testing
                # data_collection_time=calculate_data_collection_time(sampled_powermode,path,filename,offsets)
                # val_powermodes=list(set(all_powermodes) - set(sampled_powermodes))#remaining powermodes for validation = 3276 powermodes

                train_data = []
                for powermode in sampled_powermode:
                    train_data.append(populate_data([powermode], path, filename, tg_filename, offsets, start_times, end_times))
                df = pd.concat(train_data, ignore_index=True)
                df.to_csv('train.csv', index=False)
                print("Train Dataload time: ",time.time() - dl_start)

                t_time = time.time()
                df = pd.read_csv("train.csv")
                x = df.drop('power_sample', axis=1)
                y = df['power_sample']

                # 20% as test set
                trainX, testX, trainY, testY = train_test_split(x, y, test_size = 0.2, random_state=31)

                # Standardize features by removing the mean and scaling to unit variance
                sc = StandardScaler()
                scaler = sc.fit(trainX)
                trainX_scaled = scaler.transform(trainX)
                testX_scaled = scaler.transform(testX)

                model = neural_network(trainX_scaled.shape[1])
                history = model.fit(trainX_scaled, trainY, epochs=100, batch_size=32, validation_data=(testX_scaled, testY), verbose=1)
                # plot_loss_curve(history)

                # y_pred = model.predict(testX_scaled).flatten()
                print("Training Time :",time.time() - t_time)

                # validation data
                val_time = time.time()
                df = pd.read_csv('val_power.csv')
                x_val = df.drop('power_sample', axis=1)
                y_val = df['power_sample']
                valX_scaled = scaler.transform(x_val)

                # validation
                y_pred_val = model.predict(valX_scaled).flatten()

                list_stats.append([sample_count, run, metrics.mean_absolute_percentage_error(y_val, y_pred_val), sampled_powermode])
                print('Mean Absolute Percentage Error:', metrics.mean_absolute_percentage_error(y_val, y_pred_val))
              
    if os.path.exists("train.csv"):
        os.remove("train.csv")

    df=pd.DataFrame(list_stats, columns = ['No_powermodes', 'RunNo', 'Val_MAPE','Sampled_powermodes'])
    df.to_csv("NN_loop_power.csv",index=False)

def plot_loss_curve(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.show()

def plot_MAPE_dc_violins(name):
    df=pd.read_csv(name)
    #x axis is no of sampled powermodes
    #y axis is MAPE
    #violin plot for each sample count
    fig, ax = plt.subplots(figsize=(10, 6))
    ax = sns.violinplot(x="No_powermodes", y="Val_MAPE", data=df, palette="muted")
    ax.set_xlabel("No of sampled powermodes")
    ax.set_ylabel("MAPE_Power")
    ax.set_ylim(0,1)
    ax.grid(which='major', linestyle='-', linewidth='0.5', color='black')
    ax.grid(which='minor', linestyle='--', linewidth='0.5', color='gray')
    ax.minorticks_on()
    ax.set_title("NN: MAPE_Power vs No of sampled powermodes")
    plt.show()


path="mobnet_v3_runs_merged"
filename="mn_nw4_pf2_epoch_stats.csv"
tg_filename="mn_nw4_pf2_tegrastats.csv"
offset_file = "minibatch_index_list_mobilenet_new.csv"
offsets, start_times, end_times = extract_offsets_from_csv(offset_file)
sample_count_list=[200,100,50,20]
no_runs=10

all_powermodes = generate_powermodes()
sampled_pm_list = []
for i in sample_count_list:
    print(i)
    for j in range(no_runs):
        sampled_pwd=random.sample(all_powermodes, i)
        sampled_pm_list.append(sampled_pwd)

model = path.split("_")[0]
name = "NN_loop_power.csv"
NN_run_power(all_powermodes, sampled_pm_list, sample_count_list, no_runs, path, filename, tg_filename, offsets, start_times, end_times)
plot_MAPE_dc_violins(name)
