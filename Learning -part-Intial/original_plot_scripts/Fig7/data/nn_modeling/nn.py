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

def populate_data_energy(sampled_powermodes, path, filename, offset_dict):
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
        # temp_df = temp_df[temp_df[4].apply(is_numeric_value)]

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
            print("Length :",len(temp_df[start:end]))

        temp_df = temp_df.iloc[start:end]
        temp_df = temp_df.drop(temp_df.columns[:-1], axis=1)
        temp_df['Cores'] = cores
        temp_df['CPU_frequency'] = cpu
        temp_df['GPU_frequency'] = gpu
        temp_df['Memory_frequency'] = mem
        temp_df.columns = ['Minibatch_time', 'Cores', 'CPU_frequency', 'GPU_frequency', 'Memory_frequency']

        all_data.append(temp_df)

    master_df = pd.concat(all_data, ignore_index=True)
    return master_df
def extract_offsets_from_csv(csv_file):
    df = pd.read_csv(csv_file) 
    powermode_offsets = {}
    # Iterate through the DataFrame and populate the dictionary
    for _, row in df.iterrows():
        powermode = f"{int(row['cores'])}_{int(row['cpu'])}_{int(row['gpu'])}_{int(row['mem'])}"
        offset = int(row['skip_index'])
        powermode_offsets[powermode] = offset

    return powermode_offsets


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

def NN_run_time(all_powermodes, sampled_powermodes, sample_count_list, runs, path, filename, offsets):

    list_stats=[]

    if not os.path.exists("val.csv"):
        val_time = time.time()
        val_data = []
        for powermode in all_powermodes:
            val_data.append(populate_data_time([powermode], path, filename, offsets))
        df = pd.concat(val_data, ignore_index=True)
        df.to_csv('val.csv', index=False)

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
                data_collection_time=calculate_data_collection_time(sampled_powermode,path,filename,offsets)
                # val_powermodes=list(set(all_powermodes) - set(sampled_powermodes))#remaining powermodes for validation = 3276 powermodes

                train_data = []

                for powermode in sampled_powermode:
                    train_data.append(populate_data_time([powermode], path, filename, offsets))
                df = pd.concat(train_data, ignore_index=True)
                df.to_csv('train.csv', index=False)
                print("Train Dataload time: ",time.time() - dl_start)

                t_time = time.time()
                df = pd.read_csv("train.csv")
                x = df.drop('Minibatch_time', axis=1)
                y = df['Minibatch_time']

                # 20% as test set
                trainX, testX, trainY, testY = train_test_split(x, y, test_size = 0.2, random_state=31)

                # Standardize features by removing the mean and scaling to unit variance
                sc = StandardScaler()
                scaler = sc.fit(trainX)
                trainX_scaled = scaler.transform(trainX)
                testX_scaled = scaler.transform(testX)

                model = neural_network(trainX_scaled.shape[1])
                history = model.fit(trainX_scaled, trainY, epochs=100, batch_size=32, validation_data=(testX_scaled, testY), verbose=0)
                # plot_loss_curve(history)

                y_pred = model.predict(testX_scaled).flatten()
                print("Training Time :",time.time() - t_time)

                # validation data
                val_time = time.time()
                df = pd.read_csv('val.csv')
                x_val = df.drop('Minibatch_time', axis=1)
                y_val = df['Minibatch_time']
                valX_scaled = scaler.transform(x_val)

                # validation
                y_pred_val = model.predict(valX_scaled).flatten()

                list_stats.append([sample_count, run, data_collection_time, metrics.mean_absolute_percentage_error(y_val, y_pred_val), sampled_powermode])
                print('Mean Absolute Percentage Error:', metrics.mean_absolute_percentage_error(y_val, y_pred_val))
              
    if os.path.exists("train.csv"):
        os.remove("train.csv")

    df=pd.DataFrame(list_stats, columns = ['No_powermodes', 'RunNo','Data_collection_time', 'Val_MAPE','Sampled_powermodes'])
    df.to_csv("NN_loop.csv",index=False)

def NN_run_energy(all_powermodes,sampled_powermodes, sample_count_list, runs, path, eg_filename, filename, offsets):

    dc_filename = filename
    filename = eg_filename
    
    list_stats=[]

    
    if not os.path.exists("val_energy.csv"):
        val_time = time.time()
        val_data = []
        for powermode in all_powermodes:
            val_data.append(populate_data_energy([powermode], path, filename, offsets))
        df = pd.concat(val_data, ignore_index=True)
        df.to_csv('val_energy.csv', index=False)
    count = 0
    for sample_count in sample_count_list:
        print(sample_count)
        for run in range(runs):   
                print("Run number :",run)
                dl_start = time.time()
                sampled_powermode = sampled_powermodes[count]
                count+=1
                # sampled_powermodes=random.sample(all_powermodes, sample_count) #sample 25% of 4368 powermodes = 1092 powermodes for training and testing
                data_collection_time=calculate_data_collection_time(sampled_powermode,path,dc_filename,offsets)
                # val_powermodes=list(set(all_powermodes) - set(sampled_powermodes))#remaining powermodes for validation = 3276 powermodes

                train_data = []

                for powermode in sampled_powermode:
                    train_data.append(populate_data_energy([powermode], path, filename, offsets))
                df = pd.concat(train_data, ignore_index=True)
                df.to_csv('train_energy.csv', index=False)
                print("Train Dataload time: ",time.time() - dl_start)

                t_time = time.time()
                df = pd.read_csv("train_energy.csv")
                x = df.drop('Minibatch_time', axis=1)
                y = df['Minibatch_time']

                # 20% as test set
                trainX, testX, trainY, testY = train_test_split(x, y, test_size = 0.2, random_state=31)

                # Standardize features by removing the mean and scaling to unit variance
                sc = StandardScaler()
                scaler = sc.fit(trainX)
                trainX_scaled = scaler.transform(trainX)
                testX_scaled = scaler.transform(testX)

                model = neural_network(trainX_scaled.shape[1])
                history = model.fit(trainX_scaled, trainY, epochs=100, batch_size=32, validation_data=(testX_scaled, testY), verbose=0)
                # plot_loss_curve(history)

                y_pred = model.predict(testX_scaled).flatten()
                print("Training Time :",time.time() - t_time)

                # validation data
                val_time = time.time()
                df = pd.read_csv('val_energy.csv')
                x_val = df.drop('Minibatch_time', axis=1)
                y_val = df['Minibatch_time']
                valX_scaled = scaler.transform(x_val)

                # validation
                y_pred_val = model.predict(valX_scaled).flatten()

                list_stats.append([sample_count, run, data_collection_time, metrics.mean_absolute_percentage_error(y_val, y_pred_val), sampled_powermode])
                print('Mean Absolute Percentage Error:', metrics.mean_absolute_percentage_error(y_val, y_pred_val))
    
    if os.path.exists("train_energy.csv"):
        os.remove("train_energy.csv")

    df=pd.DataFrame(list_stats, columns = ['No_powermodes', 'RunNo','Data_collection_time', 'Val_MAPE','Sampled_powermodes'])
    df.to_csv("NN_loop_energy.csv",index=False)

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
    ax = sns.violinplot(x="No_powermodes", y="Val_MAPE_time", data=df, palette="muted")
    ax.set_xlabel("No of sampled powermodes")
    ax.set_ylabel("MAPE_Time")
    ax.set_ylim(0,1)
    ax.grid(which='major', linestyle='-', linewidth='0.5', color='black')
    ax.grid(which='minor', linestyle='--', linewidth='0.5', color='gray')
    ax.minorticks_on()
    ax.set_title("NN: MAPE_Time vs No of sampled powermodes")
    plt.show()

    #x axis is no of sampled powermodes
    #y axis is MAPE
    #violin plot for each sample count
    fig, ax = plt.subplots(figsize=(10, 6))
    ax = sns.violinplot(x="No_powermodes", y="Val_MAPE_energy", data=df, palette="muted")
    ax.set_xlabel("No of sampled powermodes")
    ax.set_ylabel("MAPE_energy")
    ax.set_ylim(0,1)
    ax.grid(which='major', linestyle='-', linewidth='0.5', color='black')
    ax.grid(which='minor', linestyle='--', linewidth='0.5', color='gray')
    ax.minorticks_on()
    ax.set_title("NN: MAPE_energy vs No of sampled powermodes")
    plt.show()

    #violin plot for data collection time for each sample count
    fig, ax = plt.subplots(figsize=(10, 6))
    ax = sns.violinplot(x="No_powermodes", y="Data_collection_time", data=df, palette="muted")
    ax.set_xlabel("No of sampled powermodes")
    ax.set_ylabel("Data collection time (s)")
    ax.set_ylim(0,5000)
    ax.grid(which='major', linestyle='-', linewidth='0.5', color='black')
    ax.grid(which='minor', linestyle='--', linewidth='0.5', color='gray')
    ax.minorticks_on()
    ax.set_title("NN: Data collection time vs No of sampled powermodes")
    plt.show()

def cleanup(name):
    df_time = pd.read_csv("NN_loop.csv")
    df_energy = pd.read_csv("NN_loop_energy.csv")

    df_time = df_time.rename(columns={"Val_MAPE": "Val_MAPE_time"})
    df_energy = df_energy.rename(columns={"Val_MAPE": "Val_MAPE_energy"})

    merged_df = pd.merge(df_time, df_energy, on=['No_powermodes', 'RunNo', 'Data_collection_time', 'Sampled_powermodes'], how='inner')

    # Drop any duplicate columns if present
    merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]

    columns_order = ['No_powermodes', 'RunNo', 'Data_collection_time', 'Val_MAPE_time', 'Val_MAPE_energy', 'Sampled_powermodes']
    merged_df = merged_df[columns_order]

    # Write to a new CSV file
    merged_df.to_csv(name, index=False)



path="resnet_v3_runs_merged"
filename="mn_nw4_pf2_epoch_stats.csv"
eg_filename="mn_nw4_pf2_energy.csv"
offset_file = "minibatch_index_list_resnet_new.csv"
offsets = extract_offsets_from_csv(offset_file)
sample_count_list=[200,100,50,20]
no_runs=10

all_powermodes = generate_powermodes()
sampled_pm_list = []
for i in sample_count_list:
    print(i)
    for j in range(no_runs):
        sampled_pwd=random.sample(all_powermodes, i)
        sampled_pm_list.append(sampled_pwd)
        # print("Length of sampled powermode :",len(sampled_pwd))
# print("Length of sampled list :",len(sampled_pm_list))
# for i in sampled_pm_list:
    # print(len(i))
model = path.split("_")[0]
name = "train_data.csv"
folder_name = model+"_"+str(sample_count_list[0])+"_"+str(no_runs)+"_MAPE_data"

NN_run_time(all_powermodes, sampled_pm_list, sample_count_list, no_runs, path, filename, offsets)
NN_run_energy(all_powermodes, sampled_pm_list, sample_count_list, no_runs, path, eg_filename, filename, offsets)
cleanup(name)
plot_MAPE_dc_violins(name)
