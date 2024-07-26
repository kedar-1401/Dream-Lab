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
import ast

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
        # Changed from 4 to 5 for yolo
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

        #Resnet: 22
        #Mobnet: 40
        #Yolo: 61
        required_length = 22
        if len(power_list) < required_length:
            repeats_required = -(-required_length // len(power_list))  
            power_list = (power_list * repeats_required)[:required_length]

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
    #Removed 2 slowest freq for yolo runs
    # cpu_frequencies=[576000, 729600, 883200, 1036800, 1190400, 1344000, 1497600, 1651200, 1804800, 1958400, 2112000, 2201600] #in kHz, 14 possible values
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
    return model

def load_pretrained_model(input_dim, model_weights_path):
    model = neural_network(input_dim)
    model.load_weights(model_weights_path)    
    model.pop()
    
    return model

def transfer_learning_run_time(all_powermodes, sampled_powermodes, sample_count_list, runs, path, filename, offsets, model_weights_path):
    list_stats = []

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
            print("Run number:", run)
            dl_start = time.time()
            sampled_powermode = sampled_powermodes[count]
            count += 1
            data_collection_time = calculate_data_collection_time(sampled_powermode, path, filename, offsets)


            train_data = []

            for powermode in sampled_powermode:
                train_data.append(populate_data_time([powermode], path, filename, offsets))
            df = pd.concat(train_data, ignore_index=True)
            df.to_csv('train.csv', index=False)
            print("Train Data load time:", time.time() - dl_start)

            t_time = time.time()
            df = pd.read_csv("train.csv")
            x = df.drop('Minibatch_time', axis=1)
            y = df['Minibatch_time']

            # 20% as test set
            trainX, testX, trainY, testY = train_test_split(x, y, test_size=0.2, random_state=31)

            # Standardize features by removing the mean and scaling to unit variance
            sc = StandardScaler()
            scaler = sc.fit(trainX)
            trainX_scaled = scaler.transform(trainX)
            testX_scaled = scaler.transform(testX)


            best_local_weight = 'best_model.h5'
            checkpoint = ModelCheckpoint(best_local_weight, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
            # Load the pre-trained model and replace the output layer
            model = load_pretrained_model(trainX_scaled.shape[1], model_weights_path)

            # Add a new output layer for your specific problem
            model.add(Dense(1, activation='linear'))

            model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
            
            # Fine-tune the model with new data
            history = model.fit(trainX_scaled, trainY, epochs=100, batch_size=32, validation_data=(testX_scaled, testY), verbose=0,callbacks=[checkpoint])

            y_pred = model.predict(testX_scaled).flatten()
            train_time = time.time() - t_time

            # validation data
            val_time = time.time()
            model.load_weights(best_local_weight)

            df = pd.read_csv('val.csv')
            x_val = df.drop('Minibatch_time', axis=1)
            y_val = df['Minibatch_time']
            valX_scaled = scaler.transform(x_val)

            # validation
            y_pred_val = model.predict(valX_scaled).flatten()
            validation_time = time.time() - val_time
            list_stats.append([sample_count, run, data_collection_time,
                                   metrics.mean_absolute_percentage_error(y_val, y_pred_val),
                                   sampled_powermode, train_time, validation_time])

            print('Mean Absolute Percentage Error:', metrics.mean_absolute_percentage_error(y_val, y_pred_val))

    if os.path.exists("train.csv"):
        os.remove("train.csv")

    df = pd.DataFrame(list_stats, columns=['No_powermodes', 'RunNo', 'Val_MAPE', 'Sampled_powermodes'])
    df.to_csv("NN_loop_transfer.csv", index=False)


def transfer_learning_run_power(all_powermodes, sampled_powermodes, sample_count_list, runs, path, filename, tg_filename, offsets, start_times, end_times, model_weights_path):
    
    list_stats=[]

    if not os.path.exists("val_power.csv"):
        val_data = []
        for powermode in all_powermodes:
            val_data.append(populate_data_power([powermode], path, filename, tg_filename, offsets, start_times, end_times))
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
                data_collection_time = calculate_data_collection_time(sampled_powermode, path, filename, offsets)

                train_data = []
                for powermode in sampled_powermode:
                    train_data.append(populate_data_power([powermode], path, filename, tg_filename, offsets, start_times, end_times))
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

                best_local_weight = 'best_model.h5'
                checkpoint = ModelCheckpoint(best_local_weight, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

                model = load_pretrained_model(trainX_scaled.shape[1], model_weights_path)
                model.add(Dense(1, activation='linear'))
                model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

                history = model.fit(trainX_scaled, trainY, epochs=100, batch_size=32, validation_data=(testX_scaled, testY), verbose=0,callbacks=[checkpoint])
                # plot_loss_curve(history)

                # y_pred = model.predict(testX_scaled).flatten()
                train_time = time.time() - t_time

                val_time = time.time()
                model.load_weights(best_local_weight)

                # validation data
                
                df = pd.read_csv('val_power.csv')
                x_val = df.drop('power_sample', axis=1)
                y_val = df['power_sample']
                valX_scaled = scaler.transform(x_val)
                
                # validation
                y_pred_val = model.predict(valX_scaled).flatten()
                validation_time = time.time() - val_time
                list_stats.append([sample_count, run, data_collection_time, metrics.mean_absolute_percentage_error(y_val, y_pred_val), sampled_powermode, train_time, validation_time])
                
                print('Mean Absolute Percentage Error:', metrics.mean_absolute_percentage_error(y_val, y_pred_val))
              
    if os.path.exists("train.csv"):
        os.remove("train.csv")

    df=pd.DataFrame(list_stats, columns = ['No_powermodes', 'RunNo', 'Val_MAPE','Sampled_powermodes'])
    df.to_csv("NN_loop_transfer_power.csv",index=False)

def cleanup(name):
    df_time = pd.read_csv("NN_loop_transfer.csv")
    df_energy = pd.read_csv("NN_loop_transfer_power.csv")

    df_time = df_time.rename(columns={"Val_MAPE": "Val_MAPE_time"})
    df_energy = df_energy.rename(columns={"Val_MAPE": "Val_MAPE_power"})

    merged_df = pd.merge(df_time, df_energy, on=['No_powermodes', 'RunNo', 'Data_collection_time', 'Sampled_powermodes'], how='inner')

    # Drop any duplicate columns if present
    merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]

    columns_order = ['No_powermodes', 'RunNo', 'Data_collection_time', 'Val_MAPE_time', 'Val_MAPE_power', 'Train_time_time', 'Validation_time_time', 'Train_time_power', 'Validation_time_power','Sampled_powermodes']
    merged_df = merged_df[columns_order]

    # Write to a new CSV file
    merged_df.to_csv(name, index=False)
    # os.remove("MLP_loop.csv")
    # os.remove("MLP_loop_energy.csv")


def plot_MAPE_dc_violins(name):
    df=pd.read_csv(name)
    #x axis is no of sampled powermodes
    #y axis is MAPE
    #violin plot for each sample count
    fig, ax = plt.subplots(figsize=(10, 6))
    ax = sns.violinplot(x="No_powermodes", y="Val_MAPE", data=df, palette="muted")
    ax.set_xlabel("No of sampled powermodes")
    ax.set_ylabel("MAPE_Time")
    ax.set_ylim(0,1)
    ax.grid(which='major', linestyle='-', linewidth='0.5', color='black')
    ax.grid(which='minor', linestyle='--', linewidth='0.5', color='gray')
    ax.minorticks_on()
    ax.set_title("NN: MAPE_Time vs No of sampled powermodes")
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
    ax.set_title("MLP: MAPE_Time vs No of sampled powermodes")
    plt.show()

    #x axis is no of sampled powermodes
    #y axis is MAPE
    #violin plot for each sample count
    fig, ax = plt.subplots(figsize=(10, 6))
    ax = sns.violinplot(x="No_powermodes", y="Val_MAPE_power", data=df, palette="muted")
    ax.set_xlabel("No of sampled powermodes")
    ax.set_ylabel("MAPE_power")
    ax.set_ylim(0,1)
    ax.grid(which='major', linestyle='-', linewidth='0.5', color='black')
    ax.grid(which='minor', linestyle='--', linewidth='0.5', color='gray')
    ax.minorticks_on()
    ax.set_title("MLP: MAPE_power vs No of sampled powermodes")
    plt.show()

    #violin plot for data collection time for each sample count
    # fig, ax = plt.subplots(figsize=(10, 6))
    # ax = sns.violinplot(x="No_powermodes", y="Data_collection_time", data=df, palette="muted")
    # ax.set_xlabel("No of sampled powermodes")
    # ax.set_ylabel("Data collection time (s)")
    # ax.set_ylim(0,4000)
    # ax.grid(which='major', linestyle='-', linewidth='0.5', color='black')
    # ax.grid(which='minor', linestyle='--', linewidth='0.5', color='gray')
    # ax.minorticks_on()
    # ax.set_title("MLP: Data collection time vs No of sampled powermodes")
    # plt.show()



# path="yolo_runs"
path="resnet_v3_runs_merged"
filename="mn_nw4_pf2_epoch_stats.csv"
tg_filename="mn_nw4_pf2_tegrastats.csv"
offset_file = "minibatch_index_list_resnet_new.csv"
offsets, start_times, end_times = extract_offsets_from_csv(offset_file)
sample_count_list=[100,90,80,70,60,50,40,30,20,10]
no_runs=10

all_powermodes = generate_powermodes()
sampled_pm_list = []

og_sampled = pd.read_csv("sampled_powermodes_data.csv")
filtered_sampled = og_sampled[og_sampled['No_Powermode'].isin(sample_count_list)]
filtered_sampled = filtered_sampled.sort_values(by='No_Powermode', ascending=False)
sampled_pm_list = [ast.literal_eval(i) for i in filtered_sampled["Sampled_Powermodes"].tolist()]


pretrained_model_weights_path_time = 'NN_mobnet_time_80_10_10.h5'
pretrained_model_weights_path_power = 'NN_mobnet_power_80_10_10.h5'

model = path.split("_")[0]
pretrained_model = pretrained_model_weights_path_time.split("_")[1]
name = "NN_transfer_"+pretrained_model+"_"+model+"_MAPE.csv"


# >>>>>>>>>>>>>>>>>> Removed 2 slowest freq for yolo runs (0)
# >>>>>>>>>>>>>>>>>> Changed generate power modes for yolo from 4th column to 5th column (0)

transfer_learning_run_time(all_powermodes, sampled_pm_list, sample_count_list, no_runs, path, filename, offsets, pretrained_model_weights_path_time)
transfer_learning_run_power(all_powermodes, sampled_pm_list, sample_count_list, no_runs, path, filename, tg_filename, offsets, start_times, end_times, pretrained_model_weights_path_power)
plot_MAPE_dc_violins(name)
