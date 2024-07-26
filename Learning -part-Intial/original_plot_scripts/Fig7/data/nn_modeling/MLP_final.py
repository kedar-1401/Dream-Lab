import pandas as pd
import numpy as np
import os
import csv
import random
import time
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import shutil
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn import metrics

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

        #Resnet: 22
        #Mobnet: 40
        #Yolo: 61
        required_length = 40
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

def MLP_run_time(all_powermodes, sampled_powermodes, sample_count_list, runs, path, filename, offsets):

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
                data_collection_time = calculate_data_collection_time(sampled_powermode, path, filename, offsets)

                train_data = []

                for powermode in sampled_powermode:
                    train_data.append(populate_data_time([powermode], path, filename, offsets))
                df = pd.concat(train_data, ignore_index=True)
                df.to_csv('train.csv', index=False)
                print("Train Dataload time: ", time.time() - dl_start)

                t_time = time.time()
                df = pd.read_csv("train.csv")
                x = df.drop('Minibatch_time', axis=1)
                y = df['Minibatch_time']

                trainX, testX, trainY, testY = train_test_split(x, y, test_size=0.2, random_state=31)

                sc = StandardScaler()
                scaler = sc.fit(trainX)
                trainX_scaled = scaler.transform(trainX)
                testX_scaled = scaler.transform(testX)

                mlp_reg = MLPRegressor(hidden_layer_sizes=(150, 140, 130, 120, 110, 100),
                                       max_iter=100, activation='relu', solver='adam', random_state=31, early_stopping=True)

                mlp_reg.fit(trainX_scaled, trainY)
                y_pred = mlp_reg.predict(testX_scaled)
                train_time = time.time() - t_time

                val_time = time.time()
                df = pd.read_csv('val.csv')
                x_val = df.drop('Minibatch_time', axis=1)
                y_val = df['Minibatch_time']
                valX_scaled = scaler.transform(x_val)

                y_pred_val = mlp_reg.predict(valX_scaled)
                validation_time = time.time() - val_time

                list_stats.append([sample_count, run, data_collection_time,
                                   metrics.mean_absolute_percentage_error(y_val, y_pred_val),
                                   sampled_powermode, train_time, validation_time])

                print('Mean Absolute Percentage Error:', metrics.mean_absolute_percentage_error(y_val, y_pred_val))

    if os.path.exists("train.csv"):
        os.remove("train.csv")

    df = pd.DataFrame(list_stats, columns=['No_powermodes', 'RunNo', 'Data_collection_time', 'Val_MAPE', 'Sampled_powermodes', 'Train_time_time', 'Validation_time_time'])
    df.to_csv("MLP_loop.csv", index=False)

def MLP_run_power(all_powermodes, sampled_powermodes, sample_count_list, runs, path, filename, tg_filename, offsets, start_times, end_times):
    
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

                data_collection_time = calculate_data_collection_time(sampled_powermode, path, filename, offsets)

                train_data = []
                for powermode in sampled_powermode:
                    train_data.append(populate_data_power([powermode], path, filename, tg_filename, offsets, start_times, end_times))
                df = pd.concat(train_data, ignore_index=True)
                df.to_csv('train.csv', index=False)
                print("Train Dataload time: ", time.time() - dl_start)

                t_time = time.time()
                df = pd.read_csv("train.csv")
                x = df.drop('power_sample', axis=1)
                y = df['power_sample']

                trainX, testX, trainY, testY = train_test_split(x, y, test_size=0.2, random_state=31)

                sc = StandardScaler()
                scaler = sc.fit(trainX)
                trainX_scaled = scaler.transform(trainX)
                testX_scaled = scaler.transform(testX)

                mlp_reg = MLPRegressor(hidden_layer_sizes=(150, 140, 130, 120, 110, 100),
                                       max_iter=100, activation='relu', solver='adam', random_state=31, early_stopping=True)

                mlp_reg.fit(trainX_scaled, trainY)
                y_pred = mlp_reg.predict(testX_scaled)
                train_time = time.time() - t_time

                val_time = time.time()

                df = pd.read_csv('val_power.csv')
                x_val = df.drop('power_sample', axis=1)
                y_val = df['power_sample']
                valX_scaled = scaler.transform(x_val)

                y_pred_val = mlp_reg.predict(valX_scaled).flatten()
                validation_time = time.time() - val_time

                list_stats.append([sample_count, run, data_collection_time, metrics.mean_absolute_percentage_error(y_val, y_pred_val), sampled_powermode, train_time, validation_time])
                print('Mean Absolute Percentage Error:', metrics.mean_absolute_percentage_error(y_val, y_pred_val))
              
    if os.path.exists("train.csv"):
        os.remove("train.csv")

    df = pd.DataFrame(list_stats, columns=['No_powermodes', 'RunNo', 'Data_collection_time', 'Val_MAPE', 'Sampled_powermodes', 'Train_time_power', 'Validation_time_power'])
    df.to_csv("MLP_loop_power.csv", index=False)



def cleanup(name):
    df_time = pd.read_csv("MLP_loop.csv")
    df_energy = pd.read_csv("MLP_loop_power.csv")

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

path="mobnet_v3_runs_merged"
filename="mn_nw4_pf2_epoch_stats.csv"
tg_filename="mn_nw4_pf2_tegrastats.csv"
offset_file = "minibatch_index_list_mobilenet_new.csv"
offsets, start_times, end_times = extract_offsets_from_csv(offset_file)
sample_count_list=[200,100,50,20]
no_runs=10

all_powermodes = generate_powermodes()
sampled_pm_list = []

og_sampled = pd.read_csv("sampled_powermodes_data.csv")
filtered_sampled = og_sampled[og_sampled['No_Powermode'].isin(sample_count_list)]
filtered_sampled = filtered_sampled.sort_values(by='No_Powermode', ascending=False)
sampled_pm_list = [ast.literal_eval(i) for i in filtered_sampled["Sampled_Powermodes"].tolist()]

model = path.split("_")[0]
name = "MLP_"+model+"_MAPE.csv"

MLP_run_time(all_powermodes, sampled_pm_list, sample_count_list, no_runs, path, filename, offsets)
MLP_run_power(all_powermodes, sampled_pm_list, sample_count_list, no_runs, path, filename, tg_filename, offsets, start_times, end_times)
cleanup(name)
plot_MAPE_dc_violins(name)