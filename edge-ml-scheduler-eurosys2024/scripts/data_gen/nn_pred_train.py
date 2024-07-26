import os
from pyexpat import model
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

## Used to generate NN predictions + pareto front for train workloads. Requires observed_data ({model}_train_data_final.csv) and sampled powermodes (e24_random_250_pwmds.csv) as input. 
## Change 250 to any number of powermodes you want to sample by creating a new csv with the sampled powermodes using e24_pwd_gen.py.

def generate_powermodes():
    core_vals=[4, 8, 12] #3 possible values
    gpu_vals=[114750000, 318750000, 522750000, 726750000, 930750000, 1134750000, 1300500000]
    cpu_vals=[422400, 729600, 1036800, 1344000, 1651200, 1958400, 2201600] #in kHz, 7 possible values
    mem_vals = [665600000, 2133000000, 3199000000]
    #get combinations of all 4 as powermode Ex.2_1300500000_268800_204000000
    all_powermodes=[] #6*13*14*4=4368 possible values
    for cpu_core in core_vals:
        for cpu_frequency in cpu_vals:
            for gpu_frequency in gpu_vals:
                for mem_frequency in mem_vals:
                    all_powermodes.append(str(cpu_core)+"_"+str(cpu_frequency)+"_"+str(gpu_frequency)+"_"+str(mem_frequency))
    return all_powermodes

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


def populate_data_time(sampled_powermodes, path, filename, offset_dict, model_name):
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



        #################################
        ################################# change to 5 for yolo, else 4 iloc[:,*4*]
        temp_df = temp_df.iloc[start:end]
        if model_name == "yolo":
            temp_df = temp_df.iloc[:,5]
        else:
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

def populate_data_power(sampled_powermodes, path, filename, tg_filename, offset_dict, start_dict, end_dict, model_name):

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

        ############################################################# Change required_lenght according to model
        #Resnet: 22
        #Mobnet: 40
        #Yolo: 100
        if model_name == "yolo":
            required_length = 100
        elif model_name == "mobnet":
            required_length = 40
        elif model_name == "resnet":
            required_length = 22
        elif model_name == "bert":
            required_length = 10
        else:
            required_length = 200

        if len(power_list) == 0:
            print("Power list is empty for powermode: ", powermode)
            print("Start time: ", start_time)
            print("End time: ", end_time)

        if len(power_list) < required_length:
            repeats_required = -(-required_length // len(power_list))  
            power_list = (power_list * repeats_required)[:required_length]



        power_list = power_list[:required_length]

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

# def load_pretrained_model(input_dim, model_weights_path):
#     model = neural_network(input_dim)
#     model.load_weights(model_weights_path)    
#     model.pop()
    
#     return model

def sampled_model(path, sampled_pwd_path, filename, tg_filename, offsets, start_times, end_times, model_name):

    sampled_data = pd.read_csv(sampled_pwd_path)
    sampled_pwds = sampled_data['powermode'].tolist()
    num_samples = len(sampled_pwds)

    train_data = []

    for powermode in sampled_pwds:
        train_data.append(populate_data_time([powermode], path, filename, offsets, model_name))

    df = pd.concat(train_data, ignore_index=True)
    df['Minibatch_time'] = pd.to_numeric(df['Minibatch_time'])
    # df.to_csv('train.csv', index=False)
    x = df.drop('Minibatch_time', axis=1)
    y = df['Minibatch_time']
 
    #20% as test set
    trainX, testX, trainY, testY = train_test_split(x, y, test_size = 0.2, random_state=31)

    #Standardize features by removing the mean and scaling to unit variance
    sc=StandardScaler()
    scaler_time = sc.fit(trainX)
    trainX_scaled = scaler_time.transform(trainX)
    testX_scaled = scaler_time.transform(testX)

    best_local_weight = 'best_model.h5'
    checkpoint = ModelCheckpoint(best_local_weight, monitor='val_loss', verbose=0, save_best_only=True, mode='min')

    model = neural_network(trainX_scaled.shape[1])

    history = model.fit(trainX_scaled, trainY, epochs=200, batch_size=16, validation_data=(testX_scaled, testY), verbose=1,callbacks=[checkpoint])


    val_data = []
    for powermode in all_powermodes:
        val_data.append(populate_data_time([powermode], path, filename, offsets, model_name))
    df = pd.concat(val_data, ignore_index=True)
    df.to_csv('val.csv', index=False)


    df=pd.read_csv("val.csv")
    x_val = df.drop('Minibatch_time', axis=1)
    y_val = df['Minibatch_time']
    valX_scaled = scaler_time.transform(x_val)

    y_pred_val = model.predict(valX_scaled).flatten()
    df_temp = pd.DataFrame({'predicted_time': y_pred_val})
    # merge df_temp with x_val so columns are cores, cpu, gpu, mem, predicted
    df_temp = pd.concat([x_val, df_temp], axis=1)
    df_temp.columns = ['cores', 'cpu', 'gpu', 'mem', 'predicted_time']
    df_temp['cores'] = df_temp['cores'].astype(int)
    df_temp['cpu'] = df_temp['cpu'].astype(int)
    df_temp['gpu'] = df_temp['gpu'].astype(int)
    df_temp['mem'] = df_temp['mem'].astype(int)
    print('Mean Absolute Percentage Error:', metrics.mean_absolute_percentage_error(y_val, y_pred_val))


    df_temp = df_temp.groupby(df_temp.index // 40).median()
    

    #############################################################################################################
    df_temp.to_csv("e24_{}_NN_predicted_time_{}_samples.csv".format(model_name, num_samples),index=False)
    print("Time Done")

    train_data1 = []

    for powermode in sampled_pwds:
        train_data1.append(populate_data_power([powermode], path, filename, tg_filename, offsets, start_times, end_times, model_name))
    df = pd.concat(train_data1, ignore_index=True)
    df['power_sample'] = pd.to_numeric(df['power_sample'])
    x = df.drop('power_sample', axis=1)
    y = df['power_sample']

    #20% as test set
    trainX, testX, trainY, testY = train_test_split(x, y, test_size = 0.2, random_state=31)

    #Standardize features by removing the mean and scaling to unit variance
    sc=StandardScaler()
    scaler_energy = sc.fit(trainX)
    trainX_scaled = scaler_energy.transform(trainX)
    testX_scaled = scaler_energy.transform(testX)

    best_local_weight = 'best_model.h5'
    checkpoint = ModelCheckpoint(best_local_weight, monitor='val_loss', verbose=0, save_best_only=True, mode='min')

    model = neural_network(trainX_scaled.shape[1])


    history = model.fit(trainX_scaled, trainY, epochs=200, batch_size=16, validation_data=(testX_scaled, testY), verbose=1,callbacks=[checkpoint])


    val_data = []
    for powermode in all_powermodes:
        val_data.append(populate_data_power([powermode], path, filename, tg_filename, offsets, start_times, end_times, model_name))
    df = pd.concat(val_data, ignore_index=True)
    df.to_csv('val_power.csv', index=False)

    df = pd.read_csv('val_power.csv')
    x_val = df.drop('power_sample', axis=1)
    y_val = df['power_sample']
    valX_scaled = scaler_energy.transform(x_val)

    y_pred_val = model.predict(valX_scaled).flatten()
    print('Mean Absolute Percentage Error:', metrics.mean_absolute_percentage_error(y_val, y_pred_val))
    df_temp = pd.DataFrame({'predicted_power': y_pred_val})
    df_temp = pd.concat([x_val, df_temp], axis=1)
    df_temp.columns = ['cores', 'cpu', 'gpu', 'mem', 'predicted_power']
    # convert cores, cpu, gpu, mem to int
    df_temp['cores'] = df_temp['cores'].astype(int)
    df_temp['cpu'] = df_temp['cpu'].astype(int)
    df_temp['gpu'] = df_temp['gpu'].astype(int)
    df_temp['mem'] = df_temp['mem'].astype(int)
    df_temp['predicted_power'] = df_temp['predicted_power']/1000
    # df_temp.drop(columns=['power_sample'],inplace=True)
    # df_temp.drop_duplicates(inplace=True) 
    if model_name == "yolo":
        # take median of every 100 values
        df_temp = df_temp.groupby(df_temp.index // 100).median()
    elif model_name == "mobnet":
        # take median of every 40 values
        df_temp = df_temp.groupby(df_temp.index // 40).median()
    elif model_name == "resnet":
        # take median of every 22 values
        df_temp = df_temp.groupby(df_temp.index // 22).median()
    elif model_name == "bert":
        # take median of every 10 values
        df_temp = df_temp.groupby(df_temp.index // 10).median()
    else:
        # take median of every 200 values
        df_temp = df_temp.groupby(df_temp.index // 200).median()
       
    #####################################################################################
    
    df_temp.to_csv("e24_{}_NN_predicted_power_{}_samples.csv".format(model_name, num_samples),index=False)
    print("Power Done")


def merge_csvs(all_powermodes, observed_file, predicted_time_file, predicted_power_file, savename):
    
    # Convert observed_file to df
    observed_df = pd.read_csv(observed_file)
    observed_df['powermode'] = observed_df['cores'].astype(str) + "_" + observed_df['cpu'].astype(str) + "_" + observed_df['gpu'].astype(str) + "_" + observed_df['mem'].astype(str)
    # filter if powermode in all_powermodes
    observed_df = observed_df[observed_df['powermode'].isin(all_powermodes)]
    # drop other columns
    observed_df = observed_df.drop(columns=['cores', 'cpu', 'gpu', 'mem'])

    # Convert predicted_time_file to df
    predicted_time_df = pd.read_csv(predicted_time_file)
    # convert cores, cpu, gpu, mem to int
    predicted_time_df['cores'] = predicted_time_df['cores'].astype(int)
    predicted_time_df['cpu'] = predicted_time_df['cpu'].astype(int)
    predicted_time_df['gpu'] = predicted_time_df['gpu'].astype(int)
    predicted_time_df['mem'] = predicted_time_df['mem'].astype(int)
    predicted_time_df['powermode'] = predicted_time_df['cores'].astype(str) + "_" + predicted_time_df['cpu'].astype(str) + "_" + predicted_time_df['gpu'].astype(str) + "_" + predicted_time_df['mem'].astype(str)
    # drop other columns
    predicted_time_df = predicted_time_df.drop(columns=['cores', 'cpu', 'gpu', 'mem'])
    # Convert predicted_energy_file to df
    predicted_power_df = pd.read_csv(predicted_power_file)
    # convert cores, cpu, gpu, mem to int
    predicted_power_df['cores'] = predicted_power_df['cores'].astype(int)
    predicted_power_df['cpu'] = predicted_power_df['cpu'].astype(int)
    predicted_power_df['gpu'] = predicted_power_df['gpu'].astype(int)
    predicted_power_df['mem'] = predicted_power_df['mem'].astype(int)
    predicted_power_df['powermode'] = predicted_power_df['cores'].astype(str) + "_" + predicted_power_df['cpu'].astype(str) + "_" + predicted_power_df['gpu'].astype(str) + "_" + predicted_power_df['mem'].astype(str)
    # drop other columns
    predicted_power_df = predicted_power_df.drop(columns=['cores', 'cpu', 'gpu', 'mem'])
    # merge all 3 dfs
    merged_df = pd.merge(observed_df, predicted_time_df, on='powermode', how='inner')
    merged_df = pd.merge(merged_df, predicted_power_df, on='powermode', how='inner')   

    # Save merged_df to csv
    merged_df.to_csv(savename, index=False)

def is_pareto_efficient(costs):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        is_efficient[i] = np.all(np.any(costs[:i]>c, axis=1)) and np.all(np.any(costs[i+1:]>c, axis=1))
    return is_efficient

def plot_pareto_front(data, model_name, num_samples):

    fig, ax = plt.subplots()

    # Extract the relevant data
    x_obs = data['observed_power']
    y_obs = data['observed_time']
    x_pred = data['predicted_power']
    y_pred = data['predicted_time']

    # Find pareto points
    pareto_efficient_obs = is_pareto_efficient(np.array([x_obs, y_obs]).T)
    pareto_efficient_pred = is_pareto_efficient(np.array([x_pred, y_pred]).T)
    
    # Extract Pareto points
    pareto_data_obs = data[pareto_efficient_obs]
    pareto_x_obs = pareto_data_obs['observed_power']
    pareto_y_obs = pareto_data_obs['observed_time']

    pareto_data_pred = data[pareto_efficient_pred]
    pareto_x_pred = pareto_data_pred['predicted_power']
    pareto_y_pred = pareto_data_pred['predicted_time']
    
    # Sort Pareto points by x-values (Energy)
    sorted_indices_obs = np.argsort(pareto_x_obs)
    sorted_pareto_x_obs = pareto_x_obs.iloc[sorted_indices_obs].values
    sorted_pareto_y_obs = pareto_y_obs.iloc[sorted_indices_obs].values

    sorted_indices_pred = np.argsort(pareto_x_pred)
    sorted_pareto_x_pred = pareto_x_pred.iloc[sorted_indices_pred].values
    sorted_pareto_y_pred = pareto_y_pred.iloc[sorted_indices_pred].values

    # # Set minor gridlines
    # plt.minorticks_on()
    # ax.yaxis.grid(which = 'minor', linestyle = '-', linewidth = 0.25, color = 'grey')

    # ax.xaxis.grid(which = 'minor', linestyle = '-', linewidth = 0.25, color = 'grey')

    # # Create a scatter plot
    # # ax.figure(figsize=(10, 6))

    # ax.scatter(x_obs, y_obs, s=2, label='Observed points', color='#FF6EC7')

    # ax.scatter(x_pred, y_pred, s=2, label='Predicted points', color='#94E5FF')

    # ax.scatter(sorted_pareto_x_obs, sorted_pareto_y_obs, color='#D21404', s=2, label='Obs Pareto points')

    # ax.plot(sorted_pareto_x_obs, sorted_pareto_y_obs, color='#D21404', label='Obs Pareto front', alpha=0.6)

    # ax.scatter(sorted_pareto_x_pred, sorted_pareto_y_pred, color='blue', s=2, label='Pred Pareto points')

    # ax.plot(sorted_pareto_x_pred, sorted_pareto_y_pred, color='blue', label='Pred   Pareto front', alpha=0.6)

    # ax.set_xlim(10, 27.5)

    # ax.set_ylim(0, 5000)
    # # Adjust axis limits to zoom in on the region of Pareto points
    # # plt.xlim(sorted_pareto_x.min() * 0.95, sorted_pareto_x.max() * 1.05)
    # # plt.ylim(sorted_pareto_y.min() * 0.95, sorted_pareto_y.max() * 1.05)
    # ax.set_xlabel('Power (W)')

    # ax.set_ylabel('Time (s)')

    # plt.legend()

    # # plt.title('Scatter plot of Power vs Time with Pareto front')

    # plt.grid(True)
    # # plt.savefig('ScatterPlot_BERT.png')
    # plt.show()

    # Save the corresponding power modes of the Pareto points to a csv
    ######################################################################
    pareto_powermodes_obs = 'e24_{}_NN_pareto_powermodes_obs_{}_sampled.csv'.format(model_name, num_samples)
    pareto_powermodes_pred = 'e24_{}_NN_pareto_powermodes_pred_{}_sampled.csv'.format(model_name, num_samples)
    pareto_data_obs[['powermode','predicted_time','predicted_power','observed_time','observed_power']].to_csv(pareto_powermodes_obs, index=False)
    pareto_data_pred[['powermode','predicted_time','predicted_power','observed_time','observed_power']].to_csv(pareto_powermodes_pred, index=False)

    # obs = pd.read_csv("YOLO_pareto_powermodes_obs_new.csv")
    # pred = pd.read_csv("YOLO_pareto_powermodes_obs_new.csv")



######################################################################################
# path="resnet_v3_runs_merged"
# path = "mobnet_v3_runs_merged"
model_name = "resnet"
path=model_name+"_v3_runs_merged"
if model_name == 'bert' or model_name == 'lstm':
    path = f'/home/saisamarth/exp/training_bs16_runs/{model_name}'

num_sampled_pwds  = 1092

sampled_pwd_path = f"e24_random_{num_sampled_pwds}_pwmds.csv"
filename="mn_nw4_pf2_epoch_stats.csv"
tg_filename="mn_nw4_pf2_tegrastats.csv"
#############################################################
if model_name == "mobnet":
    model_name = "mobilenet"

offset_file = f"minibatch_index_list_{model_name}_new.csv"
offsets, start_times, end_times = extract_offsets_from_csv(offset_file)

all_powermodes = generate_powermodes()

sampled_data = pd.read_csv(sampled_pwd_path)
sampled_pwds = sampled_data['powermode'].tolist()
num_samples = len(sampled_pwds)

sampled_model(path, sampled_pwd_path, filename, tg_filename, offsets, start_times, end_times, model_name)

observed_file = f"/home/saisamarth/exp/{model_name}_train_data_final.csv"
predicted_time_file = "e24_{}_NN_predicted_time_{}_samples.csv".format(model_name, len(sampled_pwds))
predicted_power_file = "e24_{}_NN_predicted_power_{}_samples.csv".format(model_name, len(sampled_pwds))
savename = "e24_merged_observed_predicted_{}_NN_sampled_{}.csv".format(model_name, num_samples)

if model_name == "yolo":
    no_mbs_per_epoch = 1563  #1443 mobnet, 3125 resent , 1563 yolo
elif model_name == "mobnet":
    no_mbs_per_epoch = 1443
else:
    no_mbs_per_epoch = 3125

all_powermodes = generate_powermodes()
merge_csvs(all_powermodes, observed_file, predicted_time_file, predicted_power_file, savename)


data = pd.read_csv("e24_merged_observed_predicted_{}_NN_sampled_{}.csv".format(model_name, num_samples))
plot_pareto_front(data, model_name, num_samples)