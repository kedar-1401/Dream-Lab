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

## Used to generate NN predictions + pareto front for infer workloads. Requires observed_data ({model}_infer_data_final.csv) and sampled powermodes (e24_random_50_pwmds.csv) as input. 
## Change 50 to any number of powermodes you want to sample by creating a new csv with the sampled powermodes using e24_pwd_gen.py.

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



def sampled_model(all_data_path, sampled_pwds, model_name):

    all_data = pd.read_csv(all_data_path)
    all_data['powermode'] = all_data['cores'].astype(str) + "_" + all_data['cpu'].astype(str) + "_" + all_data['gpu'].astype(str) + "_" + all_data['mem'].astype(str)
    # filter out the sampled pwds
    sampled_data = all_data[all_data['powermode'].isin(sampled_pwds)]   
    sampled_pwds = sampled_data['powermode'].tolist()
    num_samples = len(sampled_pwds)

    df = sampled_data.drop(columns=['powermode','observed_power'])


    df['observed_time'] = pd.to_numeric(df['observed_time'])
    # df.to_csv('train.csv', index=False)
    x = df.drop('observed_time', axis=1)
    y = df['observed_time']
 
    #20% as test set
    trainX, testX, trainY, testY = train_test_split(x, y, test_size = 0.2, random_state=31)

    #Standardize features by removing the mean and scaling to unit variance
    sc=StandardScaler()
    scaler_time = sc.fit(trainX)
    trainX_scaled = scaler_time.transform(trainX)
    testX_scaled = scaler_time.transform(testX)

    best_local_weight = 'best_model.h5'
    checkpoint = ModelCheckpoint(best_local_weight, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

    model = neural_network(trainX_scaled.shape[1])

    history = model.fit(trainX_scaled, trainY, epochs=1000, batch_size=16, validation_data=(testX_scaled, testY), verbose=0,callbacks=[checkpoint])


    df= all_data.drop(columns=['powermode','observed_power'])

    x_val = df.drop('observed_time', axis=1)
    y_val = df['observed_time']
    valX_scaled = scaler_time.transform(x_val)

    y_pred_val = model.predict(valX_scaled).flatten()
    df_temp = pd.DataFrame({'predicted_time': y_pred_val})
    # merge df_temp with x_val so columns are cores, cpu, gpu, mem, predicted
    df_temp = pd.concat([x_val, df_temp], axis=1)
    df_temp.columns = ['cores', 'cpu', 'gpu', 'mem', 'bs', 'predicted_time']
    df_temp['cores'] = df_temp['cores'].astype(int)
    df_temp['cpu'] = df_temp['cpu'].astype(int)
    df_temp['gpu'] = df_temp['gpu'].astype(int)
    df_temp['mem'] = df_temp['mem'].astype(int)
    df_temp['bs'] = df_temp['bs'].astype(int)
    print('Mean Absolute Percentage Error:', metrics.mean_absolute_percentage_error(y_val, y_pred_val))


   

    #############################################################################################################
    df_temp.to_csv("e24_{}_NN_infer_predicted_time_{}_samples.csv".format(model_name, num_samples),index=False)
    print("Time Done")
    print("Number of samples: ", num_samples)


    df = sampled_data.drop(columns=['powermode','observed_time'])
    
    df['observed_power'] = pd.to_numeric(df['observed_power'])
    x = df.drop('observed_power', axis=1)
    y = df['observed_power']

    #20% as test set
    trainX, testX, trainY, testY = train_test_split(x, y, test_size = 0.2, random_state=31)

    #Standardize features by removing the mean and scaling to unit variance
    sc=StandardScaler()
    scaler_energy = sc.fit(trainX)
    trainX_scaled = scaler_energy.transform(trainX)
    testX_scaled = scaler_energy.transform(testX)

    best_local_weight = 'best_model.h5'
    checkpoint = ModelCheckpoint(best_local_weight, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

    model = neural_network(trainX_scaled.shape[1])


    history = model.fit(trainX_scaled, trainY, epochs=1000, batch_size=16, validation_data=(testX_scaled, testY), verbose=0,callbacks=[checkpoint])


    df= all_data.drop(columns=['powermode','observed_time'])

    x_val = df.drop('observed_power', axis=1)
    y_val = df['observed_power']
    valX_scaled = scaler_energy.transform(x_val)

    y_pred_val = model.predict(valX_scaled).flatten()
    print('Mean Absolute Percentage Error:', metrics.mean_absolute_percentage_error(y_val, y_pred_val))
    df_temp = pd.DataFrame({'predicted_power': y_pred_val})
    df_temp = pd.concat([x_val, df_temp], axis=1)
    df_temp.columns = ['cores', 'cpu', 'gpu', 'mem', 'bs', 'predicted_power']
    # convert cores, cpu, gpu, mem to int
    df_temp['cores'] = df_temp['cores'].astype(int)
    df_temp['cpu'] = df_temp['cpu'].astype(int)
    df_temp['gpu'] = df_temp['gpu'].astype(int)
    df_temp['mem'] = df_temp['mem'].astype(int)
    df_temp['bs'] = df_temp['bs'].astype(int)
    # df_temp.drop(columns=['power_sample'],inplace=True)
    # df_temp.drop_duplicates(inplace=True) 

       
    #####################################################################################
    
    df_temp.to_csv("e24_{}_NN_infer_predicted_power_{}_samples.csv".format(model_name, num_samples),index=False)
    print("Power Done")


def merge_csvs(all_powermodes, observed_file, predicted_time_file, predicted_power_file, savename):
    
    # Convert observed_file to df
    observed_df = pd.read_csv(observed_file)
    observed_df['bs'] = observed_df['bs'].astype(int)
    observed_df['powermode'] = observed_df['cores'].astype(str) + "_" + observed_df['cpu'].astype(str) + "_" + observed_df['gpu'].astype(str) + "_" + observed_df['mem'].astype(str)
    observed_df['temp_powermode'] = observed_df['cores'].astype(str) + "_" + observed_df['cpu'].astype(str) + "_" + observed_df['gpu'].astype(str) + "_" + observed_df['mem'].astype(str) + "_" + observed_df['bs'].astype(str)
    # filter if powermode in all_powermodes
    observed_df = observed_df[observed_df['powermode'].isin(all_powermodes)]
    # drop other columns
    observed_df = observed_df.drop(columns=['cores', 'cpu', 'gpu', 'mem'])
    # print(observed_df)

    # Convert predicted_time_file to df
    predicted_time_df = pd.read_csv(predicted_time_file)
    # convert cores, cpu, gpu, mem to int
    predicted_time_df['cores'] = predicted_time_df['cores'].astype(int)
    predicted_time_df['cpu'] = predicted_time_df['cpu'].astype(int)
    predicted_time_df['gpu'] = predicted_time_df['gpu'].astype(int)
    predicted_time_df['mem'] = predicted_time_df['mem'].astype(int)
    predicted_time_df['powermode'] = predicted_time_df['cores'].astype(str) + "_" + predicted_time_df['cpu'].astype(str) + "_" + predicted_time_df['gpu'].astype(str) + "_" + predicted_time_df['mem'].astype(str)
    predicted_time_df['temp_powermode'] = predicted_time_df['cores'].astype(str) + "_" + predicted_time_df['cpu'].astype(str) + "_" + predicted_time_df['gpu'].astype(str) + "_" + predicted_time_df['mem'].astype(str) + "_" + predicted_time_df['bs'].astype(str)
    # drop other columns
    predicted_time_df = predicted_time_df.drop(columns=['cores', 'cpu', 'gpu', 'mem'])
    # print(predicted_time_df)
    # Convert predicted_energy_file to df
    predicted_power_df = pd.read_csv(predicted_power_file)
    # convert cores, cpu, gpu, mem to int
    predicted_power_df['cores'] = predicted_power_df['cores'].astype(int)
    predicted_power_df['cpu'] = predicted_power_df['cpu'].astype(int)
    predicted_power_df['gpu'] = predicted_power_df['gpu'].astype(int)
    predicted_power_df['mem'] = predicted_power_df['mem'].astype(int)
    predicted_power_df['powermode'] = predicted_power_df['cores'].astype(str) + "_" + predicted_power_df['cpu'].astype(str) + "_" + predicted_power_df['gpu'].astype(str) + "_" + predicted_power_df['mem'].astype(str)
    predicted_power_df['temp_powermode'] = predicted_power_df['cores'].astype(str) + "_" + predicted_power_df['cpu'].astype(str) + "_" + predicted_power_df['gpu'].astype(str) + "_" + predicted_power_df['mem'].astype(str) + "_" + predicted_power_df['bs'].astype(str)
    # drop other columns
    predicted_power_df = predicted_power_df.drop(columns=['cores', 'cpu', 'gpu', 'mem'])
    # print(predicted_power_df)
    # merge all 3 dfs on powermode and bs
    merged_df = pd.merge(observed_df, predicted_time_df, on='temp_powermode')
    merged_df = pd.merge(merged_df, predicted_power_df, on='temp_powermode')
    merged_df = merged_df[['powermode', 'bs', 'observed_time', 'predicted_time', 'observed_power', 'predicted_power']]
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
    pareto_powermodes_obs = 'e24_{}_NN_infer_pareto_powermodes_obs_{}_sampled.csv'.format(model_name, num_samples)
    pareto_powermodes_pred = 'e24_{}_NN_infer_pareto_powermodes_pred_{}_sampled.csv'.format(model_name, num_samples)
    pareto_data_obs[['powermode', 'bs', 'predicted_time','predicted_power','observed_time','observed_power']].to_csv(pareto_powermodes_obs, index=False)
    pareto_data_pred[['powermode', 'bs', 'predicted_time','predicted_power','observed_time','observed_power']].to_csv(pareto_powermodes_pred, index=False)

    # obs = pd.read_csv("YOLO_pareto_powermodes_obs_new.csv")
    # pred = pd.read_csv("YOLO_pareto_powermodes_obs_new.csv")



######################################################################################
# path="resnet_v3_runs_merged"
# path = "mobnet_v3_runs_merged"
model_name = "bert"
all_data_path = f"/home/saisamarth/exp/{model_name}_infer_data_final.csv"
# num of unqiue bs from all_data_path
all_data = pd.read_csv(all_data_path)
num_bs = all_data['bs'].nunique()
sampled_pwd_path = "e24_random_50_pwmds.csv"

all_powermodes = generate_powermodes()

sampled_data = pd.read_csv(sampled_pwd_path)
sampled_pwds = sampled_data['powermode'].tolist()
num_samples = len(sampled_pwds)*num_bs

print("Number of samples: ", num_samples)

sampled_model(all_data_path, sampled_pwds, model_name)

observed_file = f"/home/saisamarth/exp/{model_name}_infer_data_final.csv"
predicted_time_file = "e24_{}_NN_infer_predicted_time_{}_samples.csv".format(model_name, num_samples)
predicted_power_file = "e24_{}_NN_infer_predicted_power_{}_samples.csv".format(model_name, num_samples)
savename = "e24_merged_observed_predicted_{}_NN_infer_sampled_{}.csv".format(model_name, num_samples)

all_powermodes = generate_powermodes()
merge_csvs(all_powermodes, observed_file, predicted_time_file, predicted_power_file, savename)


data = pd.read_csv("e24_merged_observed_predicted_{}_NN_infer_sampled_{}.csv".format(model_name, num_samples))
plot_pareto_front(data, model_name, num_samples)