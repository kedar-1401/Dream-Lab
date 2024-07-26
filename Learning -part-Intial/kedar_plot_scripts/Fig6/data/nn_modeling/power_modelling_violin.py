import pandas as pd
import numpy as np
import csv
import random
import time
import matplotlib.pyplot as plt
import seaborn as sns

# def populate_data(all_powermodes,path,filename, tg_filename, offset_dict, start_dict, end_dict):

#     df=pd.DataFrame(columns = ['Cores', 'CPU_frequency', 'GPU_frequency', 'Memory_frequency', 'Minibatch_time'])
#     pm_count = 0
#     minibatch_power={}
#     minibatch_power_count=[]
#     for powermode in all_powermodes:
#         #print(pm_count)
#         # sum=0 #init sum to 0 for powermode
#         pm_count+=1
#         offset = offset_dict.get(powermode, -1)
#         start_time = start_dict.get(powermode, -1)
#         end_time = end_dict.get(powermode, -1)
#         file=path+"/"+"pm_"+powermode+"/"+filename
#         tg_file=path+"/"+"pm_"+powermode+"/"+tg_filename

#         # temp_df = pd.read_csv(file, header=None)
#         tg_df = pd.read_csv(tg_file)

#         filtered_df = tg_df[(tg_df['log_time'] >= start_time) & (tg_df['log_time'] <= end_time)]
#         power_list = filtered_df['power cur'].astype(float).dropna().tolist()
#         minibatch_power_count.append({'powermode': powermode, 'number': len(power_list)})
        
#     minibatch_power_count_df = pd.DataFrame(minibatch_power_count)
#     minibatch_power_count_df.to_csv("power_distribution.csv",index=False)
#     return minibatch_power_count_df

def populate_data(all_powermodes, path, filename, tg_filename, offset_dict, start_dict, end_dict):

    rows = []
    for powermode in all_powermodes:
        offset = offset_dict.get(powermode, -1)
        start_time = start_dict.get(powermode, -1)
        end_time = end_dict.get(powermode, -1)
        file = path + "/" + "pm_" + powermode + "/" + filename
        tg_file = path + "/" + "pm_" + powermode + "/" + tg_filename

        tg_df = pd.read_csv(tg_file)
        filtered_df = tg_df[(tg_df['log_time'] >= start_time) & (tg_df['log_time'] <= end_time)]
        power_list = filtered_df['power cur'].astype(float).dropna().tolist()

        # Adjust power_list length to 12
        while len(power_list) < 12:
            power_list.append(power_list[0])
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
    minibatch_power_df.to_csv("val_power.csv", index=False)
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

    all_powermodes=[] #6*13*14*4=4368 possible values
    for cpu_core in cpu_core_vals:
        for cpu_frequency in cpu_frequencies:
            for gpu_frequency in gpu_frequencies:
                for mem_frequency in mem_frequencies:
                    all_powermodes.append(str(cpu_core)+"_"+str(cpu_frequency)+"_"+str(gpu_frequency)+"_"+str(mem_frequency))
    return all_powermodes

def plot_violin(data):

    fig, ax = plt.subplots(figsize=(6, 10))

    # Create the violin plot
    sns.violinplot(y=data["number"], orient="v", ax=ax, cut=0)

    # Setting major and minor ticks on the y-axis for better clarity
    ax.yaxis.grid(True, which='both')
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.minorticks_on()

    # Title and labels
    plt.title("Viloin plot of number of power samples per powermode")
    plt.ylabel("Number")

    plt.show()

path="mobnet_v3_runs_merged"
filename="mn_nw4_pf2_energy.csv"
tg_filename="mn_nw4_pf2_tegrastats.csv"
offsets, start_times, end_times = extract_offsets_from_csv("minibatch_index_list_mobilenet_new.csv")
all_powermodes=generate_powermodes()
# print(all_powermodes)
data=populate_data(all_powermodes,path,filename, tg_filename, offsets, start_times, end_times)
# plot_violin(data)