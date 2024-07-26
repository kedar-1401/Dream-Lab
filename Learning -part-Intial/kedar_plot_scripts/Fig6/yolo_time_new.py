import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.patches as mpatches
from matplotlib.ticker import AutoMinorLocator
import re
import ast  # Import the ast module

plt.rc('xtick', labelsize='large')
plt.rc('ytick', labelsize='large')
plt.rc('axes', labelsize='x-large', titlesize='x-large')
plt.rcParams['legend.title_fontsize'] = 'large'
plt.rcParams['legend.fontsize'] = 'large'

global_time = 43.72
local_time = 16.14

# Read your data
# data1 = pd.read_csv('/home/college/Documents/iisc/edge-ml-scheduler/nn_modeling/MLP_resnet_MAPE_all.csv')
# data2 = pd.read_csv('/home/college/Documents/iisc/edge-ml-scheduler/nn_modeling/NN_resnet_MAPE_all.csv')
# data3 = pd.read_csv('/home/college/Documents/iisc/edge-ml-scheduler/nn_modeling/NN_transfer_mobnet_resnet_MAPE.csv')

#data1 = pd.read_csv(r'C:\Users\beaut\Downloads\barplots\data\nn_modeling\MLP_yolo_MAPE_all.csv')

#data2 = pd.read_csv(r'C:\Users\saisa\Documents\GitHub\sigmetrics2024\plot_scripts\Fig7\data\nn_modeling\NN_yolo_MAPE_all.csv')
#data3 = pd.read_csv(r'C:\Users\saisa\Documents\GitHub\sigmetrics2024\plot_scripts\Fig7\data\nn_modeling\NN_transfer_resnet_yolo_MAPE_all.csv')


data2 = pd.read_csv(r'data\nn_modeling\NN_yolo_MAPE_all.csv')
data3 = pd.read_csv(r'data\nn_modeling\NN_transfer_resnet_yolo_MAPE_all.csv')

# data1['combined_obj'] = (data1['Val_MAPE_time'] + data1['Val_MAPE_power']) / 2
# data1 = data1.sort_values(by='combined_obj')
data2['combined_obj'] = (data2['Val_MAPE_time'] + data2['Val_MAPE_power']) / 2
data2 = data2.sort_values(by='combined_obj')
data3['combined_obj'] = (data3['Val_MAPE_time'] + data3['Val_MAPE_power']) / 2
data3 = data3.sort_values(by='combined_obj')


# Group the data and calculate the median of 'Val_MAPE_time'
#grouped_data1 = data1.groupby('No_powermodes')['Val_MAPE_time'].apply(lambda x: x.iloc[5])
grouped_data2 = data2.groupby('No_powermodes')['Val_MAPE_time'].apply(lambda x: x.iloc[5])
grouped_data3 = data3.groupby('No_powermodes')['Val_MAPE_time'].apply(lambda x: x.iloc[5])

#grouped_data1_all = data1.groupby('No_powermodes')['Val_MAPE_time'].quantile(0.75) - data1.groupby('No_powermodes')['Val_MAPE_time'].quantile(0.25)
grouped_data2_all = data2.groupby('No_powermodes')['Val_MAPE_time'].quantile(0.75) - data2.groupby('No_powermodes')['Val_MAPE_time'].quantile(0.25)
grouped_data3_all = data3.groupby('No_powermodes')['Val_MAPE_time'].quantile(0.75) - data3.groupby('No_powermodes')['Val_MAPE_time'].quantile(0.25)

# print(grouped_data1_all)

# Group the data and calculate the median of 'data_collection'
grouped_data_collection = data2.groupby('No_powermodes')

# Get all unique 'No_powermodes' values
all_groups = grouped_data2.index.unique()

# Create a figure with two y-axes
fig, ax1 = plt.subplots()


# Bar plot for 'Val_MAPE_time'
width = 2.5
#legend_handles = []
#legend_labels = ['MLP', 'NN','TL','DC Time']


for idx, group in enumerate(all_groups):
    #medians_data1 = grouped_data1[grouped_data1.index == group] * 100
    medians_data2 = grouped_data2[grouped_data2.index == group] * 100
    medians_data3 = grouped_data3[grouped_data3.index == group] * 100

    #ax1.bar(group - width, medians_data1, width=width, label='MLP', align='center', color='#c7e59b', zorder=3, yerr = grouped_data1_all[group] * 100, capsize = 4)
    ax1.bar(group - 1.25, medians_data2, width=width, label='NN', align='center', color='#fadc6f',  yerr = grouped_data2_all[group] * 100, zorder=3, capsize = 4)
    ax1.bar(group + 1.25, medians_data3, width=width, label='TL', align='center', color='#e36668', zorder=3,  yerr = grouped_data3_all[group] * 100, capsize = 4)
    
    #print(medians_data2)
    #legend_handles.extend([mpatches.Patch(color='blue'), mpatches.Patch(color='orange'), mpatches.Patch(color='green')]) #, mpatches.Patch(color='green')

ax1.bar(110,9.72,width=width,label='',align='center',color = '#fadc6f', zorder = 3)
ax1.set_xlabel('# Power modes used for Training')
ax1.set_ylabel('Time MAPE (%)', color='black')
#ax1.set_title('Median Val_MAPE_time and Different No_powermodes Groups')



x = np.arange(10, 120, 10)
ax1.set_xticks(x)


ax1.set_xticklabels(['10','20','30','40','50','60','70','80','90','100','All'])

#ax1.grid(True)
ax1.grid(axis='y')
ax1.yaxis.set_minor_locator(AutoMinorLocator())

vlines_x_between = [15,25,35,45,55,65,75,85,95,105,115]
ax1.vlines(x=vlines_x_between,ymin=0,ymax=80,linestyles='solid',color='darkgrey',linewidth=1.5) #,bbox_to_anchor=(0.5, 1)

ax1.vlines(x=[105,115], ymin=0, ymax=80, linestyles='solid', color='black', linewidth=1.5)

ax1.grid(which='minor', linestyle='--', linewidth='0.5', color='lightgray')
# ax1.axvline(x=115, color='black', linestyle='-', linewidth=1.5)

ax1.text(110, 53, '29.7h*', color='purple', fontsize=12, ha='center', va='center', rotation=90)

ax1.set_ylim(0,60)
ax1.set_xlim(0,115)

# s = 120

# ax1.text(s - 1.25, global_time + 0.5, str(global_time), color='black', ha='center')
# ax1.text(s + 1.25, local_time + 0.5, str(local_time), color='black', ha='center')

# ax1.bar(s - 1.25, global_time, color='#fb9a3c', width=width)

# ax1.bar(s + 1.25, local_time, color='#fcc28a', width=width)

# Create a secondary y-axis for 'Data_collection_time'

data2['Data_collection_time'] /= 60
data3['Data_collection_time'] /= 60
ax2 = ax1.twinx()

data_collection_medians = []

# Load the index_data_sorted.csv file
# index_df = pd.read_csv('/home/college/Documents/iisc/edge-ml-scheduler/data/resnet_v4_runs_merged/minibatch_index_list_resnet_new.csv')

index_df = pd.read_csv(r'data\yolo_runs\minibatch_index_list_yolo_new.csv')

data_collection_overhead = []

# Iterate over the groups
for group_name, group_data in grouped_data_collection:

    data_collection_medians.append(group_data['Data_collection_time'].iloc[5])

    data_collection_time = 0

    group_data['Sampled_powermodes'] = group_data['Sampled_powermodes'].apply(ast.literal_eval)

    for i in group_data['Sampled_powermodes'].iloc[5]:

        match = re.match(r'(\d+)_(\d+)_(\d+)_(\d+)', i)

        if match:
            #print(match.group(1))
            # Find the corresponding row in index_data_sorted.csv
            matching_row = index_df[(index_df['cores'] == int(match.group(1))) &
                                    (index_df['cpu'] == int(match.group(2))) &
                                    (index_df['gpu'] == int(match.group(3))) &
                                    (index_df['mem'] == int(match.group(4))) ]

            data_collection_time += matching_row.iloc[0]['start_time']

    #print(data_collection_time)
    data_collection_overhead.append(data_collection_time+group_data['Data_collection_time'].iloc[5])


# print('MLP')
# print(grouped_data1)
# print('NN')
# print(grouped_data2)
# print('Transfer')
# print(grouped_data3)

#print(data_collection_medians)
ax2.plot(all_groups, data_collection_medians , marker='o', linestyle='-', color='#be4986', label='Data Collection Time')
#ax2.plot(all_groups, data_collection_overhead , marker='o', linestyle='-', color='red', label='Data Collection Overhead')
ax2.set_ylabel('Data collection time (min)', color='#be4986')
ax2.set_ylim(0,60)
# Set the same x-ticks for the secondary y-axis
ax2.set_xticks(x)
#ax2.grid(True)

handles, labels = [], []
for ax in fig.axes:
    for h, l in zip(*ax.get_legend_handles_labels()):
        handles.append(h)
        labels.append(l)

plt.legend(handles=handles[-3:], labels=labels[-3:], loc='upper left',prop={'size': 10}) #,bbox_to_anchor=(0.5, 1)

# Show the plot
plt.tight_layout()

plt.savefig(f'./yolo_time_new.png', dpi=600, bbox_inches='tight')
plt.savefig(f'./yolo_time_new.pdf',dpi=300,bbox_inches='tight')


# plt.show()


