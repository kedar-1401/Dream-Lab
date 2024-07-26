import pandas as pd
import numpy as np
import csv
import random
import time
import matplotlib.pyplot as plt


def generate_powermodes():
    cpu_core_vals=[4, 8, 12] #6 possible values
    gpu_frequencies=[114750000, 318750000, 522750000, 726750000, 930750000, 1134750000, 1300500000] #in Hz, 13 possible values
    cpu_frequencies=[422400, 729600, 1036800, 1344000, 1651200, 1958400, 2201600] #in kHz, 14 possible values
    mem_frequencies=[2133000000, 3199000000, 665600000] #in Hz, 4 possible values
    #get combinations of all 4 as powermode Ex.2_1300500000_268800_204000000
    all_powermodes=[] #6*13*14*4=4368 possible values
    for cpu_frequency in cpu_frequencies:
        for gpu_frequency in gpu_frequencies:
            for cpu_core in cpu_core_vals:
                for mem_frequency in mem_frequencies:
                    all_powermodes.append(str(cpu_core)+"_"+str(cpu_frequency)+"_"+str(gpu_frequency)+"_"+str(mem_frequency))


    return all_powermodes

def detect_jtop_issues (all_powermodes, path, powerstats_filename, mb_filename):
    jtop_issue_count=0
    for powermode in all_powermodes:
        cores, cpu, gpu, mem = powermode.split("_")
        #print(powermode)
        powerstats_file=path+"/"+"pm_"+powermode+"/"+powerstats_filename
        mb_file = path + "/" + "pm_" + powermode + "/" + mb_filename
        power_df = pd.read_csv(powerstats_file)
        minibatch_df = pd.read_csv(mb_file)

        #check if  last log_time entry in minibatch - last log_time entry in powerstats > 10s
        #if yes, then jtop issue
        #get last log_time entry in powerstats
        last_log_time_powerstats = power_df['log_time'].iloc[-1]
        #get last log_time entry in minibatch
        last_log_time_minibatch = minibatch_df['log_time'].iloc[-1]
        #print(last_log_time_powerstats, last_log_time_minibatch)
        if last_log_time_minibatch - last_log_time_powerstats > 10:
            jtop_issue_count+=1
            print("jtop issue for powermode: ",powermode, last_log_time_minibatch - last_log_time_powerstats)
    print("jtop issue count: ", jtop_issue_count)

#detect transition point and generate list of minibatch indices to skip for each powermode
def generate_index_list(all_powermodes,path,powerstats_filename,mb_filename):
    pm_count = 0
    short_rt=0
    undetected_count=0
    window_size = 10
    stride = 1
    # Create an empty list to store transition points
    minibatch_index_list = []
    error_count=0
    for powermode in all_powermodes:
        cores, cpu, gpu, mem = powermode.split("_")
        #print(powermode, pm_count)
        pm_count+=1
        powerstats_file=path+"/"+"pm_"+powermode+"/"+powerstats_filename
        try:
            power_df = pd.read_csv(powerstats_file)
        except:
            cores, cpu, gpu, mem = powermode.split("_")
            #print(cpu, gpu, cores, mem)
            error_count += 1
            continue

        #testing
        #if pm_count==4:
        #    break
        # read first log time from mb file
        mb_file = path + "/" + "pm_" + powermode + "/" + mb_filename
        minibatch_df = pd.read_csv(mb_file)
        # offset is the difference between the last entry in the power log file and the last entry in the minibatch log file
        time_offset=power_df['log_time'].iloc[-1]-minibatch_df['log_time'].iloc[-1]
        # add offset to minibatch log file
        if time_offset>0:
            minibatch_df['log_time']=minibatch_df['log_time']+time_offset

        if time_offset<-10:
            print("negative time offset", powermode)
            #print(powermode)
        first_log_time = minibatch_df['log_time'].iloc[0]


        # print(first_log_time)

        detected=False
        # Iterate through the dataframe using a sliding window
        for i in range(4, len(power_df), stride):

            #if i less than 10, use smaller windows
            if i<10:
                #window_size=i
                window = power_df['power cur'].iloc[4:i]  # Extract the current window of data
            else:
                window_size=10
                window = power_df['power cur'].iloc[i - window_size:i]

            # Extract the current window of data which is from i-(window_size) to i, length of window_size


            current_value = power_df['power cur'].iloc[i]  # Get the current value
            window_average = window.mean()  # Calculate the average of the window
            #if int(cores) == 2 and int(cpu) == 1036800 and int(gpu) == 318750000 and int(mem) == 665600000:
            #    print(current_value,window_average,i)
            # Check if the current value is 25% higher than the window average

            #transition point detected
            if current_value >= 1.25 * window_average:
                #print current average and log time corresponding to current value
                #print(current_value, window_average, power_df['log_time'].iloc[i])
                #get minibatch log time closest to the transition point power_df['log_time'].iloc[i],get the number of rows before the transition point (including the tp)
                rows_before_tp = minibatch_df.iloc[(power_df['log_time'].iloc[i] - minibatch_df['log_time']).abs().argsort()[:1]].index[0]

                rows_after_tp = len(minibatch_df)-rows_before_tp

                #print any with less than 40, break and continue to the next powermode
                if rows_after_tp<40:
                    detected=True
                    print("error: rows after tp < 40")
                    print(powermode, rows_after_tp)
                    minibatch_index_list.append([cores, cpu, gpu, mem, rows_before_tp, 0, 0, rows_after_tp])
                    break

                #print("rows after tp, transition time: ", rows_after_tp, power_df['log_time'].iloc[i])
                #get log time of 40th minibatch after tp
                log40=minibatch_df['log_time'].iloc[rows_before_tp-1+40]
                #get closest powerstats row to log40
                powerlog40 = power_df.iloc[(log40 - power_df['log_time']).abs().argsort()[:1]].index[0]
                #get log_time of powerlog40
                log40_time = power_df['log_time'].iloc[powerlog40]

                #testing
                #if int(cores) == 2 and int(cpu) == 1651200 and int(gpu) == 420750000 and int(mem) == 3199000000:
                    #print("rows after tp, transition time: ", rows_after_tp, power_df['log_time'].iloc[i])

                minibatch_index_list.append([cores, cpu, gpu,mem,rows_before_tp,power_df['log_time'].iloc[i], log40_time, rows_after_tp])  # Append the index of the transition point
                detected=True
                break

        if detected==False:
            undetected_count+=1

            #get greatest log time from power df
            runtime = power_df['log_time'].iloc[-1]

            #get log time of last minibatch
            last_minibatch_time = minibatch_df['log_time'].iloc[-1]
            #get closest powerstats log time to last minibatch
            powerlog40_end = power_df.iloc[(last_minibatch_time - power_df['log_time']).abs().argsort()[:1]].index[0]
            powerlog40_end_time = power_df['log_time'].iloc[powerlog40_end]

            #get last-40th minibatch log time
            log40_end = minibatch_df['log_time'].iloc[-40]
            #get closest powerstats log time to log40_end
            powerlog40_start = power_df.iloc[(log40_end - power_df['log_time']).abs().argsort()[:1]].index[0]
            powerlog40_start_time= power_df['log_time'].iloc[powerlog40_start]

            #print("Transition point not detected for powermode: ",powermode)
            minibatch_index_list.append([cores, cpu, gpu, mem, -1,powerlog40_start_time,powerlog40_end_time, len(minibatch_df)-1])  # Append the index of the transition point

    print("undetected count: ", undetected_count)
    # Convert the list to a DataFrame
    print("error count is ", error_count)
    minibatch_index_list = pd.DataFrame(minibatch_index_list, columns=['cores', 'cpu', 'gpu','mem','skip_index','start_time','end_time','clean minibatches'])

    #convert to csv
    minibatch_index_list.to_csv("minibatch_index_list_yolo_train.csv", index=False)

path="tl/yolo"
#path="resnet_v4_runs_merged"
mb_filename="mn_nw4_pf2_epoch_stats.csv"
powerstats_filename="mn_nw4_pf2_tegrastats.csv"
all_powermodes=generate_powermodes()
#detect_jtop_issues(all_powermodes, path, powerstats_filename, mb_filename)
generate_index_list(all_powermodes,path,powerstats_filename,mb_filename)
