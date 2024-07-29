#!/bin/bash
 
prefetch_factor=2
no_workers=(4)
external_device="nvme0n1"
batch_size=16
 
rm mn_nw*
rm log_file_*
 
#cpu_cores_values=(2 4 6)
#cpu_frq_values=( )
#gpu_frq_values=(306000000 510000000 624750000)
#mem_frq_values=(2133000000 3199000000 665600000)

#cpu freq completed = (268800 422400 576000 729600 883200 1036800 1190400 1344000 1497600 1510400)
 
cpu_cores_values=(2 4 6)
cpu_frq_values=(422400)
gpu_frq_values=(624750000 510000000 306000000)
mem_frq_values=(2133000000 665600000)
 
 
for cpu_frq in "${cpu_frq_values[@]}"; do
    for gpu_frq in "${gpu_frq_values[@]}"; do
        for cpu_cores in "${cpu_cores_values[@]}"; do
            for mem_frq in "${mem_frq_values[@]}"; do
                python3 generate_nvpmodel.py $cpu_cores $cpu_frq $gpu_frq $mem_frq
                sudo nvpmodel -m 14 &> log_file_pm_14
                sudo jetson_clocks --fan
                sudo jetson_clocks --show &>> log_file_pm_14
                python3 resnet.py "/media/ssd" $no_workers $prefetch_factor $external_device > train_log_file_pm_14
                sudo pkill python
 
                mkdir "pm_${cpu_cores}_${cpu_frq}_${gpu_frq}_${mem_frq}"
                mv mn_nw* "pm_${cpu_cores}_${cpu_frq}_${gpu_frq}_${mem_frq}"
                mv log_file_* "pm_${cpu_cores}_${cpu_frq}_${gpu_frq}_${mem_frq}"
                mv train_log_file_* "pm_${cpu_cores}_${cpu_frq}_${gpu_frq}_${mem_frq}"
            done
        done
    done
done