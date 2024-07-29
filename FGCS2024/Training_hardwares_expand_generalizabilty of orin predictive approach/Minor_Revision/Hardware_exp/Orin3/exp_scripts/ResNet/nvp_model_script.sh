#!/bin/bash

rm mn_nw*
rm log_file_*

cpu_cores="4"
cpu_frq="422400"
gpu_frq="318750000"
mem_frq="665600000"

python3 generate_nvpmodel.py $cpu_cores $cpu_frq $gpu_frq $mem_frq
sudo nvpmodel -m 14 &> log_file_pm_14
sudo jetson_clocks --fan
sudo jetson_clocks --show &>> log_file_pm_14