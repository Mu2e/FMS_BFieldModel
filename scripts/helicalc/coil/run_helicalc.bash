#!/bin/bash
# Run helicalc for all coils in one region

source $CONDA_PREFIX/etc/profile.d/conda.sh
conda activate helicalc

logdir="/home/shared_data/Bmaps/helicalc_partial/logs/"

# region="DS"
region="TSd"

test="n"
# test="y"

time=$(date +"%Y-%m-%d_%H%M%S")

# run on each GPU, putting process in background
python drive_helicalc.py -r ${region} -D 0 -t ${test} > ${logdir}${time}_GPU0_calculations_${region}.log &
python drive_helicalc.py -r ${region} -D 1 -t ${test} > ${logdir}${time}_GPU1_calculations_${region}.log &
python drive_helicalc.py -r ${region} -D 2 -t ${test} > ${logdir}${time}_GPU2_calculations_${region}.log &
python drive_helicalc.py -r ${region} -D 3 -t ${test} > ${logdir}${time}_GPU3_calculations_${region}.log &

# read -p "Press any key to resume ..."
