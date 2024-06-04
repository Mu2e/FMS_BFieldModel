#!/bin/bash
# Run helicalc for all coils in one region

source $CONDA_PREFIX/etc/profile.d/conda.sh
conda activate helicalc

helicalc_data=$(python ../../get_data_dir.py)
logdir="${helicalc_data}/Bmaps/auxiliary_partial/logs/"

#region="DSCylFMSAll"
region="DSCartVal"

test="n"
# test="y"

time=$(date +"%Y-%m-%d_%H%M%S")

# run on each GPU, putting process in background
# no jacobian
python drive_busconnect.py -r ${region} -D 0 -t ${test} > ${logdir}${time}_GPU0_calculations_${region}_busconnect.log &
python drive_busconnect.py -r ${region} -D 1 -t ${test} > ${logdir}${time}_GPU1_calculations_${region}_busconnect.log &
python drive_busconnect.py -r ${region} -D 2 -t ${test} > ${logdir}${time}_GPU2_calculations_${region}_busconnect.log &
python drive_busconnect.py -r ${region} -D 3 -t ${test} > ${logdir}${time}_GPU3_calculations_${region}_busconnect.log &

# with jacobian
# python drive_busconnect.py -r ${region} -D 0 -j y -d 0.001 -t ${test} > ${logdir}${time}_GPU0_calculations_${region}_busconnect.log &
# python drive_busconnect.py -r ${region} -D 1 -j y -d 0.001 -t ${test} > ${logdir}${time}_GPU1_calculations_${region}_busconnect.log &
# python drive_busconnect.py -r ${region} -D 2 -j y -d 0.001 -t ${test} > ${logdir}${time}_GPU2_calculations_${region}_busconnect.log &
# python drive_busconnect.py -r ${region} -D 3 -j y -d 0.001 -t ${test} > ${logdir}${time}_GPU3_calculations_${region}_busconnect.log &

# read -p "Press any key to resume ..."
