#!/bin/bash
# Run helicalc for all coils in one region

source $CONDA_PREFIX/etc/profile.d/conda.sh
conda activate helicalc

logdir="/home/shared_data/Bmaps/auxiliary_partial/logs/"

region="DS"
# region="TSd" # NOT NEEDED
# region="DSCylFMS"
# region="DSCylFMSAll"

rev="n"
# rev="y"

test="n"
# test="y"

time=$(date +"%Y-%m-%d_%H%M%S")

# run on each GPU, putting process in background
python drive_radial_coil.py -r ${region} -D 0 -R ${rev} -t ${test} > ${logdir}${time}_GPU0_calculations_${region}_radial_coil.log &
python drive_radial_coil.py -r ${region} -D 1 -R ${rev} -t ${test} > ${logdir}${time}_GPU1_calculations_${region}_radial_coil.log &
python drive_radial_coil.py -r ${region} -D 2 -R ${rev} -t ${test} > ${logdir}${time}_GPU2_calculations_${region}_radial_coil.log &
python drive_radial_coil.py -r ${region} -D 3 -R ${rev} -t ${test} > ${logdir}${time}_GPU3_calculations_${region}_radial_coil.log &

# read -p "Press any key to resume ..."
