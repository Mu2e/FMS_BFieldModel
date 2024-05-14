#!/bin/bash
# Run helicalc for all coils in one region

source $CONDA_PREFIX/etc/profile.d/conda.sh
conda activate helicalc

logdir=helicalc_data+"Bmaps/helicalc_partial/logs/"

# region="DS"
# region="TSd"
# region="DSCylFMS"
region="DSCylFMSAll"
# region="DSCartVal"
# region="DSCylFMSAll_MetUnc"
# itoy=$1

test="n"
# test="y"

time=$(date +"%Y-%m-%d_%H%M%S")

# run on each GPU, putting process in background
# no jacobian
# python drive_interlayer.py -r ${region} -D 0 -t ${test} > ${logdir}${time}_GPU0_calculations_${region}_interlayer.log &
# python drive_interlayer.py -r ${region} -D 1 -t ${test} > ${logdir}${time}_GPU1_calculations_${region}_interlayer.log &
# python drive_interlayer.py -r ${region} -D 2 -t ${test} > ${logdir}${time}_GPU2_calculations_${region}_interlayer.log &
# python drive_interlayer.py -r ${region} -D 3 -t ${test} > ${logdir}${time}_GPU3_calculations_${region}_interlayer.log &

# with jacobian
python drive_interlayer.py -r ${region} -D 0 -j y -d 0.001 -t ${test} > ${logdir}${time}_GPU0_calculations_${region}_interlayer.log &
python drive_interlayer.py -r ${region} -D 1 -j y -d 0.001 -t ${test} > ${logdir}${time}_GPU1_calculations_${region}_interlayer.log &
python drive_interlayer.py -r ${region} -D 2 -j y -d 0.001 -t ${test} > ${logdir}${time}_GPU2_calculations_${region}_interlayer.log &
python drive_interlayer.py -r ${region} -D 3 -j y -d 0.001 -t ${test} > ${logdir}${time}_GPU3_calculations_${region}_interlayer.log &

# MetUnc
# python drive_interlayer.py -r ${region}${itoy} -D 0 -t ${test} -i /home/sdittmer/Mu2E/mu2e/coord_metunc_${itoy}.p > ${logdir}${time}_GPU0_calculations_${region}${itoy}_interlayer.log &
# python drive_interlayer.py -r ${region}${itoy} -D 1 -t ${test} -i /home/sdittmer/Mu2E/mu2e/coord_metunc_${itoy}.p > ${logdir}${time}_GPU1_calculations_${region}${itoy}_interlayer.log &
# python drive_interlayer.py -r ${region}${itoy} -D 2 -t ${test} -i /home/sdittmer/Mu2E/mu2e/coord_metunc_${itoy}.p > ${logdir}${time}_GPU2_calculations_${region}${itoy}_interlayer.log &
# python drive_interlayer.py -r ${region}${itoy} -D 3 -t ${test} -i /home/sdittmer/Mu2E/mu2e/coord_metunc_${itoy}.p > ${logdir}${time}_GPU3_calculations_${region}${itoy}_interlayer.log &

# read -p "Press any key to resume ..."
