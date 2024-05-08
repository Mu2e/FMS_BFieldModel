#!/bin/bash
# Run SolCalc for the different magnet regions

source $CONDA_PREFIX/etc/profile.d/conda.sh
conda activate helicalc

# run PS -- done
# python calculate_Mau13_single_region.py -r PS -t n
# run TSu
# python calculate_Mau13_single_region.py -r TSu -t n
# run TSd
# python calculate_Mau13_single_region.py -r TSd -t n
# run DS
#python calculate_Mau13_single_region.py -r DS -t n
# run PStoDumpArea
# python calculate_Mau13_single_region.py -r PStoDumpArea -t n
# run ProtonDumpArea
# python calculate_Mau13_single_region.py -r ProtonDumpArea -t n
# run DSCylFMS (BP)
# python calculate_Mau13_single_region.py -r DSCylFMS -t n
# run DSCylFMSAll (BP+SP)
# without Jacobian points
python calculate_Mau13_single_region.py -r DSCylFMSAll -j n -d 0.001 -t n
# with Jacobian points
python calculate_Mau13_single_region.py -r DSCylFMSAll -j y -d 0.001 -t n

# read -p "Press any key to resume ..."
