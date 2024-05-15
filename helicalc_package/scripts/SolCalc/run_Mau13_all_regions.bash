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
# run DSCylFMSAll_MetUnc
# python calculate_Mau13_single_region.py -r DSCylFMSAll_MetUnc0 -t n -i /home/sdittmer/Mu2E/mu2e/coord_metunc_0.p
# python calculate_Mau13_single_region.py -r DSCylFMSAll_MetUnc1 -t n -i /home/sdittmer/Mu2E/mu2e/coord_metunc_1.p
# python calculate_Mau13_single_region.py -r DSCylFMSAll_MetUnc2 -t n -i /home/sdittmer/Mu2E/mu2e/coord_metunc_2.p
# python calculate_Mau13_single_region.py -r DSCylFMSAll_MetUnc3 -t n -i /home/sdittmer/Mu2E/mu2e/coord_metunc_3.p
# python calculate_Mau13_single_region.py -r DSCylFMSAll_MetUnc4 -t n -i /home/sdittmer/Mu2E/mu2e/coord_metunc_4.p
# python calculate_Mau13_single_region.py -r DSCylFMSAll_MetUnc5 -t n -i /home/sdittmer/Mu2E/mu2e/coord_metunc_5.p
# python calculate_Mau13_single_region.py -r DSCylFMSAll_MetUnc6 -t n -i /home/sdittmer/Mu2E/mu2e/coord_metunc_6.p
# python calculate_Mau13_single_region.py -r DSCylFMSAll_MetUnc7 -t n -i /home/sdittmer/Mu2E/mu2e/coord_metunc_7.p
# python calculate_Mau13_single_region.py -r DSCylFMSAll_MetUnc8 -t n -i /home/sdittmer/Mu2E/mu2e/coord_metunc_8.p
# python calculate_Mau13_single_region.py -r DSCylFMSAll_MetUnc9 -t n -i /home/sdittmer/Mu2E/mu2e/coord_metunc_9.p
# run DSCylFine
# python calculate_Mau13_single_region.py -r DSCylFine -t n 
# run DSCartVal_grid
# python calculate_Mau13_single_region.py -r DSCartVal -t n

# read -p "Press any key to resume ..."
