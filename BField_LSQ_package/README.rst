----
Mu2E
----

by Brian Pollack
updated by Cole Kampa and Susan Dittmer

Analysis software for the Mu2E experiment at Fermilab.  This software is built upon 'pandas',
and it is meant for examining the magnetic fields within the various solenoids.

Software included for data munging, plotting, fitting.

Check out http://brovercleveland.github.io/Mu2E/ for more detailed documentation (W.I.P)!

# Installation
1. Create an anaconda environemnt called `mu2eBFit` using the environment file in the first-level directory in the repository. This environment can be used to run the PINN and LSQ fits. Note if you've already created the environment for the PINN training, start at the command `conda activate mu2eBFit`.
```
(base) ~/FMS_BField_Model/$ conda env create -f environment_mu2eBFit.yml
```

2. Activate the conda environment:
```
(base) ~/FMS_BField_Model$ conda activate mu2eBFit
(mu2eBFit) ~/FMS_BField_Model$
```

3. Navigate to `BField_LSQ_package` and create a symbolic link to the output data directory (replace "YOUR_DATA_DIR" with the directory on your computer where you want to store results):
```
(mu2eBFit) ~/FMS_BField_Model$ cd BField_LSQ_package
(mu2eBFit) ~/FMS_BField_Model/BField_LSQ_package/$ ln -s YOUR_DATA_DIR data
```

4. Install the `mu2e` (LSQ) package in the conda environment:
```
(mu2eBFit) ~/FMS_BField_Model/BField_LSQ_package/$ pip install -e .
```
