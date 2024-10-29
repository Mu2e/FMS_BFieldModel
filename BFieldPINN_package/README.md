# BFieldPINN
Development for physics-informed neural network (PINN) to aid in modeling Mu2e magnetic field.

# Installation
1. Create an anaconda environemnt called `mu2eBFit` using the environment file in the first-level directory in the repository. This environment can be used to run the PINN and LSQ fits. Note if you've already created the environment for the LSQ fit, start at the command `conda activate mu2eBFit`.
```
(base) ~/FMS_BField_Model/$ conda env create -f environment_mu2eBFit.yml
```

2. Activate the conda environment:
```
(base) ~/FMS_BField_Model$ conda activate mu2eBFit
(mu2eBFit) ~/FMS_BField_Model$
```

3. Navigate to `BFieldPINN_package` and create a symbolic link to the output data directory (replace "YOUR_DATA_DIR" with the directory on your computer where you want to store results):
```
(mu2eBFit) ~/FMS_BField_Model$ cd BFieldPINN_package
(mu2eBFit) ~/FMS_BField_Model/BFieldPINN_package/$ ln -s YOUR_DATA_DIR data
```

4. Install the `BFieldPINN` package in the conda environment:
```
(mu2eBFit) ~/FMS_BField_Model/BFieldPINN_package/$ pip install -e .
```
