# The `helicalc` Package
Biot-Savart integration tools for calculating the magnetic field due to a helical solenoid. Note that `helicalc` includes the routines `helicalc.helicalc` for helical solenoids, as well as `helicalc.solcalc` for ideal solenoids. Additionally, there is a companion GUI for `helicalc.solcalc` to quickly calculate and visualize a sparse set of field points while varying the solenoid coil configuration. The current GUI implementation only includes the Mu2e PS coils.

# Installation
1. Create an Anaconda environment called `helicalc` using the supplied environment file.
```
(base) ~/helicalc$ conda env create -f environment.yml
```
Note: on an Intel i7 laptop with a strong internet connection, this step takes ~10 minutes. A lot of time is spent downloading and installing the `pytorch` library.

2. Activate the conda environment:
```
(base) ~/helicalc$ conda activate helicalc
(helicalc) ~/helicalc$
```

3. Create a symbolic link to the output data directory (replace "YOUR_DATA_DIR" with the directory on your computer where you want to store results):
```
(helicalc) ~/helicalc$ ln -s YOUR_DATA_DIR data
```

4. The SolCalc GUI requires a pickled pandas.DataFrame with field values without the PS coils. Please contact ckampa13@gmail.com if you need this file. The file is expected to be in the following location:
```
data/Bmaps/aux/Mau13.SolCalc.PS_region.standard.PSoff.pkl
```
where `data/` is the symbolic link created in the previous step.

3. Install the `helicalc` package in the conda environment:
```
(helicalc) ~/helicalc$ pip install -e .
```


Note: The "-e" flag installs the package in development mode. Any changes to package source code is automatically propagated to the package installation in the conda environment. If you do not plan to change the source code, you can install without "-e".

# SolCalc GUI
## Preferred / Automated Running
The installation will put the script `SolCalcGUI.py` in your path. From any location:
```
(helicalc) ~/$ SolCalcGUI.py

Dash is running on http://127.0.0.1:8050/

 * Serving Flask app "solcalc" (lazy loading)
 * Environment: production
   WARNING: This is a development server. Do not use it in a production deployment.
   Use a production WSGI server instead.
 * Debug mode: on
...
```

Point a web browser to `localhost:8050`, replacing `8050` with whichever port the server is running on.

## Run "by hand"
The `SolCalcGUI.py` script can also be run from the repo:
```
(helicalc) ~/helicalc/$ cd scripts/SolCalc_GUI
(helicalc) ~/helicalc/scripts/SolCalc_GUI/$ python SolCalcGUI.py
...
```
