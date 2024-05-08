# FMS_BField Model
This repository contains a collection of custom python packages used in the BField modeling efforts. Each package can be installed independently. Certain use cases may require one or more of these packages is installed. Each package has its own "environment.yml" file for installation via conda. Some packages share an environment as they share a common task (e.g. "BField_LSQ" and "BField_PINN" are both used during the modeling step).

The following packages are available, each in a directory appended with "_package" to distinguish from the python package name:
- `helicalc`: Biot-Savart integration for a variety of complex superconductor geometries. The calcualtion is GPU accelerated by utilizing the PyTorch package.
