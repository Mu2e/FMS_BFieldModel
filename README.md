# FMS_BFieldModel
This repository contains a collection of custom python packages used in the BField modeling efforts. Each package can be installed independently. Certain use cases may require one or more of these packages is installed. Each package has its own "environment.yml" file for installation via conda. Some packages share an environment as they share a common task (e.g. "BField_LSQ" and "BFieldPINN" are both used during the modeling step).

The following packages are available, each in a directory appended with "_package" to distinguish from the python package name:
- `helicalc`: Biot-Savart integration for a variety of complex superconductor geometries. The calcualtion is GPU accelerated by utilizing the PyTorch package.
- `mu2e`: Analytical model functions with non-linear least squares fitting for describing magnetic field measuremetns. The model can incorporate many types of functions. The two functional forms used in the nominal model are a series solution to Laplace's equation in cylindrical coordinates (periodic behavior in $\phi$, $z$ and modified Bessel function of the first kind in $r$) and trivial Cartesian solutions to Laplace's equation.
  - Note that the package directory is called "BField_LSQ_package".
- `BFieldPINN`: Scalar and 3-output PINNs that obey Maxwell's equations in a source free region. Typically, the PINN is run on the residuals of the LSQ fit, and can caputre the remaining physical features of the field, e.g. features from the bus bars.
