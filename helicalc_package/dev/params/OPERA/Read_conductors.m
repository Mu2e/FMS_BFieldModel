function [ Conductor_params ] = Read_conductors( filename )
% Read_conductors reads file that contains parameters of
%  conductor from which coils are wound

% Conductor_params is a matrix in the form
% Conductor_params = [h_cable w_cable h_sc w_sc t_gi t_ci phi0 phi1]

% parameters are stored in filename in mm,
% converted to meters after being loaded into memory

% parameters:
% h_cable: height (in rho direction) of Al stabilizer
% w_cable: width (in zeta direction) of Al stabilizer
% h_sc: height of rectangular envelope containing superconducting strands
% w_sc: width of this rectangular envelope
% t_gi: thickness of ground insulation surrounding entire coil
% t_ci = thickness of cable insulation surrounding Al stabilizer
% t_il = thickness between cable layers
% phi0: starting azimuthal angle of coil winding (degrees)
% phi1: ending azimuthal angle of coil winding (degrees)

ConductorD = load(filename);

h_cable = ConductorD(:,1)./1e3;
w_cable = ConductorD(:,2)./1e3;
h_sc = ConductorD(:,3)./1e3;
w_sc = ConductorD(:,4)./1e3;
t_gi = ConductorD(:,5)./1e3;
t_ci = ConductorD(:,6)./1e3;
t_il = ConductorD(:,7)./1e3;
phi0 = ConductorD(:,8);
phi1 = ConductorD(:,9);

Conductor_params = [h_cable w_cable h_sc w_sc t_gi t_ci t_il phi0 phi1];

end
