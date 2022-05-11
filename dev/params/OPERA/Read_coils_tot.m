function [ Coils ] = Read_coils_tot( filename )
%Read_coils_tot Reads file that contains coil data
% include last 4 columns not read by Read_coils.m

%   "Coils" must be an n by 14 matrix in the format:
% Coils = [Ri Ro L origin rotation I_tot N_layers N_turns N_turns_tot I_turn]
%
% where:
% Ri: Inner radius of the coil (in meters),
% Ro: Outer radius of the coil (in meters),
% L: Length of the coil (in meters),
% origin: [X Y Z] the position of the center of the coil (in meters),
% rotation: [Alpha Beta Gamma]  the orientation of the coil axis (in deg),
% I_tot: Total current (in Amps),
% N_layers: number of layers in coil,
% N_turns: number of turns per layer,
% N_turns_tot: total number of turns (N_layers * N_turns)
% I_turn: current in one turn (I_tot / N_turns_tot)
%

CoilsD=load(filename);

Ri=CoilsD(:,1)./1e3;
Ro=CoilsD(:,2)./1e3;
L=CoilsD(:,3)./1e3;

I_tot=CoilsD(:,10);
position=[CoilsD(:,4) CoilsD(:,5) CoilsD(:,6)]./1e3;
rotation=[CoilsD(:,7) CoilsD(:,8) CoilsD(:,9)];

N_layers = CoilsD(:,11);
N_turns = CoilsD(:,12);
N_turns_tot = CoilsD(:,13);
I_turn = CoilsD(:,14);


Coils=[Ri Ro L position rotation I_tot N_layers N_turns N_turns_tot I_turn];


end

