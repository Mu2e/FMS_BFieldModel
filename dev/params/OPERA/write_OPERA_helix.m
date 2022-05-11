
% write_OPERA_helix.m
% write OPERA helix conductor file for a single coil
% Reads coil and conductor parameters and writes BR20 OPERA conductor file

% Assumes coil parameters are stored in array Coils (see Read_coils_tot.m)
% also assumes conductor parameters are stored in array Conductor_params
%   (see Read_conductors.m)
% need to specify inner layer helicity (for now)

% A helix is implemented in OPERA as a sequence of 20-node Bricks (BR20)
% Each brick spans 36 degrees, so there are 10 bricks per turn

% all length units are specified in meters

% get number of bricks/turn and opening angle of each brick
N_bricks = 10;
dphi_deg = 360/N_bricks;
dphi = dphi_deg*pi/180;

% get coil number corresponding to entry in Coil file
k = input('Enter coil number (1-66): ');

% get helicity of coil inner layer
hel0 = input('Enter helicity of innermost layer (+1 or -1): ');

% Unpack needed Coil information from Coils array
Ri = Coils(k,1);    % inner coil radius
L = Coils(k,3);     % coil length

Xc = Coils(k,4);    % x coordinate of coil center
Yc = Coils(k,5);    % y coordinate of coil center
Zc = Coils(k,6);    % z coordinate of coil center
Yaw = Coils(k,8);   % coil yaw angle (rotation about y axis, degrees)
yaw_rad = pi*Yaw./180.;

N_layers = Coils(k,11);     % number of coil layers
N_turns = Coils(k,12);      % number of turns per layer
I_turn = Coils(k,14);       % current in each turn

% Unpack additional geometrical information from Conductor_params array
h_cable = Conductor_params(k,1);    
w_cable = Conductor_params(k,2);
h_sc = Conductor_params(k,3);
w_sc = Conductor_params(k,4);
t_gi = Conductor_params(k,5);
t_ci = Conductor_params(k,6);
t_il = Conductor_params(k,7);
phi0_deg = Conductor_params(k,8);
phi1_deg = Conductor_params(k,9);

% get pitch length
hmin = w_cable + 2.*t_ci;  % minimum pitch length
h = (L - w_cable - 2.*t_gi)./N_turns;  % revised pitch length 4/9/19

% get current density
j = I_turn./(h_sc.*w_sc);

% open output file
filename = input('Enter new OPERA conductor filename (*.cond): ', 's');
[fileID, errmsg] = fopen(filename, 'w');
if (fileID < 0)
    disp(errmsg)
    return
end

% write file header
fprintf(fileID, 'CONDUCTOR\n');

% use coil name for OPERA drivelabel
coil_label = input('Enter coil name, eg, "DS-8": ', 's');

% use 1e-6 T (0.01 gauss) for absolute flux tolerance
tolerance = 1e-6;

% other constant OPERA conductor parameters
symmetry = 1;
irxy = 0;
iryz = 0;
irzx = 0;

% constant LCS1 and LCS2 parameters
phi1 = 0;
theta1 = 0;
psi1 = 0;
xcen2 = 0;
ycen2 = 0;
theta2 = 0;
psi2 = 0;

% base inner radius
rho0_a = Ri + (h_cable - h_sc)/2 + t_gi + t_ci;

% initialize node coordinates
xp = zeros(20,1);
yp = zeros(20,1);
zp = zeros(20,1);

% loop over layers
i_break = 0;
for k_layer = 1:N_layers
    % flip helicity on alternate layers
    hel = hel0.*(-1).^(k_layer-1);
    
    % get limits of brick in rho direction
    rho0 = rho0_a + (k_layer-1)*(h_cable + 2.*t_ci + t_il); 
    rho1 = rho0 + h_sc;
    
    % get base limits of brick in z direction
    zeta0 = (w_cable - w_sc)/2 + t_gi + t_ci;
    zeta1 = zeta0 + w_sc;
    
    % update drivelabel
    drivelabel = sprintf('%s_layer%d', coil_label, k_layer);
    
    % determine node coordinates 
    xp(1) = rho0;
    yp(1) = 0;
    zp(1) = zeta0;
    
    xp(2) = xp(1);
    yp(2) = 0;
    zp(2) = zeta1;
    
    xp(3) = rho1;
    yp(3) = 0;
    zp(3) = zp(2);
    
    xp(4) = xp(3);
    yp(4) = 0;
    zp(4) = zp(1);
    
    xp(5) = rho0.*cos(dphi);
    yp(5) = rho0.*sin(dphi);
    zp(5) = zeta0 + h.*dphi./(2*pi);
    
    xp(6) = xp(5);
    yp(6) = yp(5);
    zp(6) = zeta1 + h.*dphi./(2*pi);
    
    xp(7) = rho1.*cos(dphi);
    yp(7) = rho1.*sin(dphi);
    zp(7) = zp(6);
    
    xp(8) = xp(7);
    yp(8) = yp(7);
    zp(8) = zp(5);
    
    xp(9) = xp(1);
    yp(9) = 0;
    zp(9) = 0.5.*(zeta0+zeta1);
    
    xp(10) = 0.5.*(rho0+rho1);
    yp(10) = 0;
    zp(10) = zp(2);
    
    xp(11) = xp(3);
    yp(11) = 0;
    zp(11) = zp(9);
    
    xp(12) = xp(9);
    yp(12) = 0;
    zp(12) = zp(1);
    
    xp(13) = rho0.*cos(dphi/2);
    yp(13) = rho0.*sin(dphi/2);
    zp(13) = zeta0 + h.*dphi./(4*pi);
    
    xp(14) = xp(13);
    yp(14) = yp(13);
    zp(14) = zeta1 + h.*dphi./(4*pi);
    
    xp(15) = rho1.*cos(dphi/2);
    yp(15) = rho1.*sin(dphi/2);
    zp(15) = zp(14);
    
    xp(16) = xp(15);
    yp(16) = yp(15);
    zp(16) = zp(13);
    
    xp(17) = xp(5);
    yp(17) = yp(5);
    zp(17) = 0.5.*(zeta0+zeta1) + h.*dphi./(2*pi);
    
    xp(18) = 0.5.*(rho0+rho1).*cos(dphi);
    yp(18) = 0.5.*(rho0+rho1).*sin(dphi);
    zp(18) = zp(6);
    
    xp(19) = xp(7);
    yp(19) = yp(7);
    zp(19) = zp(17);
    
    xp(20) = xp(18);
    yp(20) = yp(18);
    zp(20) = zp(5);
        
    % get LCS1 parameters
    xcen1 = Xc;
    ycen1 = Yc;
    zcen1 = Zc -hel.*L/2;
    
    % loop over turns
    for j_turn = 1:N_turns
        fprintf('processing layer %3d, turn %3d\n', k_layer, j_turn); 
        
        % loop over bricks (loop segments)
        for i_brick = 1:N_bricks
            dphib_deg = (i_brick-1)*dphi_deg;
            dphib = dphib_deg*pi/180;
                      
            % update LCS2 parameters
            zcen2 = hel.*h*(j_turn-1 + dphib./(2*pi));
            phi2 = mod(phi0_deg + dphib_deg, 360);
			
			% handle special cases
			% interlayer connector - last brick of layer
			if ((k_layer < N_layers) && (j_turn == N_turns) && (i_brick == N_bricks))
				rho0p = rho0 + h_cable + 2*t_ci + t_il;
				rho1p = rho0p + h_sc;
				[xp, yp] = interlayer_connector(xp, yp, rho0, rho1, rho0p, rho1p, dphi);
			end
			% last layer, last turn, last brick
            if (k_layer == N_layers && j_turn == N_turns && mod(phi1_deg - phi2,360) <= dphi_deg)
                i_break = 1;
                dphis_deg = mod(phi1_deg - phi2,360);
                dphis = dphis_deg*pi/180;
                [xp, yp, zp] = last_brick(xp, yp, zp, h, rho0, rho1, zeta0, zeta1, dphis);
            end
            
            % write out conductor parameters
            fprintf(fileID, 'DEFINE BR20\n');
            fprintf(fileID, '%9.5f %9.5f %9.5f %4.1f %4.1f %4.1f\n', ...
                xcen1, ycen1, zcen1, phi1, theta1, psi1);
            fprintf(fileID, '%9.5f %9.5f %9.5f \n', xcen2, ycen2, zcen2);
            fprintf(fileID, '%4.1f %6.2f %4.1f\n', theta2, phi2, psi2);
            for i_node = 1:20
                fprintf(fileID, '%10.6f %10.6f %10.6f\n', xp(i_node), yp(i_node), hel.*zp(i_node));
            end
            
            fprintf(fileID, '%8.4e %3d ''%s''\n', j, symmetry, drivelabel);
            fprintf(fileID, '%2d %2d %2d\n', irxy,iryz,irzx);
            fprintf(fileID, '%8.4e\n', tolerance);
            if (i_break == 1)
                break
            end
        end
        if (i_break == 1)
            break
        end
    end
    if (i_break == 1)
        break
    end
end
        

% clean up
fprintf(fileID, 'QUIT\n');
fclose(fileID);


function [xp, yp] = interlayer_connector(xp, yp, rho0, rho1, rho0p, rho1p, dphi)
% calculate special node coordinates for interlayer connector brick

xp(5) = rho0p.*cos(dphi);
yp(5) = rho0p.*sin(dphi);
xp(6) = xp(5);
yp(6) = yp(5);
xp(7) = rho1p.*cos(dphi);
yp(7) = rho1p.*sin(dphi);
xp(8) = xp(7);
yp(8) = yp(8);

xp(13) = 0.5*(rho0+rho0p).*cos(dphi/2);
yp(13) = 0.5*(rho0+rho0p).*sin(dphi/2);
xp(14) = xp(13);
yp(14) = yp(13);
xp(15) = 0.5*(rho1+rho1p).*cos(dphi/2);
yp(15) = 0.5*(rho1+rho1p).*sin(dphi/2);
xp(16) = xp(15);
yp(16) = yp(16);

xp(17) = xp(5);
yp(17) = yp(5);
xp(18) = 0.5*(rho0p+rho1p).*cos(dphi);
yp(18) = 0.5*(rho0p+rho1p).*sin(dphi);
xp(19) = xp(7);
yp(19) = yp(7);
xp(20) = xp(18);
yp(20) = yp(18);

end

function [xp, yp, zp] = last_brick(xp, yp, zp, h, rho0, rho1, zeta0, zeta1, dphis)
% calculate special node coordinates for last brick
% last brick meets bus return at angle phi1_deg

xp(5) = rho0.*cos(dphis);
yp(5) = rho0.*sin(dphis);
zp(5) = zeta0 + h.*dphis./(2*pi);
xp(6) = xp(5);
yp(6) = yp(5);
zp(6) = zeta1 + h.*dphis./(2*pi);
xp(7) = rho1.*cos(dphis);
yp(7) = rho1.*sin(dphis);
zp(7) = zp(6);
xp(8) = xp(7);
yp(8) = yp(7);
zp(8) = zp(5);

xp(13) = rho0.*cos(dphis/2);
yp(13) = rho0.*sin(dphis/2);
zp(13) = zeta0 + h.*dphis./(4*pi);
xp(14) = xp(13);
yp(14) = yp(13);
zp(14) = zeta1 + h.*dphis./(4*pi);
xp(15) = rho1.*cos(dphis/2);
yp(15) = rho1.*sin(dphis/2);
zp(15) = zp(14);
xp(16) = xp(15);
yp(16) = yp(15);
zp(16) = zp(13);
 
xp(17) = xp(5);
yp(17) = yp(5);
zp(17) = 0.5.*(zeta0+zeta1) + h.*dphis./(2*pi);
xp(18) = 0.5.*(rho0+rho1).*cos(dphis);
yp(18) = 0.5.*(rho0+rho1).*sin(dphis);
zp(18) = zp(6);
xp(19) = xp(7);
yp(19) = yp(7);
zp(19) = zp(17);
xp(20) = xp(18);
yp(20) = yp(18);
zp(20) = zp(5);

end


