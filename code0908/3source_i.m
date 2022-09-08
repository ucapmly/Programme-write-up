% Heterogeneous Propagation Medium Example
%
% This example provides a simple demonstration of using k-Wave for the
% simulation and detection of the pressure field generated by an initial
% pressure distribution within a two-dimensional heterogeneous propagation
% medium. It builds on the Homogeneous Propagation Medium Example. 
%
% author: Bradley Treeby
% date: 30th June 2009
% last update: 29th April 2017
%  
% This function is part of the k-Wave Toolbox (http://www.k-wave.org)
% Copyright (C) 2009-2017 Bradley Treeby

% This file is part of k-Wave. k-Wave is free software: you can
% redistribute it and/or modify it under the terms of the GNU Lesser
% General Public License as published by the Free Software Foundation,
% either version 3 of the License, or (at your option) any later version.
% 
% k-Wave is distributed in the hope that it will be useful, but WITHOUT ANY
% WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
% FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for
% more details. 
% 
% You should have received a copy of the GNU Lesser General Public License
% along with k-Wave. If not, see <http://www.gnu.org/licenses/>. 

clearvars;
example_number = 1;
switch example_number
    case 1
        %input_args = {'PMLAlpha', 0};
        input_args = {'PMLInside', false};
    case 2
        %input_args = {'PMLAlpha', 1e6};  
        input_args = {'PMLInside', false};      
    case 3
        %input_args = {'PMLSize', 2};
        input_args = {'PMLInside', false};
    case 4
        input_args = {'PMLInside', false};
end

% =========================================================================
% SIMULATION
% =========================================================================

% create the computational grid
Nx = 128;           % number of grid points in the x (row) direction
Ny = 128;           % number of grid points in the y (column) direction
dx = 0.1e-3;        % grid point spacing in the x direction [m]
dy = 0.1e-3;        % grid point spacing in the y direction [m]
kgrid = kWaveGrid(Nx, dx, Ny, dy);

% define the properties of the propagation medium    
medium.sound_speed = 1500 * ones(Nx, Ny);   % [m/s]
medium.sound_speed(1:Nx/2, :) = 1800;       % [m/s]
medium.density = 1000 * ones(Nx, Ny);       % [kg/m^3]
medium.density(:, Ny/4:Ny) = 1200;          % [kg/m^3]

% create initial pressure distribution using makeDisc
disc_magnitude = 5; % [Pa]
disc_x_pos = 50;    % [grid points]
disc_y_pos = 50;    % [grid points]
disc_radius = 8;    % [grid points]
disc_1 = disc_magnitude * makeDisc(Nx, Ny, disc_x_pos, disc_y_pos, disc_radius);

disc_magnitude = 3; % [Pa]
disc_x_pos = 80;    % [grid points]
disc_y_pos = 60;    % [grid points]
disc_radius = 5;    % [grid points]
disc_2 = disc_magnitude * makeDisc(Nx, Ny, disc_x_pos, disc_y_pos, disc_radius);

disc_magnitude = 4; % [Pa]
disc_x_pos = 50;    % [grid points]
disc_y_pos = 10;    % [grid points]
disc_radius = 5;    % [grid points]
disc_3 = disc_magnitude * makeDisc(Nx, Ny, disc_x_pos, disc_y_pos, disc_radius);

source.p0 = disc_1 + disc_2 + disc_3;

% define a centered circular sensor
sensor_radius = 4e-3;   % [m]
num_sensor_points = 50;
%sensor.mask = makeCartCircle(sensor_radius, num_sensor_points);
sensor.mask = ones(Nx,Ny);


% run the simulation with optional inputs for plotting the simulation
% layout in addition to removing the PML from the display
sensor_data = kspaceFirstOrder2D(kgrid, medium, source, sensor, ...
    'PlotLayout', true, 'PlotPML', false, input_args{:});

% =========================================================================
% Data
% =========================================================================

funu=reshape(sensor_data,Nx,Ny,[]);
v_data01=funu(:,:,1);v_data02=funu(:,:,10);v_data03=funu(:,:,200);
v_data1=v_data01(:); v_data2=v_data02(:); v_data3=v_data03(:);
save datai3_1.txt v_data1 -ascii
save datai3_2.txt v_data2 -ascii
save datai3_3.txt v_data3 -ascii


d1=sensor_data(10*128+80:128:40*128+80,1:256)
d2=sensor_data(10*128+120:128:40*128+120,1:256)
d3=sensor_data(10*128+80:10*128+120,1:256)
d4=sensor_data(40*128+80:40*128+120,1:256)
sen=[d1;d2;d3;d4]
sensorlong=sen(:)
save sen_datai_3.txt sensorlong -ascii


free=sensor_data(1:128,1:256)
free_long=free(:)
save free_datai_3.txt free_long -ascii

% =========================================================================
% VISUALISATION
% =========================================================================

% plot the simulated sensor data
figure;
imagesc(sensor_data, [-1, 1]);
colormap(getColorMap);
ylabel('Sensor Position');
xlabel('Time Step');
colorbar;
figure, for j=1:300,imagesc(funu(:,:,j)),pause(0.01),end
%figure,imagesc(funu(:,:,2));
%figure,imagesc(funu(:,:,100));
%figure,imagesc(funu(:,:,200));
%figure,for i=1:31, plot(sen(i,:)),hold on, end;
%figure,for i=32:62, plot(sen(i,:)),hold on, end;
%figure,for i=63:103, plot(sen(i,:)),hold on, end;
%figure,for i=104:144, plot(sen(i,:)),hold on, end;
%figure,for i=1:128, plot(free(i,:)),hold on, end;