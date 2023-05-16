# JefiPIC
JefiPIC is a 3-D particle-in-cell (PIC) code which solves the evolution and induced electromagnetic fields of the plasmas in open physics boundary. JefiPIC employs a integral method -- Jefimenko's equations to solve the electromgnetic fields and to speed up the operation rate, all of the computations are performed on GPU. The equations are in **Fexible Unit**.

## 1. The EMsolver Module
EMsolver implements the Jefimenko's equations on GPUs. 
If you use this module to perform electromagnetic calculations, please cite us via

(to be filled later)

> **To understand how EMsolver works, please refer to**

### Installation
To run EMsolver, the following packages need to be pre-installed:
  - Numba
  - Ray
  - cupy
  - matplotlib
  - cudatoolkit

To start up, create a conda environment and install CRBMG:
```
# create a new environment
$: conda create -n JefiPIC

# install relavant package sequentially
$: conda install numba
$: pip install -U ray
$: conda install cupy matplotlib
$: conda install jupyter nobteook

$ git clone https://github.com/Juenjie/JefiPIC
$ cd JefiPIC
$ jupyter notebook
```
**Note that the installation of Ray requires pip and compatible python versions! Usually this can be solved by using a lower version of Python**
Execute the test file ---  **'PIC-JefiGPU-v-rho.ipynb'** in the repository before any real tasks.

## 2. The Particle Module
Macro particles or for short particles, which describe the electrons or ions in similar phase space states are to control the total particle number into a proper range. Particles are moving at their own initial velocties and gradually pushed by the induced and external electromagnetic fields through the Newton-Lorentz force.

Particles are distributed by linear interpolation to smooth the discretization of space grids, and the particle distribution and their velocities are used to computed the total current and charge density distributions, which are the source to feed the Jefimenko's equations.

## 3. Usage via an example
The following codes demonstrate an axample of how to use JefiPIC.
```
from EMsolver.solver import EMsolver
from RBG_Maxwell.Plasma.utils import find_largest_time_steps
from Particle_transport.main import pic_transport
import cupy
import numpy as np
from RBG_Maxwell.Unit_conversion.main import unit_conversion
import time
import os
```
Suppose that the sources of charge density ρ and current density J are in region [[0, 1],[0, 251],[0, 111]] 1e-5 m, 
while the observational region of EM fields are in region [[0, 1],[0, 251],[0, 111]] 1e-5 m. 
Then we have
```
# the regions are seperated as the source region and the observation region
x_grid_size_o, y_grid_size_o, z_grid_size_o = 1,251,111
x_grid_size_s, y_grid_size_s, z_grid_size_s = 1,251,111

# the infinitesimals of the regions
# here the source region and observational region are overlap
x_left_bound_o, y_left_bound_o, z_left_bound_o = 0, 0, 0
x_left_bound_s, y_left_bound_s, z_left_bound_s = 0, 0, 0
dx_o, dy_o, dz_o = 1e-5m, 1e-5m, 1e-5m
dx_s, dy_s, dz_s = 1e-5m, 1e-5m, 1e-5m
```
JefiPIC employs Jefimenko's equations which involve integrations of the retarded time to obtain the electromagnetic fields. 
```
# Computing time
total_comp_time = 1.0 * 1.0e-9*conversion_table['second'] # unit in second
n_step = 5000
dt = total_comp_time/n_step # unit in second

# particle initial
initial_energy = 10 # electron velocity in unit of eV
initial_velocity = 1 * 3*10**8*np.sqrt(1 - (1 / (1 + initial_energy/510000)**2))*conversion_table['meter']/conversion_table['second'] # electron velocity in unit of m/s
total_charge = -0.5e-13 * conversion_table['Coulomb'] # unit in Coulomb
number_of_particles = int(total_charge/charges[0]) + 1 # Number of macro particles
```
The particles are initally put in a stripe with velocities only in the y-axis.
```
# Set the positions of all the particles in x, y, and z axes separately
pos_x = (0 + 1 * np.random.random(number_of_particles)) * dx
pos_y = (9 + 1 * np.random.random(number_of_particles)) * dy
pos_z = (5 + 101 * np.random.random(number_of_particles)) * dz
pos = {0: np.array([pos_x, pos_y, pos_z]).transpose()} # Positions of all the particles

# Set the velocities of all the particles in x, y, and z axes separately
v_x = np.zeros(number_of_particles)
v_y = initial_velocity * np.ones(number_of_particles)
v_z = np.zeros(number_of_particles)
velocity = {0: np.array([v_x, v_y, v_z]).transpose()} # Velocities of all the particles
```
We also choose the GPU '0'
```
i_GPU = '0'
```
Now we load the remote class
```
# set up the number of time snapshots to be saved
len_time_snapshots = min(find_largest_time_steps(dx_o, dy_o, dz_o, \
                                                 x_left_bound_o, y_left_bound_o, z_left_bound_o, \
                                                 dx_s, dy_s, dz_s, \
                                                 x_left_bound_s, y_left_bound_s, z_left_bound_s, \
                                                 x_grid_size_o, y_grid_size_o, z_grid_size_o, \
                                                 x_grid_size_s, y_grid_size_s, z_grid_size_s,\
                                                 dt, c), 10000)
       
# Initialize the partile transport module
PIC = pic_transport(pos, velocity, dt, charges, number_charges, masses, \
                    x_left_bound_o, y_left_bound_o, z_left_bound_o, nx, ny, nz, dx_s, dy_s, dz_s)
                    
# Initialize the electromagnetic field Jefimenko's equations solver
PIC_provider = EMsolver(len_time_snapshots, \
                        x_grid_size_o, y_grid_size_o, z_grid_size_o, \
                        x_grid_size_s, y_grid_size_s, z_grid_size_s, \
                        dx_o, dy_o, dz_o, x_left_bound_o, y_left_bound_o, z_left_bound_o, \
                        dx_s, dy_s, dz_s, x_left_bound_s, y_left_bound_s, z_left_bound_s, \
                        dt, epsilon0, c)
```
We can save the data of E and B, current density J, particle number density n, particle velocity v, and position x, and the execution will be
```
# Create the diagnosing list
electron_rho = []
EMfield = []
current_rho = []
pos_save = []
velocity_save = []
num_rho = []

# Start the time
start_time = time.time()

for i_step in range(n_steps):

    rho_to_Jefi, Jx_to_Jefi, Jy_to_Jefi, Jz_to_Jefi = PIC.evaluate_rho_J(dx, dy, dz, nx, ny, nz)
   
    rho, Jx, Jy, Jz = rho_to_Jefi.flatten(), \
                      Jx_to_Jefi.flatten(), \
                      Jy_to_Jefi.flatten(), \
                      Jz_to_Jefi.flatten()
    
    PIC_provider.Jefimenko_solver(rho, Jx, Jy, Jz)
    
    # atrack EM fields
    Ex, Ey, Ez, Bx, By, Bz = PIC_provider.acquire_EB_field()

    # proceed particle transport
    PIC.proceed_one_step(Ex, Ey, Ez, Bx, By, Bz, dx, dy, dz, nx, ny, nz)
    
    if i_step%sample_point == 0 or i_step == n_steps:
        current_rho.append([Jx_to_Jefi.get(), Jy_to_Jefi.get(), Jz_to_Jefi.get()])
        electron_rho.append(rho_to_Jefi.get())
        pos_save.append(PIC.pos[i_key].get())
        velocity_save.append(PIC.velocity[i_key].get())
        num_rho.append(PIC.evaluate_n_nu(dx, dy, dz, nx, ny, nz).get())
        EMfield.append([Ex.get(), Ey.get(), Ez.get(), Bx.get(), By.get(), Bz.get()])

# End the time
end_time = time.time()
t_time = end_time - start_time # Give the total computing time
```

## License
The package is coded by Jiannan Chen and Jun-Jie Zhang.

This package is free you can redistribute it and/or modify it under the terms of the Apache License Version 2.0, January 2004 (http://www.apache.org/licenses/).

For further questions and technical issues, please contact us at

chennn1994@alumni.sjtu.edu.cn (Jiannan Chen 陈剑楠) or zjacob@mail.ustc.edu.cn (Jun-Jie Zhang 张俊杰)

**File Structure**
```
JefiPIC
│   README.md 
│   LICENSE
│   setup.py 
│   test of EMsolver.ipynb
│
└───EMsolver
    │   cuda_functions.py
    │   region_distance.py
    │   slover.py
    │   __init__.py
    Paticle transport
    │   cuda_kernels.py
    │   main.py
    │   __init__.py
```
