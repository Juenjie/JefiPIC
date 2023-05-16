import numpy as np
import math
import os
import cupy
from numba import cuda
import ray
from .cuda_functions import *

# give the index of where to copy the newly calculated J and rho in
# GPU memory of Jx_GPU, Jy_GPU, Jz_GPU, rho_GPU
# this is a host function
def time_index_in_GPU(i_time, len_time_snapshots):
    return i_time%len_time_snapshots

class EMsolver():
    def __init__(self, \
                 len_time_snapshots, \
                 x_grid_size_o, y_grid_size_o, z_grid_size_o, \
                 x_grid_size_s, y_grid_size_s, z_grid_size_s, \
                 dx_o, dy_o, dz_o, x_left_boundary_o, y_left_boundary_o, z_left_boundary_o, \
                 dx_s, dy_s, dz_s, x_left_boundary_s, y_left_boundary_s, z_left_boundary_s, \
                 dt, epsilon0, c):
        '''
        Main class for solving E and B from Jefimenko's equations.
        The calculated rho and J are always stored as the self.data.

        Params
        ======
        len_time_snapshots: int, the maximum time sequence that can be stored in GPU memory
        x_grid_size_o, y_grid_size_o, z_grid_size_o: number of grid sizes in the obervation region
        x_grid_size_s, y_grid_size_s, z_grid_size_s: number of grid sizes in the source region
        dx_o, dy_o, dz_o: infinitesimal difference of the spatial coordinates in the observation region
        dx_s, dy_s, dz_s: infinitesimal difference of the spatial coordinates in the source region
        x_left_boundary_o, y_left_boundary_o, z_left_boundary_o: the left boundary of the observation region
        x_left_boundary_s, y_left_boundary_s, z_left_boundary_s: the left boundary of the source region
        dt: time step, epsilon0: the numerical value of epsilon0 used in FU (Flexible Unit), 
        c: the numerical value of c used in FU (Flexible Unit)
        '''
        
        # save the variables in the class
        self.len_time_snapshots = len_time_snapshots
        self.dx_o, self.dy_o, self.dz_o = dx_o, dy_o, dz_o
        self.dx_s, self.dy_s, self.dz_s = dx_s, dy_s, dz_s
        self.x_left_boundary_o, self.y_left_boundary_o, self.z_left_boundary_o = x_left_boundary_o, y_left_boundary_o, z_left_boundary_o
        self.x_left_boundary_s, self.y_left_boundary_s, self.z_left_boundary_s = x_left_boundary_s, y_left_boundary_s, z_left_boundary_s
        self.x_grid_size_o, self.y_grid_size_o, self.z_grid_size_o = x_grid_size_o, y_grid_size_o, z_grid_size_o
        self.x_grid_size_s, self.y_grid_size_s, self.z_grid_size_s = x_grid_size_s, y_grid_size_s, z_grid_size_s
        self.dt = dt
        self.epsilon0 = epsilon0
        self.c = c
        
        
        # total grid numbers and total time ticks in GPU
        self.total_grid_o = x_grid_size_o*y_grid_size_o*z_grid_size_o     
        self.total_grid_s = x_grid_size_s*y_grid_size_s*z_grid_size_s        
        self.all_grids = self.total_grid_o*self.total_grid_s
   
        # Configure the blocks
        self.threadsperblock = 32
        # configure the grids
        self.blockspergrid_o = (self.total_grid_o + (self.threadsperblock - 1)) // self.threadsperblock
        self.blockspergrid_s = (self.total_grid_s + (self.threadsperblock - 1)) // self.threadsperblock
        self.blockspergrid_all = (self.all_grids + (self.threadsperblock - 1)) // self.threadsperblock
        
        # define device array of rho and J with zero initial values of the source region
        self.rho_GPU, self.Jx_GPU, self.Jy_GPU, self.Jz_GPU = \
        (cupy.zeros([len_time_snapshots, self.total_grid_s], dtype=np.float64) for _ in range(4))
        
        # coefficient for E and B when doing integration in the source region
        self.coeff_E = 1/(4*math.pi*epsilon0)*dx_s*dy_s*dz_s
        self.coeff_B = -1/(4*math.pi*epsilon0*c**2)*dx_s*dy_s*dz_s
        
        # time tick
        self.time_tick = 0
        
        
    def Jefimenko_solver(self, rho, Jx, Jy, Jz):
        '''
        Using Jefimenko's equation to solve for E and B with given rho and J of relavant times.
         
        Params
        ======
        rho, Jx, Jy, Jz: physical quantities of shape [total_grid_s]
        quasi_neutral: if quasi_neutral == 1, the system only considers the contribution of electric current
        
        Obtain
        ======
        Ex, Ey, Ez, Bx, By, Bz at time time_snapshots[i_time]
        '''
        
        # i_time: int, index at which to copy the data
        self.i_time = time_index_in_GPU(self.time_tick, self.len_time_snapshots)
        self.t = self.time_tick*self.dt
        self.time_tick += 1
  
        # copy newly calculated rho and J to GPU
        self.rho_GPU[self.i_time], self.Jx_GPU[self.i_time], self.Jy_GPU[self.i_time], self.Jz_GPU[self.i_time] = rho, Jx, Jy, Jz
        
        # define device array of E and B with zero initial values
        GEx, GEy, GEz, GBx, GBy, GBz = (cupy.zeros(self.total_grid_o, dtype=np.float64) for _ in range(6))        
        # calculate E and B
        Jefimenko_kernel[self.blockspergrid_o, self.threadsperblock](self.rho_GPU, self.Jx_GPU, self.Jy_GPU, self.Jz_GPU, \
                                                                     self.len_time_snapshots, self.i_time, self.dt, self.t, \
                                                                     self.total_grid_o, self.total_grid_s, \
                                                                     self.x_grid_size_o, self.y_grid_size_o, self.z_grid_size_o, \
                                                                     self.x_grid_size_s, self.y_grid_size_s, self.z_grid_size_s, \
                                                                     self.dx_o, self.dy_o, self.dz_o, \
                                                                     self.dx_s, self.dy_s, self.dz_s, \
                                                                     self.x_left_boundary_o, self.y_left_boundary_o, self.z_left_boundary_o, \
                                                                     self.x_left_boundary_s, self.y_left_boundary_s, self.z_left_boundary_s, \
                                                                     GEx, GEy, GEz, GBx, GBy, GBz, self.coeff_E, self.coeff_B, self.c)
            
    
        # self.GEx, self.GEy, self.GEz, self.GBx, self.GBy, self.GBz = GEx*self.coeff_E, GEy*self.coeff_E, GEz*self.coeff_E, GBx*self.coeff_B, GBy*self.coeff_B, GBz*self.coeff_B
        self.GEx, self.GEy, self.GEz, self.GBx, self.GBy, self.GBz = GEx*self.coeff_E, GEy*self.coeff_E, GEz*self.coeff_E, GBx*self.coeff_B, GBy*self.coeff_B, GBz*self.coeff_B
    
    def terminate(self):
        '''Terminate the entire context. After the calling, GPU cannot be used in this class'''
        cuda.close()
        
    def acquire_EB_field(self):
        '''Acquire EB field from GPU'''
        
        return self.GEx, self.GEy, self.GEz, self.GBx, self.GBy, self.GBz