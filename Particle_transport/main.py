import cupy
import numpy as np
from .cuda_kernels import *

class pic_transport():
    def __init__(self, pos, velocity, dt, charges, number_charges, masses, left_bound_x, left_bound_y, left_bound_z, nx, ny, nz, dx, dy, dz):
        '''
        params
        ======
        pos: {0: of shape [number of particles, 3], 1: of shape [number of particles, 3]}
        velocity: {0: of shape [number of particles, 3], 1: of shape [number of particles, 3]}
        dt: the time step of the evolution
        number_charges: each particle represent how many particles, of shape [number of particle species]
        charges: of shape [number of particle species]
        masses: of shape [number of particle species]
        '''
        
        # send pos and velocty to GPU
        self.pos, self.velocity = {}, {}
        for i_key in pos.keys():
            self.pos[i_key], self.velocity[i_key] = cupy.array(pos[i_key]), cupy.array(velocity[i_key])
        
        self.dt = dt
        self.masses, self.charges, self.number_charges = masses, charges, number_charges
        
        # Configure the blocks
        self.threadsperblock = 32
        
        self.left_bound_x, self.left_bound_y, self.left_bound_z = left_bound_x, left_bound_y, left_bound_z
        self.right_bound_x, self.right_bound_y, self.right_bound_z = nx*dx - left_bound_x, ny*dy - left_bound_y, nz*dz - left_bound_z

    def proceed_one_step(self, Ex, Ey, Ez, Bx, By, Bz, dx, dy, dz, nx, ny, nz):
        '''
        electric_field: on GPU from JefiGPU, of shape [nx, ny, nz, 3]
        magnetic_field: on GPU from JefiGPU, of shape [nx, ny, nz, 3]
        '''

        # loop through particle species
        
        for i_species in self.pos.keys():
            
            # take the arraies for each species
            num_particles = len(self.pos[i_species])

            # configure the grids
            blockspergrid = (num_particles + (self.threadsperblock - 1)) // self.threadsperblock
       
            updata_particle_info[blockspergrid, self.threadsperblock]\
            (num_particles, self.pos[i_species], self.velocity[i_species], nx, ny, nz,\
             Ex.reshape([nx, ny, nz]), Ey.reshape([nx, ny, nz]), Ez.reshape([nx, ny, nz]),\
             Bx.reshape([nx, ny, nz]), By.reshape([nx, ny, nz]), Bz.reshape([nx, ny, nz]),\
             dx, dy, dz, self.masses[i_species], self.charges[i_species], self.dt, self.left_bound_x, self.left_bound_y, self.left_bound_z, self.right_bound_x, self.right_bound_y, self.right_bound_z)
            
    def evaluate_rho_J(self, dx, dy, dz, nx, ny, nz):

        output_rho = 0.
        output_Jx = 0.
        output_Jy = 0.
        output_Jz = 0.
        dv = dx*dy*dz
        
        # loop through particle species
        for i_species in self.pos.keys():

            # take the arraies for each species
            num_particles = len(self.pos[i_species])

            # define empty rho J
            rho = cupy.zeros([nx, ny, nz])
            Jx = cupy.zeros([nx, ny, nz])
            Jy = cupy.zeros([nx, ny, nz])
            Jz = cupy.zeros([nx, ny, nz])
            
            # configure the grids
            blockspergrid = (num_particles + (self.threadsperblock - 1)) // self.threadsperblock
 
            find_rho_J[blockspergrid, self.threadsperblock]\
            (num_particles, self.pos[i_species], self.velocity[i_species], \
             dx, dy, dz, nx, ny, nz, self.charges[i_species],\
             rho, Jx, Jy, Jz, self.left_bound_x, self.left_bound_y, self.left_bound_z, self.right_bound_x, self.right_bound_y, self.right_bound_z)

            # since total rho and J are accumulations of all species, we need to add up these to output
            output_rho += rho/dv
            output_Jx += Jx/dv
            output_Jy += Jy/dv
            output_Jz += Jz/dv
  
        return output_rho, output_Jx, output_Jy, output_Jz

    def evaluate_n_nu(self, dx, dy, dz, nx, ny, nz):

        output_rho_n = 0.
        dv = dx*dy*dz
        
        # loop through particle species
        for i_species in self.pos.keys():

            # take the arraies for each species
            num_particles = len(self.pos[i_species])

            # define empty rho J
            rho_n = cupy.zeros([nx, ny, nz])
            
            # configure the grids
            blockspergrid = (num_particles + (self.threadsperblock - 1)) // self.threadsperblock

            find_n_nu[blockspergrid, self.threadsperblock]\
            (num_particles, self.pos[i_species], self.velocity[i_species], \
             dx, dy, dz, self.number_charges[i_species],\
             rho_n, self.left_bound_x, self.left_bound_y, self.left_bound_z, self.right_bound_x, self.right_bound_y, self.right_bound_z)

            # since total rho and J are accumulations of all species, we need to add up these to output
            output_rho_n += rho_n/dv
  
        return output_rho_n
