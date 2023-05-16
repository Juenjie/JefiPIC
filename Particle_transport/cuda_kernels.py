import cupy
from numba import cuda

@cuda.jit
def find_rho_J(num_particles, pos, velocity, dx, dy, dz, nx, ny, nz, charge, rho, Jx, Jy, Jz, left_bound_x, left_bound_y, left_bound_z, right_bound_x, right_bound_y, right_bound_z):
    
    # threads loop in one dimension
    i_grid = cuda.grid(1)
    if i_grid < num_particles:
        
        # the particles must be in the box
        if pos[i_grid, 0] > left_bound_x and pos[i_grid, 0] < right_bound_x and pos[i_grid, 1] > left_bound_y and pos[i_grid, 1] < right_bound_y and pos[i_grid, 2] > left_bound_z and pos[i_grid, 2] < right_bound_z:
            
            
            # electromagnetic fields and currents are all set at the corner of the grid
            # we first need to figure out the regions the particle locates, like the corner? the boundary? or the main region?
            # we allocate the three directions to three regions, namely 0: pos_i(i=x,y,z) < 0.5*di(i=x,y,z), 1: pos_i>0.5*di, 2: pos_i+0.5*di>ni*di
            # thus, there are 27 regions for us to distinguish, and 26 of them are the boundary conditions. To avoid the much more conditions, we adopt
            # the following method. The lower boundary is different from the upper one, so we separate them into two conditions
            
            # find the index of positions in electric_field and magnetic_fleid.

            ind_x = int((pos[i_grid, 0] - left_bound_x - 0.5 * dx)//dx)
            ind_y = int((pos[i_grid, 1] - left_bound_y - 0.5 * dy)//dy)
            ind_z = int((pos[i_grid, 2] - left_bound_z - 0.5 * dz)//dz)
                         
            q1 = (pos[i_grid, 0] - (ind_x + 0.5) * dx)/dx
            q2 = (pos[i_grid, 1] - (ind_y + 0.5) * dy)/dy
            q3 = (pos[i_grid, 2] - (ind_z + 0.5) * dz)/dz                                     
                

#             if ind_x >= 0 and ind_y >= 0 and ind_z >= 0: # the particles belong to the regions with no lower boundary
#                 cuda.atomic.add(rho, (ind_x, ind_y, ind_z), charge*(1-q1)*(1-q2)*(1-q3))
#                 cuda.atomic.add(Jx, (ind_x, ind_y, ind_z), charge*velocity[i_grid, 0]*(1-q1)*(1-q2)*(1-q3))
#                 cuda.atomic.add(Jy, (ind_x, ind_y, ind_z), charge*velocity[i_grid, 1]*(1-q1)*(1-q2)*(1-q3))
#                 cuda.atomic.add(Jz, (ind_x, ind_y, ind_z), charge*velocity[i_grid, 2]*(1-q1)*(1-q2)*(1-q3))
#                 if ind_x+1 < nx:
#                     cuda.atomic.add(rho, (ind_x+1, ind_y, ind_z), charge*q1*(1-q2)*(1-q3))
#                     cuda.atomic.add(Jx, (ind_x+1, ind_y, ind_z), charge*velocity[i_grid, 0]*q1*(1-q2)*(1-q3))
#                     cuda.atomic.add(Jy, (ind_x+1, ind_y, ind_z), charge*velocity[i_grid, 1]*q1*(1-q2)*(1-q3))
#                     cuda.atomic.add(Jz, (ind_x+1, ind_y, ind_z), charge*velocity[i_grid, 2]*q1*(1-q2)*(1-q3))
#                 if ind_y+1 < ny:
#                     cuda.atomic.add(rho, (ind_x, ind_y+1, ind_z), charge*(1-q1)*q2*(1-q3))
#                     cuda.atomic.add(Jx, (ind_x, ind_y+1, ind_z), charge*velocity[i_grid, 0]*(1-q1)*q2*(1-q3))
#                     cuda.atomic.add(Jy, (ind_x, ind_y+1, ind_z), charge*velocity[i_grid, 1]*(1-q1)*q2*(1-q3))
#                     cuda.atomic.add(Jz, (ind_x, ind_y+1, ind_z), charge*velocity[i_grid, 2]*(1-q1)*q2*(1-q3))
#                 if ind_x+1 < nx and ind_y+1 < ny:
#                     cuda.atomic.add(rho, (ind_x+1, ind_y+1, ind_z), charge*q1*q2*(1-q3))
#                     cuda.atomic.add(Jx, (ind_x+1, ind_y+1, ind_z), charge*velocity[i_grid, 0]*q1*q2*(1-q3))
#                     cuda.atomic.add(Jy, (ind_x+1, ind_y+1, ind_z), charge*velocity[i_grid, 1]*q1*q2*(1-q3))
#                     cuda.atomic.add(Jz, (ind_x+1, ind_y+1, ind_z), charge*velocity[i_grid, 2]*q1*q2*(1-q3))
#                 if ind_z+1 < nz:
#                     cuda.atomic.add(rho, (ind_x, ind_y, ind_z+1), charge*(1-q1)*(1-q2)*q3)
#                     cuda.atomic.add(Jx, (ind_x, ind_y, ind_z+1), charge*velocity[i_grid, 0]*(1-q1)*(1-q2)*q3)
#                     cuda.atomic.add(Jy, (ind_x, ind_y, ind_z+1), charge*velocity[i_grid, 1]*(1-q1)*(1-q2)*q3)
#                     cuda.atomic.add(Jz, (ind_x, ind_y, ind_z+1), charge*velocity[i_grid, 2]*(1-q1)*(1-q2)*q3)
#                 if ind_x+1 < nx and ind_z+1 < nz:
#                     cuda.atomic.add(rho, (ind_x+1, ind_y, ind_z+1), charge*q1*(1-q2)*q3)
#                     cuda.atomic.add(Jx, (ind_x+1, ind_y, ind_z+1), charge*velocity[i_grid, 0]*q1*(1-q2)*q3)
#                     cuda.atomic.add(Jy, (ind_x+1, ind_y, ind_z+1), charge*velocity[i_grid, 1]*q1*(1-q2)*q3)
#                     cuda.atomic.add(Jz, (ind_x+1, ind_y, ind_z+1), charge*velocity[i_grid, 2]*q1*(1-q2)*q3)
#                 if ind_y+1 < ny and ind_z+1 < nz:
#                     cuda.atomic.add(rho, (ind_x, ind_y+1, ind_z+1), charge*(1-q1)*q2*q3)
#                     cuda.atomic.add(Jx, (ind_x, ind_y+1, ind_z+1), charge*velocity[i_grid, 0]*(1-q1)*q2*q3)
#                     cuda.atomic.add(Jy, (ind_x, ind_y+1, ind_z+1), charge*velocity[i_grid, 1]*(1-q1)*q2*q3)
#                     cuda.atomic.add(Jz, (ind_x, ind_y+1, ind_z+1), charge*velocity[i_grid, 2]*(1-q1)*q2*q3)
#                 if ind_x+1 < nx and ind_y+1 < ny and ind_z+1 < nz:
#                     cuda.atomic.add(rho, (ind_x+1, ind_y+1, ind_z+1), charge*q1*q2*q3)
#                     cuda.atomic.add(Jx, (ind_x+1, ind_y+1, ind_z+1), charge*velocity[i_grid, 0]*q1*q2*q3)
#                     cuda.atomic.add(Jy, (ind_x+1, ind_y+1, ind_z+1), charge*velocity[i_grid, 1]*q1*q2*q3)
#                     cuda.atomic.add(Jz, (ind_x+1, ind_y+1, ind_z+1), charge*velocity[i_grid, 2]*q1*q2*q3)
#             else: # regions with at least one lower boundary
#                 if ind_x != nx - 1 and ind_y != ny - 1 and ind_z != nz - 1:
#                     cuda.atomic.add(rho, (ind_x+1, ind_y+1, ind_z+1), charge*q1*q2*q3)
#                     cuda.atomic.add(Jx, (ind_x+1, ind_y+1, ind_z+1), charge*velocity[i_grid, 0]*q1*q2*q3)
#                     cuda.atomic.add(Jy, (ind_x+1, ind_y+1, ind_z+1), charge*velocity[i_grid, 1]*q1*q2*q3)
#                     cuda.atomic.add(Jz, (ind_x+1, ind_y+1, ind_z+1), charge*velocity[i_grid, 2]*q1*q2*q3)
#                 if ind_x != -1:
#                     cuda.atomic.add(rho, (ind_x, ind_y+1, ind_z+1), charge*(1-q1)*q2*q3)
#                     cuda.atomic.add(Jx, (ind_x, ind_y+1, ind_z+1), charge*velocity[i_grid, 0]*(1-q1)*q2*q3)
#                     cuda.atomic.add(Jy, (ind_x, ind_y+1, ind_z+1), charge*velocity[i_grid, 1]*(1-q1)*q2*q3)
#                     cuda.atomic.add(Jz, (ind_x, ind_y+1, ind_z+1), charge*velocity[i_grid, 2]*(1-q1)*q2*q3)
#                 if ind_y != -1:
#                     cuda.atomic.add(rho, (ind_x+1, ind_y, ind_z+1), charge*q1*(1-q2)*q3)
#                     cuda.atomic.add(Jx, (ind_x+1, ind_y, ind_z+1), charge*velocity[i_grid, 0]*q1*(1-q2)*q3)
#                     cuda.atomic.add(Jy, (ind_x+1, ind_y, ind_z+1), charge*velocity[i_grid, 1]*q1*(1-q2)*q3)
#                     cuda.atomic.add(Jz, (ind_x+1, ind_y, ind_z+1), charge*velocity[i_grid, 2]*q1*(1-q2)*q3)
#                 if ind_x != -1 and ind_y != -1:
#                     cuda.atomic.add(rho, (ind_x, ind_y, ind_z+1), charge*(1-q1)*(1-q2)*q3)
#                     cuda.atomic.add(Jx, (ind_x, ind_y, ind_z+1), charge*velocity[i_grid, 0]*(1-q1)*(1-q2)*q3)
#                     cuda.atomic.add(Jy, (ind_x, ind_y, ind_z+1), charge*velocity[i_grid, 1]*(1-q1)*(1-q2)*q3)
#                     cuda.atomic.add(Jz, (ind_x, ind_y, ind_z+1), charge*velocity[i_grid, 2]*(1-q1)*(1-q2)*q3)
#                 if ind_z != -1:
#                     cuda.atomic.add(rho, (ind_x+1, ind_y+1, ind_z), charge*q1*q2*(1-q3))
#                     cuda.atomic.add(Jx, (ind_x+1, ind_y+1, ind_z), charge*velocity[i_grid, 0]*q1*q2*(1-q3))
#                     cuda.atomic.add(Jy, (ind_x+1, ind_y+1, ind_z), charge*velocity[i_grid, 1]*q1*q2*(1-q3))
#                     cuda.atomic.add(Jz, (ind_x+1, ind_y+1, ind_z), charge*velocity[i_grid, 2]*q1*q2*(1-q3))
#                 if ind_x != -1 and ind_z != -1:
#                     cuda.atomic.add(rho, (ind_x, ind_y+1, ind_z), charge*(1-q1)*q2*(1-q3))
#                     cuda.atomic.add(Jx, (ind_x, ind_y+1, ind_z), charge*velocity[i_grid, 0]*(1-q1)*q2*(1-q3))
#                     cuda.atomic.add(Jy, (ind_x, ind_y+1, ind_z), charge*velocity[i_grid, 1]*(1-q1)*q2*(1-q3))
#                     cuda.atomic.add(Jz, (ind_x, ind_y+1, ind_z), charge*velocity[i_grid, 2]*(1-q1)*q2*(1-q3))
#                 if ind_y != -1 and ind_z != -1:
#                     cuda.atomic.add(rho, (ind_x+1, ind_y, ind_z), charge*q1*(1-q2)*(1-q3))
#                     cuda.atomic.add(Jx, (ind_x+1, ind_y, ind_z), charge*velocity[i_grid, 0]*q1*(1-q2)*(1-q3))
#                     cuda.atomic.add(Jy, (ind_x+1, ind_y, ind_z), charge*velocity[i_grid, 1]*q1*(1-q2)*(1-q3))
#                     cuda.atomic.add(Jz, (ind_x+1, ind_y, ind_z), charge*velocity[i_grid, 2]*q1*(1-q2)*(1-q3))
            
            
            if ind_x > -1 and ind_x < nx - 1 and ind_y > -1 and ind_y < ny - 1 and ind_z > -1 and ind_z < nz - 1: #(1,1,1)
                cuda.atomic.add(rho, (ind_x, ind_y, ind_z), charge*(1-q1)*(1-q2)*(1-q3))
                cuda.atomic.add(Jx, (ind_x, ind_y, ind_z), charge*velocity[i_grid, 0]*(1-q1)*(1-q2)*(1-q3))
                cuda.atomic.add(Jy, (ind_x, ind_y, ind_z), charge*velocity[i_grid, 1]*(1-q1)*(1-q2)*(1-q3))
                cuda.atomic.add(Jz, (ind_x, ind_y, ind_z), charge*velocity[i_grid, 2]*(1-q1)*(1-q2)*(1-q3))
                cuda.atomic.add(rho, (ind_x+1, ind_y, ind_z), charge*q1*(1-q2)*(1-q3))
                cuda.atomic.add(Jx, (ind_x+1, ind_y, ind_z), charge*velocity[i_grid, 0]*q1*(1-q2)*(1-q3))
                cuda.atomic.add(Jy, (ind_x+1, ind_y, ind_z), charge*velocity[i_grid, 1]*q1*(1-q2)*(1-q3))
                cuda.atomic.add(Jz, (ind_x+1, ind_y, ind_z), charge*velocity[i_grid, 2]*q1*(1-q2)*(1-q3))
                cuda.atomic.add(rho, (ind_x, ind_y+1, ind_z), charge*(1-q1)*q2*(1-q3))
                cuda.atomic.add(Jx, (ind_x, ind_y+1, ind_z), charge*velocity[i_grid, 0]*(1-q1)*q2*(1-q3))
                cuda.atomic.add(Jy, (ind_x, ind_y+1, ind_z), charge*velocity[i_grid, 1]*(1-q1)*q2*(1-q3))
                cuda.atomic.add(Jz, (ind_x, ind_y+1, ind_z), charge*velocity[i_grid, 2]*(1-q1)*q2*(1-q3))
                cuda.atomic.add(rho, (ind_x, ind_y, ind_z+1), charge*(1-q1)*(1-q2)*q3)
                cuda.atomic.add(Jx, (ind_x, ind_y, ind_z+1), charge*velocity[i_grid, 0]*(1-q1)*(1-q2)*q3)
                cuda.atomic.add(Jy, (ind_x, ind_y, ind_z+1), charge*velocity[i_grid, 1]*(1-q1)*(1-q2)*q3)
                cuda.atomic.add(Jz, (ind_x, ind_y, ind_z+1), charge*velocity[i_grid, 2]*(1-q1)*(1-q2)*q3)
                cuda.atomic.add(rho, (ind_x+1, ind_y+1, ind_z), charge*q1*q2*(1-q3))
                cuda.atomic.add(Jx, (ind_x+1, ind_y+1, ind_z), charge*velocity[i_grid, 0]*q1*q2*(1-q3))
                cuda.atomic.add(Jy, (ind_x+1, ind_y+1, ind_z), charge*velocity[i_grid, 1]*q1*q2*(1-q3))
                cuda.atomic.add(Jz, (ind_x+1, ind_y+1, ind_z), charge*velocity[i_grid, 2]*q1*q2*(1-q3))              
                cuda.atomic.add(rho, (ind_x, ind_y+1, ind_z+1), charge*(1-q1)*q2*q3)
                cuda.atomic.add(Jx, (ind_x, ind_y+1, ind_z+1), charge*velocity[i_grid, 0]*(1-q1)*q2*q3)
                cuda.atomic.add(Jy, (ind_x, ind_y+1, ind_z+1), charge*velocity[i_grid, 1]*(1-q1)*q2*q3)
                cuda.atomic.add(Jz, (ind_x, ind_y+1, ind_z+1), charge*velocity[i_grid, 2]*(1-q1)*q2*q3)
                cuda.atomic.add(rho, (ind_x+1, ind_y, ind_z+1), charge*q1*(1-q2)*q3)
                cuda.atomic.add(Jx, (ind_x+1, ind_y, ind_z+1), charge*velocity[i_grid, 0]*q1*(1-q2)*q3)
                cuda.atomic.add(Jy, (ind_x+1, ind_y, ind_z+1), charge*velocity[i_grid, 1]*q1*(1-q2)*q3)
                cuda.atomic.add(Jz, (ind_x+1, ind_y, ind_z+1), charge*velocity[i_grid, 2]*q1*(1-q2)*q3)
                cuda.atomic.add(rho, (ind_x+1, ind_y+1, ind_z+1), charge*q1*q2*q3)
                cuda.atomic.add(Jx, (ind_x+1, ind_y+1, ind_z+1), charge*velocity[i_grid, 0]*q1*q2*q3)
                cuda.atomic.add(Jy, (ind_x+1, ind_y+1, ind_z+1), charge*velocity[i_grid, 1]*q1*q2*q3)
                cuda.atomic.add(Jz, (ind_x+1, ind_y+1, ind_z+1), charge*velocity[i_grid, 2]*q1*q2*q3)
            elif ind_x > -1 and ind_x < nx - 1 and ind_y > -1 and ind_y < ny - 1 and ind_z == (nz - 1): #(1,1,2)
                cuda.atomic.add(rho, (ind_x, ind_y, ind_z), charge*(1-q1)*(1-q2)*(1-q3))
                cuda.atomic.add(Jx, (ind_x, ind_y, ind_z), charge*velocity[i_grid, 0]*(1-q1)*(1-q2)*(1-q3))
                cuda.atomic.add(Jy, (ind_x, ind_y, ind_z), charge*velocity[i_grid, 1]*(1-q1)*(1-q2)*(1-q3))
                cuda.atomic.add(Jz, (ind_x, ind_y, ind_z), charge*velocity[i_grid, 2]*(1-q1)*(1-q2)*(1-q3))
                cuda.atomic.add(rho, (ind_x+1, ind_y, ind_z), charge*q1*(1-q2)*(1-q3))
                cuda.atomic.add(Jx, (ind_x+1, ind_y, ind_z), charge*velocity[i_grid, 0]*q1*(1-q2)*(1-q3))
                cuda.atomic.add(Jy, (ind_x+1, ind_y, ind_z), charge*velocity[i_grid, 1]*q1*(1-q2)*(1-q3))
                cuda.atomic.add(Jz, (ind_x+1, ind_y, ind_z), charge*velocity[i_grid, 2]*q1*(1-q2)*(1-q3))
                cuda.atomic.add(rho, (ind_x, ind_y+1, ind_z), charge*(1-q1)*q2*(1-q3))
                cuda.atomic.add(Jx, (ind_x, ind_y+1, ind_z), charge*velocity[i_grid, 0]*(1-q1)*q2*(1-q3))
                cuda.atomic.add(Jy, (ind_x, ind_y+1, ind_z), charge*velocity[i_grid, 1]*(1-q1)*q2*(1-q3))
                cuda.atomic.add(Jz, (ind_x, ind_y+1, ind_z), charge*velocity[i_grid, 2]*(1-q1)*q2*(1-q3))
                cuda.atomic.add(rho, (ind_x+1, ind_y+1, ind_z), charge*q1*q2*(1-q3))
                cuda.atomic.add(Jx, (ind_x+1, ind_y+1, ind_z), charge*velocity[i_grid, 0]*q1*q2*(1-q3))
                cuda.atomic.add(Jy, (ind_x+1, ind_y+1, ind_z), charge*velocity[i_grid, 1]*q1*q2*(1-q3))
                cuda.atomic.add(Jz, (ind_x+1, ind_y+1, ind_z), charge*velocity[i_grid, 2]*q1*q2*(1-q3))  
            elif ind_x > -1 and ind_x < nx - 1 and ind_y == (ny - 1) and ind_z > -1 and ind_z < nz - 1: #(1,2,1)
                cuda.atomic.add(rho, (ind_x, ind_y, ind_z), charge*(1-q1)*(1-q2)*(1-q3))
                cuda.atomic.add(Jx, (ind_x, ind_y, ind_z), charge*velocity[i_grid, 0]*(1-q1)*(1-q2)*(1-q3))
                cuda.atomic.add(Jy, (ind_x, ind_y, ind_z), charge*velocity[i_grid, 1]*(1-q1)*(1-q2)*(1-q3))
                cuda.atomic.add(Jz, (ind_x, ind_y, ind_z), charge*velocity[i_grid, 2]*(1-q1)*(1-q2)*(1-q3))
                cuda.atomic.add(rho, (ind_x+1, ind_y, ind_z), charge*q1*(1-q2)*(1-q3))
                cuda.atomic.add(Jx, (ind_x+1, ind_y, ind_z), charge*velocity[i_grid, 0]*q1*(1-q2)*(1-q3))
                cuda.atomic.add(Jy, (ind_x+1, ind_y, ind_z), charge*velocity[i_grid, 1]*q1*(1-q2)*(1-q3))
                cuda.atomic.add(Jz, (ind_x+1, ind_y, ind_z), charge*velocity[i_grid, 2]*q1*(1-q2)*(1-q3))
                cuda.atomic.add(rho, (ind_x, ind_y, ind_z+1), charge*(1-q1)*(1-q2)*q3)
                cuda.atomic.add(Jx, (ind_x, ind_y, ind_z+1), charge*velocity[i_grid, 0]*(1-q1)*(1-q2)*q3)
                cuda.atomic.add(Jy, (ind_x, ind_y, ind_z+1), charge*velocity[i_grid, 1]*(1-q1)*(1-q2)*q3)
                cuda.atomic.add(Jz, (ind_x, ind_y, ind_z+1), charge*velocity[i_grid, 2]*(1-q1)*(1-q2)*q3)
                cuda.atomic.add(rho, (ind_x+1, ind_y, ind_z+1), charge*q1*(1-q2)*q3)
                cuda.atomic.add(Jx, (ind_x+1, ind_y, ind_z+1), charge*velocity[i_grid, 0]*q1*(1-q2)*q3)
                cuda.atomic.add(Jy, (ind_x+1, ind_y, ind_z+1), charge*velocity[i_grid, 1]*q1*(1-q2)*q3)
                cuda.atomic.add(Jz, (ind_x+1, ind_y, ind_z+1), charge*velocity[i_grid, 2]*q1*(1-q2)*q3)
            elif ind_x == (nx - 1) and ind_y > -1 and ind_y < ny - 1 and ind_z > -1 and ind_z < nz - 1: #(2,1,1)
                cuda.atomic.add(rho, (ind_x, ind_y, ind_z), charge*(1-q1)*(1-q2)*(1-q3))
                cuda.atomic.add(Jx, (ind_x, ind_y, ind_z), charge*velocity[i_grid, 0]*(1-q1)*(1-q2)*(1-q3))
                cuda.atomic.add(Jy, (ind_x, ind_y, ind_z), charge*velocity[i_grid, 1]*(1-q1)*(1-q2)*(1-q3))
                cuda.atomic.add(Jz, (ind_x, ind_y, ind_z), charge*velocity[i_grid, 2]*(1-q1)*(1-q2)*(1-q3))
                cuda.atomic.add(rho, (ind_x, ind_y+1, ind_z), charge*(1-q1)*q2*(1-q3))
                cuda.atomic.add(Jx, (ind_x, ind_y+1, ind_z), charge*velocity[i_grid, 0]*(1-q1)*q2*(1-q3))
                cuda.atomic.add(Jy, (ind_x, ind_y+1, ind_z), charge*velocity[i_grid, 1]*(1-q1)*q2*(1-q3))
                cuda.atomic.add(Jz, (ind_x, ind_y+1, ind_z), charge*velocity[i_grid, 2]*(1-q1)*q2*(1-q3))
                cuda.atomic.add(rho, (ind_x, ind_y, ind_z+1), charge*(1-q1)*(1-q2)*q3)
                cuda.atomic.add(Jx, (ind_x, ind_y, ind_z+1), charge*velocity[i_grid, 0]*(1-q1)*(1-q2)*q3)
                cuda.atomic.add(Jy, (ind_x, ind_y, ind_z+1), charge*velocity[i_grid, 1]*(1-q1)*(1-q2)*q3)
                cuda.atomic.add(Jz, (ind_x, ind_y, ind_z+1), charge*velocity[i_grid, 2]*(1-q1)*(1-q2)*q3)          
                cuda.atomic.add(rho, (ind_x, ind_y+1, ind_z+1), charge*(1-q1)*q2*q3)
                cuda.atomic.add(Jx, (ind_x, ind_y+1, ind_z+1), charge*velocity[i_grid, 0]*(1-q1)*q2*q3)
                cuda.atomic.add(Jy, (ind_x, ind_y+1, ind_z+1), charge*velocity[i_grid, 1]*(1-q1)*q2*q3)
                cuda.atomic.add(Jz, (ind_x, ind_y+1, ind_z+1), charge*velocity[i_grid, 2]*(1-q1)*q2*q3)
            elif ind_x > -1 and ind_x < nx - 1 and ind_y == (ny - 1) and ind_z == (nz - 1): #(1,2,2)
                cuda.atomic.add(rho, (ind_x, ind_y, ind_z), charge*(1-q1)*(1-q2)*(1-q3))
                cuda.atomic.add(Jx, (ind_x, ind_y, ind_z), charge*velocity[i_grid, 0]*(1-q1)*(1-q2)*(1-q3))
                cuda.atomic.add(Jy, (ind_x, ind_y, ind_z), charge*velocity[i_grid, 1]*(1-q1)*(1-q2)*(1-q3))
                cuda.atomic.add(Jz, (ind_x, ind_y, ind_z), charge*velocity[i_grid, 2]*(1-q1)*(1-q2)*(1-q3))
                cuda.atomic.add(rho, (ind_x+1, ind_y, ind_z), charge*q1*(1-q2)*(1-q3))
                cuda.atomic.add(Jx, (ind_x+1, ind_y, ind_z), charge*velocity[i_grid, 0]*q1*(1-q2)*(1-q3))
                cuda.atomic.add(Jy, (ind_x+1, ind_y, ind_z), charge*velocity[i_grid, 1]*q1*(1-q2)*(1-q3))
                cuda.atomic.add(Jz, (ind_x+1, ind_y, ind_z), charge*velocity[i_grid, 2]*q1*(1-q2)*(1-q3))
            elif ind_x == (nx - 1) and ind_y > -1 and ind_y < ny - 1 and ind_z == (nz - 1): #(2,1,2)
                cuda.atomic.add(rho, (ind_x, ind_y, ind_z), charge*(1-q1)*(1-q2)*(1-q3))
                cuda.atomic.add(Jx, (ind_x, ind_y, ind_z), charge*velocity[i_grid, 0]*(1-q1)*(1-q2)*(1-q3))
                cuda.atomic.add(Jy, (ind_x, ind_y, ind_z), charge*velocity[i_grid, 1]*(1-q1)*(1-q2)*(1-q3))
                cuda.atomic.add(Jz, (ind_x, ind_y, ind_z), charge*velocity[i_grid, 2]*(1-q1)*(1-q2)*(1-q3))
                cuda.atomic.add(rho, (ind_x, ind_y+1, ind_z), charge*(1-q1)*q2*(1-q3))
                cuda.atomic.add(Jx, (ind_x, ind_y+1, ind_z), charge*velocity[i_grid, 0]*(1-q1)*q2*(1-q3))
                cuda.atomic.add(Jy, (ind_x, ind_y+1, ind_z), charge*velocity[i_grid, 1]*(1-q1)*q2*(1-q3))
                cuda.atomic.add(Jz, (ind_x, ind_y+1, ind_z), charge*velocity[i_grid, 2]*(1-q1)*q2*(1-q3))
            elif ind_x == (nx - 1) and ind_y == (ny - 1) and ind_z > -1 and ind_z < nz - 1: #(2,2,1)
                cuda.atomic.add(rho, (ind_x, ind_y, ind_z), charge*(1-q1)*(1-q2)*(1-q3))
                cuda.atomic.add(Jx, (ind_x, ind_y, ind_z), charge*velocity[i_grid, 0]*(1-q1)*(1-q2)*(1-q3))
                cuda.atomic.add(Jy, (ind_x, ind_y, ind_z), charge*velocity[i_grid, 1]*(1-q1)*(1-q2)*(1-q3))
                cuda.atomic.add(Jz, (ind_x, ind_y, ind_z), charge*velocity[i_grid, 2]*(1-q1)*(1-q2)*(1-q3))
                cuda.atomic.add(rho, (ind_x, ind_y, ind_z+1), charge*(1-q1)*(1-q2)*q3)
                cuda.atomic.add(Jx, (ind_x, ind_y, ind_z+1), charge*velocity[i_grid, 0]*(1-q1)*(1-q2)*q3)
                cuda.atomic.add(Jy, (ind_x, ind_y, ind_z+1), charge*velocity[i_grid, 1]*(1-q1)*(1-q2)*q3)
                cuda.atomic.add(Jz, (ind_x, ind_y, ind_z+1), charge*velocity[i_grid, 2]*(1-q1)*(1-q2)*q3)          
            elif ind_x == (nx - 1) and ind_y == (ny - 1) and ind_z == (nz - 1): #(2,2,2)
                cuda.atomic.add(rho, (ind_x, ind_y, ind_z), charge*(1-q1)*(1-q2)*(1-q3))
                cuda.atomic.add(Jx, (ind_x, ind_y, ind_z), charge*velocity[i_grid, 0]*(1-q1)*(1-q2)*(1-q3))
                cuda.atomic.add(Jy, (ind_x, ind_y, ind_z), charge*velocity[i_grid, 1]*(1-q1)*(1-q2)*(1-q3))
                cuda.atomic.add(Jz, (ind_x, ind_y, ind_z), charge*velocity[i_grid, 2]*(1-q1)*(1-q2)*(1-q3))
            elif ind_x == -1 and ind_y == -1 and ind_z == -1: #(0,0,0)
                cuda.atomic.add(rho, (ind_x+1, ind_y+1, ind_z+1), charge*q1*q2*q3)
                cuda.atomic.add(Jx, (ind_x+1, ind_y+1, ind_z+1), charge*velocity[i_grid, 0]*q1*q2*q3)
                cuda.atomic.add(Jy, (ind_x+1, ind_y+1, ind_z+1), charge*velocity[i_grid, 1]*q1*q2*q3)
                cuda.atomic.add(Jz, (ind_x+1, ind_y+1, ind_z+1), charge*velocity[i_grid, 2]*q1*q2*q3)
            elif ind_x == (nx - 1) and ind_y == -1 and ind_z == -1: #(2,0,0)
                cuda.atomic.add(rho, (ind_x, ind_y+1, ind_z+1), charge*(1-q1)*q2*q3)
                cuda.atomic.add(Jx, (ind_x, ind_y+1, ind_z+1), charge*velocity[i_grid, 0]*(1-q1)*q2*q3)
                cuda.atomic.add(Jy, (ind_x, ind_y+1, ind_z+1), charge*velocity[i_grid, 1]*(1-q1)*q2*q3)
                cuda.atomic.add(Jz, (ind_x, ind_y+1, ind_z+1), charge*velocity[i_grid, 2]*(1-q1)*q2*q3)
            elif ind_x == -1 and ind_y == (ny - 1) and ind_z == -1: #(0,2,0)
                cuda.atomic.add(rho, (ind_x+1, ind_y, ind_z+1), charge*q1*(1-q2)*q3)
                cuda.atomic.add(Jx, (ind_x+1, ind_y, ind_z+1), charge*velocity[i_grid, 0]*q1*(1-q2)*q3)
                cuda.atomic.add(Jy, (ind_x+1, ind_y, ind_z+1), charge*velocity[i_grid, 1]*q1*(1-q2)*q3)
                cuda.atomic.add(Jz, (ind_x+1, ind_y, ind_z+1), charge*velocity[i_grid, 2]*q1*(1-q2)*q3)
            elif ind_x == -1 and ind_y == -1 and ind_z == (nz - 1): #(0,0,2)
                cuda.atomic.add(rho, (ind_x+1, ind_y+1, ind_z), charge*q1*q2*(1-q3))
                cuda.atomic.add(Jx, (ind_x+1, ind_y+1, ind_z), charge*velocity[i_grid, 0]*q1*q2*(1-q3))
                cuda.atomic.add(Jy, (ind_x+1, ind_y+1, ind_z), charge*velocity[i_grid, 1]*q1*q2*(1-q3))
                cuda.atomic.add(Jz, (ind_x+1, ind_y+1, ind_z), charge*velocity[i_grid, 2]*q1*q2*(1-q3))
            elif ind_x > -1 and ind_x < nx - 1 and ind_y == -1 and ind_z == -1: #(1,0,0)
                cuda.atomic.add(rho, (ind_x, ind_y+1, ind_z+1), charge*(1-q1)*q2*q3)
                cuda.atomic.add(Jx, (ind_x, ind_y+1, ind_z+1), charge*velocity[i_grid, 0]*(1-q1)*q2*q3)
                cuda.atomic.add(Jy, (ind_x, ind_y+1, ind_z+1), charge*velocity[i_grid, 1]*(1-q1)*q2*q3)
                cuda.atomic.add(Jz, (ind_x, ind_y+1, ind_z+1), charge*velocity[i_grid, 2]*(1-q1)*q2*q3)
                cuda.atomic.add(rho, (ind_x+1, ind_y+1, ind_z+1), charge*q1*q2*q3)
                cuda.atomic.add(Jx, (ind_x+1, ind_y+1, ind_z+1), charge*velocity[i_grid, 0]*q1*q2*q3)
                cuda.atomic.add(Jy, (ind_x+1, ind_y+1, ind_z+1), charge*velocity[i_grid, 1]*q1*q2*q3)
                cuda.atomic.add(Jz, (ind_x+1, ind_y+1, ind_z+1), charge*velocity[i_grid, 2]*q1*q2*q3)
            elif ind_x == -1 and ind_y > -1 and ind_y < ny - 1 and ind_z == -1: #(0,1,0)
                cuda.atomic.add(rho, (ind_x+1, ind_y, ind_z+1), charge*q1*(1-q2)*q3)
                cuda.atomic.add(Jx, (ind_x+1, ind_y, ind_z+1), charge*velocity[i_grid, 0]*q1*(1-q2)*q3)
                cuda.atomic.add(Jy, (ind_x+1, ind_y, ind_z+1), charge*velocity[i_grid, 1]*q1*(1-q2)*q3)
                cuda.atomic.add(Jz, (ind_x+1, ind_y, ind_z+1), charge*velocity[i_grid, 2]*q1*(1-q2)*q3)
                cuda.atomic.add(rho, (ind_x+1, ind_y+1, ind_z+1), charge*q1*q2*q3)
                cuda.atomic.add(Jx, (ind_x+1, ind_y+1, ind_z+1), charge*velocity[i_grid, 0]*q1*q2*q3)
                cuda.atomic.add(Jy, (ind_x+1, ind_y+1, ind_z+1), charge*velocity[i_grid, 1]*q1*q2*q3)
                cuda.atomic.add(Jz, (ind_x+1, ind_y+1, ind_z+1), charge*velocity[i_grid, 2]*q1*q2*q3)
            elif ind_x == -1 and ind_y == -1 and ind_z > -1 and ind_z < nz - 1: #(0,0,1)
                cuda.atomic.add(rho, (ind_x+1, ind_y+1, ind_z), charge*q1*q2*(1-q3))
                cuda.atomic.add(Jx, (ind_x+1, ind_y+1, ind_z), charge*velocity[i_grid, 0]*q1*q2*(1-q3))
                cuda.atomic.add(Jy, (ind_x+1, ind_y+1, ind_z), charge*velocity[i_grid, 1]*q1*q2*(1-q3))
                cuda.atomic.add(Jz, (ind_x+1, ind_y+1, ind_z), charge*velocity[i_grid, 2]*q1*q2*(1-q3))
                cuda.atomic.add(rho, (ind_x+1, ind_y+1, ind_z+1), charge*q1*q2*q3)
                cuda.atomic.add(Jx, (ind_x+1, ind_y+1, ind_z+1), charge*velocity[i_grid, 0]*q1*q2*q3)
                cuda.atomic.add(Jy, (ind_x+1, ind_y+1, ind_z+1), charge*velocity[i_grid, 1]*q1*q2*q3)
                cuda.atomic.add(Jz, (ind_x+1, ind_y+1, ind_z+1), charge*velocity[i_grid, 2]*q1*q2*q3)
            elif ind_x == -1 and ind_y > -1 and ind_y < ny - 1 and ind_z > -1 and ind_z < nz - 1: #(0,1,1)
                cuda.atomic.add(rho, (ind_x+1, ind_y, ind_z), charge*q1*(1-q2)*(1-q3))
                cuda.atomic.add(Jx, (ind_x+1, ind_y, ind_z), charge*velocity[i_grid, 0]*q1*(1-q2)*(1-q3))
                cuda.atomic.add(Jy, (ind_x+1, ind_y, ind_z), charge*velocity[i_grid, 1]*q1*(1-q2)*(1-q3))
                cuda.atomic.add(Jz, (ind_x+1, ind_y, ind_z), charge*velocity[i_grid, 2]*q1*(1-q2)*(1-q3))
                cuda.atomic.add(rho, (ind_x+1, ind_y+1, ind_z), charge*q1*q2*(1-q3))
                cuda.atomic.add(Jx, (ind_x+1, ind_y+1, ind_z), charge*velocity[i_grid, 0]*q1*q2*(1-q3))
                cuda.atomic.add(Jy, (ind_x+1, ind_y+1, ind_z), charge*velocity[i_grid, 1]*q1*q2*(1-q3))
                cuda.atomic.add(Jz, (ind_x+1, ind_y+1, ind_z), charge*velocity[i_grid, 2]*q1*q2*(1-q3))
                cuda.atomic.add(rho, (ind_x+1, ind_y, ind_z+1), charge*q1*(1-q2)*q3)
                cuda.atomic.add(Jx, (ind_x+1, ind_y, ind_z+1), charge*velocity[i_grid, 0]*q1*(1-q2)*q3)
                cuda.atomic.add(Jy, (ind_x+1, ind_y, ind_z+1), charge*velocity[i_grid, 1]*q1*(1-q2)*q3)
                cuda.atomic.add(Jz, (ind_x+1, ind_y, ind_z+1), charge*velocity[i_grid, 2]*q1*(1-q2)*q3)
                cuda.atomic.add(rho, (ind_x+1, ind_y+1, ind_z+1), charge*q1*q2*q3)
                cuda.atomic.add(Jx, (ind_x+1, ind_y+1, ind_z+1), charge*velocity[i_grid, 0]*q1*q2*q3)
                cuda.atomic.add(Jy, (ind_x+1, ind_y+1, ind_z+1), charge*velocity[i_grid, 1]*q1*q2*q3)
                cuda.atomic.add(Jz, (ind_x+1, ind_y+1, ind_z+1), charge*velocity[i_grid, 2]*q1*q2*q3)
            elif ind_x > -1 and ind_x < nx - 1 and ind_y == -1 and ind_z > -1 and ind_z < nz - 1: #(1,0,1)
                cuda.atomic.add(rho, (ind_x, ind_y+1, ind_z), charge*(1-q1)*q2*(1-q3))
                cuda.atomic.add(Jx, (ind_x, ind_y+1, ind_z), charge*velocity[i_grid, 0]*(1-q1)*q2*(1-q3))
                cuda.atomic.add(Jy, (ind_x, ind_y+1, ind_z), charge*velocity[i_grid, 1]*(1-q1)*q2*(1-q3))
                cuda.atomic.add(Jz, (ind_x, ind_y+1, ind_z), charge*velocity[i_grid, 2]*(1-q1)*q2*(1-q3))
                cuda.atomic.add(rho, (ind_x+1, ind_y+1, ind_z), charge*q1*q2*(1-q3))
                cuda.atomic.add(Jx, (ind_x+1, ind_y+1, ind_z), charge*velocity[i_grid, 0]*q1*q2*(1-q3))
                cuda.atomic.add(Jy, (ind_x+1, ind_y+1, ind_z), charge*velocity[i_grid, 1]*q1*q2*(1-q3))
                cuda.atomic.add(Jz, (ind_x+1, ind_y+1, ind_z), charge*velocity[i_grid, 2]*q1*q2*(1-q3))
                cuda.atomic.add(rho, (ind_x, ind_y+1, ind_z+1), charge*(1-q1)*q2*q3)
                cuda.atomic.add(Jx, (ind_x, ind_y+1, ind_z+1), charge*velocity[i_grid, 0]*(1-q1)*q2*q3)
                cuda.atomic.add(Jy, (ind_x, ind_y+1, ind_z+1), charge*velocity[i_grid, 1]*(1-q1)*q2*q3)
                cuda.atomic.add(Jz, (ind_x, ind_y+1, ind_z+1), charge*velocity[i_grid, 2]*(1-q1)*q2*q3)
                cuda.atomic.add(rho, (ind_x+1, ind_y+1, ind_z+1), charge*q1*q2*q3)
                cuda.atomic.add(Jx, (ind_x+1, ind_y+1, ind_z+1), charge*velocity[i_grid, 0]*q1*q2*q3)
                cuda.atomic.add(Jy, (ind_x+1, ind_y+1, ind_z+1), charge*velocity[i_grid, 1]*q1*q2*q3)
                cuda.atomic.add(Jz, (ind_x+1, ind_y+1, ind_z+1), charge*velocity[i_grid, 2]*q1*q2*q3)
            elif ind_x > -1 and ind_x < nx - 1 and ind_y > -1 and ind_y < ny - 1 and ind_z == -1: #(1,1,0)
                cuda.atomic.add(rho, (ind_x, ind_y, ind_z+1), charge*(1-q1)*(1-q2)*q3)
                cuda.atomic.add(Jx, (ind_x, ind_y, ind_z+1), charge*velocity[i_grid, 0]*(1-q1)*(1-q2)*q3)
                cuda.atomic.add(Jy, (ind_x, ind_y, ind_z+1), charge*velocity[i_grid, 1]*(1-q1)*(1-q2)*q3)
                cuda.atomic.add(Jz, (ind_x, ind_y, ind_z+1), charge*velocity[i_grid, 2]*(1-q1)*(1-q2)*q3)
                cuda.atomic.add(rho, (ind_x, ind_y+1, ind_z+1), charge*(1-q1)*q2*q3)
                cuda.atomic.add(Jx, (ind_x, ind_y+1, ind_z+1), charge*velocity[i_grid, 0]*(1-q1)*q2*q3)
                cuda.atomic.add(Jy, (ind_x, ind_y+1, ind_z+1), charge*velocity[i_grid, 1]*(1-q1)*q2*q3)
                cuda.atomic.add(Jz, (ind_x, ind_y+1, ind_z+1), charge*velocity[i_grid, 2]*(1-q1)*q2*q3)
                cuda.atomic.add(rho, (ind_x+1, ind_y, ind_z+1), charge*q1*(1-q2)*q3)
                cuda.atomic.add(Jx, (ind_x+1, ind_y, ind_z+1), charge*velocity[i_grid, 0]*q1*(1-q2)*q3)
                cuda.atomic.add(Jy, (ind_x+1, ind_y, ind_z+1), charge*velocity[i_grid, 1]*q1*(1-q2)*q3)
                cuda.atomic.add(Jz, (ind_x+1, ind_y, ind_z+1), charge*velocity[i_grid, 2]*q1*(1-q2)*q3)
                cuda.atomic.add(rho, (ind_x+1, ind_y+1, ind_z+1), charge*q1*q2*q3)
                cuda.atomic.add(Jx, (ind_x+1, ind_y+1, ind_z+1), charge*velocity[i_grid, 0]*q1*q2*q3)
                cuda.atomic.add(Jy, (ind_x+1, ind_y+1, ind_z+1), charge*velocity[i_grid, 1]*q1*q2*q3)
                cuda.atomic.add(Jz, (ind_x+1, ind_y+1, ind_z+1), charge*velocity[i_grid, 2]*q1*q2*q3)
            elif ind_x == -1 and ind_y > -1 and ind_y < ny - 1 and ind_z == (nz - 1): #(0,1,2)
                cuda.atomic.add(rho, (ind_x+1, ind_y, ind_z), charge*q1*(1-q2)*(1-q3))
                cuda.atomic.add(Jx, (ind_x+1, ind_y, ind_z), charge*velocity[i_grid, 0]*q1*(1-q2)*(1-q3))
                cuda.atomic.add(Jy, (ind_x+1, ind_y, ind_z), charge*velocity[i_grid, 1]*q1*(1-q2)*(1-q3))
                cuda.atomic.add(Jz, (ind_x+1, ind_y, ind_z), charge*velocity[i_grid, 2]*q1*(1-q2)*(1-q3))
                cuda.atomic.add(rho, (ind_x+1, ind_y+1, ind_z), charge*q1*q2*(1-q3))
                cuda.atomic.add(Jx, (ind_x+1, ind_y+1, ind_z), charge*velocity[i_grid, 0]*q1*q2*(1-q3))
                cuda.atomic.add(Jy, (ind_x+1, ind_y+1, ind_z), charge*velocity[i_grid, 1]*q1*q2*(1-q3))
                cuda.atomic.add(Jz, (ind_x+1, ind_y+1, ind_z), charge*velocity[i_grid, 2]*q1*q2*(1-q3))
            elif ind_x > -1 and ind_x < nx - 1 and ind_y == -1 and ind_z == (nz - 1): #(1,0,2)
                cuda.atomic.add(rho, (ind_x, ind_y+1, ind_z), charge*(1-q1)*q2*(1-q3))
                cuda.atomic.add(Jx, (ind_x, ind_y+1, ind_z), charge*velocity[i_grid, 0]*(1-q1)*q2*(1-q3))
                cuda.atomic.add(Jy, (ind_x, ind_y+1, ind_z), charge*velocity[i_grid, 1]*(1-q1)*q2*(1-q3))
                cuda.atomic.add(Jz, (ind_x, ind_y+1, ind_z), charge*velocity[i_grid, 2]*(1-q1)*q2*(1-q3))
                cuda.atomic.add(rho, (ind_x+1, ind_y+1, ind_z), charge*q1*q2*(1-q3))
                cuda.atomic.add(Jx, (ind_x+1, ind_y+1, ind_z), charge*velocity[i_grid, 0]*q1*q2*(1-q3))
                cuda.atomic.add(Jy, (ind_x+1, ind_y+1, ind_z), charge*velocity[i_grid, 1]*q1*q2*(1-q3))
                cuda.atomic.add(Jz, (ind_x+1, ind_y+1, ind_z), charge*velocity[i_grid, 2]*q1*q2*(1-q3))
            elif ind_x > -1 and ind_x < nx - 1 and ind_y == (ny - 1) and ind_z == -1: #(1,2,0)
                cuda.atomic.add(rho, (ind_x, ind_y, ind_z+1), charge*(1-q1)*(1-q2)*q3)
                cuda.atomic.add(Jx, (ind_x, ind_y, ind_z+1), charge*velocity[i_grid, 0]*(1-q1)*(1-q2)*q3)
                cuda.atomic.add(Jy, (ind_x, ind_y, ind_z+1), charge*velocity[i_grid, 1]*(1-q1)*(1-q2)*q3)
                cuda.atomic.add(Jz, (ind_x, ind_y, ind_z+1), charge*velocity[i_grid, 2]*(1-q1)*(1-q2)*q3)
                cuda.atomic.add(rho, (ind_x+1, ind_y, ind_z+1), charge*q1*(1-q2)*q3)
                cuda.atomic.add(Jx, (ind_x+1, ind_y, ind_z+1), charge*velocity[i_grid, 0]*q1*(1-q2)*q3)
                cuda.atomic.add(Jy, (ind_x+1, ind_y, ind_z+1), charge*velocity[i_grid, 1]*q1*(1-q2)*q3)
                cuda.atomic.add(Jz, (ind_x+1, ind_y, ind_z+1), charge*velocity[i_grid, 2]*q1*(1-q2)*q3)
            elif ind_x == -1 and ind_y == (ny - 1) and ind_z > -1 and ind_z < nz - 1: #(0,2,1)
                cuda.atomic.add(rho, (ind_x+1, ind_y, ind_z), charge*q1*(1-q2)*(1-q3))
                cuda.atomic.add(Jx, (ind_x+1, ind_y, ind_z), charge*velocity[i_grid, 0]*q1*(1-q2)*(1-q3))
                cuda.atomic.add(Jy, (ind_x+1, ind_y, ind_z), charge*velocity[i_grid, 1]*q1*(1-q2)*(1-q3))
                cuda.atomic.add(Jz, (ind_x+1, ind_y, ind_z), charge*velocity[i_grid, 2]*q1*(1-q2)*(1-q3))
                cuda.atomic.add(rho, (ind_x+1, ind_y, ind_z+1), charge*q1*(1-q2)*q3)
                cuda.atomic.add(Jx, (ind_x+1, ind_y, ind_z+1), charge*velocity[i_grid, 0]*q1*(1-q2)*q3)
                cuda.atomic.add(Jy, (ind_x+1, ind_y, ind_z+1), charge*velocity[i_grid, 1]*q1*(1-q2)*q3)
                cuda.atomic.add(Jz, (ind_x+1, ind_y, ind_z+1), charge*velocity[i_grid, 2]*q1*(1-q2)*q3)
            elif ind_x == (nx - 1) and ind_y == -1 and ind_z > -1 and ind_z < nz - 1: #(2,0,1)
                cuda.atomic.add(rho, (ind_x, ind_y+1, ind_z), charge*(1-q1)*q2*(1-q3))
                cuda.atomic.add(Jx, (ind_x, ind_y+1, ind_z), charge*velocity[i_grid, 0]*(1-q1)*q2*(1-q3))
                cuda.atomic.add(Jy, (ind_x, ind_y+1, ind_z), charge*velocity[i_grid, 1]*(1-q1)*q2*(1-q3))
                cuda.atomic.add(Jz, (ind_x, ind_y+1, ind_z), charge*velocity[i_grid, 2]*(1-q1)*q2*(1-q3))
                cuda.atomic.add(rho, (ind_x, ind_y+1, ind_z+1), charge*(1-q1)*q2*q3)
                cuda.atomic.add(Jx, (ind_x, ind_y+1, ind_z+1), charge*velocity[i_grid, 0]*(1-q1)*q2*q3)
                cuda.atomic.add(Jy, (ind_x, ind_y+1, ind_z+1), charge*velocity[i_grid, 1]*(1-q1)*q2*q3)
                cuda.atomic.add(Jz, (ind_x, ind_y+1, ind_z+1), charge*velocity[i_grid, 2]*(1-q1)*q2*q3)
            elif ind_x == (nx - 1) and ind_y > -1 and ind_y < nz - 1 and ind_z == -1: #(2,1,0)
                cuda.atomic.add(rho, (ind_x, ind_y, ind_z+1), charge*(1-q1)*(1-q2)*q3)
                cuda.atomic.add(Jx, (ind_x, ind_y, ind_z+1), charge*velocity[i_grid, 0]*(1-q1)*(1-q2)*q3)
                cuda.atomic.add(Jy, (ind_x, ind_y, ind_z+1), charge*velocity[i_grid, 1]*(1-q1)*(1-q2)*q3)
                cuda.atomic.add(Jz, (ind_x, ind_y, ind_z+1), charge*velocity[i_grid, 2]*(1-q1)*(1-q2)*q3)
                cuda.atomic.add(rho, (ind_x, ind_y+1, ind_z+1), charge*(1-q1)*q2*q3)
                cuda.atomic.add(Jx, (ind_x, ind_y+1, ind_z+1), charge*velocity[i_grid, 0]*(1-q1)*q2*q3)
                cuda.atomic.add(Jy, (ind_x, ind_y+1, ind_z+1), charge*velocity[i_grid, 1]*(1-q1)*q2*q3)
                cuda.atomic.add(Jz, (ind_x, ind_y+1, ind_z+1), charge*velocity[i_grid, 2]*(1-q1)*q2*q3)
            elif ind_x == -1 and ind_y == (ny - 1) and ind_z == (nz - 1): #(0,2,2)
                cuda.atomic.add(rho, (ind_x+1, ind_y, ind_z), charge*q1*(1-q2)*(1-q3))
                cuda.atomic.add(Jx, (ind_x+1, ind_y, ind_z), charge*velocity[i_grid, 0]*q1*(1-q2)*(1-q3))
                cuda.atomic.add(Jy, (ind_x+1, ind_y, ind_z), charge*velocity[i_grid, 1]*q1*(1-q2)*(1-q3))
                cuda.atomic.add(Jz, (ind_x+1, ind_y, ind_z), charge*velocity[i_grid, 2]*q1*(1-q2)*(1-q3))
            elif ind_x == (nx - 1) and ind_y == -1 and ind_z == (nz - 1): #(2,0,2)
                cuda.atomic.add(rho, (ind_x, ind_y+1, ind_z), charge*(1-q1)*q2*(1-q3))
                cuda.atomic.add(Jx, (ind_x, ind_y+1, ind_z), charge*velocity[i_grid, 0]*(1-q1)*q2*(1-q3))
                cuda.atomic.add(Jy, (ind_x, ind_y+1, ind_z), charge*velocity[i_grid, 1]*(1-q1)*q2*(1-q3))
                cuda.atomic.add(Jz, (ind_x, ind_y+1, ind_z), charge*velocity[i_grid, 2]*(1-q1)*q2*(1-q3))
            elif ind_x == (nx - 1) and ind_y == (ny - 1) and ind_z == -1: #(2,2,0) 
                cuda.atomic.add(rho, (ind_x, ind_y, ind_z+1), charge*(1-q1)*(1-q2)*q3)
                cuda.atomic.add(Jx, (ind_x, ind_y, ind_z+1), charge*velocity[i_grid, 0]*(1-q1)*(1-q2)*q3)
                cuda.atomic.add(Jy, (ind_x, ind_y, ind_z+1), charge*velocity[i_grid, 1]*(1-q1)*(1-q2)*q3)
                cuda.atomic.add(Jz, (ind_x, ind_y, ind_z+1), charge*velocity[i_grid, 2]*(1-q1)*(1-q2)*q3)
                    
        
@cuda.jit
def find_n_nu(num_particles, pos, velocity, dx, dy, dz, number_charge, rho_n, left_bound_x, left_bound_y, left_bound_z, right_bound_x, right_bound_y, right_bound_z):
    
    # threads loop in one dimension
    i_grid = cuda.grid(1)
    
    if i_grid < num_particles:
        
        # the particles must be in the box
        if pos[i_grid, 0] > left_bound_x and pos[i_grid, 0] < right_bound_x and pos[i_grid, 1] > left_bound_y and pos[i_grid, 1] < right_bound_y and pos[i_grid, 2] > left_bound_z and pos[i_grid, 2] < right_bound_z:
        
            # find the index of positions in electric_field and magnetic_fleid
            ind_x, ind_y, ind_z = int((pos[i_grid, 0] - 0. * dx - left_bound_x)//dx), int((pos[i_grid, 1] - 0. * dy - left_bound_y)//dy), int((pos[i_grid, 2] - 0. * dz - left_bound_z)//dz)

            cuda.atomic.add(rho_n, (ind_x, ind_y, ind_z), number_charge)
        
@cuda.jit
def updata_particle_info(num_particles, pos, velocity, nx, ny, nz, Ex_Jefi, Ey_Jefi, Ez_Jefi, Bx_Jefi, By_Jefi, Bz_Jefi, dx, dy, dz, mass, charge, dt, left_bound_x, left_bound_y, left_bound_z, right_bound_x, right_bound_y, right_bound_z):
    
    # threads loop in one dimension
    i_grid = cuda.grid(1)

    if i_grid < num_particles:
        
        # only update those particles that are in the computational domain
        if pos[i_grid, 0] > left_bound_x and pos[i_grid, 0] < right_bound_x and pos[i_grid, 1] > left_bound_y and pos[i_grid, 1] < right_bound_y and pos[i_grid, 2] > left_bound_z and pos[i_grid, 2] < right_bound_z:
            
            # find the index of positions in electric_field and magnetic_fleid.

            ind_x = int((pos[i_grid, 0] - left_bound_x - 0.5 * dx)//dx)
            ind_y = int((pos[i_grid, 1] - left_bound_y - 0.5 * dy)//dy)
            ind_z = int((pos[i_grid, 2] - left_bound_z - 0.5 * dz)//dz)
                
            bx, by, bz, ex, ey, ez = 0, 0, 0, 0, 0, 0
            
            # we also need to set
            q1 = (pos[i_grid, 0] - (ind_x + 0.5) * dx)/dx
            q2 = (pos[i_grid, 1] - (ind_y + 0.5) * dy)/dy
            q3 = (pos[i_grid, 2] - (ind_z + 0.5) * dz)/dz
            
            if ind_x > -1 and ind_x < nx - 1 and ind_y > -1 and ind_y < ny - 1 and ind_z > -1 and ind_z < nz - 1: #(1,1,1)
                bx += Bx_Jefi[ind_x, ind_y, ind_z]*(1-q1)*(1-q2)*(1-q3)
                by += By_Jefi[ind_x, ind_y, ind_z]*(1-q1)*(1-q2)*(1-q3)
                bz += Bz_Jefi[ind_x, ind_y, ind_z]*(1-q1)*(1-q2)*(1-q3)
                ex += Ex_Jefi[ind_x, ind_y, ind_z]*(1-q1)*(1-q2)*(1-q3)
                ey += Ey_Jefi[ind_x, ind_y, ind_z]*(1-q1)*(1-q2)*(1-q3)
                ez += Ez_Jefi[ind_x, ind_y, ind_z]*(1-q1)*(1-q2)*(1-q3)
                bx += Bx_Jefi[ind_x+1, ind_y, ind_z]*q1*(1-q2)*(1-q3)
                by += By_Jefi[ind_x+1, ind_y, ind_z]*q1*(1-q2)*(1-q3)
                bz += Bz_Jefi[ind_x+1, ind_y, ind_z]*q1*(1-q2)*(1-q3)
                ex += Ex_Jefi[ind_x+1, ind_y, ind_z]*q1*(1-q2)*(1-q3)
                ey += Ey_Jefi[ind_x+1, ind_y, ind_z]*q1*(1-q2)*(1-q3)
                ez += Ez_Jefi[ind_x+1, ind_y, ind_z]*q1*(1-q2)*(1-q3)
                bx += Bx_Jefi[ind_x, ind_y+1, ind_z]*(1-q1)*q2*(1-q3)
                by += By_Jefi[ind_x, ind_y+1, ind_z]*(1-q1)*q2*(1-q3)
                bz += Bz_Jefi[ind_x, ind_y+1, ind_z]*(1-q1)*q2*(1-q3)
                ex += Ex_Jefi[ind_x, ind_y+1, ind_z]*(1-q1)*q2*(1-q3)
                ey += Ey_Jefi[ind_x, ind_y+1, ind_z]*(1-q1)*q2*(1-q3)
                ez += Ez_Jefi[ind_x, ind_y+1, ind_z]*(1-q1)*q2*(1-q3)
                bx += Bx_Jefi[ind_x, ind_y, ind_z+1]*(1-q1)*(1-q2)*q3
                by += By_Jefi[ind_x, ind_y, ind_z+1]*(1-q1)*(1-q2)*q3
                bz += Bz_Jefi[ind_x, ind_y, ind_z+1]*(1-q1)*(1-q2)*q3
                ex += Ex_Jefi[ind_x, ind_y, ind_z+1]*(1-q1)*(1-q2)*q3
                ey += Ey_Jefi[ind_x, ind_y, ind_z+1]*(1-q1)*(1-q2)*q3
                ez += Ez_Jefi[ind_x, ind_y, ind_z+1]*(1-q1)*(1-q2)*q3                
                bx += Bx_Jefi[ind_x+1, ind_y+1, ind_z]*q1*q2*(1-q3)
                by += By_Jefi[ind_x+1, ind_y+1, ind_z]*q1*q2*(1-q3)
                bz += Bz_Jefi[ind_x+1, ind_y+1, ind_z]*q1*q2*(1-q3)
                ex += Ex_Jefi[ind_x+1, ind_y+1, ind_z]*q1*q2*(1-q3)
                ey += Ey_Jefi[ind_x+1, ind_y+1, ind_z]*q1*q2*(1-q3)
                ez += Ez_Jefi[ind_x+1, ind_y+1, ind_z]*q1*q2*(1-q3)
                bx += Bx_Jefi[ind_x+1, ind_y, ind_z+1]*q1*(1-q2)*q3
                by += By_Jefi[ind_x+1, ind_y, ind_z+1]*q1*(1-q2)*q3
                bz += Bz_Jefi[ind_x+1, ind_y, ind_z+1]*q1*(1-q2)*q3
                ex += Ex_Jefi[ind_x+1, ind_y, ind_z+1]*q1*(1-q2)*q3
                ey += Ey_Jefi[ind_x+1, ind_y, ind_z+1]*q1*(1-q2)*q3
                ez += Ez_Jefi[ind_x+1, ind_y, ind_z+1]*q1*(1-q2)*q3
                bx += Bx_Jefi[ind_x, ind_y+1, ind_z+1]*(1-q1)*q2*q3
                by += By_Jefi[ind_x, ind_y+1, ind_z+1]*(1-q1)*q2*q3
                bz += Bz_Jefi[ind_x, ind_y+1, ind_z+1]*(1-q1)*q2*q3
                ex += Ex_Jefi[ind_x, ind_y+1, ind_z+1]*(1-q1)*q2*q3
                ey += Ey_Jefi[ind_x, ind_y+1, ind_z+1]*(1-q1)*q2*q3
                ez += Ez_Jefi[ind_x, ind_y+1, ind_z+1]*(1-q1)*q2*q3
                bx += Bx_Jefi[ind_x+1, ind_y+1, ind_z+1]*q1*q2*q3
                by += By_Jefi[ind_x+1, ind_y+1, ind_z+1]*q1*q2*q3
                bz += Bz_Jefi[ind_x+1, ind_y+1, ind_z+1]*q1*q2*q3
                ex += Ex_Jefi[ind_x+1, ind_y+1, ind_z+1]*q1*q2*q3
                ey += Ey_Jefi[ind_x+1, ind_y+1, ind_z+1]*q1*q2*q3
                ez += Ez_Jefi[ind_x+1, ind_y+1, ind_z+1]*q1*q2*q3
            elif ind_x > -1 and ind_x < nx - 1 and ind_y > -1 and ind_y < ny - 1 and ind_z == (nz - 1): #(1,1,2)
                bx += Bx_Jefi[ind_x, ind_y, ind_z]*(1-q1)*(1-q2)*(1-q3)
                by += By_Jefi[ind_x, ind_y, ind_z]*(1-q1)*(1-q2)*(1-q3)
                bz += Bz_Jefi[ind_x, ind_y, ind_z]*(1-q1)*(1-q2)*(1-q3)
                ex += Ex_Jefi[ind_x, ind_y, ind_z]*(1-q1)*(1-q2)*(1-q3)
                ey += Ey_Jefi[ind_x, ind_y, ind_z]*(1-q1)*(1-q2)*(1-q3)
                ez += Ez_Jefi[ind_x, ind_y, ind_z]*(1-q1)*(1-q2)*(1-q3)
                bx += Bx_Jefi[ind_x+1, ind_y, ind_z]*q1*(1-q2)*(1-q3)
                by += By_Jefi[ind_x+1, ind_y, ind_z]*q1*(1-q2)*(1-q3)
                bz += Bz_Jefi[ind_x+1, ind_y, ind_z]*q1*(1-q2)*(1-q3)
                ex += Ex_Jefi[ind_x+1, ind_y, ind_z]*q1*(1-q2)*(1-q3)
                ey += Ey_Jefi[ind_x+1, ind_y, ind_z]*q1*(1-q2)*(1-q3)
                ez += Ez_Jefi[ind_x+1, ind_y, ind_z]*q1*(1-q2)*(1-q3)
                bx += Bx_Jefi[ind_x, ind_y+1, ind_z]*(1-q1)*q2*(1-q3)
                by += By_Jefi[ind_x, ind_y+1, ind_z]*(1-q1)*q2*(1-q3)
                bz += Bz_Jefi[ind_x, ind_y+1, ind_z]*(1-q1)*q2*(1-q3)
                ex += Ex_Jefi[ind_x, ind_y+1, ind_z]*(1-q1)*q2*(1-q3)
                ey += Ey_Jefi[ind_x, ind_y+1, ind_z]*(1-q1)*q2*(1-q3)
                ez += Ez_Jefi[ind_x, ind_y+1, ind_z]*(1-q1)*q2*(1-q3)             
                bx += Bx_Jefi[ind_x+1, ind_y+1, ind_z]*q1*q2*(1-q3)
                by += By_Jefi[ind_x+1, ind_y+1, ind_z]*q1*q2*(1-q3)
                bz += Bz_Jefi[ind_x+1, ind_y+1, ind_z]*q1*q2*(1-q3)
                ex += Ex_Jefi[ind_x+1, ind_y+1, ind_z]*q1*q2*(1-q3)
                ey += Ey_Jefi[ind_x+1, ind_y+1, ind_z]*q1*q2*(1-q3)
                ez += Ez_Jefi[ind_x+1, ind_y+1, ind_z]*q1*q2*(1-q3)
            elif ind_x > -1 and ind_x < nx - 1 and ind_y == (ny - 1) and ind_z > -1 and ind_z < nz - 1: #(1,2,1)
                bx += Bx_Jefi[ind_x, ind_y, ind_z]*(1-q1)*(1-q2)*(1-q3)
                by += By_Jefi[ind_x, ind_y, ind_z]*(1-q1)*(1-q2)*(1-q3)
                bz += Bz_Jefi[ind_x, ind_y, ind_z]*(1-q1)*(1-q2)*(1-q3)
                ex += Ex_Jefi[ind_x, ind_y, ind_z]*(1-q1)*(1-q2)*(1-q3)
                ey += Ey_Jefi[ind_x, ind_y, ind_z]*(1-q1)*(1-q2)*(1-q3)
                ez += Ez_Jefi[ind_x, ind_y, ind_z]*(1-q1)*(1-q2)*(1-q3)
                bx += Bx_Jefi[ind_x+1, ind_y, ind_z]*q1*(1-q2)*(1-q3)
                by += By_Jefi[ind_x+1, ind_y, ind_z]*q1*(1-q2)*(1-q3)
                bz += Bz_Jefi[ind_x+1, ind_y, ind_z]*q1*(1-q2)*(1-q3)
                ex += Ex_Jefi[ind_x+1, ind_y, ind_z]*q1*(1-q2)*(1-q3)
                ey += Ey_Jefi[ind_x+1, ind_y, ind_z]*q1*(1-q2)*(1-q3)
                ez += Ez_Jefi[ind_x+1, ind_y, ind_z]*q1*(1-q2)*(1-q3)
                bx += Bx_Jefi[ind_x, ind_y, ind_z+1]*(1-q1)*(1-q2)*q3
                by += By_Jefi[ind_x, ind_y, ind_z+1]*(1-q1)*(1-q2)*q3
                bz += Bz_Jefi[ind_x, ind_y, ind_z+1]*(1-q1)*(1-q2)*q3
                ex += Ex_Jefi[ind_x, ind_y, ind_z+1]*(1-q1)*(1-q2)*q3
                ey += Ey_Jefi[ind_x, ind_y, ind_z+1]*(1-q1)*(1-q2)*q3
                ez += Ez_Jefi[ind_x, ind_y, ind_z+1]*(1-q1)*(1-q2)*q3                
                bx += Bx_Jefi[ind_x+1, ind_y, ind_z+1]*q1*(1-q2)*q3
                by += By_Jefi[ind_x+1, ind_y, ind_z+1]*q1*(1-q2)*q3
                bz += Bz_Jefi[ind_x+1, ind_y, ind_z+1]*q1*(1-q2)*q3
                ex += Ex_Jefi[ind_x+1, ind_y, ind_z+1]*q1*(1-q2)*q3
                ey += Ey_Jefi[ind_x+1, ind_y, ind_z+1]*q1*(1-q2)*q3
                ez += Ez_Jefi[ind_x+1, ind_y, ind_z+1]*q1*(1-q2)*q3
            elif ind_x == (nx - 1) and ind_y > -1 and ind_y < ny - 1 and ind_z > -1 and ind_z < nz - 1: #(2,1,1)
                bx += Bx_Jefi[ind_x, ind_y, ind_z]*(1-q1)*(1-q2)*(1-q3)
                by += By_Jefi[ind_x, ind_y, ind_z]*(1-q1)*(1-q2)*(1-q3)
                bz += Bz_Jefi[ind_x, ind_y, ind_z]*(1-q1)*(1-q2)*(1-q3)
                ex += Ex_Jefi[ind_x, ind_y, ind_z]*(1-q1)*(1-q2)*(1-q3)
                ey += Ey_Jefi[ind_x, ind_y, ind_z]*(1-q1)*(1-q2)*(1-q3)
                ez += Ez_Jefi[ind_x, ind_y, ind_z]*(1-q1)*(1-q2)*(1-q3)
                bx += Bx_Jefi[ind_x, ind_y+1, ind_z]*(1-q1)*q2*(1-q3)
                by += By_Jefi[ind_x, ind_y+1, ind_z]*(1-q1)*q2*(1-q3)
                bz += Bz_Jefi[ind_x, ind_y+1, ind_z]*(1-q1)*q2*(1-q3)
                ex += Ex_Jefi[ind_x, ind_y+1, ind_z]*(1-q1)*q2*(1-q3)
                ey += Ey_Jefi[ind_x, ind_y+1, ind_z]*(1-q1)*q2*(1-q3)
                ez += Ez_Jefi[ind_x, ind_y+1, ind_z]*(1-q1)*q2*(1-q3)
                bx += Bx_Jefi[ind_x, ind_y, ind_z+1]*(1-q1)*(1-q2)*q3
                by += By_Jefi[ind_x, ind_y, ind_z+1]*(1-q1)*(1-q2)*q3
                bz += Bz_Jefi[ind_x, ind_y, ind_z+1]*(1-q1)*(1-q2)*q3
                ex += Ex_Jefi[ind_x, ind_y, ind_z+1]*(1-q1)*(1-q2)*q3
                ey += Ey_Jefi[ind_x, ind_y, ind_z+1]*(1-q1)*(1-q2)*q3
                ez += Ez_Jefi[ind_x, ind_y, ind_z+1]*(1-q1)*(1-q2)*q3                
                bx += Bx_Jefi[ind_x, ind_y+1, ind_z+1]*(1-q1)*q2*q3
                by += By_Jefi[ind_x, ind_y+1, ind_z+1]*(1-q1)*q2*q3
                bz += Bz_Jefi[ind_x, ind_y+1, ind_z+1]*(1-q1)*q2*q3
                ex += Ex_Jefi[ind_x, ind_y+1, ind_z+1]*(1-q1)*q2*q3
                ey += Ey_Jefi[ind_x, ind_y+1, ind_z+1]*(1-q1)*q2*q3
                ez += Ez_Jefi[ind_x, ind_y+1, ind_z+1]*(1-q1)*q2*q3
            elif ind_x > -1 and ind_x < nx - 1 and ind_y == (ny - 1) and ind_z == (nz - 1): #(1,2,2)
                bx += Bx_Jefi[ind_x, ind_y, ind_z]*(1-q1)*(1-q2)*(1-q3)
                by += By_Jefi[ind_x, ind_y, ind_z]*(1-q1)*(1-q2)*(1-q3)
                bz += Bz_Jefi[ind_x, ind_y, ind_z]*(1-q1)*(1-q2)*(1-q3)
                ex += Ex_Jefi[ind_x, ind_y, ind_z]*(1-q1)*(1-q2)*(1-q3)
                ey += Ey_Jefi[ind_x, ind_y, ind_z]*(1-q1)*(1-q2)*(1-q3)
                ez += Ez_Jefi[ind_x, ind_y, ind_z]*(1-q1)*(1-q2)*(1-q3)
                bx += Bx_Jefi[ind_x+1, ind_y, ind_z]*q1*(1-q2)*(1-q3)
                by += By_Jefi[ind_x+1, ind_y, ind_z]*q1*(1-q2)*(1-q3)
                bz += Bz_Jefi[ind_x+1, ind_y, ind_z]*q1*(1-q2)*(1-q3)
                ex += Ex_Jefi[ind_x+1, ind_y, ind_z]*q1*(1-q2)*(1-q3)
                ey += Ey_Jefi[ind_x+1, ind_y, ind_z]*q1*(1-q2)*(1-q3)
                ez += Ez_Jefi[ind_x+1, ind_y, ind_z]*q1*(1-q2)*(1-q3)
            elif ind_x == (nx - 1) and ind_y > -1 and ind_y < ny - 1 and ind_z == (nz - 1): #(2,1,2)
                bx += Bx_Jefi[ind_x, ind_y, ind_z]*(1-q1)*(1-q2)*(1-q3)
                by += By_Jefi[ind_x, ind_y, ind_z]*(1-q1)*(1-q2)*(1-q3)
                bz += Bz_Jefi[ind_x, ind_y, ind_z]*(1-q1)*(1-q2)*(1-q3)
                ex += Ex_Jefi[ind_x, ind_y, ind_z]*(1-q1)*(1-q2)*(1-q3)
                ey += Ey_Jefi[ind_x, ind_y, ind_z]*(1-q1)*(1-q2)*(1-q3)
                ez += Ez_Jefi[ind_x, ind_y, ind_z]*(1-q1)*(1-q2)*(1-q3)
                bx += Bx_Jefi[ind_x, ind_y+1, ind_z]*(1-q1)*q2*(1-q3)
                by += By_Jefi[ind_x, ind_y+1, ind_z]*(1-q1)*q2*(1-q3)
                bz += Bz_Jefi[ind_x, ind_y+1, ind_z]*(1-q1)*q2*(1-q3)
                ex += Ex_Jefi[ind_x, ind_y+1, ind_z]*(1-q1)*q2*(1-q3)
                ey += Ey_Jefi[ind_x, ind_y+1, ind_z]*(1-q1)*q2*(1-q3)
                ez += Ez_Jefi[ind_x, ind_y+1, ind_z]*(1-q1)*q2*(1-q3)             
            elif ind_x == (nx - 1) and ind_y == (ny - 1) and ind_z > -1 and ind_z < nz - 1: #(2,2,1)
                bx += Bx_Jefi[ind_x, ind_y, ind_z]*(1-q1)*(1-q2)*(1-q3)
                by += By_Jefi[ind_x, ind_y, ind_z]*(1-q1)*(1-q2)*(1-q3)
                bz += Bz_Jefi[ind_x, ind_y, ind_z]*(1-q1)*(1-q2)*(1-q3)
                ex += Ex_Jefi[ind_x, ind_y, ind_z]*(1-q1)*(1-q2)*(1-q3)
                ey += Ey_Jefi[ind_x, ind_y, ind_z]*(1-q1)*(1-q2)*(1-q3)
                ez += Ez_Jefi[ind_x, ind_y, ind_z]*(1-q1)*(1-q2)*(1-q3)
                bx += Bx_Jefi[ind_x, ind_y, ind_z+1]*(1-q1)*(1-q2)*q3
                by += By_Jefi[ind_x, ind_y, ind_z+1]*(1-q1)*(1-q2)*q3
                bz += Bz_Jefi[ind_x, ind_y, ind_z+1]*(1-q1)*(1-q2)*q3
                ex += Ex_Jefi[ind_x, ind_y, ind_z+1]*(1-q1)*(1-q2)*q3
                ey += Ey_Jefi[ind_x, ind_y, ind_z+1]*(1-q1)*(1-q2)*q3
                ez += Ez_Jefi[ind_x, ind_y, ind_z+1]*(1-q1)*(1-q2)*q3                      
            elif ind_x == (nx - 1) and ind_y == (ny - 1) and ind_z == (nz - 1): #(2,2,2)
                bx += Bx_Jefi[ind_x, ind_y, ind_z]*(1-q1)*(1-q2)*(1-q3)
                by += By_Jefi[ind_x, ind_y, ind_z]*(1-q1)*(1-q2)*(1-q3)
                bz += Bz_Jefi[ind_x, ind_y, ind_z]*(1-q1)*(1-q2)*(1-q3)
                ex += Ex_Jefi[ind_x, ind_y, ind_z]*(1-q1)*(1-q2)*(1-q3)
                ey += Ey_Jefi[ind_x, ind_y, ind_z]*(1-q1)*(1-q2)*(1-q3)
                ez += Ez_Jefi[ind_x, ind_y, ind_z]*(1-q1)*(1-q2)*(1-q3)
            elif ind_x == -1 and ind_y == -1 and ind_z == -1: #(0,0,0)
                bx += Bx_Jefi[ind_x+1, ind_y+1, ind_z+1]*q1*q2*q3
                by += By_Jefi[ind_x+1, ind_y+1, ind_z+1]*q1*q2*q3
                bz += Bz_Jefi[ind_x+1, ind_y+1, ind_z+1]*q1*q2*q3
                ex += Ex_Jefi[ind_x+1, ind_y+1, ind_z+1]*q1*q2*q3
                ey += Ey_Jefi[ind_x+1, ind_y+1, ind_z+1]*q1*q2*q3
                ez += Ez_Jefi[ind_x+1, ind_y+1, ind_z+1]*q1*q2*q3
            elif ind_x == (nx - 1) and ind_y == -1 and ind_z == -1: #(2,0,0)
                bx += Bx_Jefi[ind_x, ind_y+1, ind_z+1]*(1-q1)*q2*q3
                by += By_Jefi[ind_x, ind_y+1, ind_z+1]*(1-q1)*q2*q3
                bz += Bz_Jefi[ind_x, ind_y+1, ind_z+1]*(1-q1)*q2*q3
                ex += Ex_Jefi[ind_x, ind_y+1, ind_z+1]*(1-q1)*q2*q3
                ey += Ey_Jefi[ind_x, ind_y+1, ind_z+1]*(1-q1)*q2*q3
                ez += Ez_Jefi[ind_x, ind_y+1, ind_z+1]*(1-q1)*q2*q3
            elif ind_x == -1 and ind_y == (ny - 1) and ind_z == -1: #(0,2,0)
                bx += Bx_Jefi[ind_x+1, ind_y, ind_z+1]*q1*(1-q2)*q3
                by += By_Jefi[ind_x+1, ind_y, ind_z+1]*q1*(1-q2)*q3
                bz += Bz_Jefi[ind_x+1, ind_y, ind_z+1]*q1*(1-q2)*q3
                ex += Ex_Jefi[ind_x+1, ind_y, ind_z+1]*q1*(1-q2)*q3
                ey += Ey_Jefi[ind_x+1, ind_y, ind_z+1]*q1*(1-q2)*q3
                ez += Ez_Jefi[ind_x+1, ind_y, ind_z+1]*q1*(1-q2)*q3
            elif ind_x == -1 and ind_y == -1 and ind_z == (nz - 1): #(0,0,2)
                bx += Bx_Jefi[ind_x+1, ind_y+1, ind_z]*q1*q2*(1-q3)
                by += By_Jefi[ind_x+1, ind_y+1, ind_z]*q1*q2*(1-q3)
                bz += Bz_Jefi[ind_x+1, ind_y+1, ind_z]*q1*q2*(1-q3)
                ex += Ex_Jefi[ind_x+1, ind_y+1, ind_z]*q1*q2*(1-q3)
                ey += Ey_Jefi[ind_x+1, ind_y+1, ind_z]*q1*q2*(1-q3)
                ez += Ez_Jefi[ind_x+1, ind_y+1, ind_z]*q1*q2*(1-q3)
            elif ind_x > -1 and ind_x < nx - 1 and ind_y == -1 and ind_z == -1: #(1,0,0)
                bx += Bx_Jefi[ind_x, ind_y+1, ind_z+1]*(1-q1)*q2*q3
                by += By_Jefi[ind_x, ind_y+1, ind_z+1]*(1-q1)*q2*q3
                bz += Bz_Jefi[ind_x, ind_y+1, ind_z+1]*(1-q1)*q2*q3
                ex += Ex_Jefi[ind_x, ind_y+1, ind_z+1]*(1-q1)*q2*q3
                ey += Ey_Jefi[ind_x, ind_y+1, ind_z+1]*(1-q1)*q2*q3
                ez += Ez_Jefi[ind_x, ind_y+1, ind_z+1]*(1-q1)*q2*q3
                bx += Bx_Jefi[ind_x+1, ind_y+1, ind_z+1]*q1*q2*q3
                by += By_Jefi[ind_x+1, ind_y+1, ind_z+1]*q1*q2*q3
                bz += Bz_Jefi[ind_x+1, ind_y+1, ind_z+1]*q1*q2*q3
                ex += Ex_Jefi[ind_x+1, ind_y+1, ind_z+1]*q1*q2*q3
                ey += Ey_Jefi[ind_x+1, ind_y+1, ind_z+1]*q1*q2*q3
                ez += Ez_Jefi[ind_x+1, ind_y+1, ind_z+1]*q1*q2*q3
            elif ind_x == -1 and ind_y > -1 and ind_y < ny - 1 and ind_z == -1: #(0,1,0)
                bx += Bx_Jefi[ind_x+1, ind_y, ind_z+1]*q1*(1-q2)*q3
                by += By_Jefi[ind_x+1, ind_y, ind_z+1]*q1*(1-q2)*q3
                bz += Bz_Jefi[ind_x+1, ind_y, ind_z+1]*q1*(1-q2)*q3
                ex += Ex_Jefi[ind_x+1, ind_y, ind_z+1]*q1*(1-q2)*q3
                ey += Ey_Jefi[ind_x+1, ind_y, ind_z+1]*q1*(1-q2)*q3
                ez += Ez_Jefi[ind_x+1, ind_y, ind_z+1]*q1*(1-q2)*q3
                bx += Bx_Jefi[ind_x+1, ind_y+1, ind_z+1]*q1*q2*q3
                by += By_Jefi[ind_x+1, ind_y+1, ind_z+1]*q1*q2*q3
                bz += Bz_Jefi[ind_x+1, ind_y+1, ind_z+1]*q1*q2*q3
                ex += Ex_Jefi[ind_x+1, ind_y+1, ind_z+1]*q1*q2*q3
                ey += Ey_Jefi[ind_x+1, ind_y+1, ind_z+1]*q1*q2*q3
                ez += Ez_Jefi[ind_x+1, ind_y+1, ind_z+1]*q1*q2*q3
            elif ind_x == -1 and ind_y == -1 and ind_z > -1 and ind_z < nz - 1: #(0,0,1)
                bx += Bx_Jefi[ind_x+1, ind_y+1, ind_z]*q1*q2*(1-q3)
                by += By_Jefi[ind_x+1, ind_y+1, ind_z]*q1*q2*(1-q3)
                bz += Bz_Jefi[ind_x+1, ind_y+1, ind_z]*q1*q2*(1-q3)
                ex += Ex_Jefi[ind_x+1, ind_y+1, ind_z]*q1*q2*(1-q3)
                ey += Ey_Jefi[ind_x+1, ind_y+1, ind_z]*q1*q2*(1-q3)
                ez += Ez_Jefi[ind_x+1, ind_y+1, ind_z]*q1*q2*(1-q3)
                bx += Bx_Jefi[ind_x+1, ind_y+1, ind_z+1]*q1*q2*q3
                by += By_Jefi[ind_x+1, ind_y+1, ind_z+1]*q1*q2*q3
                bz += Bz_Jefi[ind_x+1, ind_y+1, ind_z+1]*q1*q2*q3
                ex += Ex_Jefi[ind_x+1, ind_y+1, ind_z+1]*q1*q2*q3
                ey += Ey_Jefi[ind_x+1, ind_y+1, ind_z+1]*q1*q2*q3
                ez += Ez_Jefi[ind_x+1, ind_y+1, ind_z+1]*q1*q2*q3
            elif ind_x == -1 and ind_y > -1 and ind_y < ny - 1 and ind_z > -1 and ind_z < nz - 1: #(0,1,1)
                bx += Bx_Jefi[ind_x+1, ind_y, ind_z]*q1*(1-q2)*(1-q3)
                by += By_Jefi[ind_x+1, ind_y, ind_z]*q1*(1-q2)*(1-q3)
                bz += Bz_Jefi[ind_x+1, ind_y, ind_z]*q1*(1-q2)*(1-q3)
                ex += Ex_Jefi[ind_x+1, ind_y, ind_z]*q1*(1-q2)*(1-q3)
                ey += Ey_Jefi[ind_x+1, ind_y, ind_z]*q1*(1-q2)*(1-q3)
                ez += Ez_Jefi[ind_x+1, ind_y, ind_z]*q1*(1-q2)*(1-q3)
                bx += Bx_Jefi[ind_x+1, ind_y+1, ind_z]*q1*q2*(1-q3)
                by += By_Jefi[ind_x+1, ind_y+1, ind_z]*q1*q2*(1-q3)
                bz += Bz_Jefi[ind_x+1, ind_y+1, ind_z]*q1*q2*(1-q3)
                ex += Ex_Jefi[ind_x+1, ind_y+1, ind_z]*q1*q2*(1-q3)
                ey += Ey_Jefi[ind_x+1, ind_y+1, ind_z]*q1*q2*(1-q3)
                ez += Ez_Jefi[ind_x+1, ind_y+1, ind_z]*q1*q2*(1-q3)
                bx += Bx_Jefi[ind_x+1, ind_y, ind_z+1]*q1*(1-q2)*q3
                by += By_Jefi[ind_x+1, ind_y, ind_z+1]*q1*(1-q2)*q3
                bz += Bz_Jefi[ind_x+1, ind_y, ind_z+1]*q1*(1-q2)*q3
                ex += Ex_Jefi[ind_x+1, ind_y, ind_z+1]*q1*(1-q2)*q3
                ey += Ey_Jefi[ind_x+1, ind_y, ind_z+1]*q1*(1-q2)*q3
                ez += Ez_Jefi[ind_x+1, ind_y, ind_z+1]*q1*(1-q2)*q3
                bx += Bx_Jefi[ind_x+1, ind_y+1, ind_z+1]*q1*q2*q3
                by += By_Jefi[ind_x+1, ind_y+1, ind_z+1]*q1*q2*q3
                bz += Bz_Jefi[ind_x+1, ind_y+1, ind_z+1]*q1*q2*q3
                ex += Ex_Jefi[ind_x+1, ind_y+1, ind_z+1]*q1*q2*q3
                ey += Ey_Jefi[ind_x+1, ind_y+1, ind_z+1]*q1*q2*q3
                ez += Ez_Jefi[ind_x+1, ind_y+1, ind_z+1]*q1*q2*q3
            elif ind_x > -1 and ind_x < nx - 1 and ind_y == -1 and ind_z > -1 and ind_z < nz - 1: #(1,0,1)
                bx += Bx_Jefi[ind_x, ind_y+1, ind_z]*(1-q1)*q2*(1-q3)
                by += By_Jefi[ind_x, ind_y+1, ind_z]*(1-q1)*q2*(1-q3)
                bz += Bz_Jefi[ind_x, ind_y+1, ind_z]*(1-q1)*q2*(1-q3)
                ex += Ex_Jefi[ind_x, ind_y+1, ind_z]*(1-q1)*q2*(1-q3)
                ey += Ey_Jefi[ind_x, ind_y+1, ind_z]*(1-q1)*q2*(1-q3)
                ez += Ez_Jefi[ind_x, ind_y+1, ind_z]*(1-q1)*q2*(1-q3)
                bx += Bx_Jefi[ind_x+1, ind_y+1, ind_z]*q1*q2*(1-q3)
                by += By_Jefi[ind_x+1, ind_y+1, ind_z]*q1*q2*(1-q3)
                bz += Bz_Jefi[ind_x+1, ind_y+1, ind_z]*q1*q2*(1-q3)
                ex += Ex_Jefi[ind_x+1, ind_y+1, ind_z]*q1*q2*(1-q3)
                ey += Ey_Jefi[ind_x+1, ind_y+1, ind_z]*q1*q2*(1-q3)
                ez += Ez_Jefi[ind_x+1, ind_y+1, ind_z]*q1*q2*(1-q3)
                bx += Bx_Jefi[ind_x, ind_y+1, ind_z+1]*(1-q1)*q2*q3
                by += By_Jefi[ind_x, ind_y+1, ind_z+1]*(1-q1)*q2*q3
                bz += Bz_Jefi[ind_x, ind_y+1, ind_z+1]*(1-q1)*q2*q3
                ex += Ex_Jefi[ind_x, ind_y+1, ind_z+1]*(1-q1)*q2*q3
                ey += Ey_Jefi[ind_x, ind_y+1, ind_z+1]*(1-q1)*q2*q3
                ez += Ez_Jefi[ind_x, ind_y+1, ind_z+1]*(1-q1)*q2*q3
                bx += Bx_Jefi[ind_x+1, ind_y+1, ind_z+1]*q1*q2*q3
                by += By_Jefi[ind_x+1, ind_y+1, ind_z+1]*q1*q2*q3
                bz += Bz_Jefi[ind_x+1, ind_y+1, ind_z+1]*q1*q2*q3
                ex += Ex_Jefi[ind_x+1, ind_y+1, ind_z+1]*q1*q2*q3
                ey += Ey_Jefi[ind_x+1, ind_y+1, ind_z+1]*q1*q2*q3
                ez += Ez_Jefi[ind_x+1, ind_y+1, ind_z+1]*q1*q2*q3
            elif ind_x > -1 and ind_x < nx - 1 and ind_y > -1 and ind_y < ny - 1 and ind_z == -1: #(1,1,0)
                bx += Bx_Jefi[ind_x, ind_y, ind_z+1]*(1-q1)*(1-q2)*q3
                by += By_Jefi[ind_x, ind_y, ind_z+1]*(1-q1)*(1-q2)*q3
                bz += Bz_Jefi[ind_x, ind_y, ind_z+1]*(1-q1)*(1-q2)*q3
                ex += Ex_Jefi[ind_x, ind_y, ind_z+1]*(1-q1)*(1-q2)*q3
                ey += Ey_Jefi[ind_x, ind_y, ind_z+1]*(1-q1)*(1-q2)*q3
                ez += Ez_Jefi[ind_x, ind_y, ind_z+1]*(1-q1)*(1-q2)*q3
                bx += Bx_Jefi[ind_x, ind_y+1, ind_z+1]*(1-q1)*q2*q3
                by += By_Jefi[ind_x, ind_y+1, ind_z+1]*(1-q1)*q2*q3
                bz += Bz_Jefi[ind_x, ind_y+1, ind_z+1]*(1-q1)*q2*q3
                ex += Ex_Jefi[ind_x, ind_y+1, ind_z+1]*(1-q1)*q2*q3
                ey += Ey_Jefi[ind_x, ind_y+1, ind_z+1]*(1-q1)*q2*q3
                ez += Ez_Jefi[ind_x, ind_y+1, ind_z+1]*(1-q1)*q2*q3
                bx += Bx_Jefi[ind_x+1, ind_y, ind_z+1]*q1*(1-q2)*q3
                by += By_Jefi[ind_x+1, ind_y, ind_z+1]*q1*(1-q2)*q3
                bz += Bz_Jefi[ind_x+1, ind_y, ind_z+1]*q1*(1-q2)*q3
                ex += Ex_Jefi[ind_x+1, ind_y, ind_z+1]*q1*(1-q2)*q3
                ey += Ey_Jefi[ind_x+1, ind_y, ind_z+1]*q1*(1-q2)*q3
                ez += Ez_Jefi[ind_x+1, ind_y, ind_z+1]*q1*(1-q2)*q3
                bx += Bx_Jefi[ind_x+1, ind_y+1, ind_z+1]*q1*q2*q3
                by += By_Jefi[ind_x+1, ind_y+1, ind_z+1]*q1*q2*q3
                bz += Bz_Jefi[ind_x+1, ind_y+1, ind_z+1]*q1*q2*q3
                ex += Ex_Jefi[ind_x+1, ind_y+1, ind_z+1]*q1*q2*q3
                ey += Ey_Jefi[ind_x+1, ind_y+1, ind_z+1]*q1*q2*q3
                ez += Ez_Jefi[ind_x+1, ind_y+1, ind_z+1]*q1*q2*q3
            elif ind_x == -1 and ind_y > -1 and ind_y < ny - 1 and ind_z == (nz - 1): #(0,1,2)
                bx += Bx_Jefi[ind_x+1, ind_y, ind_z]*q1*(1-q2)*(1-q3)
                by += By_Jefi[ind_x+1, ind_y, ind_z]*q1*(1-q2)*(1-q3)
                bz += Bz_Jefi[ind_x+1, ind_y, ind_z]*q1*(1-q2)*(1-q3)
                ex += Ex_Jefi[ind_x+1, ind_y, ind_z]*q1*(1-q2)*(1-q3)
                ey += Ey_Jefi[ind_x+1, ind_y, ind_z]*q1*(1-q2)*(1-q3)
                ez += Ez_Jefi[ind_x+1, ind_y, ind_z]*q1*(1-q2)*(1-q3)
                bx += Bx_Jefi[ind_x+1, ind_y+1, ind_z]*q1*q2*(1-q3)
                by += By_Jefi[ind_x+1, ind_y+1, ind_z]*q1*q2*(1-q3)
                bz += Bz_Jefi[ind_x+1, ind_y+1, ind_z]*q1*q2*(1-q3)
                ex += Ex_Jefi[ind_x+1, ind_y+1, ind_z]*q1*q2*(1-q3)
                ey += Ey_Jefi[ind_x+1, ind_y+1, ind_z]*q1*q2*(1-q3)
                ez += Ez_Jefi[ind_x+1, ind_y+1, ind_z]*q1*q2*(1-q3)
            elif ind_x > -1 and ind_x < nx - 1 and ind_y == -1 and ind_z == (nz - 1): #(1,0,2)
                bx += Bx_Jefi[ind_x, ind_y+1, ind_z]*(1-q1)*q2*(1-q3)
                by += By_Jefi[ind_x, ind_y+1, ind_z]*(1-q1)*q2*(1-q3)
                bz += Bz_Jefi[ind_x, ind_y+1, ind_z]*(1-q1)*q2*(1-q3)
                ex += Ex_Jefi[ind_x, ind_y+1, ind_z]*(1-q1)*q2*(1-q3)
                ey += Ey_Jefi[ind_x, ind_y+1, ind_z]*(1-q1)*q2*(1-q3)
                ez += Ez_Jefi[ind_x, ind_y+1, ind_z]*(1-q1)*q2*(1-q3)
                bx += Bx_Jefi[ind_x+1, ind_y+1, ind_z]*q1*q2*(1-q3)
                by += By_Jefi[ind_x+1, ind_y+1, ind_z]*q1*q2*(1-q3)
                bz += Bz_Jefi[ind_x+1, ind_y+1, ind_z]*q1*q2*(1-q3)
                ex += Ex_Jefi[ind_x+1, ind_y+1, ind_z]*q1*q2*(1-q3)
                ey += Ey_Jefi[ind_x+1, ind_y+1, ind_z]*q1*q2*(1-q3)
                ez += Ez_Jefi[ind_x+1, ind_y+1, ind_z]*q1*q2*(1-q3)
            elif ind_x > -1 and ind_x < nx - 1 and ind_y == (ny - 1) and ind_z == -1: #(1,2,0)
                bx += Bx_Jefi[ind_x, ind_y, ind_z+1]*(1-q1)*(1-q2)*q3
                by += By_Jefi[ind_x, ind_y, ind_z+1]*(1-q1)*(1-q2)*q3
                bz += Bz_Jefi[ind_x, ind_y, ind_z+1]*(1-q1)*(1-q2)*q3
                ex += Ex_Jefi[ind_x, ind_y, ind_z+1]*(1-q1)*(1-q2)*q3
                ey += Ey_Jefi[ind_x, ind_y, ind_z+1]*(1-q1)*(1-q2)*q3
                ez += Ez_Jefi[ind_x, ind_y, ind_z+1]*(1-q1)*(1-q2)*q3
                bx += Bx_Jefi[ind_x+1, ind_y, ind_z+1]*q1*(1-q2)*q3
                by += By_Jefi[ind_x+1, ind_y, ind_z+1]*q1*(1-q2)*q3
                bz += Bz_Jefi[ind_x+1, ind_y, ind_z+1]*q1*(1-q2)*q3
                ex += Ex_Jefi[ind_x+1, ind_y, ind_z+1]*q1*(1-q2)*q3
                ey += Ey_Jefi[ind_x+1, ind_y, ind_z+1]*q1*(1-q2)*q3
                ez += Ez_Jefi[ind_x+1, ind_y, ind_z+1]*q1*(1-q2)*q3
            elif ind_x == -1 and ind_y == (ny - 1) and ind_z > -1 and ind_z < nz - 1: #(0,2,1)
                bx += Bx_Jefi[ind_x+1, ind_y, ind_z]*q1*(1-q2)*(1-q3)
                by += By_Jefi[ind_x+1, ind_y, ind_z]*q1*(1-q2)*(1-q3)
                bz += Bz_Jefi[ind_x+1, ind_y, ind_z]*q1*(1-q2)*(1-q3)
                ex += Ex_Jefi[ind_x+1, ind_y, ind_z]*q1*(1-q2)*(1-q3)
                ey += Ey_Jefi[ind_x+1, ind_y, ind_z]*q1*(1-q2)*(1-q3)
                ez += Ez_Jefi[ind_x+1, ind_y, ind_z]*q1*(1-q2)*(1-q3)
                bx += Bx_Jefi[ind_x+1, ind_y, ind_z+1]*q1*(1-q2)*q3
                by += By_Jefi[ind_x+1, ind_y, ind_z+1]*q1*(1-q2)*q3
                bz += Bz_Jefi[ind_x+1, ind_y, ind_z+1]*q1*(1-q2)*q3
                ex += Ex_Jefi[ind_x+1, ind_y, ind_z+1]*q1*(1-q2)*q3
                ey += Ey_Jefi[ind_x+1, ind_y, ind_z+1]*q1*(1-q2)*q3
                ez += Ez_Jefi[ind_x+1, ind_y, ind_z+1]*q1*(1-q2)*q3
            elif ind_x == (nx - 1) and ind_y == -1 and ind_z > -1 and ind_z < nz - 1: #(2,0,1)
                bx += Bx_Jefi[ind_x, ind_y+1, ind_z]*(1-q1)*q2*(1-q3)
                by += By_Jefi[ind_x, ind_y+1, ind_z]*(1-q1)*q2*(1-q3)
                bz += Bz_Jefi[ind_x, ind_y+1, ind_z]*(1-q1)*q2*(1-q3)
                ex += Ex_Jefi[ind_x, ind_y+1, ind_z]*(1-q1)*q2*(1-q3)
                ey += Ey_Jefi[ind_x, ind_y+1, ind_z]*(1-q1)*q2*(1-q3)
                ez += Ez_Jefi[ind_x, ind_y+1, ind_z]*(1-q1)*q2*(1-q3)
                bx += Bx_Jefi[ind_x, ind_y+1, ind_z+1]*(1-q1)*q2*q3
                by += By_Jefi[ind_x, ind_y+1, ind_z+1]*(1-q1)*q2*q3
                bz += Bz_Jefi[ind_x, ind_y+1, ind_z+1]*(1-q1)*q2*q3
                ex += Ex_Jefi[ind_x, ind_y+1, ind_z+1]*(1-q1)*q2*q3
                ey += Ey_Jefi[ind_x, ind_y+1, ind_z+1]*(1-q1)*q2*q3
                ez += Ez_Jefi[ind_x, ind_y+1, ind_z+1]*(1-q1)*q2*q3
            elif ind_x == (nx - 1) and ind_y > -1 and ind_y < nz - 1 and ind_z == -1: #(2,1,0)
                bx += Bx_Jefi[ind_x, ind_y, ind_z+1]*(1-q1)*(1-q2)*q3
                by += By_Jefi[ind_x, ind_y, ind_z+1]*(1-q1)*(1-q2)*q3
                bz += Bz_Jefi[ind_x, ind_y, ind_z+1]*(1-q1)*(1-q2)*q3
                ex += Ex_Jefi[ind_x, ind_y, ind_z+1]*(1-q1)*(1-q2)*q3
                ey += Ey_Jefi[ind_x, ind_y, ind_z+1]*(1-q1)*(1-q2)*q3
                ez += Ez_Jefi[ind_x, ind_y, ind_z+1]*(1-q1)*(1-q2)*q3
                bx += Bx_Jefi[ind_x, ind_y+1, ind_z+1]*(1-q1)*q2*q3
                by += By_Jefi[ind_x, ind_y+1, ind_z+1]*(1-q1)*q2*q3
                bz += Bz_Jefi[ind_x, ind_y+1, ind_z+1]*(1-q1)*q2*q3
                ex += Ex_Jefi[ind_x, ind_y+1, ind_z+1]*(1-q1)*q2*q3
                ey += Ey_Jefi[ind_x, ind_y+1, ind_z+1]*(1-q1)*q2*q3
                ez += Ez_Jefi[ind_x, ind_y+1, ind_z+1]*(1-q1)*q2*q3
            elif ind_x == -1 and ind_y == (ny - 1) and ind_z == (nz - 1): #(0,2,2)
                bx += Bx_Jefi[ind_x+1, ind_y, ind_z]*q1*(1-q2)*(1-q3)
                by += By_Jefi[ind_x+1, ind_y, ind_z]*q1*(1-q2)*(1-q3)
                bz += Bz_Jefi[ind_x+1, ind_y, ind_z]*q1*(1-q2)*(1-q3)
                ex += Ex_Jefi[ind_x+1, ind_y, ind_z]*q1*(1-q2)*(1-q3)
                ey += Ey_Jefi[ind_x+1, ind_y, ind_z]*q1*(1-q2)*(1-q3)
                ez += Ez_Jefi[ind_x+1, ind_y, ind_z]*q1*(1-q2)*(1-q3)
            elif ind_x == (nx - 1) and ind_y == -1 and ind_z == (nz - 1): #(2,0,2)
                bx += Bx_Jefi[ind_x, ind_y+1, ind_z]*(1-q1)*q2*(1-q3)
                by += By_Jefi[ind_x, ind_y+1, ind_z]*(1-q1)*q2*(1-q3)
                bz += Bz_Jefi[ind_x, ind_y+1, ind_z]*(1-q1)*q2*(1-q3)
                ex += Ex_Jefi[ind_x, ind_y+1, ind_z]*(1-q1)*q2*(1-q3)
                ey += Ey_Jefi[ind_x, ind_y+1, ind_z]*(1-q1)*q2*(1-q3)
                ez += Ez_Jefi[ind_x, ind_y+1, ind_z]*(1-q1)*q2*(1-q3)
            elif ind_x == (nx - 1) and ind_y == (ny - 1) and ind_z == -1: #(2,2,0) 
                bx += Bx_Jefi[ind_x, ind_y, ind_z+1]*(1-q1)*(1-q2)*q3
                by += By_Jefi[ind_x, ind_y, ind_z+1]*(1-q1)*(1-q2)*q3
                bz += Bz_Jefi[ind_x, ind_y, ind_z+1]*(1-q1)*(1-q2)*q3
                ex += Ex_Jefi[ind_x, ind_y, ind_z+1]*(1-q1)*(1-q2)*q3
                ey += Ey_Jefi[ind_x, ind_y, ind_z+1]*(1-q1)*(1-q2)*q3
                ez += Ez_Jefi[ind_x, ind_y, ind_z+1]*(1-q1)*(1-q2)*q3
            elif ind_x == -1 and ind_y == -1 and ind_z == -1: #(0,0,0) 
                bx += Bx_Jefi[ind_x+1, ind_y+1, ind_z+1]*q1*q2*q3
                by += By_Jefi[ind_x+1, ind_y+1, ind_z+1]*q1*q2*q3
                bz += Bz_Jefi[ind_x+1, ind_y+1, ind_z+1]*q1*q2*q3
                ex += Ex_Jefi[ind_x+1, ind_y+1, ind_z+1]*q1*q2*q3
                ey += Ey_Jefi[ind_x+1, ind_y+1, ind_z+1]*q1*q2*q3
                ez += Ez_Jefi[ind_x+1, ind_y+1, ind_z+1]*q1*q2*q3
            
            # if ind_x >= 0 and ind_y >= 0 and ind_z >= 0: # the particles belong to the regions with no lower boundary
            #     bx += Bx_Jefi[ind_x, ind_y, ind_z]*(1-q1)*(1-q2)*(1-q3)
            #     by += By_Jefi[ind_x, ind_y, ind_z]*(1-q1)*(1-q2)*(1-q3)
            #     bz += Bz_Jefi[ind_x, ind_y, ind_z]*(1-q1)*(1-q2)*(1-q3)
            #     ex += Ex_Jefi[ind_x, ind_y, ind_z]*(1-q1)*(1-q2)*(1-q3)
            #     ey += Ey_Jefi[ind_x, ind_y, ind_z]*(1-q1)*(1-q2)*(1-q3)
            #     ez += Ez_Jefi[ind_x, ind_y, ind_z]*(1-q1)*(1-q2)*(1-q3)
            #     if ind_x+1 < nx:
            #         bx += Bx_Jefi[ind_x+1, ind_y, ind_z]*q1*(1-q2)*(1-q3)
            #         by += By_Jefi[ind_x+1, ind_y, ind_z]*q1*(1-q2)*(1-q3)
            #         bz += Bz_Jefi[ind_x+1, ind_y, ind_z]*q1*(1-q2)*(1-q3)
            #         ex += Ex_Jefi[ind_x+1, ind_y, ind_z]*q1*(1-q2)*(1-q3)
            #         ey += Ey_Jefi[ind_x+1, ind_y, ind_z]*q1*(1-q2)*(1-q3)
            #         ez += Ez_Jefi[ind_x+1, ind_y, ind_z]*q1*(1-q2)*(1-q3)
            #     if ind_y+1 < ny:
            #         bx += Bx_Jefi[ind_x, ind_y+1, ind_z]*(1-q1)*q2*(1-q3)
            #         by += By_Jefi[ind_x, ind_y+1, ind_z]*(1-q1)*q2*(1-q3)
            #         bz += Bz_Jefi[ind_x, ind_y+1, ind_z]*(1-q1)*q2*(1-q3)
            #         ex += Ex_Jefi[ind_x, ind_y+1, ind_z]*(1-q1)*q2*(1-q3)
            #         ey += Ey_Jefi[ind_x, ind_y+1, ind_z]*(1-q1)*q2*(1-q3)
            #         ez += Ez_Jefi[ind_x, ind_y+1, ind_z]*(1-q1)*q2*(1-q3)
            #     if ind_x+1 < nx and ind_y+1 < ny:
            #         bx += Bx_Jefi[ind_x+1, ind_y+1, ind_z]*q1*q2*(1-q3)
            #         by += By_Jefi[ind_x+1, ind_y+1, ind_z]*q1*q2*(1-q3)
            #         bz += Bz_Jefi[ind_x+1, ind_y+1, ind_z]*q1*q2*(1-q3)
            #         ex += Ex_Jefi[ind_x+1, ind_y+1, ind_z]*q1*q2*(1-q3)
            #         ey += Ey_Jefi[ind_x+1, ind_y+1, ind_z]*q1*q2*(1-q3)
            #         ez += Ez_Jefi[ind_x+1, ind_y+1, ind_z]*q1*q2*(1-q3)
            #     if ind_z+1 < nz:
            #         bx += Bx_Jefi[ind_x, ind_y, ind_z+1]*(1-q1)*(1-q2)*q3
            #         by += By_Jefi[ind_x, ind_y, ind_z+1]*(1-q1)*(1-q2)*q3
            #         bz += Bz_Jefi[ind_x, ind_y, ind_z+1]*(1-q1)*(1-q2)*q3
            #         ex += Ex_Jefi[ind_x, ind_y, ind_z+1]*(1-q1)*(1-q2)*q3
            #         ey += Ey_Jefi[ind_x, ind_y, ind_z+1]*(1-q1)*(1-q2)*q3
            #         ez += Ez_Jefi[ind_x, ind_y, ind_z+1]*(1-q1)*(1-q2)*q3
            #     if ind_x+1 < nx and ind_z+1 < nz:
            #         bx += Bx_Jefi[ind_x+1, ind_y, ind_z+1]*q1*(1-q2)*q3
            #         by += By_Jefi[ind_x+1, ind_y, ind_z+1]*q1*(1-q2)*q3
            #         bz += Bz_Jefi[ind_x+1, ind_y, ind_z+1]*q1*(1-q2)*q3
            #         ex += Ex_Jefi[ind_x+1, ind_y, ind_z+1]*q1*(1-q2)*q3
            #         ey += Ey_Jefi[ind_x+1, ind_y, ind_z+1]*q1*(1-q2)*q3
            #         ez += Ez_Jefi[ind_x+1, ind_y, ind_z+1]*q1*(1-q2)*q3
            #     if ind_y+1 < ny and ind_z+1 < nz:
            #         bx += Bx_Jefi[ind_x, ind_y+1, ind_z+1]*(1-q1)*q2*q3
            #         by += By_Jefi[ind_x, ind_y+1, ind_z+1]*(1-q1)*q2*q3
            #         bz += Bz_Jefi[ind_x, ind_y+1, ind_z+1]*(1-q1)*q2*q3
            #         ex += Ex_Jefi[ind_x, ind_y+1, ind_z+1]*(1-q1)*q2*q3
            #         ey += Ey_Jefi[ind_x, ind_y+1, ind_z+1]*(1-q1)*q2*q3
            #         ez += Ez_Jefi[ind_x, ind_y+1, ind_z+1]*(1-q1)*q2*q3
            #     if ind_x+1 < nx and ind_y+1 < ny and ind_z+1 < nz:
            #         bx += Bx_Jefi[ind_x+1, ind_y+1, ind_z+1]*q1*q2*q3
            #         by += By_Jefi[ind_x+1, ind_y+1, ind_z+1]*q1*q2*q3
            #         bz += Bz_Jefi[ind_x+1, ind_y+1, ind_z+1]*q1*q2*q3
            #         ex += Ex_Jefi[ind_x+1, ind_y+1, ind_z+1]*q1*q2*q3
            #         ey += Ey_Jefi[ind_x+1, ind_y+1, ind_z+1]*q1*q2*q3
            #         ez += Ez_Jefi[ind_x+1, ind_y+1, ind_z+1]*q1*q2*q3
            # else: # regions with at least one lower boundary
            #     if ind_x != nx - 1 and ind_y != ny - 1 and ind_z != nz - 1:
            #         bx += Bx_Jefi[ind_x+1, ind_y+1, ind_z+1]*q1*q2*q3
            #         by += By_Jefi[ind_x+1, ind_y+1, ind_z+1]*q1*q2*q3
            #         bz += Bz_Jefi[ind_x+1, ind_y+1, ind_z+1]*q1*q2*q3
            #         ex += Ex_Jefi[ind_x+1, ind_y+1, ind_z+1]*q1*q2*q3
            #         ey += Ey_Jefi[ind_x+1, ind_y+1, ind_z+1]*q1*q2*q3
            #         ez += Ez_Jefi[ind_x+1, ind_y+1, ind_z+1]*q1*q2*q3
            #     if ind_x != -1:
            #         bx += Bx_Jefi[ind_x, ind_y+1, ind_z+1]*(1-q1)*q2*q3
            #         by += By_Jefi[ind_x, ind_y+1, ind_z+1]*(1-q1)*q2*q3
            #         bz += Bz_Jefi[ind_x, ind_y+1, ind_z+1]*(1-q1)*q2*q3
            #         ex += Ex_Jefi[ind_x, ind_y+1, ind_z+1]*(1-q1)*q2*q3
            #         ey += Ey_Jefi[ind_x, ind_y+1, ind_z+1]*(1-q1)*q2*q3
            #         ez += Ez_Jefi[ind_x, ind_y+1, ind_z+1]*(1-q1)*q2*q3
            #     if ind_y != -1:
            #         bx += Bx_Jefi[ind_x+1, ind_y, ind_z+1]*q1*(1-q2)*q3
            #         by += By_Jefi[ind_x+1, ind_y, ind_z+1]*q1*(1-q2)*q3
            #         bz += Bz_Jefi[ind_x+1, ind_y, ind_z+1]*q1*(1-q2)*q3
            #         ex += Ex_Jefi[ind_x+1, ind_y, ind_z+1]*q1*(1-q2)*q3
            #         ey += Ey_Jefi[ind_x+1, ind_y, ind_z+1]*q1*(1-q2)*q3
            #         ez += Ez_Jefi[ind_x+1, ind_y, ind_z+1]*q1*(1-q2)*q3
            #     if ind_x != -1 and ind_y != -1:
            #         bx += Bx_Jefi[ind_x, ind_y, ind_z+1]*(1-q1)*(1-q2)*q3
            #         by += By_Jefi[ind_x, ind_y, ind_z+1]*(1-q1)*(1-q2)*q3
            #         bz += Bz_Jefi[ind_x, ind_y, ind_z+1]*(1-q1)*(1-q2)*q3
            #         ex += Ex_Jefi[ind_x, ind_y, ind_z+1]*(1-q1)*(1-q2)*q3
            #         ey += Ey_Jefi[ind_x, ind_y, ind_z+1]*(1-q1)*(1-q2)*q3
            #         ez += Ez_Jefi[ind_x, ind_y, ind_z+1]*(1-q1)*(1-q2)*q3
            #     if ind_z != -1:
            #         bx += Bx_Jefi[ind_x+1, ind_y+1, ind_z]*q1*q2*(1-q3)
            #         by += By_Jefi[ind_x+1, ind_y+1, ind_z]*q1*q2*(1-q3)
            #         bz += Bz_Jefi[ind_x+1, ind_y+1, ind_z]*q1*q2*(1-q3)
            #         ex += Ex_Jefi[ind_x+1, ind_y+1, ind_z]*q1*q2*(1-q3)
            #         ey += Ey_Jefi[ind_x+1, ind_y+1, ind_z]*q1*q2*(1-q3)
            #         ez += Ez_Jefi[ind_x+1, ind_y+1, ind_z]*q1*q2*(1-q3)
            #     if ind_x != -1 and ind_z != -1:
            #         bx += Bx_Jefi[ind_x, ind_y+1, ind_z]*(1-q1)*q2*(1-q3)
            #         by += By_Jefi[ind_x, ind_y+1, ind_z]*(1-q1)*q2*(1-q3)
            #         bz += Bz_Jefi[ind_x, ind_y+1, ind_z]*(1-q1)*q2*(1-q3)
            #         ex += Ex_Jefi[ind_x, ind_y+1, ind_z]*(1-q1)*q2*(1-q3)
            #         ey += Ey_Jefi[ind_x, ind_y+1, ind_z]*(1-q1)*q2*(1-q3)
            #         ez += Ez_Jefi[ind_x, ind_y+1, ind_z]*(1-q1)*q2*(1-q3)
            #     if ind_y != -1 and ind_z != -1:
            #         bx += Bx_Jefi[ind_x+1, ind_y, ind_z]*q1*(1-q2)*(1-q3)
            #         by += By_Jefi[ind_x+1, ind_y, ind_z]*q1*(1-q2)*(1-q3)
            #         bz += Bz_Jefi[ind_x+1, ind_y, ind_z]*q1*(1-q2)*(1-q3)
            #         ex += Ex_Jefi[ind_x+1, ind_y, ind_z]*q1*(1-q2)*(1-q3)
            #         ey += Ey_Jefi[ind_x+1, ind_y, ind_z]*q1*(1-q2)*(1-q3)
            #         ez += Ez_Jefi[ind_x+1, ind_y, ind_z]*q1*(1-q2)*(1-q3)
                    
            # ind_x = int((pos[i_grid, 0] - 0. * dx - left_bound_x)//dx)
            # ind_y = int((pos[i_grid, 1] - 0. * dy - left_bound_y)//dy)
            # ind_z = int((pos[i_grid, 2] - 0. * dz - left_bound_z)//dz)  
            # bx, by, bz, ex, ey, ez = 0, 0, 0, 0, 0, 0
            # bx = Bx_Jefi[ind_x, ind_y, ind_z]
            # by = By_Jefi[ind_x, ind_y, ind_z]
            # bz = Bz_Jefi[ind_x, ind_y, ind_z]
            # ex = Ex_Jefi[ind_x, ind_y, ind_z]
            # ey = Ey_Jefi[ind_x, ind_y, ind_z]
            # ez = Ez_Jefi[ind_x, ind_y, ind_z]
    
            tx = 0.5 * charge * bx * dt / mass
            ty = 0.5 * charge * by * dt / mass
            tz = 0.5 * charge * bz * dt / mass
            t2 = tx * tx + ty * ty + tz * tz
            
            sx = 2 * tx / (1 + t2)
            sy = 2 * ty / (1 + t2)
            sz = 2 * tz / (1 + t2) # s = 2*t/(1+|t|2)

            velocity[i_grid, 0] += 0.5 * charge * ex * dt/mass
            velocity[i_grid, 1] += 0.5 * charge * ey * dt/mass
            velocity[i_grid, 2] += 0.5 * charge * ez * dt/mass  # v- = v(n-1/2) + qE*dt/2m
            
            velocity_x = velocity[i_grid, 0] + velocity[i_grid, 1] * tz - velocity[i_grid, 2] * ty
            velocity_y = velocity[i_grid, 1] + velocity[i_grid, 2] * tx - velocity[i_grid, 0] * tz
            velocity_z = velocity[i_grid, 2] + velocity[i_grid, 0] * ty - velocity[i_grid, 1] * tx # v' = v- + v- × t
            
            velocity__x = velocity[i_grid, 0] + velocity_y * sz - velocity_z * sy
            velocity__y = velocity[i_grid, 1] + velocity_z * sx - velocity_x * sz
            velocity__z = velocity[i_grid, 2] + velocity_x * sy - velocity_y * sx # v+ = v- + v' × s
            
            velocity[i_grid, 0] = velocity__x + 0.5 * charge * ex * dt/mass
            velocity[i_grid, 1] = velocity__y + 0.5 * charge * ey * dt/mass
            velocity[i_grid, 2] = velocity__z + 0.5 * charge * ez * dt/mass  # v(n+1/2) = v+ + qE*dt/2m
            
            # velocity[i_grid, 0] += charge * ex * dt/mass
            # velocity[i_grid, 1] += charge * ey * dt/mass
            # velocity[i_grid, 2] += charge * ez * dt/mass  # v(n+1/2) = v+ + qE*dt/2m

            # update position
            pos[i_grid, 0] += velocity[i_grid, 0]*dt
            pos[i_grid, 1] += velocity[i_grid, 1]*dt
            pos[i_grid, 2] += velocity[i_grid, 2]*dt
            
            
            # print(velocity[i_grid, 1], dt, pos[i_grid, 1], 2000)
        