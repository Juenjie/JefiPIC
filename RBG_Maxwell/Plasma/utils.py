import math
import numpy as np
    
def find_largest_time_steps(dx_o, dy_o, dz_o, x_left_boundary_o, y_left_boundary_o, z_left_boundary_o, \
                            dx_s, dy_s, dz_s, x_left_boundary_s, y_left_boundary_s, z_left_boundary_s, \
                            x_grid_size_o, y_grid_size_o, z_grid_size_o, \
                            x_grid_size_s, y_grid_size_s, z_grid_size_s,\
                            dt, c):
    '''
    Estimate the largest time steps for light to transmit between the two regions.
    
    Params
    ======
    o indicates observational and s indicates source
    dx_o, dy_o, dz_o, x_left_boundary_o, y_left_boundary_o, z_left_boundary_o
    dx_s, dy_s, dz_s, x_left_boundary_s, y_left_boundary_s, z_left_boundary_s
    x_grid_size_o, y_grid_size_o, z_grid_size_o
    x_grid_size_s, y_grid_size_s, z_grid_size_s
    c:
        numerical of c in flexible unit (FU)
    
    Return
    ======
    distance:
        The largest distance between the two regions.
    '''
    
    # find the left and right boundaries
    x_right_boundary_o, y_right_boundary_o, z_right_boundary_o = x_left_boundary_o + dx_o*x_grid_size_o, \
                                                                 y_left_boundary_o + dy_o*y_grid_size_o, \
                                                                 z_left_boundary_o + dz_o*z_grid_size_o

    x_right_boundary_s, y_right_boundary_s, z_right_boundary_s = x_left_boundary_s + dx_s*x_grid_size_s, \
                                                                 y_left_boundary_s + dy_o*y_grid_size_s, \
                                                                 z_left_boundary_s + dz_o*z_grid_size_s
    
    xro,yro,zro,xlo,ylo,zlo,xrs,yrs,zrs,xls,yls,zls = x_right_boundary_o, y_right_boundary_o, z_right_boundary_o,\
                                                      x_left_boundary_o, y_left_boundary_o, z_left_boundary_o, \
                                                      x_right_boundary_s, y_right_boundary_s, z_right_boundary_s, \
                                                      x_left_boundary_s, y_left_boundary_s, z_left_boundary_s
    
    distances = []
    for xo in [xro, xlo]:
        for yo in [yro, ylo]:
            for zo in [zro, zlo]:
                for xs in [xrs, xls]:
                    for ys in [yrs, yls]:
                        for zs in [zrs, zls]:
                            distances.append(math.sqrt((xo-xs)**2+(yo-ys)**2+(zo-zs)**2))
                            
    return int(math.ceil(max(distances)/(dt*c)))

def check_input_legacy(f, dt, \
                       nx_o, ny_o, nz_o, dx, dy, dz, boundary_configuration, \
                       x_left_bound_o, y_left_bound_o, z_left_bound_o, \
                       npx, npy, npz, half_px, half_py, half_pz,\
                       masses, charges,\
                       sub_region_relations,\
                       collision_type_for_all_species, num_gpus_for_each_region,\
                       num_samples,\
                       flavor, collision_type, particle_type,\
                       degeneracy, expected_collision_type):
    
    '''Check the legacy of the initial input'''
    
    if type(num_samples) != int:
        raise AssertionError("Type of num_samples should be int, but {} detected.".format(type(num_samples))) 
        
    if type(num_gpus_for_each_region) != float and type(num_gpus_for_each_region) != int:
        raise AssertionError("Type of num_gpus_for_each_region should be (float or int), but {} detected.".format(type(num_gpus_for_each_region)))
        
    if type(dt) != float and type(dt) != int:
        raise AssertionError("Type of dt should be (float or int), but {} detected.".format(type(dt))) 
        
    if type(dx) != float and type(dx) != int:
        raise AssertionError("Type of dx should be (float or int), but {} detected.".format(type(dx)))
        
    if type(dy) != float and type(dy) != int:
        raise AssertionError("Type of dy should be (float or int), but {} detected.".format(type(dy)))
        
    if type(dz) != float and type(dz) != int:
        raise AssertionError("Type of dz should be (float or int), but {} detected.".format(type(dz)))
    
    if type(f) == dict:
        for i_reg in range(len(f)):
            if type(f[i_reg]) != np.ndarray:
                raise AssertionError("Types of values in f should be numpy.ndarray, but {} detected.".format(type(f[i_reg]))) 
    else:
        raise AssertionError("Type of f should be dict, but {} detected.".format(type(f))) 
        
    if type(nx_o) == list:
        for i_reg in range(len(nx_o)):
            if type(nx_o[i_reg]) != int:
                raise AssertionError("Types of values in nx_o should be int, but {} detected.".format(type(nx_o[i_reg]))) 
    else:
        raise AssertionError("Type of nx_o should be list, but {} detected.".format(type(nx_o)))
        
    if type(ny_o) == list:
        for i_reg in range(len(ny_o)):
            if type(ny_o[i_reg]) != int:
                raise AssertionError("Types of values in ny_o should be int, but {} detected.".format(type(ny_o[i_reg]))) 
    else:
        raise AssertionError("Type of ny_o should be list, but {} detected.".format(type(ny_o)))
        
    if type(nz_o) == list:
        for i_reg in range(len(nz_o)):
            if type(nz_o[i_reg]) != int:
                raise AssertionError("Types of values in nz_o should be int, but {} detected.".format(type(nz_o[i_reg]))) 
    else:
        raise AssertionError("Type of nz_o should be list, but {} detected.".format(type(nz_o)))
        
    if type(npx) != int:
        raise AssertionError("Type of npx should be int, but {} detected.".format(type(npx))) 
         
    if type(npy) != int:
        raise AssertionError("Type of npy should be int, but {} detected.".format(type(npy)))
        
    if type(npz) != int:
        raise AssertionError("Type of npz should be int, but {} detected.".format(type(npz)))
        
    if type(boundary_configuration) == dict:
        for i_reg in range(len(boundary_configuration)):
            if type(boundary_configuration[i_reg]) == tuple:
                for i in range(len(boundary_configuration[i_reg])):
                    if type(boundary_configuration[i_reg][i]) != np.ndarray:
                        raise AssertionError("Types of boundary_configuration[#][#] should be numpy.ndarray, but {} detected.".format(type(boundary_configuration[i_reg][i]))) 
            else:
                raise AssertionError("Types of boundary_configuration[#] should be tuple, but {} detected.".format(type(boundary_configuration[i_reg]))) 
    else:
        raise AssertionError("Type of boundary_configuration should be dict, but {} detected.".format(type(boundary_configuration)))
        
    if type(x_left_bound_o) == list:
        for i_reg in range(len(x_left_bound_o)):
            if type(x_left_bound_o[i_reg]) != int and type(x_left_bound_o[i_reg]) != float:
                raise AssertionError("Types of x_left_bound_o[#] should be (float or int), but {} detected.".format(type(x_left_bound_o[i_reg]))) 
    else:
        raise AssertionError("Type of x_left_bound_o should be list, but {} detected.".format(type(x_left_bound_o)))
        
    if type(y_left_bound_o) == list:
        for i_reg in range(len(y_left_bound_o)):
            if type(y_left_bound_o[i_reg]) != int and type(y_left_bound_o[i_reg]) != float:
                raise AssertionError("Types of y_left_bound_o[#] should be (float or int), but {} detected.".format(type(y_left_bound_o[i_reg]))) 
    else:
        raise AssertionError("Type of y_left_bound_o should be list, but {} detected.".format(type(y_left_bound_o)))
        
    if type(z_left_bound_o) == list:
        for i_reg in range(len(z_left_bound_o)):
            if type(z_left_bound_o[i_reg]) != int and type(z_left_bound_o[i_reg]) != float:
                raise AssertionError("Types of z_left_bound_o[#] should be (float or int), but {} detected.".format(type(z_left_bound_o[i_reg]))) 
    else:
        raise AssertionError("Type of z_left_bound_o should be list, but {} detected.".format(type(z_left_bound_o)))
        
    name = ['half_px', 'half_py', 'half_pz', 'masses', 'charges']
    para = [half_px, half_py, half_pz, masses, charges]
    shape = []
    for i in range(len(name)):
        shape.append(para[i].shape)
        if type(para[i]) != np.ndarray:
            raise AssertionError("Type of {} should be numpy.ndarray, but {} detected.".format(name[i], type(para[i])))
    for i in range(len(shape)):
        for j in range(i+1, len(shape)):
            if shape[i] != shape[j]:
                raise AssertionError("Shape of {} should be same with {}, but detected {} is of shape {} and {} is of shape {}."\
                                     .format(name[i], name[j], name[i], shape[i], name[j], shape[j]))
            
    if type(sub_region_relations) == dict:
        for i_reg in ['indicator', 'position']:
            if type(sub_region_relations[i_reg]) != list:
                raise AssertionError("Types sub_region_relations[#] should be list, but {} detected.".format(type(sub_region_relations[i_reg]))) 
            else:
                for i in range(len(sub_region_relations[i_reg])):
                    if type(sub_region_relations[i_reg][i]) == list:
                        for j in range(len(sub_region_relations[i_reg][i])):
                            if type(sub_region_relations[i_reg][i][j]) != int and type(sub_region_relations[i_reg][i][j]) != type(None):
                                raise AssertionError("Types sub_region_relations[#][#][#] should be (int or None), but {} detected.".format(type(sub_region_relations[i_reg][i][j]))) 
                    else:
                        raise AssertionError("Types sub_region_relations[#][#] should be list, but {} detected.".format(type(sub_region_relations[i_reg][i]))) 
    else:
        raise AssertionError("Type of sub_region_relations should be dict, but {} detected.".format(type(sub_region_relations))) 
        
        if type(flavor) != dict:
            raise AssertionError("Type of flavor should be dict, but {} detected.".format(type(flavor)))
        if type(collision_type) != dict:
            raise AssertionError("Type of collision_type should be dict, but {} detected.".format(type(collision_type)))
            
        for key in expected_collision_type:
            if flavor[key]==None:
                raise AssertionError("flavor[{}] is None, remove {} in expected_collision_type".format(key,key))
            if collision_type[key]==None:
                raise AssertionError("collision_type[{}] is None, remove {} in expected_collision_type".format(key,key))

    
