import numpy as np
import math

def step_func(x, l, r):
    '''compare x with l and r. return -1 or 0 or 1.'''
    if x<l:
        return -1
    elif x>r:
        return 1
    else:
        return 0
    
def conut_eights_or_minus_eights(x):
    '''given x = [x1,x2,x3], count how many of x1,x2,x3 are 8 or -8'''
    count = 0
    index = []
    sign = []
    for i in range(3):
        if abs(x[i]) == 8:
            count+=1 
            index.append(i)
            sign.append(x[i]/8)
    return count,index,sign

def sign_distance(sign, xlo,ylo,zlo, xrs,yrs,zrs, xls,yls,zls, xro,yro,zro, index):
    '''return the distance according to the index'''
    if sign > 0:
        return [xlo,ylo,zlo][index]  - [xrs,yrs,zrs][index]
    else:
        return [xls,yls,zls][index] - [xro,yro,zro][index]
    
def signal_indicator(dx_o, dy_o, dz_o, x_left_boundary_o, y_left_boundary_o, z_left_boundary_o, \
                     dx_s, dy_s, dz_s, x_left_boundary_s, y_left_boundary_s, z_left_boundary_s, \
                     x_grid_size_o, y_grid_size_o, z_grid_size_o, \
                     x_grid_size_s, y_grid_size_s, z_grid_size_s):
    '''
    Estimate if the signal from the source region has reached the observational region.
    
    Params
    ======
    o indicates observational and s indicates source
    dx_o, dy_o, dz_o, x_left_boundary_o, y_left_boundary_o, z_left_boundary_o
    dx_s, dy_s, dz_s, x_left_boundary_s, y_left_boundary_s, z_left_boundary_s
    x_grid_size_o, y_grid_size_o, z_grid_size_o
    x_grid_size_s, y_grid_size_s, z_grid_size_s
    
    Return
    ======
    time_indicator: the signal arrives at the observational region when t > time_indicator
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
    
    # there are four cases for the relavent spatial positions of the two regions
    # we use the boundaries points in observational region to subtract the boundaries in source region
    # first we compare the points in observational region with source region
    ob = []
    for i in (xlo,xro):
        ii=step_func(i,xls,xrs)
        for j in (ylo,yro):
            jj=step_func(j,yls,yrs)
            for k in (zlo,zro):
                kk=step_func(k,zls,zrs)
                ob.append([ii,jj,kk])
    
    # count how many 8 or -8 and where are they
    sob = np.sum(ob,0)
    count, index, sign = conut_eights_or_minus_eights(sob)
    
    # the two regions overlap
    if count == 0:
        return 0.
    # one direction exceeds
    elif count == 1:
         return sign_distance(sign[0], xlo,ylo,zlo, xrs,yrs,zrs, xls,yls,zls, xro,yro,zro, index[0])
    # two directions exceed
    elif count == 2:
        r1 = sign_distance(sign[0], xlo,ylo,zlo, xrs,yrs,zrs, xls,yls,zls, xro,yro,zro, index[0])
        r2 = sign_distance(sign[1], xlo,ylo,zlo, xrs,yrs,zrs, xls,yls,zls, xro,yro,zro, index[1])
        return math.sqrt(r1**2+r2**2)
    elif count ==3:
        r1 = sign_distance(sign[0], xlo,ylo,zlo, xrs,yrs,zrs, xls,yls,zls, xro,yro,zro, index[0])
        r2 = sign_distance(sign[1], xlo,ylo,zlo, xrs,yrs,zrs, xls,yls,zls, xro,yro,zro, index[1])
        r3 = sign_distance(sign[2], xlo,ylo,zlo, xrs,yrs,zrs, xls,yls,zls, xro,yro,zro, index[2])
        return math.sqrt(r1**2+r2**2+r3**2)   