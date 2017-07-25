import sys

import numpy as np
import ctypes
import h5py
from mpi4py import MPI

import osiris_interface as oi
import pic_calculations as pic
import utilities

import matplotlib.pyplot as plt

library_filename = '/Users/stotor/Desktop/proton_radiography/rad/create_radiograph.so'

lib = ctypes.cdll[library_filename]
create_radiograph = lib['create_radiograph']
    
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

###################################
# Input parameters
save_filename = '/Users/stotor/Desktop/radiograph-1.h5'
# Proton source properties
u_mag = 0.177707
rqm = 1836.0
n_p = 174467
# Simulation plasma properties
dx = 2.0
dt = 1.14
mi_mez_sim = 128.0
b1_filename = '/Users/stotor/Desktop/proton_radiography/rad/b1-savg-000001.h5'
b2_filename = '/Users/stotor/Desktop/proton_radiography/rad/b2-savg-000001.h5'
b3_filename = '/Users/stotor/Desktop/proton_radiography/rad/b3-savg-000001.h5'
# Physical plasma properties
mi_mez_exp = 2.0 * 1836.0
v_sim = 0.815
v_exp = 1.0 / 300.0
# n_e in units of cm^-3
n_e = 4.4 * 10.0**19
# All length scales in units of m
source_width = 19.11 * 10**-6
l_source_detector = 300.0 * 10**-3
l_source_midpoint = 10.0 * 10**-3
plasma_width = 5.0 * 10**-3
radiograph_width = 10.0 * 10**-2
radiograph_grid = [512, 512]
theta_max = 0.3

############################
# Convert length scales to c/wpe
di = 2.28 * 10.0**7 * np.sqrt(mi_mez_exp / 1836.0) / np.sqrt(n_e) * 10.0**-2
l_source_plasma = (l_source_midpoint - plasma_width / 2.0) * np.sqrt(128.0) / di
l_plasma_detector = (l_source_detector - (l_source_midpoint + plasma_width / 2.0)) * np.sqrt(128.0) / di
plasma_width = plasma_width * np.sqrt(128.0) / di
radiograph_width = radiograph_width * np.sqrt(128.0) / di
source_width = source_width * np.sqrt(128.0) / di

field_scaling = (v_exp / v_sim) * np.sqrt(mi_mez_exp / mi_mez_sim)
b1_h5f = h5py.File(b1_filename, 'r', driver='mpio', comm=comm)
b2_h5f = h5py.File(b2_filename, 'r', driver='mpio', comm=comm)
b3_h5f = h5py.File(b3_filename, 'r', driver='mpio', comm=comm)
# Need to implement shifting to deal with Yee lattice and spatial averaging
b1 = np.array(b1_h5f['b1'][:], copy=True).astype('double')
b2 = np.array(b2_h5f['b2'][:], copy=True).astype('double')
b3 = np.array(b3_h5f['b3'][:], copy=True).astype('double')
b1_h5f.close()
b2_h5f.close()
b3_h5f.close()

b1 = b1 * field_scaling
b2 = b2 * field_scaling
b3 = b3 * field_scaling

radiograph_grid = np.array(radiograph_grid, 'intc')
radiograph = np.zeros(radiograph_grid, dtype='double')
field_grid = np.array(b1.shape, 'intc')

c_double_p = ctypes.POINTER(ctypes.c_double)
c_int_p = ctypes.POINTER(ctypes.c_int)

create_radiograph(b1.ctypes.data_as(c_double_p),
                  b2.ctypes.data_as(c_double_p),
                  b3.ctypes.data_as(c_double_p),
		  field_grid.ctypes.data_as(c_int_p),
                  ctypes.c_double(dx),
                  ctypes.c_double(dt),
		  radiograph.ctypes.data_as(c_double_p),
                  radiograph_grid.ctypes.data_as(c_int_p),
                  ctypes.c_double(radiograph_width),
                  ctypes.c_double(source_width),
                  ctypes.c_double(theta_max),
		  ctypes.c_int(n_p),
                  ctypes.c_double(u_mag),
                  ctypes.c_double(rqm),
		  ctypes.c_double(l_source_plasma),
                  ctypes.c_double(l_plasma_detector),
                  ctypes.c_double(plasma_width),
                  ctypes.c_int(rank))

radiograph_total = np.zeros_like(radiograph)
comm.Reduce([radiograph, MPI.DOUBLE], [radiograph_total, MPI.DOUBLE],
            op = MPI.SUM, root = 0)

if rank == 0:
    plt.imshow(radiograph, cmap='gnuplot2')
    plt.colorbar()
    plt.show()

if rank == 0:
    h5f = h5py.File(save_filename, 'w')
    h5f.create_dataset('radiograph', data=radiograph_total)
    h5f.close()

#     save_folder = simulation_folder + '/radiography/' + species_save + '/'
#     utilities.ensure_folder_exists(save_folder)
#     filename = save_folder + 'radiography-' + species_save + '-' + str(t).zfill(6) + '.h5'
#     h5f.attrs['detector_distance'] = detector_distance
#     h5f.attrs['detector_width'] = detector_width
#     time = t_array[t]
#     h5f.attrs['time'] = time
#     h5f.attrs['u_mag'] = u_mag
#     h5f.attrs['plasma_width'] = plasma_width

