import sys

import numpy as np
import ctypes
import os
import h5py
from mpi4py import MPI

import osiris_interface as oi
import pic_calculations as pic
import utilities

# Only for debugging
import matplotlib.pyplot as plt

filename = '/Users/stotor/Desktop/proton_radiography/rad/create_radiograph.so'

lib = ctypes.cdll[filename]
create_radiograph = lib['create_radiograph']
    
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# if len(sys.argv) != 3:
#     if rank == 0:
#         print 'Usage:\n    python create_radiograph.py <simulation_folder> <species>'
#     sys.exit()

# Input parameters

# Source properties
## Proton energy
## Proton rqm
## Total number of protons
## Source width

# Physical plasma properties
## Ion skin depth (m)
## Mass ratio
## Flow velocity

# Simulation plasma properties
## Simulation mass ratio
## Simulation velocity

# Experimental geometry
## Source to center of plasma (m)
## Center of plasma to detector (m)
## plasma width (m)
## Detector width (m)
## Radiograph pixel grid

# Load in B-field
b1_filename = '/Users/stotor/Desktop/proton_radiography/rad/b1-savg-000001.h5'
b2_filename = '/Users/stotor/Desktop/proton_radiography/rad/b2-savg-000001.h5'
b3_filename = '/Users/stotor/Desktop/proton_radiography/rad/b3-savg-000001.h5'

b1_h5f = h5py.File(b1_filename, 'r', driver='mpio', comm=comm)
b2_h5f = h5py.File(b2_filename, 'r', driver='mpio', comm=comm)
b3_h5f = h5py.File(b3_filename, 'r', driver='mpio', comm=comm)

# Shift away from Yee lattice, necessary?  Need to figure out
# exactly what OSIRIS is outputting

b1 = b1_h5f['b1'][:]
b2 = b2_h5f['b2'][:]
b3 = b3_h5f['b3'][:]

b1_h5f.close()
b2_h5f.close()
b3_h5f.close()

# Scale B-field

field_grid = np.array([256, 256, 256], dtype='int16')
dx = 2.0
dt = 1.14
radiograph_grid = np.array([512, 512], dtype='int16')
radiograph_width = 23284.0
source_width = 0.0
n_p = 1000
u_mag = 0.177707
rqm = 83811.8
l_source_plasma = 1667.14
l_plasma_detector = 66941.6
plasma_width = 2560.0

radiograph = np.zeros(radiograph_grid, dtype='double')
print('Before c')
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
		  ctypes.c_int(n_p),
                  ctypes.c_double(u_mag),
                  ctypes.c_double(rqm),
		  ctypes.c_double(l_source_plasma),
                  ctypes.c_double(l_plasma_detector),
                  ctypes.c_double(plasma_width),
                  ctypes.c_int16(rank))
print('After c')

radiograph_total = np.zeros_like(radiograph)
comm.Reduce([radiograph, MPI.DOUBLE], [radiograph_total, MPI.DOUBLE],
            op = MPI.SUM, root = 0)
if rank == 0:
    plt.imshow(radiograph)
    plt.colorbar()
    plt.show()

# # Save field
# if rank == 0:
#     save_folder = simulation_folder + '/radiography/' + species_save + '/'
#     utilities.ensure_folder_exists(save_folder)
#     filename = save_folder + 'radiography-' + species_save + '-' + str(t).zfill(6) + '.h5'
#     h5f = h5py.File(filename, 'w')
#     h5f.create_dataset('radiograph', data=radiograph_total)
#     h5f.attrs['n_p'] = n_p
#     h5f.attrs['detector_distance'] = detector_distance
#     h5f.attrs['detector_width'] = detector_width
#     time = t_array[t]
#     h5f.attrs['time'] = time
#     if species_save == 'protons_3':
#         h5f.attrs['species'] = spe
#         h5f.attrs['penetration'] = penetration
#     elif species_save == 'protons_147':
#         v0 = 0.174965
#         penetration = v0 * time 
#         h5f.attrs['v0'] = v0
#         h5f.attrs['penetration'] = penetration
#         h5f.close()

