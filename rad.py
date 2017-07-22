import sys

import numpy as np
import h5py
from mpi4py import MPI

import osiris_interface as oi
import pic_calculations as pic
import utilities

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if len(sys.argv) != 3:
    if rank == 0:
        print 'Usage:\n    python create_radiograph.py <simulation_folder> <species>'
    sys.exit()

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
b1_h5f = h5py.File(b1_filename, 'r', driver='mpio', comm=comm)
b2_h5f = h5py.File(b2_filename, 'r', driver='mpio', comm=comm)
b3_h5f = h5py.File(b3_filename, 'r', driver='mpio', comm=comm)

# Shift away from Yee lattice, necessary?  Need to figure out
# exactly what OSIRIS is outputting

b1 = b1_h5f['b1'][:]
b2 = b1_h5f['b2'][:]
b3 = b1_h5f['b3'][:]

b1_h5f.close()
b2_h5f.close()
b3_h5f.close()

# Scale B-field

# Weight to a 2d grid
radiograph = np.zeros_like(radiograph_grid, dtype='double')

# Call c code to loop over number of particles
#(b1, b2, b3, n_x, n_y, n_z, n_p, radiograph, source_width)

# Reduce 2d grids
radiograph_total = np.zeros_like(radiograph)
comm.Reduce([radiograph, MPI.DOUBLE], [radiograph_total, MPI.DOUBLE], op = MPI.SUM, root = 0)

# Save field
if rank == 0:
    save_folder = simulation_folder + '/radiography/' + species_save + '/'
    utilities.ensure_folder_exists(save_folder)
    filename = save_folder + 'radiography-' + species_save + '-' + str(t).zfill(6) + '.h5'
    h5f = h5py.File(filename, 'w')
    h5f.create_dataset('radiograph', data=radiograph_total)
    h5f.attrs['n_p'] = n_p
    h5f.attrs['detector_distance'] = detector_distance
    h5f.attrs['detector_width'] = detector_width
    time = t_array[t]
    h5f.attrs['time'] = time
    if species_save == 'protons_3':
        h5f.attrs['species'] = spe
        h5f.attrs['penetration'] = penetration
    elif species_save == 'protons_147':
        v0 = 0.174965
        penetration = v0 * time 
        h5f.attrs['v0'] = v0
        h5f.attrs['penetration'] = penetration
        h5f.close()

