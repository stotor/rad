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

simulation_folder = sys.argv[1]
species = 'protons_147'

chunk_size = 10000
detector_distance = 86267.8
n_x = 512
n_y = 512

detector_width = 15215.0 * 2.0 / 5.0

x1_min = -1.0 * detector_width / 2.0
x1_max = detector_width / 2.0
x2_min = -1.0 * detector_width / 2.0
x2_max = detector_width / 2.0
dx = float(x1_max - x1_min) / n_x

t_array = oi.get_HIST_time(simulation_folder)
timesteps = range(len(t_array))

for t in timesteps:
    if rank == 0:
        t_start = MPI.Wtime()
        print 'Starting timestep ' + str(t)

    filename = simulation_folder + '/MS/RAW/' + species + '/RAW-' + species + '-' + str(t).zfill(6) + '.h5'
    h5f = h5py.File(filename, 'r', driver='mpio', comm=comm)
    
    n_p = h5f['x1'].shape[0]

    processor_boundaries = np.linspace(0, n_p, size+1).astype('int')
    processor_start = processor_boundaries[rank]
    processor_end = processor_boundaries[rank+1]

    indices = np.append(range(processor_start, processor_end, chunk_size),
                        processor_end)
    
    radiograph = np.zeros([n_x, n_y], dtype='float64')

    # First find x3_max

    x3_max_i = []

    for i in range(len(indices)-1):
        i_start = indices[i]
        i_end = indices[i+1]

        position = oi.create_position_array(h5f, i_start, i_end, dim=3)
        x3_max_i.append(np.amax(position[:, 2]))

    x3_max_local = np.amax(x3_max_i)
    x3_max_global = np.zeros(1, dtype='float64')

    comm.Allreduce([x3_max_local, MPI.DOUBLE], [x3_max_global, MPI.DOUBLE], op = MPI.MAX)
    print(x3_max_global)

    for i in range(len(indices)-1):
#        if rank == 0:
#            print float(i) / float(len(indices))

        i_start = indices[i]
        i_end = indices[i+1]

        momentum = oi.create_momentum_array(h5f, i_start, i_end)
        position = oi.create_position_array(h5f, i_start, i_end, dim=3)

        delta_z = x3_max_global - position[:, 2]

        # Calculate final x1 and x2 positions
        position[:, 0] = position[:, 0] + (detector_distance + delta_z) * momentum[:, 0] / np.abs(momentum[:, 2])
        position[:, 1] = position[:, 1] + (detector_distance + delta_z) * momentum[:, 1] / np.abs(momentum[:, 2])

        charges = np.ones(position.shape[0], dtype='float64')

        in_box = np.where((position[:, 0] <= x1_max) * (position[:, 0] >= x1_min) * (position[:, 1] <= x2_max) * (position[:, 1] >= x2_min))[0]
        position = position[in_box, :]

        # Weight particles to grid
        position[:, 0] = position[:, 0] - x1_min
        position[:, 1] = position[:, 1] - x2_min

        radiograph = pic.deposit_species(position[:,1::-1], radiograph, charges, n_x, n_y, dx, deposit_type='cic')

    h5f.close()

    # Reduce radiographs
    radiograph_total = np.zeros_like(radiograph)
    comm.Reduce([radiograph, MPI.DOUBLE], [radiograph_total, MPI.DOUBLE], op = MPI.SUM, root = 0)

    species_save = 'protons_3'

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
            v0 = 0.0797759
            penetration = v0 * time 
            h5f.attrs['v0'] = v0
            h5f.attrs['penetration'] = penetration
        elif species_save == 'protons_147':
            v0 = 0.174965
            penetration = v0 * time 
            h5f.attrs['v0'] = v0
            h5f.attrs['penetration'] = penetration
        h5f.close()

    if rank == 0:
        t_end = MPI.Wtime()
        t_elapsed = t_end - t_start
        print 'Total time for timestep:'
        print t_elapsed
