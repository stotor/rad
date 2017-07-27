import os

import numpy as np
import ctypes
import h5py
from mpi4py import MPI

import utilities

library_folder = os.path.dirname(os.path.realpath(__file__))
lib = ctypes.cdll[library_folder + '/create_radiograph.so']
create_radiograph = lib['create_radiograph']

def t_wpe_to_ns(t_wpe, mi_mez_sim, mi_mez_exp, v_sim, v_exp, n_e):
    t_wpi = t_wpe / np.sqrt(mi_mez_sim)
    wpi = 1.32 * 10.0**3 * np.sqrt(n_e) / np.sqrt(mi_mez_exp / 1836.0)
    t_ns = (t_wpi / wpi) * (v_sim / v_exp)
    return t_ns

def shift_field(field, shift, axis):
    w1 = 1.0 - shift
    w2 = shift
    field = field * w1 + np.roll(field, -1, axis=axis) * w2
    return field

def rad(params):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    ############################
    # Load parameters
    t = params['t']
    field_folder = params['field_folder']
    save_folder = params['save_folder']
    # Proton source properties
    u_mag = params['u_mag']
    rqm = params['rqm']
    proton_yield = params['proton_yield']
    # Simulation plasma properties
    dx = params['dx']
    dt = params['dt']
    mi_mez_sim = params['mi_mez_sim']
    b1_filename = params['b1_filename']
    b2_filename = params['b2_filename']
    b3_filename = params['b3_filename']
    # Physical plasma properties
    mi_mez_exp = params['mi_mez_exp']
    v_sim = params['v_sim']
    v_exp = params['v_exp']
    # n_e in units of cm^-3
    n_e = params['n_e']
    # All length scales in units of m
    source_width = params['source_width']
    l_source_detector = params['l_source_detector']
    l_source_midpoint = params['l_source_midpoint']
    plasma_width = params['plasma_width']
    radiograph_width = params['radiograph_width']
    radiograph_grid = params['radiograph_grid']
    theta_max = params['theta_max']

    ###############################
    # Manipulate loaded parameters
    n_p = float(proton_yield) / size
    n_p = n_p * (1.0 - np.cos(theta_max)) / 2.0
    n_p = int(np.ceil(n_p))
    
    # Convert length scales to c/wpe
    di = 2.28 * 10.0**7 * np.sqrt(mi_mez_exp / 1836.0) / np.sqrt(n_e) * 10.0**-2
    l_source_plasma = (l_source_midpoint - plasma_width / 2.0) * np.sqrt(128.0) / di
    l_plasma_detector = (l_source_detector - (l_source_midpoint + plasma_width / 2.0)) * np.sqrt(128.0) / di
    plasma_width = plasma_width * np.sqrt(128.0) / di
    radiograph_width = radiograph_width * np.sqrt(128.0) / di
    source_width = source_width * np.sqrt(128.0) / di
    
    b1_h5f = h5py.File(b1_filename, 'r', driver='mpio', comm=comm)
    b2_h5f = h5py.File(b2_filename, 'r', driver='mpio', comm=comm)
    b3_h5f = h5py.File(b3_filename, 'r', driver='mpio', comm=comm)
    time = b1_h5f.attrs['TIME'][0]
    b1 = np.array(b1_h5f['b1'][:], copy=True).astype('double')
    b2 = np.array(b2_h5f['b2'][:], copy=True).astype('double')
    b3 = np.array(b3_h5f['b3'][:], copy=True).astype('double')
    b1_h5f.close()
    b2_h5f.close()
    b3_h5f.close()
    field_scaling = (v_exp / v_sim) * np.sqrt(mi_mez_exp / mi_mez_sim)
    b1 = b1 * field_scaling
    b2 = b2 * field_scaling
    b3 = b3 * field_scaling
    # Need to make this general, currently is for the Weibel savg fields
    dx_original = 0.5
    dx_new = 2.0
    shift = (dx_original / 2.0) / dx_new
    b1 = shift_field(b1, shift, axis=2)
    b2 = shift_field(b2, shift, axis=1)
    b3 = shift_field(b3, shift, axis=0)
    
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
        utilities.ensure_folder_exists(save_folder)
        save_filename = save_folder + '/radiograph-' + str(t).zfill(6) + '.h5'
        h5f = h5py.File(save_filename, 'w')
        h5f.create_dataset('radiograph', data=radiograph_total)
        h5f.attrs['u_mag'] = u_mag
        h5f.attrs['di_m'] = di
        h5f.attrs['plasma_width_m'] = plasma_width
        h5f.attrs['source_width_m'] = source_width
        h5f.attrs['radiograph_width_m'] = radiograph_width
        h5f.attrs['l_source_detector_m'] = l_source_detector
        h5f.attrs['l_source_midpoint_m'] = l_source_midpoint
        h5f.attrs['magnification'] = l_source_detector / l_source_midpoint
        h5f.attrs['time_wpe'] = time
        h5f.attrs['time_ns'] = t_wpe_to_ns(time, mi_mez_sim, mi_mez_exp, v_sim, v_exp, n_e)
        h5f.attrs['field_scaling'] = field_scaling
        h5f.close()
