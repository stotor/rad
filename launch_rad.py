import rad

for t in [1]:
    params = {}
    # Input parameters
    params['t'] = t
    params['field_folder'] = '/Users/stotor/Desktop/proton_radiography/rad/'
    params['save_folder'] = params['field_folder'] + '/point_source/'
    # Proton source properties
    params['u_mag'] = 0.177707
    params['rqm'] = 1836.0
    params['proton_yield'] = 10**6
    # Simulation plasma properties
    params['dx'] = 2.0
    params['dt'] = 1.14
    params['mi_mez_sim'] = 128.0
    params['b1_filename'] = params['field_folder'] + '/b1-savg-' + str(t).zfill(6) + '.h5'
    params['b2_filename'] = params['field_folder'] + '/b2-savg-' + str(t).zfill(6) + '.h5'
    params['b3_filename'] = params['field_folder'] + '/b3-savg-' + str(t).zfill(6) + '.h5'
    # Physical plasma properties
    params['mi_mez_exp'] = 2.0 * 1836.0
    params['v_sim'] = 0.815
    params['v_exp'] = 1.0 / 300.0
    # n_e in units of cm^-3
    params['n_e'] = 4.4 * 10.0**19
    # All length scales in units of m
    params['source_width'] = 0.0
    params['l_source_detector'] = 300.0 * 10**-3
    params['l_source_midpoint'] = 10.0 * 10**-3
    params['plasma_width'] = 5.0 * 10**-3
    params['radiograph_width'] = 10.0 * 10**-2
    params['radiograph_grid'] = [512, 512]
    params['theta_max'] = 0.3
    rad.rad(params)
