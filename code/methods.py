# ~~~
# This file is part of the paper:
#
#           "Certified MPC For Switched Evolution Equations Using MOR"
#
#   https://github.com/michikartmann
#
# Copyright 2024 all developers. All rights reserved.
# License: Licensed as BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)
# Contributor: Michael Kartmann
# ~~~
# Description: this file contains additional methods.

import numpy as np
import numpy.matlib
import statistics

def get_mpc_options(kf, 
                    kp, 
                    mpcplot_, 
                    PODmethod, 
                    State_update_tol_start_end,
                    Control_update_tol_start_end,
                    restart, 
                    l_POD, 
                    len_old_snaps, 
                    coarse,
                    innersolver_options,
                    fom_predictor_innersolver_options,
                    error_est_type = 'expensive',
                    type_ = 'FOMROM',
                    print_ = True
                    ):
    
    mpc_options = collection()
    
    # set sampling and prediction horizons
    mpc_options.kf = kf
    mpc_options.kp = kp
    # mpc_options.Tf = None
    # mpc_options.Tp = None
    mpc_options.plot_ = mpcplot_
    mpc_options.PODmethod = PODmethod
    
    # error estimation and tolerances
    mpc_options.print_ = True
    mpc_options.type = type_
    mpc_options.error_est_type = error_est_type
    mpc_options.adaptive_tolerance = True
    mpc_options.tol_update_type = 4
    mpc_options.error_est_corr = False
    mpc_options.error_est_correction_constant_for_cheap = 1
    mpc_options.measure_true_error = False
    mpc_options.error_scale_cons = 1
    mpc_options.err_lowernbound = 0.8*1e-4
    mpc_options.control_tol_update = None
    mpc_options.control_tol_init = None
    mpc_options.MPC_trajectory_tol = None
    mpc_options.State_update_tol_start_end = State_update_tol_start_end
    mpc_options.Control_update_tol_start_end = Control_update_tol_start_end
    mpc_options.restart = restart
    
    # pod mpc options
    mpc_options.l_init = l_POD
    mpc_options.POD_update_l = 0
    mpc_options.target_snapshots = True
    mpc_options.track_basis_coeff = False
    mpc_options.POD_type = 2
    mpc_options.len_old_snapshots = len_old_snaps
    
    # mpc coarse options
    mpc_options.coarse = coarse
    mpc_options.coarse_tolerance = 2e-10
    mpc_options.coarse_threshold = 0.2*1
    mpc_options.accept_count_threshold = 50
    
    # performance index options
    mpc_options.perf_bound = False
    mpc_options.perf_lower_bound = 0.005
    mpc_options.not_accept_threshhold = 8
    
    # solver options
    mpc_options.innersolver_options = innersolver_options
    mpc_options.fom_predictor_innersolver_options = fom_predictor_innersolver_options
    
    return mpc_options
    
def get_reductor_and_rom(kp, fom, dx, debug, options, 
                         nonsmooth, energy_prod, random_switching_law, 
                         U_0, PODmethod, errorest_assembled = False, l_POD = 5):
    
    # create local in time fom that is used for training (it is not used by the adaptive methods)
    from discretizer import discretize
    from reductor import pod_reductor
    
    # define training horizon
    ind_K = kp+1  #K
    T_train = fom.time_disc.t_v[ind_K-1] 
    K_train = ind_K
    
    # create fom train
    fom_train = discretize(T=T_train, dx=dx, K=K_train,
                               debug=debug,
                               state_dependent_switching=False,
                               model_options=options, 
                               nonsmooth=nonsmooth, 
                               use_energy_products=energy_prod
                               )
    fom_train.pde.sigma = random_switching_law
    fom_train.update_cost_data(Yd=fom.cost_data.Yd[:, :K_train],
                               YT=fom.cost_data.Yd[:, K_train-1],
                               Ud=fom.cost_data.Ud,
                               weights=fom.cost_data.weights,
                               input_product=fom.cost_data.input_product,
                               output_product=fom.cost_data.output_product)
    
    # choose training data
    U_train = U_0[:, :K_train]
    y0_train = fom.pde.y0
    
    # get snapshots using the data for adjoint and state
    Y_train, P_train, Out_train, _, _ = fom_train.get_snapshots(U=U_train,
                                                             y0=y0_train)
    # POD
    r_pod = pod_reductor(fom_train,
                         model_toproject=fom,
                         H_prod=fom_train.pde.products['L2'],
                         space_product=fom_train.pde.products['H1'],
                         errorest_assembled=errorest_assembled)
    
    rom_pod = r_pod.get_rom(l=l_POD,
                            Snapshots=[Y_train, P_train],
                            # fom_train.pde.products['H1'],
                            space_product=None,
                            time_product=fom_train.time_disc.D_diag,
                            PODmethod=PODmethod,
                            plot=True)
    if 0:
        fom_train.print_info()
        rom_pod.print_info()
    return rom_pod, r_pod

class collection():
    pass
    
class save_field():
    
    def __init__(self, name):
        self.name = name
        self.d = {
        'basissize' : [],
        'enrichedtimes' : [],
        'eu' : [],
        'ey' : [],
        'estate' : [],
        'eJ' : [],
        'J' : [],
        'time' : [],
        'time_err' : [],
        'time_fomsub' : [],
        'time_romsub' : [],
        'time_model' : [],
        'speedup' : [],
        'time_coarse': []
        
         }
        self.hists = []
        
    def appenddd(self,basissize,enrichedtimes, eu, ey, estate, eJ, J,speedup, history):
            self.d['basissize'].append(basissize)
            self.d['enrichedtimes'].append(enrichedtimes)
            self.d['eu'].append(eu)
            self.d['ey'].append(ey)
            self.d['estate'].append(estate)
            self.d['eJ'].append(eJ)
            self.d['J'].append(J)
            self.d['speedup'].append(speedup)
            
            self.d['time'].append(history['time'])
            self.d['time_err'].append(history['errorest_time'])
            self.d['time_fomsub'].append(history['FOMsubproblem_time'])
            self.d['time_romsub'].append(history['ROMsubproblem_time'])
            self.d['time_model'].append(history['ROMupdate_time'])
            self.d['time_coarse'].append(history['coarse_time'])
                      
            self.hists.append(history)
            
    def average_min_max(self):
            aaa = len(self.d['time'])
            print(f'############ length {aaa} ##########################')
            for key, value in self.d.items():
                print(f'{key}: average {statistics.mean(value)}, min {min(value)}, max {max(value)}')
    def print_lists(self):
        aaa = len(self.d['time'])
        print(f'############ length {aaa} ##########################')
        for key, value in self.d.items():
            print(f'{key}: {value}')
            
def simple_to_switch(EiMples):
    
    switch_model = collection()
    switch_model.A = []
    switch_model.M = []
    switch_model.F = None
    switch_model.B = []
    switch_model.C = []
    
    switch_model.y0 = EiMples[0].initial_data.array.to_numpy().T
    switch_model.y0 = switch_model.y0.reshape((len( switch_model.y0),))
    switch_model.state_dim = EiMples[0].order
    switch_model.input_dim = EiMples[0].dim_input
    switch_model.output_dim = EiMples[0].dim_output
    
    for i in range(len(EiMples)):
        switch_model.A.append(-EiMples[i].A.matrix)
        switch_model.M.append(EiMples[i].E.matrix)
        switch_model.B.append(EiMples[i].B.matrix)
        switch_model.C.append(EiMples[i].C.matrix)
        
    return switch_model

def compare_fom_rom(fom, roms, Us, Sigmas):
    
    count = 0
    for u in Us:
        for sig in Sigmas:
            
            # solve fom
            fom.pde.sigma = sig
            Y, Out, time, _ = fom.solve_state(u, theta = fom.theta, print_ = False)
            out_l2 = fom.L2_scalar_norm((Out)[1,:])
            
            # solve roms
            errs = []
            output_errors = []
            outputs = [Out]
            label = ['FOM']
            for rom in roms:
                    
                rom.pde.sigma = sig
                Y_rom, Out_rom, time_rom, _ = rom.solve_state(u, theta = rom.theta, print_ = False)
                # rom.visualize_output(Out_pod, title = 'ROM POD')
                
                error = fom.L2_scalar_norm((Out-Out_rom)[1,:])
                errs.append(error)
                outputs.append(Out_rom)
                label.append(rom.type + f' l = {rom.pde.state_dim}')
                # output_errors.append(abs((Out-Out_rom)/error))
                output_errors.append(abs(Out-Out_rom)/np.max(abs(Out)))
            
            # print
            if len(roms)>1:
                print(f'{count}: rel l2-error output: {label[1]}: {errs[0]/out_l2:2.4e}, {label[2]}: {errs[1]/out_l2:2.4e}')
            else:
                print(f'{count}: rel l2-error output: {label[1]}: {errs[0]/out_l2:2.4e}')
                
            # plot
            rom.visualize_outputs(outputs, only_room2 = True, title = f'{count}: Outputs', labels = label)
            rom.visualize_outputs(output_errors, only_room2 = True, title = f'{count}: Output Errors', labels = label[1:])
            count += 1
            
def get_random_switching_law(T, numer_random_switch_points, init_sigma):
    
    if init_sigma == 1:
        else_sigma = 2
    elif init_sigma == 2:
        else_sigma = 1
    else:
        assert 0, 'sigma can only be 1 and 2...'
    
    np.random.seed(0)     
    time_points = list(np.sort(T*np.random.rand(numer_random_switch_points)))
    time_points.insert(0,0); time_points.append(T)
    def random_switch(t, y, sigma = None):
        ind = None
        for i in range(len(time_points)-1):
            if time_points[i] <= t < time_points[i+1]:
                ind = i
        if t == T:
           ind = i
        assert ind is not None, 't is not in the range here ...'  
        if ind%2: 
           return else_sigma 
        else:
           return init_sigma 
    print(f'switching time points are: {time_points}')  
    return random_switch

def get_switching_law(T, time_points, init_sigma):
    
    if init_sigma == 1:
        else_sigma = 2
    elif init_sigma == 2:
        else_sigma = 1
    else:
        assert 0, 'sigma can only be 1 and 2...'
        
    def random_switch(t, y, sigma = None):
        ind = None
        for i in range(len(time_points)-1):
            if time_points[i] <= t < time_points[i+1]:
                ind = i
        if t == T:
           ind = i
        assert ind is not None, 't is not in the range here ...'  
        if ind%2:
           return else_sigma 
        else:
           return init_sigma
       
    return random_switch