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
# Description: this file can be used to test the adjoint and state error estimators.

from discretizer import discretize, get_y0
import numpy as np
from methods import collection, get_random_switching_law
import matplotlib.pyplot as plt
import fenics as fe
plt.rcParams.update({'font.size': 18})
plt.rcParams['lines.linewidth'] = 3

#%% init

# assemble error estimator or not
energy_prod = True
number_switches = 2
l = 10
 
# stuff
random = True
derivative_check = True
errorest_assembled = True
POD = True                                                            
debug = False
print_ = False
state_dependent_switching = False
visualize_only_room2 = True

# Choose nonsmoothness
nonsmooth = 'l1box' 

#%% get fom

options = collection()
options.factorize = True                                                        # only gets used if switch model
options.energy_prod = energy_prod

# get pde model
T = 0.4  
K = 21 
dx = 1

fom = discretize(T = T, dx = dx, K = K , debug = debug,
                 state_dependent_switching = state_dependent_switching,
                 model_options = options, nonsmooth = nonsmooth, use_energy_products = energy_prod)

print('FOM solving ...')
# get input U
if random:
    int_yd = 15
    U = np.random.uniform(-int_yd, int_yd, (fom.pde.input_dim, fom.time_disc.K)) 
else:
    U = (1*np.ones((fom.pde.input_dim, fom.time_disc.K)))

# set init guess
y0 = get_y0(fom.space_disc.V, fe.Expression('0.5', degree = 1))

# solve state to get yd
if fom.isSwitchModel():
    
    # choose seitching law
    # random_switching_law = get_switching_law(T = T, time_points = [0, T], init_sigma = 2)
    random_switching_law = get_random_switching_law(T = T, numer_random_switch_points = number_switches, init_sigma = 2)
    fom.pde.sigma = random_switching_law
    
    # solve state
    Y, Out, time, switch_profil = fom.solve_state(U, theta = fom.theta, print_ = print_, 
                                              y0 = y0)
    if 0:
       fom.visualize_output(Out, only_room2 = visualize_only_room2, title = 'FOM', semi = True)

Yd = (Out)
# fom.pde.y0 = get_y0(fom.space_disc.V, Expression('0', degree = 1))
fom.update_cost_data(Yd = Yd,#np.ones(Out.shape), 
                     YT = Yd[:,-1], 
                     Ud = np.zeros((fom.input_dim, fom.time_disc.K)), 
                     weights= [1, 1e-1, 0], 
                     input_product= fom.cost_data.input_product, 
                     output_product= fom.cost_data.output_product)

_, _, _, _, _ = fom.get_snapshots(U = U, y0 = None, test_residual = True)

#%% reduce

from reductor import pod_reductor
estpod = []

# starting value for optimization
if random:
    int_yd = 20
    U_0 = np.random.uniform(-int_yd, int_yd, (fom.pde.input_dim, fom.time_disc.K)) 
else:
    U_0 = 5*np.ones((fom.input_dim, fom.time_disc.K))

# create training fom
if POD:
    
    # define training horizon
    kp = K-1
    ind_K = kp+1
    T_train = fom.time_disc.t_v[ind_K-1] #T
    K_train = ind_K 
    
    # create fom train
    fom_train = discretize(T = T_train, dx = dx, K = K_train,
                                debug = debug,
                                state_dependent_switching = state_dependent_switching,
                                model_options = options, nonsmooth = nonsmooth , use_energy_products = energy_prod)
        
    # choose switching law
    fom_train.pde.sigma = random_switching_law

    
    # update fom train
    fom_train.update_cost_data(Yd = fom.cost_data.Yd[:,:K_train], 
                          YT = fom.cost_data.Yd[:,K_train-1], 
                          Ud = fom.cost_data.Ud, 
                          weights= fom.cost_data.weights, 
                          input_product= fom.cost_data.input_product, 
                          output_product= fom.cost_data.output_product)

if POD:
    
    # choose training data (maybe choose different switching law etc...)
    U_train = U[:,:K_train]
    y0_train = fom.pde.y0
    
    # get snapshots using the data for adjoint and state
    Y_train, P_train, Out_train, _, _ = fom_train.get_snapshots(U = U_train,
                                                          y0 = y0_train)
    
    # POD
    l_POD = l
    r_pod = pod_reductor(fom_train, 
                          model_toproject = fom , 
                          H_prod = fom_train.pde.products['L2'],
                          space_product = fom_train.pde.products['H1'], 
                          errorest_assembled = errorest_assembled)
    
    rom_pod =  r_pod.get_rom(l = l_POD,
                            Snapshots = [Y_train, P_train],
                            space_product = None,
                            time_product = fom_train.time_disc.D_diag, 
                            PODmethod = 0, 
                            plot = True,
                            use_energy_content = False)
    
    # check orthogonality
    if 0:
        r_pod.check_orthogonality()

#%% test error estimator

if 1:
    KK = [K-1]
    for ktest in KK:
    
        if POD:
            
            # STATE
            Yr, Pr, Outr, switching_profile, gaps_r = rom_pod.get_snapshots(U)
            
            if 1:
                print('STATE')
                if 1:
                    est_off = rom_pod.state_est(U, k = ktest, Yr = Yr, switching_profile = switching_profile, 
                                      norm_type = 'L2V_list', computationtype = 'offline_online')
                    
                    # L2V
                    est_online = rom_pod.state_est(U, k = ktest, Yr = Yr, switching_profile = switching_profile, 
                                      norm_type = 'L2V_list', computationtype = 'online')
                    
                    est_true = rom_pod.state_est(U, k = ktest, Y = None, Yr = Yr, switching_profile = switching_profile, norm_type = 'L2V_list', computationtype = 'true')
                    
                    
                    for k in range(len(est_online)):
                        print(f'L2V {k}: online est = {est_online[k]}, offline est = {est_off[k]}, true error = {est_true[k]}, effectivity online = {est_true[k]/est_online[k]}, diff online/offline {est_online[k]-est_off[k]}')
                        
    
            # ADJOINT
            if 1:
                print('ADJOINT')
                type_ = 'LinfH_list'
                est_online = rom_pod.adjoint_est(U = U, Yr = Yr, Y = None, Pr = Pr , P = None, type_ = type_,
                          k = ktest, switching_profile_r = switching_profile, switching_profile = switching_profile, 
                          out = None, out_r = Outr, state_H_est_list = None, computationtype = 'online', gaps_r = gaps_r)
                
                est_true = rom_pod.adjoint_est(U = U, Yr = Yr, Y = None, Pr = Pr , P = None, type_ = type_,
                          k = ktest, switching_profile_r = switching_profile, switching_profile = switching_profile, 
                          out = None, out_r = Outr, state_H_est_list = None, computationtype = 'true', gaps_r = gaps_r)
                
                est_offline = rom_pod.adjoint_est(U = U, Yr = Yr, Y = None, Pr = Pr , P = None, type_ = type_,
                          k = ktest, switching_profile_r = switching_profile, switching_profile = switching_profile, 
                          out = None, out_r = Outr, state_H_est_list = None, computationtype = 'offline_online', gaps_r = gaps_r)
                
                for k in range(len(est_online)):
                    print(f'LinfH {k}: online residual = {est_online[k]}, offline est = {est_offline[k]}, true error = {est_true[k]}, effectivity = {est_true[k]/est_online[k]}, diff online/offline {est_online[k]-est_offline[k]}')
                    
