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
# Description: this file can be used to test the open-loop optimal control error estimators.

from discretizer import discretize, get_y0
import numpy as np
from methods import collection, get_random_switching_law
import matplotlib.pyplot as plt
import fenics as fe
plt.rcParams.update({'font.size': 18})
plt.rcParams['lines.linewidth'] = 3

#%% flags

data_folder = 'data/test_error_est/'

# choose which test: test optimization error ests
test = 'errorest_and_effect_optimization'

number_switches = 3

# assemble error estimator or not
random = False
derivative_check = False
errorest_assembled = True
POD = True                                                            
debug = False
print_ = False
state_dependent_switching = False
visualize_only_room2 = True

# Choose nonsmoothness
nonsmooth = 'l1box'

options = collection()
options.factorize = True                                                        # only gets used if switch model
options.energy_prod = True

#%% test error est and effectivities

if test == 'errorest_and_effect_optimization':
        
        # get pde model
        T = 0.4  
        K = 21
        dx = 1

        fom = discretize(T = T, dx = dx, K = K , debug = debug, reac_cons = 0.01,
                     state_dependent_switching = state_dependent_switching,
                     model_options = options, nonsmooth = nonsmooth)

#%% create fom

        np.random.seed(2)

        print('FOM solving ...')
        # get input U
        if random:
            int_yd = 15
            U = np.random.uniform(-int_yd, int_yd, (fom.pde.input_dim, fom.time_disc.K)) #(1*np.ones((fom.pde.input_dim, fom.time_disc.K)))
        else:
            U = (1*np.ones((fom.pde.input_dim, fom.time_disc.K)))
            #U = np.cos(0.5*np.pi*np.tile(fom.time_disc.t_v, (fom.pde.input_dim, 1)))

        # set init guess
        if random:
            u_low =-1
            u_up = 1
            y0 = np.random.uniform(u_low, u_up, size=(fom.state_dim,))
        else: 
            y0 = get_y0(fom.space_disc.V, fe.Expression('0.2', degree = 1))

        # solve state to get yd
        if fom.isSwitchModel():
            
            # choose seitching law
            # random_switching_law = get_switching_law(T = T, time_points = [0, T], init_sigma = 2)
            random_switching_law = get_random_switching_law(T = T, numer_random_switch_points = number_switches, init_sigma = 1)
            fom.pde.sigma = random_switching_law
            
            # solve state
            Y, Out, time, switch_profil = fom.solve_state(U, theta = fom.theta, print_ = print_, 
                                                      y0 = y0)
            fom.visualize_output(Out, only_room2 = visualize_only_room2, title = 'FOM', semi = True)
        else:
            # maybe modify time data
            
            # solve state
            Y, Out, time, _ = fom.solve_state(U, theta = fom.theta, print_ = print_, 
                                                  y0 = y0)
            
            # get output trajectory and visualize it
            OUT_ = fom.space_norm_trajectory(Y, norm = 'output')
            fom.visualize_1d(OUT_, title = 'Output norm y', semi = True)

        Yd = (Out)
        fom.update_cost_data(Yd = Yd,#np.ones(Out.shape), 
                             YT = Yd[:,-1], 
                             Ud = np.zeros((fom.input_dim, fom.time_disc.K)), 
                             weights= [1, 1e-2, 0], 
                             input_product= fom.cost_data.input_product, 
                             output_product= fom.cost_data.output_product)
#%% create rom
        
        from reductor import pod_reductor
        
        max_size = 60

        # modify yd, y0, u0, sigma 
        # starting value for optimization
        if random:
            int_yd = 20
            U_0 = np.random.uniform(-int_yd, int_yd, (fom.pde.input_dim, fom.time_disc.K)) #(1*np.ones((fom.pde.input_dim, fom.time_disc.K)))
        else:
            U_0 = 1*np.ones((fom.input_dim, fom.time_disc.K))


#%% optimization setup

        # initial guess (see above)
        # U_0 = np.zeros((fom.input_dim, fom.time_disc.K))
        # U_0 = 10*np.sin(fom.time_disc.t_v).reshape((fom.pde.input_dim, fom.time_disc.K))
        u_TRUE = None

        # inner solver options
        tol = 1e-12
        maxit = 3000
        # BB optimization options
        optionsBB = fom.set_default_options(tol = tol,
                                            maxit = maxit, 
                                            save = False, 
                                            plot = False,
                                            print_info = False)

#%% optimization

        history_TRUE = None
        
        if derivative_check:
            fom.derivative_check()
            fom.print_info()
        if 1:  
            print('FOM Optimize ...')
            u_TRUE, history_TRUE = fom.solve_ocp(U_0, 
                                              "BB",
                                              options = optionsBB)
            
            if 0:
                u_TRUE2uncons, history_TRUE2uncons = fom.solve_ocp(U_0, 
                                                  "BB",
                                                  options = optionsBB,
                                                  solve_unconstrained = True)
                cons_error = fom.space_time_norm(u_TRUE-u_TRUE2uncons , space_norm = "control")
                print(f'Unconstrained to constrained error: {cons_error}')
            
            J_true_val = fom.J(u_TRUE)
            J_true_track, J_true_cont, J_true_traj = fom.Jtracking_trajectory(u_TRUE)
            
        
        #looop over roms
        SIZES = list(np.arange(2, max_size, 5))
        bt = collection()
        bt.speedups = []
        bt.timings = []
        bt.true_error = []
        bt.error_cheap = []
        bt.error_expensive = []
        bt.iter = []
        
        # error ests
        pod = collection()
        pod.speedups = []
        pod.error_cheap_online = []
        pod.timings = []
        pod.true_error = []
        pod.error_cheap = []
        pod.error_expensive = []
        pod.iter = []
        pod.error_exp_split = []
        pod.error_exp_perturb = []
        pod.error_cheapA = []
        pod.error_cheapA_online = []
    
        
        # output ests
        pod_output = collection()
        pod_output.A = []
        pod_output.Acheap = []
        pod_output.B = []
        pod_output.Bcheap = []
        pod_output.Atrue = []
        pod_output.Btrue = []
        
        # effectivities
        pod.eff_perturb = []
        pod.eff_exp1 = []
        pod.eff_exp2 = []
        pod.eff_cheap1 = []
        pod.eff_cheap2 = []
        
        # state, adjoint 
        state_col = collection()
        state_col.off = []
        state_col.on = []
        state_col.true = []
        state_col.effoff = []
        state_col.effon = []
        
        adstate_col = collection()
        adstate_col.off = []
        adstate_col.on = []
        adstate_col.true = []
        adstate_col.effoff = []
        adstate_col.effon = []
        
        count = 0
        if POD:
            for size in SIZES:
                
                
                count += 1
                print('------------------------------------------------------------------------------------------------------------------------------')
                print(f'Count {count}/{len(SIZES)+1}, size {size} #####################################################################################')
                print('------------------------------------------------------------------------------------------------------------------------------')
                
                if POD:
                    
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
                                                   model_options = options, nonsmooth = nonsmooth)
                            
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
                        if random and 0:
                            int_u_0 = 15
                            U_tr = np.random.uniform(-int_u_0, int_u_0, (fom.pde.input_dim, fom.time_disc.K))
                            U_train = U_tr[:,:K_train] 
                        else:
                            # U_0
                            U_train = U[:,:K_train]
                        y0_train = fom.pde.y0
                        
                        UTRAIN = [u_TRUE, U_train, U_0]
                        SNAPS = []
                        for u in UTRAIN:
                            # get snapshots using the data for adjoint and state
                            Y_train, P_train, Out_train, _,_ = fom_train.get_snapshots(U = u,
                                                                                  y0 = y0_train)
                            
                            SNAPS.append(Y_train)
                            SNAPS.append(P_train)
                            
                        l_POD = max_size
                     
                    # create POD model
                    est_options = collection()
                    est_options.product = fom_train.pde.products['H1'] 
                    r_pod = pod_reductor(fom_train, 
                                          model_toproject = fom, 
                                          H_prod = fom_train.pde.products['L2'],
                                          space_product = fom_train.pde.products['H1'], 
                                          errorest_assembled = errorest_assembled)
                
                    rom_pod =  r_pod.get_rom(l = size,
                                            Snapshots = SNAPS,
                                            space_product = None,
                                            time_product = fom_train.time_disc.D_diag, 
                                            PODmethod = 0, 
                                            plot = True,
                                            use_energy_content = False)   
                    
                    if derivative_check:
                        rom_pod.derivative_check()
                        rom_pod.print_info()
                    print('POD ROM Optimize ...')
                    u_BB_POD, history_BB_POD = rom_pod.solve_ocp(U_0, 
                                                      "BB",
                                                      options = optionsBB)
                    
                    #state adjoint error
                    if 1:
                        Yr, Pr, Outr, switching_profile_r, gaps_r = rom_pod.get_snapshots(u_BB_POD)
                        
                        ktest = K-1
                        
                        print('STATE')
                        est_offline = rom_pod.state_est(U = u_BB_POD, k = ktest, Yr = Yr, switching_profile = switching_profile_r, 
                                          norm_type = 'L2V_list', computationtype = 'offline_online')
                        
                        # L2V
                        est_online = rom_pod.state_est(U = u_BB_POD, k = ktest, Yr = Yr, switching_profile = switching_profile_r, 
                                          norm_type = 'L2V_list', computationtype = 'online')
                        
                        est_true = rom_pod.state_est(U = u_BB_POD, k = ktest, Y = None, Yr = Yr, switching_profile = switching_profile_r, norm_type = 'L2V_list', computationtype = 'true')
                        
                        
                        # for k in range(len(est_online)):
                        #     print(f'L2V {k}: online est = {est_online[k]}, offline est = {est_off[k]}, true error = {est_true[k]}, effectivity online = {est_true[k]/est_online[k]}, diff online/offline {est_online[k]-est_off[k]}')
                            
                        state_col.off.append(est_offline[-1])
                        state_col.on.append(est_online[-1])
                        state_col.true.append(est_true[-1])
                        state_col.effoff.append(est_true[-1]/est_offline[-1])
                        state_col.effon.append(est_true[-1]/est_online[-1])
                        
                        print('ADJOINT')
                        type_ = 'LinfH_list'
                        est_online = rom_pod.adjoint_est(U = u_BB_POD, Yr = Yr, Y = None, Pr = Pr , P = None, type_ = type_,
                                  k = ktest, switching_profile_r = switching_profile_r, switching_profile = None, 
                                  out = None, out_r = Outr, state_H_est_list = None, computationtype = 'online', gaps_r = gaps_r)
                        
                        est_true = rom_pod.adjoint_est(U = u_BB_POD, Yr = Yr, Y = None, Pr = Pr , P = None, type_ = type_,
                                  k = ktest, switching_profile_r = switching_profile_r, switching_profile = None, 
                                  out = None, out_r = Outr, state_H_est_list = None, computationtype = 'true', gaps_r = gaps_r)
                        
                        est_offline = rom_pod.adjoint_est(U = u_BB_POD, Yr = Yr, Y = None, Pr = Pr , P = None, type_ = type_,
                                  k = ktest, switching_profile_r = switching_profile_r, switching_profile = None, 
                                  out = None, out_r = Outr, state_H_est_list = None, computationtype = 'offline_online', gaps_r = gaps_r)
                        
                        adstate_col.off.append(est_offline[-1])
                        adstate_col.on.append(est_online[-1])
                        adstate_col.true.append(est_true[-1])
                        adstate_col.effoff.append(est_true[-1]/est_offline[-1])
                        adstate_col.effon.append(est_true[-1]/est_online[-1])
                        
                        # for k in range(len(est_online)):
                        #     print(f'LinfH {k}: online residual = {est_online[k]}, offline est = {est_offline[k]}, true error = {est_true[k]}, effectivity = {est_true[k]/est_online[k]}, diff online/offline {est_online[k]-est_offline[k]}')
                            
                    
                    ######### compare estimates
                    # true error
                    true_error = fom.space_time_norm(u_TRUE-u_BB_POD , space_norm = "control")
                    true_error_out = fom.space_time_norm(history_TRUE['out_opt']-history_BB_POD['out_opt'] , space_norm = "output")
                    
                    # apply ROM control to FOM system and get output
                    _, out_FOMROM, _, _ = fom.solve_state(u_BB_POD)
                    true_error_outFOMROM = fom.space_time_norm(history_TRUE['out_opt']- out_FOMROM , space_norm = "output")
                    
                    # new expensive bound A
                    est_pod, Y_est, P_est, est_collA = fom.optimal_control_est(u_BB_POD, P = None, B_listTP =None, B_listTPr =  history_BB_POD['B_listTP_opt'], type_ = 'new', return_init_bound = True)
                    
                    # new expensive bound split up B
                    est_pod_split, Y_est_split, P_est_split, est_collB = fom.optimal_control_est(u_BB_POD, P = None, B_listTP = None,
                                                                                                        B_listTPr =  history_BB_POD['B_listTP_opt'], 
                                                                                                        out_r =history_BB_POD['out_opt'], 
                                                                                                        Yr = history_BB_POD['Y_opt'], 
                                                                                                        switch_profile_r = history_BB_POD['switch_profil_opt'],
                                                                                                        type_= 'new_split_up',
                                                                                                        return_init_bound = True)
                    
                    # perturbation bound old
                    if fom.cost_data.g_reg.type == 'box' or not fom.isNonSmoothlyRegularized():
                        est_pod_perturb, _, _, _ = fom.optimal_control_est(u_BB_POD, P = None, B_listTP =None, B_listTPr =  history_BB_POD['B_listTP_opt'], type_ = 'perturbation_standard')
                        pod.eff_perturb.append(true_error/est_pod_perturb)
                    else:
                        est_pod_perturb = None
                    
                    # prepare cheap bounds 
                    Yr, Pr, switch_profile_r, out_r, gaps_r = history_BB_POD['Y_opt'], history_BB_POD['P_opt'], history_BB_POD['switch_profil_opt'], history_BB_POD['out_opt'], history_BB_POD['gaps'],
                    
                    # bound B' online and offline
                    control_est_pod_cheap_online, _ = rom_pod.cheap_optimal_control_error_est(U = u_BB_POD, Yr = Yr, Pr = Pr , type_ = 'new_split_up_cheap',
                              k = None, switching_profile_r = switch_profile_r, out_r = out_r, state_H_est_list = None, computationtype = 'online', return_init_bound = True, gaps_r = gaps_r)
                    control_est_pod_cheap_offline, est_coll_cheapB = rom_pod.cheap_optimal_control_error_est(U = u_BB_POD, Yr = Yr, Pr = Pr , type_ = 'new_split_up_cheap',
                              k = None, switching_profile_r = switch_profile_r, out_r = out_r, state_H_est_list = None, computationtype = 'offline_online', return_init_bound = True, gaps_r = gaps_r)
                    
                    # bound A' online and offline
                    control_est_pod_cheapA_onlineA, _ = rom_pod.cheap_optimal_control_error_est(U = u_BB_POD, Yr = Yr, Pr = Pr , type_ = 'new_cheap',
                              k = None, switching_profile_r = switch_profile_r, out_r = out_r, state_H_est_list = None, computationtype = 'online', return_init_bound = True, gaps_r = gaps_r)
                    control_est_pod_cheapA_offlineA, est_coll_cheapA = rom_pod.cheap_optimal_control_error_est(U = u_BB_POD, Yr = Yr, Pr = Pr , type_ = 'new_cheap',
                              k = None, switching_profile_r = switch_profile_r, out_r = out_r, state_H_est_list = None, computationtype = 'offline_online', return_init_bound = True, gaps_r = gaps_r)
                    
                    print('OPTIMIZATION POD ROM - FOM control')
                    print(f'control errors:\n-true error: {true_error}\n-expensive A: {est_pod}\n-expensive B: {est_pod_split}\n-expensive perturb: {est_pod_perturb}\n-cheap online A: {control_est_pod_cheapA_onlineA}, cheap offline A {control_est_pod_cheapA_offlineA}\n-cheap online B: {control_est_pod_cheap_online}, cheap offline B {control_est_pod_cheap_offline}')
                    print(f'effectivities:\n-expensive A: {true_error/est_pod}\n-expensive B: {true_error/est_pod_split}\n-cheap A: {true_error/control_est_pod_cheapA_offlineA}\n-cheap B: {true_error/control_est_pod_cheap_offline}')
                    print(f'timings pod rom {history_BB_POD["time"]}, fom {history_TRUE["time"]}, speedup {history_TRUE["time"]/history_BB_POD["time"]}')
                    
                    print(f'output errors ROM ROM:\n-true error: {true_error_out}\n-expensive B: {est_collB.est_output_current}\n-cheap offline B {est_coll_cheapB.est_output_current}\n')
                    print(f'output errors FOM ROM:\n-true error: {true_error_outFOMROM}\n-expensive A: {est_collA.est_output_current}\n-cheap offline A {est_coll_cheapA.est_output_current}\n')
                    
                    ests = []
                    # control bounds
                    pod.speedups.append(history_TRUE["time"]/history_BB_POD["time"])
                    pod.timings.append(history_BB_POD["time"])
                    pod.true_error.append(true_error)
                    pod.error_cheap.append(control_est_pod_cheap_offline)
                    pod.error_cheap_online.append(control_est_pod_cheap_online)
                    pod.error_expensive.append(est_pod)
                    pod.iter.append(history_BB_POD['k'])
                    pod.error_exp_split.append(est_pod_split)
                    pod.error_exp_perturb.append(est_pod_perturb)
                    pod.error_cheapA.append(control_est_pod_cheapA_offlineA)
                    pod.error_cheapA_online.append(control_est_pod_cheapA_onlineA)
                    
                    # output bounds
                    pod_output.A.append(est_collA.est_output_current)
                    pod_output.Acheap.append(est_coll_cheapA.est_output_current)
                    pod_output.B.append(est_collB.est_output_current)
                    pod_output.Bcheap.append(est_coll_cheapB.est_output_current)
                    pod_output.Atrue.append(true_error_outFOMROM)
                    pod_output.Btrue.append(true_error_out)
                    
                    # effectivities
                    pod.eff_exp1.append(true_error/est_pod)
                    pod.eff_exp2.append(true_error/est_pod_split)
                    pod.eff_cheap1.append(true_error/control_est_pod_cheapA_offlineA)
                    pod.eff_cheap2.append(true_error/control_est_pod_cheap_online)
                
        print('FINISHED #####################################################################################')
        
#%% control est visualization
        
        if POD:
           # POD all estimates
            plt.figure()
            plt.title('All estimates',fontsize=10)
            try:
                plt.semilogy(SIZES, pod.true_error, label= r'true error')
            except:
                pass
            try:
                plt.semilogy(SIZES, pod.error_cheap, label= r'cheap B off')
            except:
                pass
            try:
                plt.semilogy(SIZES, pod.error_cheap_online, label= r'cheap B on')
            except:
                pass
            try:
                plt.semilogy(SIZES, pod.error_cheapA, label= r'cheap A off')
            except:
                pass
            try:
                plt.semilogy(SIZES, pod.error_cheapA_online, label= r'cheap A on')
            except:
                pass
            try:
                plt.semilogy(SIZES, pod.error_exp_split, label= r'expensive B')
            except:
                # pod.error_exp_split
                pass
            try:
                plt.semilogy(SIZES, pod.error_expensive, label= r'expensive A')
            except:
                pass
            try:
                plt.semilogy(SIZES, pod.error_exp_perturb, label= r'perturb')
            except:
                pass
            plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=10)
            # plt.legend(loc='lower right', ncol=1)
            plt.xlabel('Basis size', fontsize=12)
            # plt.ylabel('y-Achse', fontsize=12)
            
            if POD:
                # plost for apaper and effeectiviities
                # POD all estimates
                plt.figure()
                plt.title('All estimates 2',fontsize=10)
                try:
                    plt.semilogy(SIZES, pod.true_error, label= r'true error')
                except:
                    pass
                try:
                    plt.semilogy(SIZES, pod.error_cheap, label= r'cheap B off')
                except:
                    pass
                # try:
                #     plt.semilogy(SIZES, pod.error_cheap_online, label= r'cheap B on')
                # except:
                #     pass
                try:
                    plt.semilogy(SIZES, pod.error_cheapA, label= r'cheap A off')
                except:
                    pass
                # try:
                #     plt.semilogy(SIZES, pod.error_cheapA_online, label= r'cheap A on')
                # except:
                #     pass
                try:
                    plt.semilogy(SIZES, pod.error_exp_split, label= r'expensive B')
                except:
                    # pod.error_exp_split
                    pass
                try:
                    plt.semilogy(SIZES, pod.error_expensive, label= r'expensive A')
                except:
                    pass
                # try:
                #     plt.semilogy(SIZES, pod.error_exp_perturb, label= r'perturb')
                # except:
                #     pass
                fom.plot_to_csv(ax = plt.gca(), name =  data_folder +'basissize_control_error_est.csv')
                plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=10)
                # plt.legend(loc='lower right', ncol=1)
                plt.xlabel('Basis size', fontsize=12)
                # plt.ylabel('y-Achse', fontsize=12)
                
                plt.figure()
                plt.title('All estimates effectivities',fontsize=10)
                try:
                    plt.semilogy(SIZES, pod.eff_cheap2 , label= r'cheap B off')
                except:
                    pass
                # try:
                #     plt.semilogy(SIZES, pod.error_cheap_online, label= r'cheap B on')
                # except:
                #     pass
                try:
                    plt.semilogy(SIZES, pod.eff_cheap1, label= r'cheap A off')
                except:
                    pass
                # try:
                #     plt.semilogy(SIZES, pod.error_cheapA_online, label= r'cheap A on')
                # except:
                #     pass
                try:
                    plt.semilogy(SIZES, pod.eff_exp2 , label= r'expensive B')
                except:
                    # pod.error_exp_split
                    pass
                try:
                    plt.semilogy(SIZES, pod.eff_exp1, label= r'expensive A')
                except:
                    pass
                # try:
                #     plt.semilogy(SIZES, pod.error_exp_perturb, label= r'perturb')
                # except:
                #     pass
                fom.plot_to_csv(ax = plt.gca(), name =  data_folder +'basissize_control_eff.csv')
                plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=10)
                # plt.legend(loc='lower right', ncol=1)
                plt.xlabel('Basis size', fontsize=12)
                # plt.ylabel('y-Achse', fontsize=12)
            
            # POD all offline estimates
            plt.figure()
            plt.title('All estimates',fontsize=10)
            try:
                plt.semilogy(SIZES, pod.true_error, label= r'true error')
            except:
                pass
            try:
                plt.semilogy(SIZES, pod.error_cheap, label= r'cheap B off')
            except:
                pass
            try:
                # plt.semilogy(SIZES, pod.error_cheap_online, label= r'cheap B on')
                pass
            except:
                pass
            try:
                plt.semilogy(SIZES, pod.error_cheapA, label= r'cheap A off')
            except:
                pass
            try:
                # plt.semilogy(SIZES, pod.error_cheapA_online, label= r'cheap A on')
                pass
            except:
                pass
            # try:
            #     plt.semilogy(SIZES, pod.error_exp_split, label= r'expensive B')
            # except:
            #     # pod.error_exp_split
            #     pass
            # try:
            #     plt.semilogy(SIZES, pod.error_expensive, label= r'expensive A')
            # except:
            #     pass
            # try:
            #     plt.semilogy(SIZES, pod.error_exp_perturb, label= r'perturb')
            # except:
            #     pass
            plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=10)
            # plt.legend(loc='lower right', ncol=1)
            plt.xlabel('Basis size', fontsize=12)
            # plt.ylabel('y-Achse', fontsize=12)
        
            
            # POD expensive estimates
            plt.figure()
            plt.title('Better scaling in regularization parameter',fontsize=10)
            try:
                plt.semilogy(SIZES, pod.true_error, label= r' true error')
            except:
                pass
            try:
                plt.semilogy(SIZES, pod.error_expensive, 'x-', label= r'A')
            except:
                pass
            try:
                plt.semilogy(SIZES, pod.error_exp_split, label= r'B')
            except:
                pass
            try:
                plt.semilogy(SIZES, pod.error_exp_perturb, '--', label= r'perturb')
            except:
                pass
            plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=10)
            plt.xlabel('Basis size', fontsize=12)
            # plt.legend(loc='lower right', ncol=1)
            
            if 0:
                # basis vs speedups
                plt.figure()
                try:
                    plt.plot(SIZES, bt.speedups, label= r'BT speedup')
                except:
                    pass
                try:
                    plt.plot(SIZES, pod.speedups, label= r'POD speedup')
                    pass
                except:
                    pass
                plt.legend(loc='lower right', ncol=1)
                
                plt.figure()
                try:
                    plt.plot(SIZES, bt.timings, label= r'BT timing')
                except:
                    pass
                try:
                    plt.plot(SIZES, pod.timings, label= r'POD timing')
                    pass
                except:
                    pass
                plt.legend(loc='lower right', ncol=1)
                
                # iter
                plt.figure()
                try:
                    plt.plot(SIZES, bt.iter, label= r'BT iter')
                except:
                    pass
                try:
                    plt.plot(SIZES, pod.iter, label= r'POD iter')
                    pass
                except:
                    pass
                plt.legend(loc='lower right', ncol=1)
         
#%% output est visu
                
            ####### OUTPUT ESTS
            # FOM ROM output error estimates A
            plt.figure()
            plt.title('Output FOM ROM A',fontsize=10)
            try:
                plt.semilogy(SIZES, pod_output.Atrue, label= r'true error (FOM ROM)')
            except:
                pass
            try:
                plt.semilogy(SIZES, pod_output.A, 'x-', label= r'A')
            except:
                pass
            try:
                plt.semilogy(SIZES, pod_output.Acheap, label= r'Acheap')
            except:
                pass
            fom.plot_to_csv(ax = plt.gca(), name =  data_folder +'basissize_outA.csv')
            plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=10)
            plt.xlabel('Basis size', fontsize=12)
            
            # ROM ROM output error estimates B
            plt.figure()
            plt.title('Output ROM ROM B',fontsize=10)
            try:
                plt.semilogy(SIZES, pod_output.Btrue, label= r'true error (ROM ROM)')
            except:
                pass
            try:
                plt.semilogy(SIZES, pod_output.B, 'x-', label= r'B')
            except:
                pass
            try:
                plt.semilogy(SIZES, pod_output.Bcheap, label= r'Bcheap')
            except:
                pass
            fom.plot_to_csv(ax = plt.gca(), name =  data_folder +'basissize_outB.csv')
            plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=10)
            plt.xlabel('Basis size', fontsize=12)
            
            # all output error estimates
            plt.figure()
            plt.title('ALL output bounds',fontsize=10)
            try:
                plt.semilogy(SIZES, pod_output.Atrue, label= r'true error (FOM ROM)')
            except:
                pass
            try:
                plt.semilogy(SIZES, pod_output.A, 'x-', label= r'A')
            except:
                pass
            try:
                plt.semilogy(SIZES, pod_output.Acheap, label= r'Acheap')
            except:
                pass
            try:
                plt.semilogy(SIZES, pod_output.Btrue, label= r'true error (ROM ROM)')
            except:
                pass
            try:
                plt.semilogy(SIZES, pod_output.B, 'x-', label= r'B')
            except:
                pass
            try:
                plt.semilogy(SIZES, pod_output.Bcheap, label= r'Bcheap')
            except:
                pass
            plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=10)
            plt.xlabel('Basis size', fontsize=12)
        
#%% state est visu
        
        if 1:
            title = ['state est', 'adjoint est']
            ests = [state_col, adstate_col]
            
            for i in range(len(title)):
                
                struct = ests[i]
                
                plt.figure()
                plt.title(title[i],fontsize=10)
                try:
                    plt.semilogy(SIZES, struct.on, label= r'online')
                except:
                    pass
                try:
                    plt.semilogy(SIZES, struct.off, label= r'offline')
                except:
                    pass
                try:
                    plt.semilogy(SIZES, struct.true, label= r'true')
                except:
                    pass
                fom.plot_to_csv(ax = plt.gca(), name =  data_folder + title[i] +'est_basis.csv')
                plt.legend(loc='best', ncol=1)
                plt.show()
                
                plt.title(title[i]+' effectivities',fontsize=10)
                try:
                    plt.semilogy(SIZES, struct.effon, label= r'online')
                except:
                    pass
                try:
                    plt.semilogy(SIZES, struct.effoff, label= r'offline')
                except:
                    pass
                fom.plot_to_csv(ax = plt.gca(), name =  data_folder + title[i] +'effec_basis.csv')
                plt.legend(loc='best', ncol=1)
                
                plt.show()