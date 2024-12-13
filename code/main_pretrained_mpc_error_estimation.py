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
# Description: this file can be used to test the MPC error estimators.

import statistics
from mpc import mpc
from discretizer import discretize, get_y0
import numpy as np
from methods import collection, get_random_switching_law, get_switching_law
import fenics as fe
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})
plt.rcParams['lines.linewidth'] = 3

#%% flags

data_folder = 'data/test_error_est/'

# repetitions for random y0, u0
repeat = 1

PODmethod = 2
number_switches = 20
PLOT = True
print_lists = False

# assemble error est or not
energy_prod = True
FOM = True
POD = True
optimization = False
debug = False
print_ = False
state_dependent_switching = False
visualize_only_room2 = False
nonsmooth = 'l1box'

#%% get fom

options = collection()
# only gets used if switch model
options.factorize = True
options.energy_prod = True

# get pde model
T = 10  
K = 501
dx = 1

fom = discretize(T=T, dx=dx, K=K, debug=debug,
                     state_dependent_switching=state_dependent_switching,
                     model_options=options, nonsmooth=nonsmooth, use_energy_products=energy_prod)

if 1:
    U = (1*np.ones((fom.pde.input_dim, fom.time_disc.K)))

else:
    funs_u = [lambda x: 1*np.cos(x),
        lambda x: np.sin(2*np.pi*x),lambda x:  1,lambda x: 3*np.sin(x),lambda x: np.cos(2*np.pi*x)
        ]
    U = np.zeros((fom.pde.input_dim, fom.time_disc.K))
    for i in range(fom.pde.input_dim):  
        xx = funs_u[i%len(funs_u)](fom.time_disc.t_v)
        U[i,:] = xx

# choose switching signal
if 0:
    random_switching_law = get_random_switching_law(T=T, numer_random_switch_points=number_switches, init_sigma=1)
else:
    random_switching_law = get_switching_law(T, [0, 0.5,1,1.5,2, 2.5, 3, 3.5, 4,4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, T], init_sigma= 1)
fom.pde.sigma = random_switching_law

if 1:
    # get YD
    print('FOM solving ...')
    y0 = get_y0(fom.space_disc.V, fe.Expression('1', degree=1))
    
    # solve state to get yd
    if fom.isSwitchModel():
    
        # solve state
        Y, Out, time, switch_profil = fom.solve_state(U, theta=fom.theta, print_=print_,
                                                      y0=y0)
        fom.visualize_output(Out, only_room2=visualize_only_room2, title='FOM', semi=True)
    Yd = Out
else:
    funs = [lambda x: np.cos(x),
            lambda x: np.cos(x),
            ]
    Yd = np.zeros((fom.pde.output_dim, fom.time_disc.K))
    for i in range(fom.pde.output_dim):  
        xx = funs[i](fom.time_disc.t_v)
        Yd[i,:] = xx

fom.update_cost_data(Yd=Yd,
                     YT=Yd[:, -1],
                     Ud=np.zeros((fom.input_dim, fom.time_disc.K)),  # U
                     weights=[1, 1e-2, 0],
                     input_product=fom.cost_data.input_product,
                     output_product=fom.cost_data.output_product)

# uncontrolled quantities
Y_uncontrolled, Out_uncontrolled, _, _ = fom.solve_state(
    0*U, theta=fom.theta, print_=print_)
J_uncontrolled_val = fom.J(0*U)
J_uncontrolled_track, J_uncontrolled_cont, J_uncontrolled_traj = fom.Jtracking_trajectory(
    0*U)

# ud controlled quantities
Y_ud, Out_ud, _, _ = fom.solve_state(
    fom.cost_data.Ud, theta=fom.theta, print_=print_)
J_ud_val = fom.J(U)
J_ud_track, J_ud_cont, J_ud_traj = fom.Jtracking_trajectory(U)

# print info
fom.print_info()

#%% set up for loop

if 1:
    u_low =-20
    u_up = 20
    U_0 = np.random.uniform(u_low, u_up, size=(fom.input_dim, fom.time_disc.K))
else: 
    U_0 = 5*np.ones((fom.input_dim, fom.time_disc.K))

# choose initial value randomly
if 0:
    u_low =-20
    u_up = 20
    y0_init = np.random.uniform(u_low, u_up, size=(fom.state_dim,))
    fom.pde.y0 = y0_init
else: 
    pass
    
for ind in range(repeat):
        print(f'############ Repeat {ind+1} of {repeat}#####################')
        
#%% mpc/optimization setup
        
        # start
        u_TRUE = None
        history_TRUE = None
        
        # mpc setup
        l_POD = 100
        kp = min(20, K-1)
        kf = min(1, kp)   
        coarse = False
        State_update_tol_start_end = [1e5, 1e5]
        Control_update_tol_start_end = [1e5, 1e5]
        restart = False
        mpcplot_ = False
        len_old_snaps = 4
        
        # other global options for all methods
        # mpc inner solver
        tol = 1e-14
        maxit = 500
        innersolver_options = fom.set_default_options(tol=tol, maxit=maxit,
                                                                  save=False,
                                                                  plot=False,
                                                                  print_info=False)
        
        # fom inner solver for adaptive methods to obtain new snapshots
        tol_fom = 1e-14
        maxit_fom = 500
        fom_predictor_innersolver_options = fom.set_default_options(tol=tol_fom, maxit=maxit_fom,
                                                                                save=False,
                                                                                plot=False,
                                                                                print_info=False)
        
#%% setup fom and fom rom mpc expensive A
        
        from methods import get_mpc_options
        
        mpc_options = get_mpc_options(kf, 
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
                                      error_est_type = 'compare',
                                      type_ = 'FOMROM'
                                      )
        
#%% optimization
        
        if optimization:
            tol_all_at_once = 1e-12
            maxit_all_at_once = 500
            optionsBB = fom.set_default_options(tol=tol_all_at_once, maxit=maxit_all_at_once,
                                                save=False,
                                                plot=False,
                                                print_info=False)
            print('FOM Optimize ...')
            u_TRUE, history_TRUE = fom.solve_ocp(U_0,
                                                 "BB",
                                                 options=optionsBB)
            J_true_val = fom.J(u_TRUE)
            J_true_track, J_true_cont, J_true_traj = fom.Jtracking_trajectory(u_TRUE)
        
        
#%% fom mpc
        
        if FOM:
            
            # FOM MPC
            fom_mpc = mpc(prediction_model=None,
                          model=fom,
                          options=mpc_options)
            U_fom, Y_fom, P_fom, out_fom, history_fom = fom_mpc.solve(U_0)
        
#%% rom mpc
        
        if POD:
            
            # create training fom
            if POD:
                
                # create reduce object and rom (rom is directly over written in the adaptive methods)
                from methods import get_reductor_and_rom
                rom_pod, r_pod = get_reductor_and_rom(min(kp,K-1), fom, dx, debug, options, 
                                                      nonsmooth, energy_prod, 
                                                      random_switching_law, 
                                                      U_0, PODmethod, 
                                                      errorest_assembled = True,
                                                      l_POD = 200)

            if 1:
                # FOMROM
                mpc_fompod = mpc(prediction_model=rom_pod,
                                 model=fom,
                                 options=mpc_options,
                                 reductor=r_pod,
                                 options_add = {'test_estimate': True, 'FOMROM': True})
                
                U_fomrom, Y_fomrom, P_fomrom, out_fomrom, history_fomrom = mpc_fompod.solveFOMROM(
                    U_0)
                
            if 1:
                # ROMROM
                mpc_romrom = mpc(prediction_model=rom_pod,
                                       model=fom,
                                       options=mpc_options,
                                       reductor=r_pod,
                                       options_add = {'test_estimate': True, 'FOMROM': False})
                # run
                U_romrom, Y_romrom, P_romrom, output_romrom, historY_romrom = mpc_romrom.solveFOMROM(
                    U_0)
             
#%% print results
        
        print(f'-----------------------------RESULTS {ind+1}/{repeat}---------------------------------------------------------------')
        try:
            print(f'MPC FOM: time {history_fom["time"]} ------------------')
            print(f'value J: {history_fom["J"]}')
            if history_TRUE is not None:
                print(f'true val J: {J_true_val}')
                print(
                    f'rel FOM exakt output error {fom.rel_error_norm(history_TRUE["out_opt"], history_fom["output"], "output")}, rel FOM exact - control error {fom.rel_error_norm(u_TRUE, U_fom, "control")}, rel cost error = {abs(J_true_val-history_fom["J"])}')
                print(f'rel J value error: {abs((history_fom["J"]-J_true_val)/J_true_val)}')
                print('----------')
        except:
            pass
            
        #### FOMROM 
        try:
            print(
                f'FOMROM: time {history_fomrom["time"]}, final basis size {history_fomrom["final_basissize"]}, average basisize over iterations {statistics.mean(history_fomrom["basis_size"])}, speed-up {history_fom["time"]/history_fomrom["time"]} ------------------')
            print(
                f'enriched in {history_fomrom["enriched_in_iter"]} and {len(history_fomrom["enriched_in_iter"])}  times')
            print(f'rel FOM  L2(H) state error {fom.rel_error_norm(Y_fom,Y_fomrom, "L2")}, rel FOM control error {fom.space_time_norm(U_fomrom-U_fom , space_norm = "control")/fom.space_time_norm(U_fom , space_norm = "control")}')
            print(
                f'rel output error {fom.rel_error_norm(out_fom, out_fomrom, "output")}')
            print(f'rel J value error: {abs((history_fomrom["J"]-history_fom["J"])/history_fom["J"])}')
            print(f'value J: {history_fomrom["J"]}')
            print('----------')
        except:
            pass
        
        #### ROMROM
        try:
            print(
                f'ROMROM: time {historY_romrom["time"]}, final basis size {historY_romrom["final_basissize"]} average basisize over iterations {statistics.mean(historY_romrom["basis_size"])},  speed-up {history_fom["time"]/historY_romrom["time"]}------------------')
            print(f'enriched in {historY_romrom["enriched_in_iter"]} and {len(historY_romrom["enriched_in_iter"])}  times')
            print(f'rel FOM  L2(H) state error {fom.rel_error_norm(Y_fom, Y_romrom, "L2")}, rel FOM control error {fom.space_time_norm( U_romrom-U_fom , space_norm = "control")/fom.space_time_norm(U_fom , space_norm = "control")}')
            print(
                f'rel output error {fom.rel_error_norm(out_fom, output_romrom, "output")}')
            print(f'rel J value error: {abs((historY_romrom["J"]-history_fom["J"])/history_fom["J"])}')
            print(f'value J: {historY_romrom["J"]}')
            print('----------')
        except:
            pass
        
        k = 150
        hists = [history_fomrom, historY_romrom]
        title = ['FOMROM', 'ROMROM']
        
        true_errorFOMROM = fom_mpc.compute_true_errors(history_fom['control_list'], history_fomrom['control_list'], Y_fom, Y_fomrom , history_fomrom['switch_profile'])
        true_errorROMROM = fom_mpc.compute_true_errors(history_fom['control_list'], history_fomrom['control_list'], Y_fom, Y_romrom, historY_romrom['switch_profile'])
        
        true_error_control = [true_errorFOMROM['control_L2error'], true_errorROMROM['control_L2error']]
        true_error_state = [true_errorFOMROM['state_MA_error'], true_errorROMROM['state_MA_error']]
        
        # # compute true control on each prediction intervall
        # control_trueFOMROM = np.sqrt(fom.space_time_product( U_fomrom -U_fom, U_fomrom -U_fom , space_norm = "control", return_trajectory = True))
        # control_trueROMROM = np.sqrt(fom.space_time_product( U_romrom-U_fom, U_romrom-U_fom , space_norm = "control", return_trajectory = True))
        
        # # compute true state error on each feedback intervall and tn
        # MPC_trueFOMROM = np.sqrt(fom.space_time_product(Y_fom-Y_fomrom, Y_fom-Y_fomrom, space_norm = 'M_switch', return_trajectory = True, switch_profile = history_fomrom['switch_profile']))
        # MPC_trueROMROM = np.sqrt(fom.space_time_product(Y_fom-Y_romrom, Y_fom-Y_romrom, space_norm = 'M_switch', return_trajectory = True, switch_profile = historY_romrom['switch_profile']))
       
        # true_error_control = [control_trueFOMROM, control_trueROMROM]
        # true_error_state = [MPC_trueFOMROM, MPC_trueROMROM]
        
        for i in range(len(title)):
            
            struct = hists[i]['compare_ests']
            true_control = np.array(true_error_control[i])
            true_state = np.array(true_error_state[i])
            
            plt.figure()
            plt.title(title[i] + ' control',fontsize=10)
            if title[i] == 'FOMROM' or 1:
                try:
                    plt.semilogy(struct.control_initA[:k], label= r'A')
                except:
                    pass
                try:
                    plt.semilogy(struct.control_initB[:k], label= r'B')
                except:
                    pass
            try:
                plt.semilogy(struct.control_initAcheap[:k], label= r'Acheap')
            except:
                pass
            try:
                plt.semilogy(struct.control_initBcheap[:k], label= r'Bcheap')
            except:
                pass
            try:
                plt.semilogy(true_control[:k], label= r'true')
            except:
                pass
            plt.ylim(1e-10, 1e1)
            fom.plot_to_csv(ax = plt.gca(), name =  data_folder + title[i] +'control_pretr.csv')
            plt.legend(loc='best', ncol=1)
            plt.show()
            
            plt.figure()
            plt.title(title[i] + ' effetctivities control',fontsize=10)
            if title[i] == 'FOMROM' or 1:
                try:
                    plt.semilogy((true_control/struct.control_initA)[:k], label= r'A')
                except:
                    pass
                try:
                    plt.semilogy((true_control/struct.control_initB)[:k], label= r'B')
                except:
                    pass
            try:
                plt.semilogy((true_control/struct.control_initAcheap)[:k], label= r'Acheap')
            except:
                pass
            try:
                plt.semilogy((true_control/struct.control_initBcheap)[:k], label= r'Bcheap')
            except:
                pass
            # try:
            #     plt.semilogy(true_control, label= r'true')
            # except:
            #     pass
            # plt.ylim(1e-8, 1e1)
            fom.plot_to_csv(ax = plt.gca(), name =  data_folder + title[i] +'effect_control_pretr.csv')
            plt.legend(loc='best', ncol=1)
            plt.show()
            
            # plt.figure()
            # plt.title(title[i] + ' current_control',fontsize=10)
            # if title[i] == 'FOMROM' or 1:
            #     try:
            #         plt.semilogy(struct.controlA[:k], label= r'Acurrent')
            #     except:
            #         pass
            #     try:
            #         plt.semilogy(struct.controlB[:k], label= r'Bcurrent')
            #     except:
            #         pass
            # try:
            #     plt.semilogy(struct.controlAcheap[:k], label= r'Acheap current')
            # except:
            #     pass
            # try:
            #     plt.semilogy(struct.controlBcheap[:k], label= r'Bcheap current')
            # except:
            #     pass
            # try:
            #     plt.semilogy(true_control[:k], label= r'true')
            # except:
            #     pass
            # # plt.ylim(1e-6, 1e-1)
            # fom.plot_to_csv(ax = plt.gca(), name =  data_folder + title[i] +'current_control_pretr.csv')
            # plt.legend(loc='best', ncol=1)
            # plt.show()
            
            plt.figure()
            plt.title(title[i]+' state',fontsize=10)
            if title[i] == 'FOMROM'or 1:
                try:
                    plt.semilogy(struct.MPC_initA[:k], label= r'A')
                except:
                    pass
                try:
                    plt.semilogy(struct.MPC_initB[:k], label= r'B')
                except:
                    pass
            try:
                plt.semilogy(struct.MPC_initAcheap[:k], label= r'Acheap')
            except:
                pass
            try:
                plt.semilogy(struct.MPC_initBcheap[:k], label= r'Bcheap')
            except:
                pass
            try:
                plt.semilogy(true_state[:k], label= r'true')
            except:
                pass
            # plt.ylim(0, 12)
            fom.plot_to_csv(ax = plt.gca(), name =  data_folder + title[i] +'state_pretr.csv')
            plt.legend(loc='best', ncol=1)
            plt.show()
            
            plt.figure()
            plt.title(title[i]+' effect_state',fontsize=10)
            if title[i] == 'FOMROM'or 1:
                try:
                    plt.semilogy((true_state/struct.MPC_initA)[:k], label= r'A')
                except:
                    pass
                try:
                    plt.semilogy((true_state/struct.MPC_initB)[:k], label= r'B')
                except:
                    pass
            try:
                plt.semilogy((true_state/struct.MPC_initAcheap)[:k], label= r'Acheap')
            except:
                pass
            try:
                plt.semilogy((true_state/struct.MPC_initBcheap)[:k], label= r'Bcheap')
            except:
                pass
            # try:
            #     plt.semilogy(true_state, label= r'true')
            # except:
            #     pass
            # plt.ylim(0, 12)
            fom.plot_to_csv(ax = plt.gca(), name =  data_folder + title[i] +'effect_state_pretr.csv')
            plt.legend(loc='best', ncol=1)
            plt.show()
            