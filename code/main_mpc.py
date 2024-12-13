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
# Description: main file for comparing FOM-MPC with its reduced variants.
    
from mpc import mpc
from discretizer import discretize, get_y0
import numpy as np
from methods import collection, get_random_switching_law, get_switching_law
import fenics as fe
import matplotlib.pyplot as plt
import statistics
from methods import get_reductor_and_rom

#%% init

# folders
data_folder = 'data/test/'
plot_folder = 'plots/'

# update tolerance
constant_tol = 1e-3
State_update_tol_start_end = [constant_tol, constant_tol]
Control_update_tol_start_end = [constant_tol, constant_tol]

# options
PODmethod = 2                                                                   # POD computation type
number_switches = 20                                                            # number switches
PLOT = True                                                                     # plot option
energy_prod = True                                                              # use energy product
FOM = True                                                                      # perform FOM calculations
POD = True                                                                      # perform PODROM calculations
true_optimization = False                                                       # compute true solution
debug = False                                                                   # debug option
print_ = False                                                                  # print option                                                            
nonsmooth = 'l1box'                                                             # nonsmooth regularization type                                        
restart = True                                                                  # reset Delta_t_n after ROM update

#%% create fom

# fom options
options = collection()
options.factorize = True
options.energy_prod = energy_prod

# time, space discretization
T = 10  
K = 501 
dx = 1

# get pde model
print('Discretize problem...')
fom = discretize(T=T, dx=dx, K=K, debug = debug,
                 state_dependent_switching = False,
                 model_options = options, 
                 nonsmooth=nonsmooth, 
                 use_energy_products = energy_prod)

# get control to construct yd
if 1:
    U = 1*np.ones((fom.pde.input_dim, fom.time_disc.K))

else:
    funs_u = [lambda x: 1*np.cos(x),
              lambda x: np.sin(2*np.pi*x),
              lambda x:  1,
              lambda x: 3*np.sin(x),
              lambda x: np.cos(2*np.pi*x)
              ]
    
    U = np.zeros((fom.pde.input_dim, fom.time_disc.K))
    for i in range(fom.pde.input_dim):  
        xx = funs_u[i%len(funs_u)](fom.time_disc.t_v)
        U[i,:] = xx

# get switching law
if 0:
    random_switching_law = get_random_switching_law(T=T, numer_random_switch_points=number_switches, init_sigma=1)
else:
    random_switching_law = get_switching_law(T, [0, 0.5,1,1.5,2, 2.5, 3, 3.5, 4,4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, T], init_sigma= 1)
fom.pde.sigma = random_switching_law

# get target yd
if 1:
    print('Solving FOM to construct target yd ...')
    y0 = get_y0(fom.space_disc.V, fe.Expression('1', degree=1))

    # solve state
    Y, Yd, time, switch_profil = fom.solve_state(U, theta=fom.theta, print_=print_,
                                                  y0=y0)
    # visualize output
    fom.visualize_output(Yd, only_room2=False, title=r'Target $y_d$', semi=True)
    
else:
    funs = [lambda x: np.sin(3.141*x),
            lambda x: np.cos(x),
            ]
    Yd = np.zeros((fom.pde.output_dim, fom.time_disc.K))
    for i in range(fom.pde.output_dim):  
        Yd[i,:] = funs[i](fom.time_disc.t_v)
    fom.visualize_output(Yd, only_room2=False, title='FOM', semi=False)

# update fom with true_optimization data
fom.update_cost_data(Yd=Yd,
                     YT=Yd[:, -1],
                     Ud=np.zeros((fom.input_dim, fom.time_disc.K)),  # U
                     weights=[1, 1e-2, 0],
                     input_product=fom.cost_data.input_product,
                     output_product=fom.cost_data.output_product)

# get uncontrolled quantities for comaprison
Y_uncontrolled, Out_uncontrolled, _, _ = fom.solve_state(0*U, 
                                                         theta=fom.theta, 
                                                         print_=print_)
J_uncontrolled_val = fom.J(0*U)
J_uncontrolled_track, J_uncontrolled_cont, J_uncontrolled_traj = fom.Jtracking_trajectory(0*U)

# get ud uncontrolled quantities for comaprison
Y_ud, Out_ud, _, _ = fom.solve_state(fom.cost_data.Ud, 
                                     theta=fom.theta, 
                                     print_=print_)
J_ud_val = fom.J(U)
J_ud_track, J_ud_cont, J_ud_traj = fom.Jtracking_trajectory(U)

# print info
fom.print_info()

#%% set up for loop
     
# get control initial guess       
if 1:
    u_low = fom.cost_data.g_reg.u_low
    u_up = fom.cost_data.g_reg.u_up
    U_0 = np.random.uniform(u_low, u_up, size=(fom.input_dim, fom.time_disc.K))
else: 
    U_0 = 5*np.ones((fom.input_dim, fom.time_disc.K))
        
#%% mpc options
        
# general setup
l_POD = 100                                                                     # maximum basis size
kp = min(20, K-1)                                                               # prediciton horizon
kf = min(1, kp)                                                                 # feedback horizon
coarse = False
mpcplot_ = False
len_old_snaps = 7
tol = 1e-11
maxit = 500
innersolver_options = fom.set_default_options(tol=tol, 
                                              maxit=maxit,
                                              save=False,
                                              plot=False,
                                              print_info=False)
# fom inner solver for adaptive methods to obtain new snapshots
tol_fom = 1e-6
maxit_fom = 500
fom_predictor_innersolver_options = fom.set_default_options(tol=tol_fom,
                                                            maxit=maxit_fom,
                                                            save=False,
                                                            plot=False,
                                                            print_info=False)

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
                              error_est_type = 'expensive',
                              type_ = 'FOMROM'
                              )

mpc_optionsFOMROMcheap =  get_mpc_options(kf, 
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
                              error_est_type = 'cheap',
                              type_ = 'FOMROM'
                              )

mpc_optionsROMROMcheap =  get_mpc_options(kf, 
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
                              error_est_type = 'cheap',
                              type_ = 'ROMROM'
                              )
        
#%% run true optimization
        
if true_optimization:
    tol_all_at_once = 1e-12
    maxit_all_at_once = 500
    optionsBB = fom.set_default_options(tol= tol_all_at_once, 
                                        maxit= maxit_all_at_once,
                                        save= False,
                                        plot= False,
                                        print_info= False)
    print('Perform true (global in time) FOM optimization ...')
    u_TRUE, history_TRUE = fom.solve_ocp(U_0,
                                         "BB",
                                         options=optionsBB)
    J_true_val = fom.J(u_TRUE)
    J_true_track, J_true_cont, J_true_traj = fom.Jtracking_trajectory(u_TRUE)
else:
    u_TRUE = None
    history_TRUE = None
     
#%% run mpc
        
#### FOM MPC
if FOM:
    
    # set up mpc object
    fom_mpc = mpc(prediction_model=None,
                  model=fom,
                  options=mpc_options)
    # solve
    U_fom, Y_fom, P_fom, out_fom, history_fom = fom_mpc.solve(U_0)

if POD:

    #### FOM ROM MPC with expensive-to-evaluate error est
    if 1:
        
        # create reduce object and rom (rom is directly over written in the adaptive methods)
        rom_pod, r_pod_unassembled = get_reductor_and_rom(kp, fom, dx, debug, options, 
                                              nonsmooth, energy_prod, 
                                              random_switching_law, 
                                              U_0, PODmethod, 
                                              errorest_assembled = False)
        
        # set up mpc object
        mpc_fompod = mpc(prediction_model=rom_pod,
                         model=fom,
                         options=mpc_options,
                         reductor=r_pod_unassembled)
                
        # solve
        U_fomrom_exp, Y_fomrom_exp, P_fomrom_exp, out_fomrom_exp, history_fomrom_exp = mpc_fompod.solveFOMROM(U_0)
    
    #### FOM ROM MPC with cheap-to-evaluate error est
    if 1:
        
        # create reduce object and rom (rom is directly over written in the adaptive methods)
        rom_podFOMROMcheap, r_podFOMROMcheap = get_reductor_and_rom(kp, fom, dx, debug, options, 
                                              nonsmooth, energy_prod, 
                                              random_switching_law, 
                                              U_0, PODmethod, 
                                              errorest_assembled = True)        
        
        # setup mpc object
        mpc_fomrom_cheap = mpc(prediction_model=rom_podFOMROMcheap,
                               model=fom,
                               options=mpc_optionsFOMROMcheap,
                               reductor=r_podFOMROMcheap)
        # solve
        U_fomrom_cheap, Y_fomrom_cheap, P_fomrom_cheap, output_fomrom_cheap, history_fomrom_cheap = mpc_fomrom_cheap.solveFOMROM(U_0)
        
    #### ROM ROM MPC with cheap-to-evaluate error est
    if 1:
        
        # create reduce object and rom (rom is directly over written in the adaptive methods)
        rom_podROMROMcheap, r_podROMROMcheap = get_reductor_and_rom(kp, fom, dx, debug, options, 
                                             nonsmooth, energy_prod, 
                                             random_switching_law, 
                                             U_0, PODmethod, 
                                             errorest_assembled = True)
                
        # setup mpc object
        mpc_romrom_cheap = mpc(prediction_model=rom_podROMROMcheap,
                               model=fom,
                               options=mpc_optionsROMROMcheap,
                               reductor=r_podROMROMcheap)
        # solve
        U_romrom_cheap, Y_romrom_cheap, P_romrom_cheap, output_romrom_cheap, history_romrom_cheap = mpc_romrom_cheap.solveFOMROM(U_0)   
             
#%% print results
        
print('-----------------------------RESULTS---------------------------------------------------------------')

#### print FOM MPC and true values
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
    
if POD:
    
    #### print FOMROM expensive
    try:
        print(
            f'POD FOMROM expensive: time {history_fomrom_exp["time"]}, final basis size {history_fomrom_exp["final_basissize"]}, average basisize over iterations {statistics.mean(history_fomrom_exp["basis_size"])}, speed-up {history_fom["time"]/history_fomrom_exp["time"]} ------------------')
        print(
            f'enriched in {history_fomrom_exp["enriched_in_iter"]} and {len(history_fomrom_exp["enriched_in_iter"])}  times')
        print(f'rel FOM  L2(H) state error {fom.rel_error_norm(Y_fom,Y_fomrom_exp, "L2")}, rel FOM control error {fom.space_time_norm(U_fomrom_exp-U_fom , space_norm = "control")/fom.space_time_norm(U_fom , space_norm = "control"):.4g}')
        print(
            f'rel output error {fom.rel_error_norm(out_fom, out_fomrom_exp, "output")}')
        print(f'rel J value error: {abs((history_fomrom_exp["J"]-history_fom["J"])/history_fom["J"])}')
        print(f'value J: {history_fomrom_exp["J"]}')
        print('----------')
    except:
        pass
    
    #### print FOMROM cheap
    try:
        print(
            f'POD FOMROM cheap: time {history_fomrom_cheap["time"]}, final basis size {history_fomrom_cheap["final_basissize"]}, average basisize over iterations {statistics.mean(history_fomrom_cheap["basis_size"])},  speed-up {history_fom["time"]/history_fomrom_cheap["time"]} ------------------')
        print(
            f'enriched in {history_fomrom_cheap["enriched_in_iter"]} and {len(history_fomrom_cheap["enriched_in_iter"])}  times')
        print(f'rel FOM  L2(H) state error {fom.rel_error_norm(Y_fom,Y_fomrom_cheap, "L2")}, rel FOM control error {fom.space_time_norm(U_fomrom_cheap-U_fom , space_norm = "control")/fom.space_time_norm(U_fom , space_norm = "control")}')
        print(
            f'rel output error {fom.rel_error_norm(out_fom, output_fomrom_cheap, "output")}')
        print(f'rel J value error: {abs((history_fomrom_cheap["J"]-history_fom["J"])/history_fom["J"])}')
        print(f'value J: {history_fomrom_cheap["J"]}')
        print('----------')
    except:
        pass
    
    #### print ROMROM cheap
    try:
        print(
            f'POD ROMROM cheap: time {history_romrom_cheap["time"]}, final basis size {history_romrom_cheap["final_basissize"]} average basisize over iterations {statistics.mean(history_romrom_cheap["basis_size"])},  speed-up {history_fom["time"]/history_romrom_cheap["time"]}------------------')
        print(f'enriched in {history_romrom_cheap["enriched_in_iter"]} and {len(history_romrom_cheap["enriched_in_iter"])}  times')
        print(f'rel FOM  L2(H) state error {fom.rel_error_norm(Y_fom, Y_romrom_cheap, "L2")}, rel FOM control error {fom.space_time_norm(U_romrom_cheap-U_fom , space_norm = "control")/fom.space_time_norm(U_fom , space_norm = "control")}')
        print(
            f'rel output error {fom.rel_error_norm(out_fom, output_romrom_cheap, "output")}')
        print(f'rel J value error: {abs((history_romrom_cheap["J"]-history_fom["J"])/history_fom["J"])}')
        print(f'value J: {history_romrom_cheap["J"]}')
        print('----------')
    except:
        pass

        
#%% plot results
            
if PLOT:

    plt.rcParams.update({'font.size': 18})
    plt.rcParams['lines.linewidth'] = 3
    repeat_mark = 100
    format_ = 'png'
    markersize_ = 8
    
    #### yd tracking
    if 1:
        plt.figure()
        plt.title('Tracking Room 1')
        try:
            plt.semilogy(fom.time_disc.t_v, Yd[0, :], label=r'$y_{d_1}$')
        except:
            pass
        try:
            plt.semilogy(fom.time_disc.t_v,
                         output_romrom_cheap[0, :], 'o', label=r'ROMROM cheap')
        except:
            pass
        try:
            plt.semilogy(fom.time_disc.t_v,
                         history_TRUE['out_opt'][0, :], label='True', alpha=0.8)
        except:
            pass
        try:
            plt.semilogy(fom.time_disc.t_v, out_fom[0, :], '--',  marker='o',
                         markevery=repeat_mark, markersize=markersize_, label='FOM', alpha=1)
        except:
            pass
        try:
            plt.semilogy(fom.time_disc.t_v,
                         out_fomrom_exp[0, :], 'r--', label='FOMROM expensive', alpha=0.8)
        except:
            pass
        try:
            plt.semilogy(fom.time_disc.t_v,
                         output_fomrom_cheap[0, :], 'b--', label='FOMROM cheap', alpha=0.8)
        except:
            pass
        plt.xlabel(r'$t$')
        # plt.ylim([2e0, 3.2e0])
        plt.legend(loc='best', ncol=1)
        # fom.plot_to_csv(ax = plt.gca(), name =  data_folder +'yd1.csv')
        plt.savefig(plot_folder+'yd1.'+format_, format=format_,
                    dpi=1200, bbox_inches='tight')
    
        plt.figure()
        plt.title('Tracking Room 2')
        try:
            plt.semilogy(fom.time_disc.t_v, Yd[1, :], label=r'$y_{d_2}$')
        except:
            pass
        try:
            plt.semilogy(fom.time_disc.t_v,
                         output_romrom_cheap[1, :], 'o', label=r'ROMROM cheap')
        except:
            pass
        try:
            plt.semilogy(fom.time_disc.t_v,
                         history_TRUE['out_opt'][1, :], label='True', alpha=0.8)
        except:
            pass
        try:
            plt.semilogy(fom.time_disc.t_v, out_fom[1, :], '--',  marker='o',
                         markevery=repeat_mark, markersize=markersize_, label='FOM', alpha=1)
        except:
            pass
        try:
            plt.semilogy(fom.time_disc.t_v,
                         out_fomrom_exp[1, :], 'r--', label='FOMROM expensive', alpha=0.8)
        except:
            pass
        try:
            plt.semilogy(fom.time_disc.t_v,
                         output_fomrom_cheap[1, :], 'b--', label='FOMROM cheap', alpha=0.8)
        except:
            pass
        plt.xlabel(r'$t$')
        # plt.ylim([1.5e-1, 5e-1])
        plt.legend(loc='best', ncol=1)
        plt.savefig(plot_folder+'yd2.'+format_, format=format_,
                    dpi=1200, bbox_inches='tight')
    
    #### J trajectories
    plt.figure()
    plt.title(label='Cost trajectories')
    try:
        plt.semilogy(fom.time_disc.t_v,
                     history_romrom_cheap['Jtrack_traj'], 'o', label=r'ROMROM cheap')
    except:
        pass
    try:
        pass
        # plt.semilogy(fom.time_disc.t_v, J_uncontrolled_traj, label= 'uncontrolled')
    except:
        pass
    try:
        pass
        # plt.semilogy(fom.time_disc.t_v, J_ud_traj, label= r'$u_d$ controlled')
    except:
        pass
    try:
        plt.semilogy(fom.time_disc.t_v, J_true_track, label='True')
    except:
        pass
    try:
        plt.semilogy(fom.time_disc.t_v, history_fom['Jtrack_traj'], '--', marker='o',
                     markevery=repeat_mark, markersize=markersize_, label='FOM', alpha=1)
    except:
        pass
    try:
        plt.semilogy(fom.time_disc.t_v,
                     history_fomrom_exp['Jtrack_traj'], 'r--', markersize=markersize_, label='FOMROM expensive')
    except:
        pass
    try:
        plt.semilogy(fom.time_disc.t_v, history_fomrom_cheap['Jtrack_traj'],
                     markevery=repeat_mark, markersize=markersize_, label='FOMROM cheap', alpha=1)
    except:
        pass
    plt.xlabel(r'$t$')
    # plt.ylabel(r'$l$')
    plt.legend(loc='lower left')
    plt.savefig(plot_folder+'J.'+format_, format=format_, dpi=1200, bbox_inches='tight')
    
    # POD
    if POD:
        
        #### ROM error in control, state, output
        # CONTROL
        plt.figure()
        plt.title('Pointwise control error')
        rel_error_traj = [fom.rel_space_norm_trajectory(U_fom,U_fomrom_exp, norm='control'),
                          fom.rel_space_norm_trajectory(U_fom,U_fomrom_cheap, norm='control'),
                          fom.rel_space_norm_trajectory(U_fom,U_romrom_cheap, norm='control')
                          ]
        try:
            plt.semilogy(fom.time_disc.t_v, rel_error_traj[0], '-', markersize=2, label='FOMROM expensive')
        except:
            pass
        try:
            plt.semilogy(fom.time_disc.t_v, rel_error_traj[1], '-', markersize=2, label='FOMROM cheap')
        except:
            pass
        try:
            plt.semilogy(fom.time_disc.t_v, rel_error_traj[2], '-', markersize=2, label='ROMROM cheap')
        except:
            pass
        plt.xlabel(r'$t$')
        plt.legend()
        fom.plot_to_csv(ax = plt.gca(), name =  data_folder +'control_error.csv')
        plt.savefig(plot_folder+'control_error.'+format_,
                    format=format_, dpi=1200, bbox_inches='tight')
        
        # state error
        plt.figure()
        plt.title(r'Pointwise $L^2$ state error')
        rel_error_traj = [fom.rel_space_norm_trajectory(Y_fom,Y_fomrom_exp, norm='L2'),
                          fom.rel_space_norm_trajectory(Y_fom,Y_fomrom_cheap, norm='L2'),
                          fom.rel_space_norm_trajectory(Y_fom,Y_romrom_cheap, norm='L2')]
        try:
            plt.semilogy(fom.time_disc.t_v, fom.rel_space_norm_trajectory(
                Y_fom,Y_fomrom_exp, norm='L2'), '-', markersize=2, label='FOMROM expensive')
        except:
            pass
        try:
            plt.semilogy(fom.time_disc.t_v, fom.rel_space_norm_trajectory(
                Y_fom,Y_fomrom_cheap, norm='L2'), '-', markersize=2, label='FOMROM cheap')
        except:
            pass
        try:
            plt.semilogy(fom.time_disc.t_v, fom.rel_space_norm_trajectory(
                Y_fom,Y_romrom_cheap, norm='L2'), '-', markersize=2, label='ROMROM cheap')
        except:
            pass
        plt.xlabel(r'$t$')
        plt.legend()
        fom.plot_to_csv(ax = plt.gca(), name =  data_folder +'state_error.csv')
        plt.savefig(plot_folder+'state_error.'+format_,
                    format=format_, dpi=1200, bbox_inches='tight')
      
        # l error
        plt.figure()
        plt.title(r'Pointwise $\ell$ error')
        rel_error_traj = [fom.rel_scalar_trajectory(history_fom['Jtrack_traj'],history_fomrom_exp['Jtrack_traj']),
                          fom.rel_scalar_trajectory(history_fom['Jtrack_traj'],history_fomrom_cheap['Jtrack_traj']),
                          fom.rel_scalar_trajectory(history_fom['Jtrack_traj'],history_romrom_cheap['Jtrack_traj'])
                          ]
        try:
            plt.semilogy(fom.time_disc.t_v, fom.rel_scalar_trajectory(
                history_fom['Jtrack_traj'],history_fomrom_exp['Jtrack_traj']), '-', markersize=2, label='FOMROM expensive')
        except:
            pass
        try:
            plt.semilogy(fom.time_disc.t_v, fom.rel_scalar_trajectory(
                history_fom['Jtrack_traj'],history_fomrom_cheap['Jtrack_traj']), '-', markersize=2, label='FOMROM cheap')
        except:
            pass
        try:
            plt.semilogy(fom.time_disc.t_v, fom.rel_scalar_trajectory(
                history_fom['Jtrack_traj'],history_romrom_cheap['Jtrack_traj']), '-', markersize=2, label='ROMROM cheap')
        except:
            pass
        plt.xlabel(r'$t$')
        plt.legend()
        fom.plot_to_csv(ax = plt.gca(), name =  data_folder +'l_error.csv')
        plt.savefig(plot_folder+'l_error.'+format_,
                    format=format_, dpi=1200, bbox_inches='tight')

        # output
        plt.figure()
        plt.title('Pointwise output error')
        rel_error_traj = [fom.rel_space_norm_trajectory(out_fom,out_fomrom_exp, norm='output'),  
                          fom.rel_space_norm_trajectory(out_fom,output_fomrom_cheap, norm='output'), 
                          fom.rel_space_norm_trajectory(out_fom,output_romrom_cheap, norm='output')]
        try:
            plt.semilogy(fom.time_disc.t_v, fom.rel_space_norm_trajectory(
                out_fom,out_fomrom_exp, norm='output'), '-', markersize=2, label='FOMROM expensive')
        except:
            pass
        try:
            plt.semilogy(fom.time_disc.t_v, fom.rel_space_norm_trajectory(
                out_fom,output_fomrom_cheap, norm='output'), '-', markersize=2, label='FOMROM cheap')
        except:
            pass
        try:
            plt.semilogy(fom.time_disc.t_v, fom.rel_space_norm_trajectory(
                out_fom,output_romrom_cheap, norm='output'), '-', markersize=2, label='ROMROM cheap')
        except:
            pass
        plt.xlabel(r'$t$')
        plt.legend()
        fom.plot_to_csv(ax = plt.gca(), name =  data_folder +'output_error.csv')
        plt.savefig(plot_folder+'output_error.'+format_,
                    format=format_, dpi=1200, bbox_inches='tight')
                
    #### error estimator plots in control, state and output
    strings_ = ['FOMROM expensive - ', 'FOMROM cheap - ', 'ROMROM cheap - ']
    hists = [history_fomrom_exp, history_fomrom_cheap, history_romrom_cheap ]
    
    for i in range(len(strings_)):

        # read
        hist_ = hists[i]
        string = strings_[i]
        
        # control est
        plt.figure()
        plt.title(label=string+'control estimator')
        plt.semilogy(
            hist_['control_error_est'], 'bo', label=r'$\Delta_u(0)$')
        plt.semilogy(
            hist_['MPC_control_apost'], 'g', label=r' $\Delta_u(\Delta_{t_n})$')
        plt.semilogy(
            hist_['tol'], 'r-', label=r'$\varepsilon$')
        # plt.axhline(y=self.tol, color='r', linestyle='-', label = 'tol')
        plt.legend(loc='best')
        plt.xlabel(r'MPC steps $n$')
        fom.plot_to_csv(ax = plt.gca(), name =  data_folder +string+'control_est.csv')
        plt.savefig(plot_folder+string+'control_est.'+format_,
                    format=format_, dpi=1200, bbox_inches='tight')
        
        # recursive state estimator
        plt.figure()
        plt.title(label=string+'state estimator')
        plt.semilogy(
            hist_['MPC_traj_apost'], 'bo', label=r'$\Delta_{t_n}$')
        plt.semilogy(
            hist_['MPC_traj_apost_tol'], 'r-', label=r'$\epsilon_n$')
        fom.plot_to_csv(ax = plt.gca(), name =  data_folder +string+'mpctraj_est.csv')
        plt.vlines(x=hist_['enriched_in_iter'], ymin=0, ymax=max(
            hist_['MPC_traj_apost_too_large']), colors='purple', ls='-', lw=1, label='ROM update')
        plt.legend(loc='lower right')
        plt.xlabel(r'MPC steps $n$')
        plt.savefig(plot_folder+string+'state_est.'+format_,
                    format=format_, dpi=1200, bbox_inches='tight')
        
        # plot output estimator
        plt.figure()
        plt.title(label=string+'output estimator')
        plt.semilogy(
            hist_['MPC_output_current'], 'bo', label=r'$\Delta_y(0)$')
        plt.semilogy(
            hist_['MPC_output_apost'], 'g', label=r' $\Delta_y(\Delta_{t_n})$')
        plt.semilogy(
            hist_['tol'], 'r-', label=r'$\varepsilon$')
       
        plt.legend(loc='best')
        plt.xlabel(r'$n$')
        fom.plot_to_csv(ax = plt.gca(), name =  data_folder +string+'out_est.csv')
        plt.savefig(plot_folder+string+'output_est.'+format_,
                    format=format_, dpi=1200, bbox_inches='tight')
                    
        # basissize over the course of mpc steps
        plt.figure()
        plt.title(string+'Basis size')
        plt.plot(hist_['basis_size'], 'b')
        plt.vlines(x=hist_['coarsed_in_iter'], ymin=0, ymax=max(hist_['basis_size']), colors='green', ls='-', lw=1, label='ROM coarse')
        plt.vlines(x=hist_['enriched_in_iter'], ymin=0, ymax=max(hist_['basis_size']), colors='purple', ls='-', lw=1, label='ROM update')
        plt.xlabel(r'$n$')
        plt.legend(loc='lower right')
    
        #### bar plot timings
        try:
            plt.figure()
            plt.title(label=string+'Timings')
            categories = ['Error estimation', 'FOM subproblem', 'ROM subproblem',
                          'ROM update', 'ROM coarsening', 'total main', 'total']
            values = [hist_['errorest_time'],
                      hist_['FOMsubproblem_time'],
                      hist_['ROMsubproblem_time'],
                      hist_['ROMupdate_time'],
                      hist_['coarse_time'],
                      hist_['mainroutines_time'],
                      hist_['time']
                      ]
            plt.bar(categories, values)
            plt.xticks(rotation=45, fontsize=12)
            # plt.xlabel('Routines')
            plt.ylabel(r'Time')
        except:
            pass
    