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
# Description: this files contains the MPC algorithms.
    
from methods import collection
import numpy as np
from scipy.sparse import diags
from time import perf_counter
import matplotlib.pyplot as plt
import copy

class mpc():
    
    def __init__(self, 
                 model, 
                 prediction_model = None, 
                 options = None, 
                 reductor = None, 
                 debug_model = None, 
                 checkstuff = None,
                 options_add = None):
        
        # set options
        assert options is not None
        self.options = options
        self.options_add = options_add
        
        # set models
        self.model = model
        self.reductor = reductor
        if prediction_model is not None:
            self.prediction_model = prediction_model
            self.same_prediction_and_feedback_model = False
            assert self.reductor is not None    
        else:
            self.prediction_model = model
            self.same_prediction_and_feedback_model = True
        
        # save global in time data
        self.global_model_time_disc, self.global_model_data = copy.deepcopy(self.model.get_mpc_parameters())
        self.global_prediction_model_time_disc, self.global_prediction_model_data =  copy.deepcopy(self.prediction_model.get_mpc_parameters())
        if self.model.isStateDep() and 0:
            self.plant_switching_law = copy.deepcopy(self.model.pde.sigma)
            self.target_switching_law = self.options.tracking_switched_profile
         
        # get time data
        self.build_mpc_time_data()
        
        # debug options
        if  debug_model is not None:
            self.debug_model = debug_model
            self.global_debug_model_time_disc, self.global_debug_model_data =  copy.deepcopy(self.debug_model.get_mpc_parameters())
        else:
            self.debug_model = None
        self.checkstuff = checkstuff  
    
    def update_adaptive_tol(self, lower_bound, scaling, norm_current, J_current, n):
        if self.options.adaptive_tolerance:
            if self.options.tol_update_type == 1:
                if n == 200 or n == 500 or 0:
                    self.options.control_tol_update = 0.2*self.options.control_tol_update
            elif self.options.tol_update_type == 2:
                self.options.control_tol_update = max(lower_bound, scaling * norm_current)
            elif self.options.tol_update_type == 3:
                self.options.control_tol_update = max( min(self.options.control_tol_update, J_current), lower_bound)
            else:
                pass 
        return self.options.control_tol_update
    
    def build_mpc_time_data(self):
        
        mpc_time_data = collection()
        mpc_time_data.T = self.global_model_time_disc.T
        mpc_time_data.t0 = self.global_model_time_disc.t0
        mpc_time_data.dt = self.global_model_time_disc.dt
        mpc_time_data.t_v_global = self.global_model_time_disc.t_v
        mpc_time_data.K = self.global_model_time_disc.K
        
        # insert kf, kp and get Tf, Tp
        mpc_time_data.kf = self.options.kf
        mpc_time_data.kp = self.options.kp
        mpc_time_data.Tp = mpc_time_data.kp*mpc_time_data.dt+mpc_time_data.t0
        mpc_time_data.Tf = mpc_time_data.kf*mpc_time_data.dt+mpc_time_data.t0
        
        # compute mpc steps and check compatibility
        assert mpc_time_data.t0 <= mpc_time_data.Tf <= mpc_time_data.Tp <= mpc_time_data.T, 'time horizons not compatible...'
        mpc_time_data.mpc_steps = round((mpc_time_data.T-mpc_time_data.t0)/mpc_time_data.Tf)
        assert round(mpc_time_data.mpc_steps) == mpc_time_data.mpc_steps, 'not conforming'
        mpc_time_data.mpc_grid = np.linspace(mpc_time_data.t0, mpc_time_data.T, mpc_time_data.mpc_steps+1)
        self.mpc_time_data = mpc_time_data
    
    def get_mpc_time_data(self):
        return self.mpc_time_data.kp, self.mpc_time_data.kf, self.mpc_time_data.Tp, self.mpc_time_data.Tf, self.mpc_time_data.mpc_steps, self.mpc_time_data.dt, self.mpc_time_data.t0, self.mpc_time_data.T, self.mpc_time_data.t_v_global, self.mpc_time_data.K, self.mpc_time_data.mpc_grid
        
            
    def update_predictionmodel_data(self, n, y0_pred):
        
        # read mpc_time_data
        kp, kf, Tp, Tf, mpc_steps, dt, t0, T, t_v, K, mpc_grid = self.get_mpc_time_data()
        
        # update time window
        local_time_disc = collection()
        local_time_disc.t0 = mpc_grid[n]
        local_time_disc.T = min(mpc_grid[n]+Tp,T)
        local_time_disc.K = round((local_time_disc.T-local_time_disc.t0)/dt)+1
        local_time_disc.dt = dt
        local_time_disc.t_v = t_v[n*kf:min(n*kf+kp+1,K)]
        local_time_disc.D = local_time_disc.dt * np.ones(local_time_disc.K)
        local_time_disc.D_diag = diags(local_time_disc.D)
        
        # use prediction model update routine n, mpc_time_data, local_time_disc, y0, global_data
        self.prediction_model.update_mpc_parameters_pred(n = n,
                                                         mpc_time_data = self.mpc_time_data,
                                                         local_time_disc = local_time_disc,
                                                         y0 = y0_pred,
                                                         global_data = self.global_prediction_model_data)
        
        return self.prediction_model
        
    def update_plant_data(self, n, y0_plant):
        
        # read mpc_time_data
        kp, kf, Tp, Tf, mpc_steps, dt, t0, T, t_v, K, mpc_grid = self.get_mpc_time_data()
        
        # update time window
        feed_time_disc = collection()
        feed_time_disc.t0 = mpc_grid[n]
        feed_time_disc.T = min(mpc_grid[n]+Tf,T)
        feed_time_disc.K = round((feed_time_disc.T-feed_time_disc.t0)/dt)+1
        feed_time_disc.dt = dt
        feed_time_disc.t_v = t_v[n*kf:min(n*kf+kf+1,K)]
        feed_time_disc.D = feed_time_disc.dt * np.ones(feed_time_disc.K)
        feed_time_disc.D_diag = diags(feed_time_disc.D)
        
        self.model.update_mpc_parameters_plant(n = n,
                                               mpc_time_data = self.mpc_time_data,
                                               local_time_disc = feed_time_disc,
                                               y0 = y0_plant,
                                               global_data = self.global_model_data)
    
    def update_plant_prediction(self, n, y0_plant):
        
        # read mpc_time_data
        kp, kf, Tp, Tf, mpc_steps, dt, t0, T, t_v, K, mpc_grid = self.get_mpc_time_data()
        
        # update time window
        local_time_disc = collection()
        local_time_disc.t0 = mpc_grid[n]
        local_time_disc.T = min(mpc_grid[n]+Tp,T)
        local_time_disc.K = round((local_time_disc.T-local_time_disc.t0)/dt)+1
        local_time_disc.dt = dt
        local_time_disc.t_v = t_v[n*kf:min(n*kf+kp+1,K)]
        local_time_disc.D = local_time_disc.dt * np.ones(local_time_disc.K)
        local_time_disc.D_diag = diags(local_time_disc.D)
        
        # use prediction model update routine n, mpc_time_data, local_time_disc, y0, global_data
        self.model.update_mpc_parameters_pred(n = n,
                                              mpc_time_data = self.mpc_time_data,
                                              local_time_disc = local_time_disc,
                                              y0 = y0_plant,
                                              global_data = self.global_model_data)
        
        return self.model
        
    def performance_bound(self, n, alpha_old = None, V_old = None, stage_cost_old = None, u_opt = None, y_opt= None, out_opt= None, y_opt_FOM = None, out_opt_FOM = None, u_opt_FOM = None):
        if n == 0:
            self.performance_index = []
            alpha = 1
            V_new = self.prediction_model.J(u_opt, y_opt, out_opt)
            stage_cost_new = self.prediction_model.stage_cost(y_opt, out_opt, u_opt, m1 = 0, m2 = self.options.kf)
            return alpha, alpha, V_new, stage_cost_new
        assert self.prediction_model.cost_data.weights[2] == 0, '... modify this in terms of terminal cost'
        V_new = self.prediction_model.J(u_opt, y_opt, out_opt)
        alpha_local =  min(1,(V_old - V_new)/stage_cost_old)
        alpha_global = min(alpha_old, alpha_local)
        self.performance_index.append(alpha_local)
        stage_cost_new =  self.prediction_model.stage_cost(y_opt, out_opt, u_opt, m1 = 0, m2 = self.options.kf)
        return alpha_global, alpha_local, V_new, stage_cost_new

    def performance_bound_R(self, n, alpha_old = None, V_old = None, stage_cost_old = None, u_opt = None, y_opt= None, out_opt= None, y_opt_FOM = None, out_opt_FOM = None, u_opt_FOM = None):
         if n == 0:
             self.performance_index = []
             alpha = 1
             V_new = self.prediction_model.J(u_opt, y_opt, out_opt)
             return alpha, alpha, V_new
         assert self.prediction_model.cost_data.weights[2] == 0, '... modify this in terms of terminal cost'
         V_new = self.prediction_model.J(u_opt, y_opt, out_opt)
         alpha_local = min(1,(V_old - V_new)/stage_cost_old)
         alpha_global = min(alpha_old, alpha_local)
         return alpha_global, alpha_local, V_new
     
    def performance_bound_Rfom(self, n, alpha_old = None, V_old = None, stage_cost_old = None, u_opt = None, y_opt= None, out_opt= None, y_opt_FOM = None, out_opt_FOM = None, u_opt_FOM = None):
         if n == 0:
             self.performance_index = []
             alpha = 1
             V_new = self.model.J(u_opt, y_opt, out_opt)
             return alpha, alpha, V_new 
         assert self.model.cost_data.weights[2] == 0, '... modify this in terms of terminal cost'
         V_new = self.model.J(u_opt, y_opt, out_opt)
         alpha_local =  min(1,(V_old - V_new)/stage_cost_old)
         alpha_global = min(alpha_old, alpha_local)
         return alpha_global, alpha_local, V_new

#%% mpc

    def solve(self, U_0):

        #### init
        start_time = perf_counter()
        kp, kf, Tp, Tf, mpc_steps, dt, t0, T, t_v, K, _ = self.get_mpc_time_data()
        self.U_0 = U_0.copy()
        y0_pred = self.prediction_model.pde.y0
        y0_plant = self.model.pde.y0
        U_optimal = U_0[:,:kp+1]
        U_feedback = []
        Y_feedback = []
        P_feedback = []
        history = {'fom_solves': 0,
                   'ROMupdate_time': 0, 
                   'FOMsubproblem_time': 0, 
                   'coarse_time': 0,
                   'ROMsubproblem_time':0,
                   'errorest_time':0,
                   "enriched_in_iter":[],
                   "final_basissize": 0
                   }
        control_list = []
        
        # loop
        print(f'------------------------------------ MPC with prediction model: {self.prediction_model.model_type}, plant: {self.model.model_type} ---------------------------------------------')
        print(f' with options: Tf: {Tf}, Tp: {Tp}, mpc steps: {mpc_steps}, dt = {dt}, t0 = {t0}, T = {T}, kf = {kf}, kp = {kp}')
        print(f'------------------------------------ MPC with prediction model: {self.prediction_model.model_type}, plant: {self.model.model_type}  ---------------------------------------------')
        for n in range(mpc_steps):
            
            #### 1. solve subproblem
            print(f'Step {n+1}/{mpc_steps}: prediction horizon: [{t_v[n*kf]}, {t_v[min(n*kf+kp,K-1)]}], feedback horizon: [{t_v[n*kf]}, {t_v[(n+1)*kf]}]')
            self.update_predictionmodel_data(n, y0_pred)
            
            #### 2. solve mpc subproblem
            print('       Solve OCP ...')
            start_time_FOMsub = perf_counter()
            if 1:
                # warm start
                U_0 = U_optimal[:,:self.prediction_model.time_disc.K]
            else:
                U_0 = self.U_0.copy()
                U_0 = U_0[:,:self.prediction_model.time_disc.K]
            U_optimal, history_sub_solver = self.prediction_model.solve_ocp(U_0,
                                              options = self.options.innersolver_options,
                                              checkstuff = self.checkstuff)
            control_list.append(U_optimal)
            assert not np.isnan(U_optimal).any(), 'nan val found...'
            history['FOMsubproblem_time'] +=  perf_counter()-start_time_FOMsub
            if n == 0:
                history['first_sol'] = U_optimal
                
            #### 3. get feedback by applying control to feedback model 
            print('       Get feedback ...')
            if self.same_prediction_and_feedback_model and 1:
                
                # get feedback and update initial value
                if n == 0:
                    Y_feedback.append(history_sub_solver['Y_opt'][:,:kf+1])
                    U_feedback.append(U_optimal[:,:kf+1])
                    # P_feedback.append(history_BB['P_opt'][:,:kf+1])
                    y0_pred = history_sub_solver['Y_opt'][:,kf]
                else:
                    Y_feedback.append(history_sub_solver['Y_opt'][:,1:kf+1])
                    # P_feedback.append(history_BB['P_opt'][:,1:kf+1])
                    U_feedback.append(U_optimal[:,1:kf+1])
                    y0_pred = history_sub_solver['Y_opt'][:,kf]
                
                if self.model.isStateDep():
                    # extract also init switching for the next mpc iteration
                    self.model.init_switch = history_sub_solver['switch_profil_opt'][kf]
                    self.prediction_model.init_switch =history_sub_solver['switch_profil_opt'][kf]
                
            else: # then use the specialized feedback model
                self.update_plant_data(n, y0_plant)
                
                # solve state
                Y_opt, _,_, switch_feedback = self.model.solve_state(U_optimal[:,:kf+1])
                # self.update_plant_prediction(n, y0_plant)
                
                # get feedback from feedback model
                if n == 0:
                    Y_feedback.append(Y_opt[:,:kf+1])
                    U_feedback.append(U_optimal[:,:kf+1])
                    # P_feedback.append(history_BB['P_opt'][:,:kf+1])
                    y0_plant = Y_opt[:,kf]
                else:
                    Y_feedback.append(Y_opt[:,1:kf+1])
                    # P_feedback.append(history_BB['P_opt'][:,1:kf+1])
                    U_feedback.append(U_optimal[:,1:kf+1])
                    y0_plant = Y_opt[:,kf] 
                y0_pred = self.reductor.FOMtoROM(y0_plant)
            
                if self.model.isStateDep():
                    
                    # extract also initial switch for the next mpc iteration
                    self.model.init_switch = switch_feedback[kf]
                    self.prediction_model.init_switch = switch_feedback[kf]
                           
            #### 4. eval performance bound
            if self.options.perf_bound:
                if n == 0:
                    alpha_old, alpha_local, V_old, stage_cost_old = self.performance_bound(n, u_opt = U_optimal, y_opt= history_sub_solver['Y_opt'], out_opt= history_sub_solver['out_opt'])
                else:
                    alpha_old, alpha_local, V_old, stage_cost_old = self.performance_bound(n, alpha_old, V_old, stage_cost_old, u_opt = U_optimal, y_opt = history_sub_solver['Y_opt'], out_opt= history_sub_solver['out_opt'])
                print(f'Local performance index is alpha = {alpha_local} and global {alpha_old}.')
            
        #### Finalize
        print(f'MPC FINISHED with prediction model: {self.prediction_model.model_type}, plant: {self.model.model_type} -----------------------------------------------------------------')
        print('------------------------------------------------------------------------------')
        elapsed_time = perf_counter() - start_time
        history['time'] = elapsed_time
        history['final_basissize'] = self.prediction_model.pde.state_dim
        print(f'Elapsed time {elapsed_time}.')
        history['control_list'] = control_list
        
        if self.options.perf_bound:
            print(f'Performance index is {alpha_old}.')
            history['perf_index'] = self.performance_index
            if self.options.plot_ or 1:
                plt.semilogy(self.performance_index)
                plt.title('Performance index over course of mpc steps')
        
        # get feedback controls as array
        U_feedback = np.concatenate(U_feedback, axis = 1)
        Y_feedback = np.concatenate(Y_feedback, axis = 1)
        # P_feedback_NOTUSED = np.concatenate(P_feedback, axis = 1)
        
        # check first order optimality condition and reupdate the models with the global data
        self.model.update_mpc_parameters_global(time_disc = self.global_model_time_disc, global_data = self.global_model_data)                                   
        self.prediction_model.update_mpc_parameters_global(time_disc = self.global_prediction_model_time_disc, global_data = self.global_prediction_model_data)   
        if self.model.isStateDep() and 0:
            self.prediction_model.pde.sigma = self.plant_switching_law
            self.model.pde.sigma = self.plant_switching_law
        grad, output_feedback, Y, P_feedback = self.model.gradJ_OBD(U_feedback)
        J_val = self.model.J(U_feedback, Y = Y, output = output_feedback)
        
        # save data
        history['J'] = J_val
        history['out'] = output_feedback
        history['Y'] = Y
        history['P'] = P_feedback
        history['U'] = U_feedback
        print(f'J = {J_val}')
        print(f'Gradient norm = {self.model.space_time_norm(grad, "control")}')
        print(f'Y-Yf = {self.model.space_time_norm(Y_feedback-Y, "L2")}')
       
        # get trajectory and plot distance to target and distance to control
        Jtrack_traj, Jcontrol_traj, J_traj = self.model.Jtracking_trajectory(u = U_feedback, Y = Y_feedback, output = output_feedback)
        if self.options.plot_:
            self.model.visualize_1d_many([Jtrack_traj, Jcontrol_traj, J_traj], strings = ['tracking term', 'control term', 'J'], title = f' {self.prediction_model.model_type}, plant: {self.model.model_type}', semi = True, time = None)
        history['Jtrack_traj'] = Jtrack_traj
        
        print('------------------------------------------------------------------------------')
        print('------------------------------------------------------------------------------')
        return U_feedback, Y_feedback, P_feedback, output_feedback, history
    
#%% adaptive mpc
    
    def solveFOMROM(self, U_0):
        
        #### init
        start_time = perf_counter()
        if 1:
            steps = self.mpc_time_data.mpc_steps
            self.options.MPC_trajectory_tol = np.linspace(self.options.State_update_tol_start_end[0], self.options.State_update_tol_start_end[1], steps)
            self.options.control_tol_update = np.linspace(self.options.Control_update_tol_start_end[0], self.options.Control_update_tol_start_end[1], steps)
       
        else:
            self.options.MPC_trajectory_tol = np.linspace(self.options.update_tol, self.options.update_tol, steps)
            self.options.control_tol_update = np.linspace(self.options.control_tol_update, self.options.control_tol_update, steps)
        l = self.options.l_init
        self.U_0 = U_0.copy()
        error_est_correction_constant_for_cheap = 1
        kp, kf, Tp, Tf, mpc_steps, dt, t0, T, t_v, K, _ = self.get_mpc_time_data()
        y0_pred = self.prediction_model.pde.y0
        y0_plant = self.model.pde.y0
        U_optimal = U_0[:,:kp+1]
        U_feedback = []
        Y_feedback = []
        P_feedback = []
        YROMlist = []
        PROMlist = []
        init_bound = 0
        history = {'fom_solves': 0, 
                   'control_error_est': [], 
                   'enriched_in_iter': [], 
                   'tol': [], 
                   'error_est_too_large':[1*self.options.control_tol_update+1e-12], 
                   'basis_size':[], 
                   'ROMupdate_time': 0, 
                   'coarsed_in_iter': [], 
                   'FOMsubproblem_time': 0, 
                   'coarse_time': 0,
                   'ROMsubproblem_time':0,
                   'errorest_time':0,
                   'coarse_proj_time': 0,
                   'coarse_select_time': 0,
                   'Cheap_error_est_too_large': [],
                   'Cheap_control_error_est': [],
                   'true_error': [],
                    'MPC_traj_apost': [],
                    'MPC_traj_apost_too_large': [1*self.options.MPC_trajectory_tol[0]+1e-12],
                    'MPC_traj_apost_tol':[],
                    'MPC_traj_enriched_in_iter': [],
                    'MPC_output_apost': [],
                    'MPC_control_apost': [],
                    'MPC_output_current': [],
                    'update_flags': []
                   }
        n = 0
        use_FOM_predictor = True
        test_est = False
        
        if self.options_add:
            if self.options_add['test_estimate']:
                print('TEST ESTIMATE MODE ON...')
                use_FOM_predictor = False
                control_list = []
                self.options.error_est_type == 'compare'
            if self.options_add['FOMROM']:
                self.options.type = 'FOMROM'
                init_bound = 0
            else:
                self.options.type = 'ROMROM'
                init_switch = self.model.pde.sigma(0, None, None)
                init_bound = self.model.space_norm(y0_plant - self.reductor.ROMtoFOM(y0_pred), space_norm = 'M_switch', switch = init_switch)
                
            accept_count = 0
            test_est = True
        if self.options.error_est_type == 'compare':
            compare_ests = collection()
            compare_ests.controlA = []
            compare_ests.control_initA = []
            compare_ests.controlAcheap = []
            compare_ests.control_initAcheap = []
            
            compare_ests.controlB = []
            compare_ests.control_initB = []
            compare_ests.controlBcheap = []
            compare_ests.control_initBcheap = []
            
            compare_ests.MPC_initA = []
            compare_ests.MPC_initB = []
            compare_ests.MPC_initAcheap = []
            compare_ests.MPC_initBcheap = []

        else:
            compare_ests = None 
        
        # loop
        print(f'------------------------------------ MPC with prediction model: {self.prediction_model.model_type}, plant: {self.model.model_type} ---------------------------------------------')
        print(f' with options: Tf: {Tf}, Tp: {Tp}, mpc steps: {mpc_steps}, dt = {dt}, t0 = {t0}, T = {T}, kf = {kf}, kp = {kp}')
        print(f'------------------------------------ MPC with prediction model: {self.prediction_model.model_type}, plant: {self.model.model_type}  ---------------------------------------------')
        while n < mpc_steps:
            
            print(f'Step {n+1}/{mpc_steps}: prediction horizon: [{t_v[n*kf]}, {t_v[min(n*kf+kp,K-1)]}], feedback horizon: [{t_v[n*kf]}, {t_v[(n+1)*kf]}]')
            #### A FOM predictor
            if use_FOM_predictor:
                
               #### A1 solve FOM MPC subproblem
               start_time_FOMsub = perf_counter()
               self.update_plant_prediction(n, y0_plant)
               print('       Solve FOM OCP ...')
               if 1:
                   U_0 = U_optimal[:,:self.model.time_disc.K]
               else:
                   U_0 = self.U_0.copy()
                   U_0 = U_0[:,:self.model.time_disc.K]
               U_optimal, history_sub_solver = self.model.solve_ocp(U_0,
                                                 options = self.options.fom_predictor_innersolver_options,
                                                 checkstuff = self.checkstuff)
               control_list.append(U_optimal)
               assert not np.isnan(U_optimal).any(), 'nan val found...'
               history['FOMsubproblem_time'] += perf_counter()-start_time_FOMsub
               
               if 1:
                   norm_current = self.model.space_norm( history_sub_solver['out_opt'][:,kf]-self.model.cost_data.Yd[:,kf], 'output')
            
               #### A2 Error est correction update (if needed)
               self.update_predictionmodel_data(n, y0_pred) 
               error_est_correction_constant_for_cheap = self.get_error_est_correction(n, U_optimal, history_sub_solver)
                
               #### A3 FOM feedback
               if n == 0:
                   Y_feedback.append(history_sub_solver['Y_opt'][:,:kf+1])
                   U_feedback.append(U_optimal[:,:kf+1])
                   # P_feedback.append(history_BB['P_opt'][:,:kf+1])
                   y0_plant = history_sub_solver['Y_opt'][:,kf]
                       
               else:
                   Y_feedback.append(history_sub_solver['Y_opt'][:,1:kf+1])
                   # P_feedback.append(history_BB['P_opt'][:,1:kf+1])
                   U_feedback.append(U_optimal[:,1:kf+1])
                   y0_plant = history_sub_solver['Y_opt'][:,kf]
             
               #### A4 ROM update
               start_time_enrich = perf_counter()
               self.prediction_model = self.update_rom(n, history_sub_solver, dt, l)
               # get global in time predicition model back
               self.global_prediction_model_time_disc, self.global_prediction_model_data =  self.prediction_model.get_mpc_parameters()
               history['ROMupdate_time'] += perf_counter()-start_time_enrich
               
               #### A5 update history, ...
               # update tol and bound in initial guess
               self.options.control_tol_update = self.update_adaptive_tol(self.options.err_lowernbound, self.options.error_scale_cons, norm_current, J_current = None, n = n)     

                # restart
               if self.options.restart:
                   # update init bound
                   print('Restart recursive error estimation ....')
                   init_bound = 0 
                   
               history['MPC_traj_apost'].append(init_bound)
               history['tol'].append(self.options.control_tol_update[n])
               history['MPC_traj_apost_tol'].append(self.options.MPC_trajectory_tol[n])
               if n == 0 or 1:
                   history['MPC_output_current'].append(0)
                   history['MPC_output_apost'].append(0)
                   history['MPC_control_apost'].append(0)
                   history['control_error_est'].append(0)
               else:
                    history['MPC_output_current'].append(est_output_init_collection.est_output_current)
                    history['MPC_output_apost'].append(est_output_init_collection.est_with_init_output)
                    history['MPC_control_apost'].append(est_output_init_collection.est_with_init)
                    history['control_error_est'].append(est_optimal_control)
               
               self.update_plant_prediction(n, y0_plant)
               self.update_predictionmodel_data(n, y0_pred)
              
               # get initial value for ROM
               y0_pred = self.reductor.FOMtoROM(y0_plant)
               
               # extract init switching for the next mpc iteration
               if self.model.isStateDep():
                    self.model.init_switch = history_sub_solver['switch_profil_opt'][kf]
                    self.prediction_model.init_switch =history_sub_solver['switch_profil_opt'][kf]
               
               # track the basis coefficients
               if self.options.track_basis_coeff:
                   if n == 0:
                        YROMlist = []
                        PROMlist = []
                       # get projection of current state into current basis
                        y_ROM = [self.reductor.FOMtoROM(history_sub_solver['Y_opt'][:,:kf+1])]
                        p_ROM = [self.reductor.FOMtoROM(history_sub_solver['P_opt'][:,:kf+1])]
                   else:
                        try:
                            YROMlist.append(np.concatenate(y_ROM, axis = 1))
                            PROMlist.append(np.concatenate(p_ROM, axis = 1))
                        except:
                            pass           
                       # get projection of current state into current basis
                        y_ROM = [self.reductor.FOMtoROM(history_sub_solver['Y_opt'][:,1:kf+1])]
                        p_ROM = [self.reductor.FOMtoROM(history_sub_solver['P_opt'][:,1:kf+1])]
               
               # eval performance bound
               if self.options.perf_bound:
                   if n == 0:
                       alpha_old, alpha_local, V_old, stage_cost_old = self.performance_bound(n, u_opt = U_optimal, y_opt= history_sub_solver['Y_opt'], out_opt= history_sub_solver['out_opt'])
                   else:
                       alpha_old, alpha_local, V_old, stage_cost_old = self.performance_bound(n, alpha_old, V_old, stage_cost_old, u_opt = U_optimal, y_opt = history_sub_solver['Y_opt'], out_opt= history_sub_solver['out_opt'])
                   print(f'Local performance index is alpha = {alpha_local} and global { alpha_old}.')
               
               # update flags and history and counter
               if self.options.measure_true_error:
                   history['true_error'].append(0)
               use_FOM_predictor = False
               history['enriched_in_iter'].append(n)
               history['basis_size'].append(self.prediction_model.pde.state_dim)
               n += 1
               accept_count = 0
            
            #### B ROM predictor
            else:
                
                #### B1 solve ROM MPC subproblem
                print('       Solve ROM OCP ...')
                start_time_ROMsub =  perf_counter()
                self.update_predictionmodel_data(n, y0_pred)
                if 1:
                    U_0 = U_optimal[:,:self.prediction_model.time_disc.K]
                else:
                    U_0 = self.U_0.copy()
                    U_0 = U_0[:,:self.prediction_model.time_disc.K]
                U_optimal, history_sub_solver = self.prediction_model.solve_ocp(U_0,
                                                  options = self.options.innersolver_options,
                                                  checkstuff = self.checkstuff)
                control_list.append(U_optimal)
                history['ROMsubproblem_time'] += perf_counter()-start_time_ROMsub
                
                #### B2 error estimation
                self.update_plant_prediction(n, y0_plant)
                start_time_error_est =  perf_counter()   
                new_init_bound, est_optimal_control, est_output_init_collection, Y_FOMest = self.error_estimation(U_optimal, history_sub_solver, init_bound, error_est_correction_constant_for_cheap, compare_ests, history, kf, U_0)
                print(f'Control est = {est_optimal_control} and tol {self.options.control_tol_update[n]}, Delta_tn+1 {new_init_bound} ands tol {self.options.MPC_trajectory_tol[n]}, old_init_bound {init_bound} and control est with init bound {est_output_init_collection.est_with_init}')
                print(f'current output est {est_output_init_collection.est_output_current}, output est with init {est_output_init_collection.est_with_init_output}')
                history['errorest_time'] += perf_counter()-start_time_error_est
        
                # check if error criteria is fulfilled
                if (est_optimal_control > self.options.control_tol_update[n] or new_init_bound > self.options.MPC_trajectory_tol[n]) and not test_est:
                # if (new_init_bound > self.options.MPC_trajectory_tol[n]) and not test_est:
                    
                   #### B3 reject ROM control
                   if est_optimal_control > self.options.control_tol_update[n]:
                       enrich_flag = 'current_control'
                       history['error_est_too_large'].append(est_optimal_control)
                   if new_init_bound > self.options.MPC_trajectory_tol[n]:
                       enrich_flag = 'MPC_trajectory_bound'
                       history['MPC_traj_apost_too_large'].append(new_init_bound)
                   if est_optimal_control > self.options.control_tol_update[n] and new_init_bound > self.options.MPC_trajectory_tol[n]:
                       enrich_flag = 'both_cases'
                       history['error_est_too_large'].append(est_optimal_control)
                       history['MPC_traj_apost_too_large'].append(new_init_bound)
                   use_FOM_predictor = True
                   print('       Control not accepted, because of' + enrich_flag+ ', update Model ...')
                   history['update_flags'].append(enrich_flag)
                   
                else: 
                    
                  #### B4 accept ROM control, get feedback
                  print('       Get feedback ...')
                  history['tol'].append(self.options.control_tol_update[n])
                  history['MPC_traj_apost'].append(new_init_bound)
                  history['MPC_traj_apost_tol'].append(self.options.MPC_trajectory_tol[n])
                  history['MPC_output_current'].append(est_output_init_collection.est_output_current)
                  history['MPC_output_apost'].append(est_output_init_collection.est_with_init_output)
                  history['MPC_control_apost'].append(est_output_init_collection.est_with_init)
                  history['control_error_est'].append(est_optimal_control)
                  accept_count += 1
                  
                  # update init_bound
                  init_bound = new_init_bound
                  
                  #### B5 coarsening
                  if self.options.coarse and n>1 and est_optimal_control/self.options.control_tol_update < self.options.coarse_threshold and accept_count >self.options.accept_count_threshold :
                      start_time_coarse = perf_counter()
                      
                      # check fouriercoefficients, we need ROM prediciton horizon
                      self.model.update_mpc_parameters_global(time_disc = self.global_model_time_disc, global_data = self.global_model_data)
                      quantities_to_measure_basis_energy = [history_sub_solver['Y_opt'], history_sub_solver['P_opt']]#[history_sub_solver['Y_opt'][history_sub_solver['Y_opt']] #]#
                      self.prediction_model, coarse_flag, time_proj, time_selection = self.reductor.coarse_rom(quantities_to_measure_basis_energy, 
                                                                       tolerance = self.options.coarse_tolerance, 
                                                                       model_to_project = self.model)
                      history['coarse_proj_time'] += time_proj
                      history['coarse_select_time'] += time_selection
                      if coarse_flag: 
                          history['coarsed_in_iter'].append(n)
                          self.global_prediction_model_time_disc, self.global_prediction_model_data =  self.prediction_model.get_mpc_parameters()
                          
                          # track the basis coefficients
                          if self.options.track_basis_coeff:
                                   # get projection of current state into current basis
                                   y_ROM.append(history_sub_solver['Y_opt'][:,1:kf+1])
                                   p_ROM.append(history_sub_solver['P_opt'][:,1:kf+1])
                                   
                                   YROMlist.append(np.concatenate(y_ROM, axis = 1))
                                   PROMlist.append(np.concatenate(p_ROM, axis = 1))
                                  
                                  # get projection of current state into current basis
                                   y_ROM = []
                                   p_ROM = []          
                      history['coarse_time'] += perf_counter()-start_time_coarse
                  else: 
                      coarse_flag = False
                      
                  #### B6 get feedback from feedback model           
                  if self.options.type == 'FOMROM': 
                      # FOMROM
                      if self.options.error_est_type == 'cheap':
                          self.update_plant_data(n, y0_plant)
                          Y_opt, _,_,_ = self.model.solve_state(U_optimal[:,:kf+1])
                      else:
                          Y_opt = Y_FOMest
                      
                      if 0:
                          norm_current = self.model.space_norm( history_sub_solver['out_opt'][:,kf]-self.model.cost_data.Yd[:,kf], 'output')
                          
                      if n == 0:
                          Y_feedback.append(Y_opt[:,:kf+1])
                          U_feedback.append(U_optimal[:,:kf+1])
                          # P_feedback.append(history_BB['P_opt'][:,:kf+1])
                          y0_plant = Y_opt[:,kf]
                          
                          if self.options.track_basis_coeff and not coarse_flag:
                              # get projection of current state into current basis
                              y_ROM.append(history_sub_solver['Y_opt'][:,:kf+1])
                              p_ROM.append(history_sub_solver['P_opt'][:,:kf+1])
                              
                      else:
                          Y_feedback.append(Y_opt[:,1:kf+1])
                          # P_feedback.append(history_BB['P_opt'][:,1:kf+1])
                          U_feedback.append(U_optimal[:,1:kf+1])
                          y0_plant = Y_opt[:,kf] 
                          
                          if self.options.track_basis_coeff and not coarse_flag:
                              # get projection of current state into current basis
                              y_ROM.append(history_sub_solver['Y_opt'][:,1:kf+1])
                              p_ROM.append(history_sub_solver['P_opt'][:,1:kf+1])
                         
                      if self.model.isStateDep():
                           # extract also init switching for the next mpc iteration
                           self.model.init_switch = history_sub_solver['switch_profil_opt'][kf]
                           self.prediction_model.init_switch =history_sub_solver['switch_profil_opt'][kf]
                      y0_pred = self.reductor.FOMtoROM(y0_plant) 
                      
                  elif self.options.type == 'ROMROM':
                        assert not self.options.error_est_type == 'expensive', 'using ROMROM MPC with expensive error est makes no sense :) ...'
                        
                        self.update_predictionmodel_data(n, y0_pred)  
                        # Y ROM get from inner solver
                        YROM_opt = history_sub_solver['Y_opt']
                        
                        # get new ROM initial value
                        y0_pred = YROM_opt[:,kf]
                        y0_plant = self.reductor.ROMtoFOM(y0_pred)
                        Y_opt = self.reductor.ROMtoFOM(YROM_opt)
                        
                        if n == 0:
                            Y_feedback.append(Y_opt[:,:kf+1])
                            U_feedback.append(U_optimal[:,:kf+1])
                            # P_feedback.append(history_BB['P_opt'][:,:kf+1])
                            
                            if self.options.track_basis_coeff and not coarse_flag:
                                # get projection of current state into current basis
                                y_ROM.append(history_sub_solver['Y_opt'][:,:kf+1])
                                p_ROM.append(history_sub_solver['P_opt'][:,:kf+1])
                                
                        else:
                            Y_feedback.append(Y_opt[:,1:kf+1])
                            # P_feedback.append(history_BB['P_opt'][:,1:kf+1])
                            U_feedback.append(U_optimal[:,1:kf+1])
                            
                            if self.options.track_basis_coeff and not coarse_flag:
                                # get projection of current state into current basis
                                y_ROM.append(history_sub_solver['Y_opt'][:,1:kf+1])
                                p_ROM.append(history_sub_solver['P_opt'][:,1:kf+1])
                           
                        if self.model.isStateDep():
                             # extract also init switching for the next mpc iteration
                             self.model.init_switch = history_sub_solver['switch_profil_opt'][kf]
                             self.prediction_model.init_switch =history_sub_solver['switch_profil_opt'][kf]                         
                        
                  # update tol
                  if not  self.options.error_est_type == 'compare':      
                      self.options.control_tol_update = self.update_adaptive_tol(self.options.err_lowernbound, self.options.error_scale_cons, norm_current, J_current = None, n = n)
                  
                  # eval performance bound
                  self.update_plant_prediction(n, y0_plant)
                  self.update_predictionmodel_data(n, y0_pred)
                  if self.options.perf_bound:
                      if n == 0:
                         alpha_old, alpha_local, V_old, stage_cost_old = self.performance_bound(n, u_opt = U_optimal, y_opt= history_sub_solver['Y_opt'], out_opt= history_sub_solver['out_opt'])
                      else:
                         alpha_old, alpha_local, V_old, stage_cost_old = self.performance_bound(n, alpha_old, V_old, stage_cost_old, u_opt = U_optimal, y_opt = history_sub_solver['Y_opt'], out_opt= history_sub_solver['out_opt'])
                      print(f'Local performance index is alpha = {alpha_local} and global { alpha_old}.')
                  # raise counter
                  history['basis_size'].append(self.prediction_model.pde.state_dim)
                  n += 1  
                
        #### finalize
        print(f'MPC FINISHED with prediction model: {self.prediction_model.model_type}, plant: {self.model.model_type} -----------------------------------------------------------------')
        print('------------------------------------------------------------------------------')
        elapsed_time = perf_counter() - start_time
        history['time'] = elapsed_time
        history['control_list'] = control_list
        if self.options.perf_bound:
            history['perf_index'] = self.performance_index
        history['final_basissize'] = self.prediction_model.pde.state_dim
        history['mainroutines_time'] = history["ROMupdate_time"] + history["coarse_time"] + history["FOMsubproblem_time"] + history["ROMsubproblem_time"] + history["errorest_time"]
        print(f'Elapsed time is {elapsed_time}, main routine time { history["mainroutines_time"]}, POD update time {history["ROMupdate_time"]}, coarse time {history["coarse_time"]}, proj coarse {history["coarse_proj_time"]}, select coarse {history["coarse_select_time"]}, FOM subproblem time {history["FOMsubproblem_time"]}, ROM subproblem time {history["ROMsubproblem_time"]}, error_est time {history["errorest_time"]}.')
        if self.options.perf_bound:
            print(f'Performance index is time {alpha_old}.')
        print(f'Final basisize is {self.prediction_model.pde.state_dim}.')
        
        if self.options.track_basis_coeff and not coarse_flag:
           YROMlist.append(np.concatenate(y_ROM, axis = 1))
           PROMlist.append(np.concatenate(p_ROM, axis = 1))
           history['YROMlist'] = YROMlist
           history['PROMlist'] = PROMlist
           
        print(f'update flags: {history["update_flags"]}')
        
        # get feedback controls as array
        U_feedback = np.concatenate(U_feedback, axis = 1)
        Y_feedback = np.concatenate(Y_feedback, axis = 1)
        # P_feedback_NOTUSED = np.concatenate(P_feedback, axis = 1)
        
        # check first order optimality condition and reupdate the models with the global data
        self.model.update_mpc_parameters_global(time_disc = self.global_model_time_disc, global_data = self.global_model_data)                                   
        self.prediction_model.update_mpc_parameters_global(time_disc = self.global_prediction_model_time_disc, global_data = self.global_prediction_model_data)   
        grad, output_feedback, Y, P_feedback, switch_profile, BTp, gaps = self.model.gradJ_OBD(U_feedback, return_switch_profil = True)
        print(f'Gradient norm = {self.model.space_time_norm(grad, "control")}')
        print(f'Y-Yf = {self.model.space_time_norm(Y_feedback-Y, "L2")}')
        J_val = self.model.J(U_feedback, Y = Y, output = output_feedback)
        history['J'] = J_val
        history['U'] = U_feedback
        history['P'] = P_feedback
        history['switch_profile'] = switch_profile
        history['BTp'] = BTp
        history['Y'] = Y_feedback
        history['out'] = output_feedback
        history['compare_ests'] = compare_ests
        
        # plot distance to target and distance to control
        Jtrack_traj, Jcontrol_traj, J_traj = self.model.Jtracking_trajectory(u = U_feedback, Y = Y_feedback, output = output_feedback)
        if self.options.plot_:
             self.model.visualize_1d_many([Jtrack_traj, Jcontrol_traj, J_traj], strings = ['tracking term', 'control term', 'J'], title = f'adaptiv {self.prediction_model.model_type}, plant: {self.model.model_type} ', semi = True, time = None)
        
        # print performance index
        if self.options.perf_bound:
            plt.figure()
            plt.semilogy(self.performance_index)
            plt.title(f'perf index {self.prediction_model.model_type}, plant: {self.model.model_type} ')
        
        history['Jtrack_traj'] = Jtrack_traj
        print(f" coarsed in {history['coarsed_in_iter']}")
        print(f" enriched in {history['enriched_in_iter']}")
        print('------------------------------------------------------------------------------')
        print('------------------------------------------------------------------------------')
        return U_feedback, Y_feedback, P_feedback, output_feedback, history

    def get_error_est_correction(self, n, U_optimal, history_sub_solver):
        if n >0 and self.options.error_est_corr:
            est_optimal_control, Y_FOMest, P_FOMest, output_FOMest = self.model.optimal_control_est(U_optimal, 
                                                                                                    P = history_sub_solver['P_opt'],
                                                                                                    Y = history_sub_solver['Y_opt'])
            est_optimal_control_cheap = self.prediction_model.cheap_optimal_control_error_est(U = U_optimal,
                                                                                       Yr = self.reductor.FOMtoROM(history_sub_solver['Y_opt']), Pr = self.reductor.FOMtoROM(history_sub_solver['P_opt']),
                  k = None, switching_profile_r = history_sub_solver['switch_profil_opt'], out_r = history_sub_solver['out_opt'], state_H_est_list = None, computationtype = 'online')
            factor = 3 
            if n == 1:
                error_est_correction_constant_for_cheap = factor*est_optimal_control/est_optimal_control_cheap
            else:
                error_est_correction_constant_for_cheap = factor*est_optimal_control/est_optimal_control_cheap 
           
            print(f'Error est is = {est_optimal_control}, cheap est is {est_optimal_control_cheap} and tol {self.options.control_tol_update} and therefore the correction is {error_est_correction_constant_for_cheap}.')
        else:
            error_est_correction_constant_for_cheap = 1
        
        return error_est_correction_constant_for_cheap
    
    def update_rom(self, n, history_sub_solver, dt, l):
        
        if n == 0:
            if self.options.POD_type == 0:
               # standard snapshot selection
               P_snap = 1*history_sub_solver['P_opt']
               P_snap[:,0] *=0
               snapshots = [history_sub_solver['Y_opt'], P_snap]
               if self.target_snapshots:
                   Y_target, P_target, Out_target = self.model.get_snapshots(U = self.model.cost_data.Ud)
                   P_snap = 1*P_target
                   P_snap[:,0] *=0
                   snapshots.append(Y_target)
                   snapshots.append(P_snap)
                   
               D_time = 1*self.model.time_disc.D_diag + 0
            elif self.options.POD_type == 1:
                # standard snapshot selection
                P_snap = 1*history_sub_solver['P_opt']
                P_snap[:,0] *=0
                snapshots = [history_sub_solver['Y_opt'], P_snap]
                if self.target_snapshots:
                    Y_target, P_target, Out_target = self.model.get_snapshots(U = self.model.cost_data.Ud)
                    P_snap = 1*P_target
                    P_snap[:,0] *=0
                    snapshots.append(Y_target)
                    snapshots.append(P_snap)
                 # compute DQs for every snapshot
                dq_snapshots = []
                for snapshot in snapshots:
                    TMP = 0*snapshot
                    TMP[:,2:] = (snapshot[:,2:]-snapshot[:,1:-1])/dt 
                    # compute dq
                    dq_snapshots.append(TMP)
                snapshots += dq_snapshots
                D_time = 1*self.model.time_disc.D_diag + 0
                
            elif self.options.POD_type == 2:
                 data = [history_sub_solver['Y_opt'], history_sub_solver['P_opt']]
                 if self.options.target_snapshots:
                     Y_target, P_target, Out_target, _ , gaps_target= self.model.get_snapshots(U = self.model.cost_data.Ud)
                     data.append(Y_target)
                     data.append(P_target)
                 
                 snapshots = []
                 for snapshot in data:
                     TMP = 0*snapshot
                     TMP[:,2:] = (snapshot[:,2:]-snapshot[:,1:-1])/dt 
                     TMP[:,0] = snapshot[:,1]
                     snapshots.append(TMP)
                 
                 tmp = dt * np.ones(self.model.time_disc.K)
                 tmp[0] = 1
                 D_time = diags(tmp)
             
        else:
            
            l += self.options.POD_update_l
            if 1 and n<300 : 
                len_s = len(self.reductor.Snapshots)
                snapshots = self.reductor.Snapshots[-1][max(0,len_s-self.options.len_old_snapshots):] # 5
            elif 1 and n>= 300:
                len_s = len(self.reductor.Snapshots)
                snapshots = self.reductor.Snapshots[-1][max(0,len_s-self.options.len_old_snapshots):]
                     
            if self.reductor.update_type == 'incremental_svd': # throw them away
                snapshots = []
            
            if self.options.POD_type == 0:
                P_snap = 1*history_sub_solver['P_opt']
                P_snap[:,0] *=0
                new_snapshots = [history_sub_solver['Y_opt'], P_snap]
                if self.target_snapshots:
                    Y_target, P_target, Out_target,_,_ = self.model.get_snapshots(U = self.model.cost_data.Ud)
                    P_snap = 1*P_target
                    P_snap[:,0] *=0
                    new_snapshots.append(Y_target)
                    new_snapshots.append(P_snap)
                snapshots += new_snapshots 
                D_time = 1*self.model.time_disc.D_diag + 0
                
            elif self.options.POD_type == 1:
                
                # standard snapshot selection
                new_snapshots = [] 
                P_snap = 1*history_sub_solver['P_opt']
                P_snap[:,0] *=0
                snapshots.append(history_sub_solver['Y_opt'])
                snapshots.append(P_snap)
                if self.target_snapshots:
                    Y_target, P_target, Out_target,_,_ = self.model.get_snapshots(U = self.model.cost_data.Ud)
                    snapshots.append(Y_target)
                    P_snap = 1*P_target
                    P_snap[:,0] *=0
                    snapshots.append(P_snap)
                    
                # compute DQs for every snapshot
                dq_snapshots = []
                for snapshot in new_snapshots:
                   TMP = 0*snapshot
                   TMP[:,2:] = (snapshot[:,2:]-snapshot[:,1:-1])/dt 
                   # compute dq
                   dq_snapshots.append(TMP)
                snapshots += new_snapshots
                snapshots += dq_snapshots
                
                D_time = 1*self.model.time_disc.D_diag + 0
                
            elif self.options.POD_type == 2:
                data = [history_sub_solver['Y_opt'], history_sub_solver['P_opt']]
                if self.options.target_snapshots:
                    Y_target, P_target, Out_target, _,_ = self.model.get_snapshots(U = self.model.cost_data.Ud)
                    data.append(Y_target)
                    data.append(P_target)
                
                new_snapshots = []
                for snapshot in data:
                    TMP = 0*snapshot
                    TMP[:,2:] = (snapshot[:,2:]-snapshot[:,1:-1])/dt 
                    TMP[:,0] = snapshot[:,1]
                    new_snapshots.append(TMP)
                snapshots += new_snapshots
                
                tmp = dt * np.ones(self.model.time_disc.K)
                tmp[0] = 1
                D_time = diags(tmp)
                
        # update model
        self.model.update_mpc_parameters_global(time_disc = self.global_model_time_disc, global_data = self.global_model_data) 
        self.prediction_model = self.reductor.update_rom(l, 
                                                         Snapshots = snapshots, 
                                 space_product = self.reductor.space_product,
                                 time_product = D_time,
                                 PODmethod = self.options.PODmethod,
                                 plot = False,
                                 model_to_project = None,
                                 old_rom = None,
                                 n = n)
        return self.prediction_model

    def error_estimation(self, U_optimal, history_sub_solver, init_bound, error_est_correction_constant_for_cheap, compare_ests, history, kf, U_0):
        
        if self.options.error_est_type == 'expensive':
            # Eval expensive error estimator  
            
            if self.options.type == 'FOMROM': # est expesive A
                est_optimal_control, Y_FOMest, P_FOMest, est_output_init_collection = self.model.optimal_control_est(U_optimal, B_listTPr =  history_sub_solver['B_listTP_opt'], 
                                                                                                    out_r =history_sub_solver['out_opt'], 
                                                                                                    Yr = history_sub_solver['Y_opt'], 
                                                                                                    switch_profile_r = history_sub_solver['switch_profil_opt'],
                                                                                                    type_= 'new',
                                                                                                    Bound_init_n = init_bound, 
                                                                                                    return_init_bound = True)
            elif self.options.type == 'ROMROM': # est expensive B     
                    assert 0, 'Options choice makes no sense ....'   
                    #type_ = 'new_split_up',                                                                         
                
        elif self.options.error_est_type == 'cheap':
            
            if self.options.type == 'FOMROM': # est cheap A'
                est_optimal_control, est_output_init_collection = self.prediction_model.cheap_optimal_control_error_est(U = U_optimal,
                                                                                                                        Yr = history_sub_solver['Y_opt'], 
                                                                                                                        Pr = history_sub_solver['P_opt'],
                                                                                                                        k = None, 
                                                                                                                        type_ = 'new_cheap',
                                                                                                                        switching_profile_r = history_sub_solver['switch_profil_opt'], 
                                                                                                                        out_r = history_sub_solver['out_opt'], 
                                                                                                                        state_H_est_list = None, 
                                                                                                                        computationtype = 'offline_online',
                                                                                                                        return_init_bound = True,
                                                                                                                        Bound_init_n = init_bound,
                                                                                                                        gaps_r = history_sub_solver['gaps'])
                est_optimal_control = error_est_correction_constant_for_cheap*est_optimal_control
                Y_FOMest = None      
            elif self.options.type == 'ROMROM': # est cheap B' 
                est_optimal_control, est_output_init_collection = self.prediction_model.cheap_optimal_control_error_est(U = U_optimal,
                                                                                                                        Yr = history_sub_solver['Y_opt'], 
                                                                                                                        Pr = history_sub_solver['P_opt'],
                                                                                                                        k = None, 
                                                                                                                        type_ = 'new_split_up_cheap',
                                                                                                                        switching_profile_r = history_sub_solver['switch_profil_opt'], 
                                                                                                                        out_r = history_sub_solver['out_opt'], 
                                                                                                                        state_H_est_list = None, 
                                                                                                                        computationtype = 'offline_online',
                                                                                                                        return_init_bound = True,
                                                                                                                        Bound_init_n = init_bound,
                                                                                                                        gaps_r = history_sub_solver['gaps'])
                est_optimal_control = error_est_correction_constant_for_cheap*est_optimal_control
                Y_FOMest = None        
        elif self.options.error_est_type == 'compare':
            # Compute all estimators and true error
            
            compare_ests, est_optimal_control, new_init_bound, Y_FOMest, est_output_init_collection  = fun_compare_ests(compare_ests, self.model, self.prediction_model, U_optimal, history_sub_solver, init_bound, self.options, kf, type_ = self.options.type)
           
            # true error
            if self.options.measure_true_error:
                U_optimal_true, history_sub_solver_true = self.model.solve_ocp(U_0,
                                                  options = self.options.innersolver_options,
                                                  checkstuff = self.checkstuff)
                history['true_error'].append(self.model.space_time_norm(U_optimal_true-U_optimal , space_norm = "control"))
            
            
        if not  self.options.error_est_type == 'compare':      
            # EVAL NEW State INIT BOUND
            new_init_bound = self.model.state_control_perturbation(rho = None, 
                                                                   control_bound = est_output_init_collection.est_with_init, 
                                                                   init_bound = init_bound , 
                                                                   switch_profile = history_sub_solver['switch_profil_opt'],
                                                                   kf = kf,
                                                                   mpc_type = self.options.type,
                                                                   ResStateDualNormSquared = est_output_init_collection.Res_squarednorm)
        
        return new_init_bound, est_optimal_control, est_output_init_collection, Y_FOMest
        
#%% compute true errors 

    def compute_true_errors(self, Ufom, Urom, Yfom, Yrom, switchprofile):
        
        start_time = perf_counter()

        ######init
        kp, kf, Tp, Tf, mpc_steps, dt, t0, T, t_v, K = self.mpc_time_data.kp, self.mpc_time_data.kf, self.mpc_time_data.Tp, self.mpc_time_data.Tf, self.mpc_time_data.mpc_steps, self.mpc_time_data.dt, self.mpc_time_data.t0, self.mpc_time_data.T, self.mpc_time_data.t_v_global, self.mpc_time_data.K
        history = {'state_MA_error': [],
                   'control_L2error': [],
                   }
        for n in range(mpc_steps):
            
            ####### 1. prepare local in time feedback model and solve subproblem
            print(f'Step {n+1}/{mpc_steps}: prediction horizon: [{n*kf}, {min(n*kf+kp,K-1)}], feedback horizon: [{n*kf}, {(n+1)*kf}]')
            
            # compute L2 control norm on prediction intervall
            self.update_predictionmodel_data(n, None)
            Urom_local = Urom[n]#Urom[:,n*kf:min(n*kf+kp+1,K)]
            Ufom_local = Ufom[n]#Ufom[:,n*kf:min(n*kf+kp+1,K)]
            history['control_L2error'].append(self.prediction_model.space_time_norm(Urom_local-Ufom_local,'control'))
            
            # compute mA norm on feedback interval
            self.update_plant_data(n, None)
            Yrom_local = Yrom[:,n*kf:(n+1)*kf+1]
            Yfom_local = Yfom[:,n*kf:(n+1)*kf+1]
            terminal = (Yrom_local-Yfom_local)[:,-1]
            rho, info = self.model.compute_rhos(switchprofile[n*kf:(n+1)*kf+1])
            Min = self.prediction_model.space_product( terminal, terminal, space_norm = 'M_switch', switch = switchprofile[(n+1)*kf])
            Ain = 0#self.prediction_model.space_time_product(Yrom_local-Yfom_local,Yrom_local-Yfom_local,'energy_product', time_norm = info.rho_time_mat,switch_profile = switchprofile[n*kf:(n+1)*kf+1])
            out = np.sqrt(Ain+Min)
            history['state_MA_error'].append(out)
            
        # Finalize
        print('Calculation of error norms finished -----------------------------------------------------------------')
        elapsed_time = perf_counter() - start_time
        history['time'] = elapsed_time
        return history
        
#%% helpers

def fun_compare_ests(compare_ests, model, prediction_model, U_optimal, history_sub_solver, init_bound, options, kf, type_):
    
    # A'
    if  type_ == 'ROMROM':
        stop = 1
    est_Acheap, est_collectionAcheap = prediction_model.cheap_optimal_control_error_est(U = U_optimal,
                                                                                            Yr = history_sub_solver['Y_opt'], 
                                                                                            Pr = history_sub_solver['P_opt'],
                                                                                            k = None, 
                                                                                            type_ = 'new_cheap',
                                                                                            switching_profile_r = history_sub_solver['switch_profil_opt'], 
                                                                                            out_r = history_sub_solver['out_opt'], 
                                                                                            state_H_est_list = None, 
                                                                                            computationtype = 'offline_online',
                                                                                            return_init_bound = True,
                                                                                            Bound_init_n = init_bound,
                                                                                            gaps_r = history_sub_solver['gaps'])
    
    new_init_boundAcheap = model.state_control_perturbation(rho = None, 
                                                           control_bound = est_collectionAcheap.est_with_init, 
                                                           init_bound = init_bound , 
                                                           switch_profile = history_sub_solver['switch_profil_opt'],
                                                           kf = kf,
                                                           mpc_type = options.type,
                                                           ResStateDualNormSquared = est_collectionAcheap.Res_squarednorm)
    # B'
    est_Bcheap, est_collectionBcheap = prediction_model.cheap_optimal_control_error_est(U = U_optimal,
                                                                                            Yr = history_sub_solver['Y_opt'], 
                                                                                            Pr = history_sub_solver['P_opt'],
                                                                                            k = None, 
                                                                                            type_ = 'new_split_up_cheap',
                                                                                            switching_profile_r = history_sub_solver['switch_profil_opt'], 
                                                                                            out_r = history_sub_solver['out_opt'], 
                                                                                            state_H_est_list = None, 
                                                                                            computationtype = 'offline_online',
                                                                                            return_init_bound = True,
                                                                                            Bound_init_n = init_bound,
                                                                                            gaps_r = history_sub_solver['gaps'])
    new_init_boundBcheap = model.state_control_perturbation(rho = None, 
                                                           control_bound = est_collectionBcheap.est_with_init, 
                                                           init_bound = init_bound , 
                                                           switch_profile = history_sub_solver['switch_profil_opt'],
                                                           kf = kf,
                                                           mpc_type = options.type,
                                                           ResStateDualNormSquared = est_collectionBcheap.Res_squarednorm)
    
    if not type_ == 'ROMROM' or 1:
        # A
        est_A, Y_FOMest, P_FOMest, est_collectionA = model.optimal_control_est(U_optimal, B_listTPr =  history_sub_solver['B_listTP_opt'], 
                                                                                            out_r =history_sub_solver['out_opt'], 
                                                                                            Yr = history_sub_solver['Y_opt'], 
                                                                                            switch_profile_r = history_sub_solver['switch_profil_opt'],
                                                                                            type_= 'new',
                                                                                            Bound_init_n = init_bound, 
                                                                                            return_init_bound = True)
        
        new_init_boundA = model.state_control_perturbation(rho = None, 
                                                               control_bound = est_collectionA.est_with_init, 
                                                               init_bound = init_bound , 
                                                               switch_profile = history_sub_solver['switch_profil_opt'],
                                                               kf = kf,
                                                               mpc_type = options.type,
                                                               ResStateDualNormSquared = est_collectionAcheap.Res_squarednorm)
        
        # B
        est_B, _, P_FOMest, est_collectionB = model.optimal_control_est(U_optimal, B_listTPr =  history_sub_solver['B_listTP_opt'], 
                                                                                            out_r =history_sub_solver['out_opt'], 
                                                                                            Yr = history_sub_solver['Y_opt'], 
                                                                                            switch_profile_r = history_sub_solver['switch_profil_opt'],
                                                                                            type_= 'new_split_up',
                                                                                            Bound_init_n = init_bound, 
                                                                                            return_init_bound = True)
        
        new_init_boundB = model.state_control_perturbation(rho = None, 
                                                               control_bound = est_collectionB.est_with_init, 
                                                               init_bound = init_bound , 
                                                               switch_profile = history_sub_solver['switch_profil_opt'],
                                                               kf = kf,
                                                               mpc_type = options.type,
                                                               ResStateDualNormSquared = est_collectionBcheap.Res_squarednorm)
        
        compare_ests.controlA.append(est_A)
        compare_ests.control_initA.append(est_collectionA.est_with_init)
        compare_ests.controlB.append(est_B)
        compare_ests.control_initB.append(est_collectionB.est_with_init)
        compare_ests.MPC_initA.append(new_init_boundA)
        compare_ests.MPC_initB.append(new_init_boundB)
   
    # update bounds
    
    compare_ests.controlAcheap.append(est_Acheap)
    compare_ests.control_initAcheap.append(est_collectionAcheap.est_with_init)

    compare_ests.controlBcheap.append(est_Bcheap)
    compare_ests.control_initBcheap.append(est_collectionBcheap.est_with_init)
    
    compare_ests.MPC_initAcheap.append(new_init_boundAcheap)
    compare_ests.MPC_initBcheap.append(new_init_boundBcheap)
    
    if not type_ == 'ROMROM':
        return compare_ests, est_A, new_init_boundAcheap, Y_FOMest, est_collectionA
    else:
        return compare_ests, est_Bcheap, new_init_boundBcheap, None , est_collectionBcheap