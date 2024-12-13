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
# Description: this file contains the class for the switched evolution equation.

import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve, factorized
from time import time, perf_counter
from methods import collection
import fenics as fenics
import pandas as pd

class model():
    
    def __init__(self, pde_data, cost_data, time_disc, space_disc, options = None, error_estimator = None):
        self.pde = pde_data
        self.cost_data = cost_data
        self.options = options
        self.time_disc = time_disc
        self.state_dim = pde_data.state_dim
        self.input_dim = pde_data.input_dim
        self.output_dim = pde_data.output_dim
        self.products = pde_data.products
        self.n_systems = len(self.pde.A)
        self.model_type = pde_data.type
        self.space_disc = space_disc
        self.error_estimator = error_estimator
        self.init_switch = None
        
        self.theta = 1
        assert self.theta == 1, 'factorization does only work for theta equal one.... fix this'
        
        if options is not None:
            if options.factorize and self.isSwitchModel():
                self.factorize(theta = 1)
            else:
                options.factorize = False
                self.pde.factorized_op = None
                self.pde.factorized_op_adjoint = None
        else:
            options = collection()
            options.factorize = False
            options.energy_prod = False
            self.pde.factorized_op = None
            self.pde.factorized_op_adjoint = None
            self.options = options
    
    def isFOM(self):
        return 'FOM' in self.pde.type
    def isSwitchModel(self):
        return 'Switch' in self.pde.type
    def isTimeVaryingModel(self):
        return 'TimeVarying' in self.pde.type
    def isStateDep(self):
        return 'StateDep' in self.pde.type
    def isSymmetricA(self):
        return 'SymmetricA' in self.pde.type
    def isNonSmoothlyRegularized(self):
        return 'nonsmooth_g' in self.pde.type
    def print_info(self):
        print(f' Model name: {self.model_type} with state dim: {self.state_dim}, output_dim: {self.output_dim}, input_dim: {self.input_dim}')
    def factorize(self, theta = 1):
        assert self.n_systems < 20, 'too many systems, this is maybe inefficient....'

        # factorize
        self.pde.factorized_op = []
        self.pde.factorized_op_adjoint = []
        self.pde.Mfactorized = []
        for i in range(self.n_systems):
            
            if self.isFOM(): # sparse
                op = self.pde.M[i] + theta * self.time_disc.dt*self.pde.A[i] 
                op_adjoint = self.pde.M[i] + theta * self.time_disc.dt*self.pde.A[i].T         
                self.pde.factorized_op.append(factorized(op))
                self.pde.factorized_op_adjoint.append(factorized(op_adjoint))
                self.pde.Mfactorized.append(factorized(self.pde.M[i]))
            else:
                
                # TODO is sparse factorize efficient here?
                op = self.pde.M[i] + theta * self.time_disc.dt*self.pde.A[i] 
                assert isinstance(op, np.ndarray)
                op_adjoint = self.pde.M[i] + theta * self.time_disc.dt*self.pde.A[i].T         
                self.pde.factorized_op.append(factorized(op))
                self.pde.factorized_op_adjoint.append(factorized(op_adjoint))
                self.pde.Mfactorized.append(factorized(self.pde.M[i]))
       
#%% set model
    
    def update_cost_data(self, Yd = None, YT = None, Ud= None, weights= None, input_product= None, output_product= None, desired_switching_profile = None):
        if input_product is not None:
           self.cost_data.input_product = input_product
        if output_product is not None:
           self.cost_data.output_product = output_product
        if weights is not None:
           self.cost_data.weights = weights
        if Ud is not None:
           self.cost_data.Ud = Ud
        
        if Yd is not None:
            self.cost_data.Yd = Yd
            self.cost_data.Mc_Yd = self.cost_data.output_product@Yd
            self.cost_data.Yd_Mc_Yd = self.space_product_trajectory(Yd, Yd, norm = 'output')# trajectory
            
        if YT is not None:
            self.cost_data.YT = YT
            self.cost_data.Mc_YT = self.cost_data.output_product@YT
            self.cost_data.YT_Mc_YT = self.space_product(YT, YT, space_norm = 'output') #constant value

        if desired_switching_profile is not None:
            self.cost_data.desired_switching_profile = desired_switching_profile

    def get_mpc_parameters(self):
        global_data = {'y0': self.pde.y0,
                         'Yd': self.cost_data.Yd,
                        'YT': self.cost_data.YT,
                        'Ud': self.cost_data.Ud,
                        'Mc_Yd': self.cost_data.Mc_Yd,
                        'Yd_Mc_Yd': self.cost_data.Yd_Mc_Yd,
                        'Mc_YT': self.cost_data.Mc_YT,
                        'YT_Mc_YT': self.cost_data.YT_Mc_YT,
                        'F': self.pde.F,
                    }
        return self.time_disc, global_data
    
    def update_mpc_parameters_pred(self, n, mpc_time_data, local_time_disc, y0, global_data):
        self.time_disc = local_time_disc
        
        kf, kp, K = mpc_time_data.kf, mpc_time_data.kp, mpc_time_data.K
        
        # cost data
        Ud_local = global_data["Ud"][:, n*kf:min(n*kf+kp+1,K)]
        self.cost_data.Ud = Ud_local
        self.cost_data.Yd = global_data["Yd"][:, n*kf:min(n*kf+kp+1,K)]
        
        self.cost_data.Mc_Yd = global_data["Mc_Yd"][:, n*kf:min(n*kf+kp+1,K)]
        self.cost_data.Yd_Mc_Yd = global_data["Yd_Mc_Yd"][n*kf:min(n*kf+kp+1,K)]
        self.cost_data.Mc_YT = global_data["Mc_Yd"][:, min(n*kf+kp,K-1)]
        self.cost_data.YT_Mc_YT = global_data["Yd_Mc_Yd"][min(n*kf+kp,K-1)]
        
        # PDE data
        F_local = global_data["F"][:, n*kf:min(n*kf+kp+1,K)]
        self.pde.y0 = y0
        self.pde.F = F_local
        
        
    def update_mpc_parameters_plant(self, n, mpc_time_data, local_time_disc, y0, global_data):
        # (current_mpc_time_data, global_data, y0)
        self.time_disc = local_time_disc
        
        kf, kp, K = mpc_time_data.kf, mpc_time_data.kp, mpc_time_data.K
        Ud_local = global_data["Ud"][:, n*kf:min(n*kf+kf+1,K)]
        
        # COST DATA
        # self.cost_data.Yd = Yd_local
        self.cost_data.Ud = Ud_local
        self.cost_data.Yd = global_data["Yd"][:, n*kf:min(n*kf+kf+1,K)]
        self.cost_data.Mc_Yd = global_data["Mc_Yd"][:, n*kf:min(n*kf+kf+1,K)]
        self.cost_data.Yd_Mc_Yd = global_data["Yd_Mc_Yd"][n*kf:min(n*kf+kf+1,K)]
        self.cost_data.Mc_YT = global_data["Mc_Yd"][:, min(n*kf+kf,K-1)]
        self.cost_data.YT_Mc_YT = global_data["Yd_Mc_Yd"][min(n*kf+kf,K-1)]
        
        # PDE
        F_local = global_data["F"][:, n*kf:min(n*kf+kf+1,K)]
        self.pde.y0 = y0
        self.pde.F = F_local

    
    def update_mpc_parameters_global(self, time_disc, global_data):
        # update PDE
        self.time_disc = time_disc
        self.pde.y0 = global_data["y0"]
        self.pde.F = global_data["F"]
        
        # update cost data
        self.cost_data.Yd = global_data["Yd"]
        self.cost_data.Ud = global_data["Ud"]
        self.cost_data.YT = global_data["YT"]
        self.cost_data.Mc_Yd = global_data["Mc_Yd"]
        self.cost_data.Yd_Mc_Yd = global_data["Yd_Mc_Yd"]
        self.cost_data.Mc_YT = global_data["Mc_YT"]
        self.cost_data.YT_Mc_YT = global_data["YT_Mc_YT"]
        
        
#%% solve methods
    
    def solve_linear_system(self, M, A, dt, theta, switch_old, rhs, factorized_op):
        
        # delete dirichlet dofs
        if self.isFOM() and self.space_disc.DirichletBC is not None and 'Dirichlet' in self.model_type:
            _, rhs = self.space_disc.DirichletClearFun(LHS = None, rhs = rhs)
            
        
        if self.options.factorize:
           assert theta == 1, 'theta must be chosen to zero for factorization.... modify this'
           ind_switch = switch_old - 1
           out = factorized_op[ind_switch](rhs)
        else:
            LHS = M + theta*dt*A
            if self.isFOM():
                out = spsolve(LHS, rhs)
            else:
                out = np.linalg.solve(LHS, rhs)

        return out
    
    def assemble_matrices_at_time(self, t, output_old, sigma = None, switching_profile_ind = None):
        if self.isSwitchModel():
            if switching_profile_ind is None:
                switch_old = self.pde.sigma(t, output_old, sigma)
            else:
                switch_old = switching_profile_ind 
            ind_switch_old = switch_old-1
                
            C = self.pde.C[ind_switch_old] 
            A = self.pde.A[ind_switch_old] 
            B = self.pde.B[ind_switch_old] 
            M = self.pde.M[ind_switch_old] 
            return A, M, B, C, switch_old
            
        elif self.isTimeVaryingModel():
            
            A = self.pde.A[0]*self.pde.A_time_coefficient[0](t) + self.pde.A[1]*self.pde.A_time_coefficient[1](t) + self.pde.A[2]*self.pde.A_time_coefficient[2](t)
            B = self.pde.B[0]*self.pde.B_time_coefficient(t)
            C = self.pde.C[0]*self.pde.C_time_coefficient(t)
            M = self.pde.M[0]*self.pde.M_time_coefficient(t)
            return A, M, B, C, None
            
        else:
            assert 0, 'wrong model choice'
        
    def solve_state(self, U = None, theta = 1, print_ = False, T_end = None, y0 = None, fixed_switching_profil = None):
        
        start_time = perf_counter()
        F = self.pde.F
        if y0 is None:
            y0 = self.pde.y0
        time_disc = self.time_disc
        
        # init
        dt = time_disc.dt
        K = time_disc.K
        yy = y0.copy()
        Y = yy.copy().reshape(-1,1)
        t = time_disc.t_v[0]
            
        if self.init_switch is not None and t != 0 :   
            init_switch = self.init_switch
        else:
            init_switch = None
            
        A_old, M_old, B_old, C_old, switch_old = self.assemble_matrices_at_time( t, None, sigma = None, switching_profile_ind = init_switch)
        
        out = [C_old@yy]
        sigma_profile = [switch_old]
        if print_:
            if self.isSwitchModel():
                print(f'k = {0}: t = {t}, switch = {switch_old}, output = {out[-1]}')
            else:
                print(f'k = {0}: t = {t},')
        
        for k in range(1, K):
            
            # get current time and switching signal
            t = time_disc.t_v[k]
            
            A, M, B, C, switch_old = self.assemble_matrices_at_time( t, out[-1][1], sigma = switch_old)
            sigma_profile.append(switch_old)
            
            # build LHS and rhs
            RHS_mat = M+ (theta-1)*dt*A
            rhs = RHS_mat.dot(yy)
            if F is not None:
                rhs += dt*(theta*F[:,k]+(1-theta)*F[:,k-1])
            if U is not None:
                rhs += dt*B.dot((theta*U[:,k]+(1-theta)*U[:,k-1]))
            
            yy = self.solve_linear_system(M, A, dt, theta, switch_old, rhs,  self.pde.factorized_op)
            
            
            # save and compute output
            Y = np.concatenate((Y,yy.copy().reshape(-1,1)), axis=1)
            out.append(C@yy)
            if print_:
                if self.isSwitchModel():
                    print(f'k = {k}: t = {t}, switch = {switch_old}, output = {out[-1]}')
                else:
                    print(f'k = {k}: t = {t}')
        
        end_time = perf_counter()  
        if print_:
            print(f'Time stepping finished in time {end_time-start_time}')
        
        return Y, np.array(out).T, end_time, sigma_profile
    
    
    def solve_adjoint(self, Z, ZT, switching_profil, theta = 1 ):
        
        time_disc = self.time_disc
        dt = time_disc.dt
        K = time_disc.K
        t = time_disc.t_v[-1]
        A, M, B, C, _ = self.assemble_matrices_at_time(t = t, output_old = None, sigma = None,
                                                                switching_profile_ind = switching_profil[-1])
        
        ##### strategy 1 (OBD)
        if 0:
            p = np.zeros((A.shape[0],))
            p += self.cost_data.weights[2]*C.T@ZT
        
        ##### strategy 2
        else:
            rhs = dt *self.cost_data.weights[0]* C.T@Z[:,-1] 
            rhs += self.cost_data.weights[2]*C.T@ZT 
            
            p = self.solve_linear_system(M.T, A.T, dt, theta, switching_profil[-1], rhs, self.pde.factorized_op_adjoint)
        
        M_old = M
        ### init
        P = p.copy().reshape(-1,1)
        B_listTP = [B.T.dot(p)]
        
        gaps = []
        compute_gaps = True
        for k in range(K-2, -1, -1 ):
            
            # get t and switch
            t = time_disc.t_v[k]
            switch = switching_profil[k]
            
            A, M, B, C, _ = self.assemble_matrices_at_time(t = t, output_old = None, sigma = None,
                                                                    switching_profile_ind = switch)
            
            F = dt * self.cost_data.weights[0]*C.T@Z[:,k]
            rhs = M_old.T.dot(p) + F
            
            # compute gaps
            if compute_gaps:
                if abs(switch-switching_profil[k+1])<1e-14:
                    gaps.append(None)
                else:
                    p_gap = self.solve_LS(M, M_old.T.dot(p), self.pde.Mfactorized[switch-1])
                    gaps.append(p_gap)
            
            # solve system
            p = self.solve_linear_system(M.T, A.T, dt, theta, switch, rhs, self.pde.factorized_op_adjoint)
            M_old = M
            
            # append and compute output
            P = np.concatenate((p.reshape(-1,1),P), axis=1)
            B_listTP.append(B.T.dot(p))
            
        # reverse time
        B_listTP.reverse()
        gaps.reverse()
        
        return P, np.array(B_listTP).T, gaps
        
    def solve_LS(self,M, rhs, M_factor = None):
            if self.options.factorize and M_factor is not None:  
               out = M_factor(rhs)
            else:
                if self.isFOM():
                    out = spsolve(M, rhs)
                else:
                    out = np.linalg.solve(M, rhs)

            return out
            
#%% Residuals and state est matrix
  
    
    def state_residual(self, U, Y, out = None, theta = 1, switching_profile = None, compute_norm = False, norm_type = 'H1dual'):
        
        if self.options.energy_prod:
            norm_type = 'switch_energy_dual'
            
        Res = [self.pde.y0-Y[:,0]]
        Res_norm = []
        
        if compute_norm:
            if self.options.energy_prod:
                Res_norm.append(self.space_product(Res[0], Res[0], space_norm = 'M_switch', switch = switching_profile[0]))
            else:
                Res_norm.append(self.space_product(Res[0], Res[0], space_norm = 'M_switch', switch = switching_profile[0])) #'L2'
            
        assert theta == 1, 'this does only work for theta equal one'
        
        for k in range(1, self.time_disc.K):
            t = self.time_disc.t_v[k]

            A, M, B, C, switch_old = self.assemble_matrices_at_time( t, None, sigma = switching_profile[k])
            Res_k = self.pde.F[:,k] + B.dot(U[:,k]) - (M.dot(Y[:,k]) - M.dot(Y[:,k-1]))/self.time_disc.dt - A.dot(Y[:,k]) 
            Res.append(Res_k)
            if compute_norm:
                Res_norm.append(self.space_product(Res[-1], Res[-1], space_norm = norm_type, switch = switching_profile[k]))
                
        return Res, Res_norm
    
    def get_state_est_matrix(self, POD_Basis, old_data = None, norm_type = 'H1dual'):
        
        if self.options.energy_prod:
            norm_type = 'switch_energy'
            
        start_time = perf_counter()
        est_mat_sigma = []
        
        assert self.isSwitchModel() , ' ... modify this to account for affine timevarying models...'
        
        Rietz_sigma = []
        Dual_sigma = []
        
        B_dual_sigma = []
        B_rietz_sigma = []
        
        M_dual_sigma = []
        M_rietz_sigma = []
        A_dual_sigma = []
        A_rietz_sigma = []
        
        
        for i in range(len(self.pde.M)): # for all switches
            
            # constant data
            if old_data is None:
                # B_sigma
                B = self.pde.B[i]
                BDUAL = 1*B +0
                BRIETZ = self.compute_rietzrepresentant(BDUAL, norm_type = norm_type, switch = i)
            else:
                BDUAL = old_data.state_est_stuff.B_dual_sigma[i]
                BRIETZ = old_data.state_est_stuff.B_rietz_sigma[i]
            B_dual_sigma.append(BDUAL)
            B_rietz_sigma.append(BRIETZ)
            
            # changing data with basis update
            # M sigma
            M = self.pde.M[i]
            MDUAL = M@POD_Basis
            MRIETZ = self.compute_rietzrepresentant(MDUAL, norm_type = norm_type, switch = i)
            M_dual_sigma.append(MDUAL)
            M_rietz_sigma.append(MRIETZ)
            
            # A sigma 
            A = self.pde.A[i]
            ADUAL = A@POD_Basis
            ARIETZ = self.compute_rietzrepresentant(ADUAL, norm_type = norm_type, switch = i)
            A_dual_sigma.append(ADUAL)
            A_rietz_sigma.append(ARIETZ)
            
            if 1:
                RDUAL = np.concatenate((BDUAL, MDUAL, ADUAL), axis = 1)
                RRIETZ = np.concatenate((BRIETZ, MRIETZ, ARIETZ), axis = 1)
            else:
                RDUAL = ADUAL
                RRIETZ =ARIETZ
            
            R_rietz = RRIETZ
            R_dual = RDUAL
            
            # get error est mat
            est_mat = R_dual.T@R_rietz
            est_mat_sigma.append(est_mat)
            
            Rietz_sigma.append(RRIETZ)
            Dual_sigma.append(RDUAL)
            
        state_est_stuff = collection()
        state_est_stuff.est_mat_sigma = est_mat_sigma
        state_est_stuff.Rietz_sigma = Rietz_sigma
        state_est_stuff.Dual_sigma = Dual_sigma
        state_est_stuff.B_dual_sigma = B_dual_sigma
        state_est_stuff.B_rietz_sigma = B_rietz_sigma
        state_est_stuff.M_dual_sigma = M_dual_sigma
        state_est_stuff.M_rietz_sigma = M_rietz_sigma
        state_est_stuff.A_dual_sigma = A_dual_sigma
        state_est_stuff.A_rietz_sigma = A_rietz_sigma
        print(f'State error est assembly done in {perf_counter()-start_time} ')
        
        return state_est_stuff
    
    def adjoint_residual(self, P, ZT, Z, out = None, theta = 1, switching_profile = None, compute_norm = False, norm_type = 'H1dual', gaps = None):
        
        if self.options.energy_prod:
            norm_type = 'switch_energy_dual'
            
        Res = []
        Res_norm = []
        Res_switchpoints_norm = []
        R_switch = []
        K = self.time_disc.K
        dt = self.time_disc.dt
        
        # init
        t = self.time_disc.t_v[-1]
        A, M, B, C, _ = self.assemble_matrices_at_time(t = t, output_old = None, sigma = None,
                                                                switching_profile_ind = switching_profile[-1])
        rhs = self.cost_data.weights[0]* C.T@Z[:,-1] + self.cost_data.weights[2]/dt*C.T@ZT 
        LHS = M.T.dot(P[:,-1])/dt + A.T.dot(P[:,-1])
        Res_k = rhs - LHS
        Res.append(Res_k)

        if compute_norm:
            if self.options.energy_prod:
                # Res_norm.append(self.space_product(Res[-1], Res[-1], space_norm = 'M_switch', switch = switching_profile[-1] ))
                Res_norm.append(self.space_product(Res[-1], Res[-1], space_norm = 'switch_energy_dual', switch = switching_profile[-1] ))
            else:
                # Res_norm.append(self.space_product(Res[-1], Res[-1], space_norm = 'L2', switch = switching_profile[-1] ))
                Res_norm.append(self.space_product(Res[-1], Res[-1], space_norm = 'H1dual', switch = switching_profile[-1] ))
                
        assert theta == 1, 'this does only work for theta equal one'
        
        M_old = M
        
        for k in range(K-2, -1, -1 ):
            t = self.time_disc.t_v[k]
            switch = switching_profile[k]
            A, M, B, C, switch_old = self.assemble_matrices_at_time(t, None, sigma = None, switching_profile_ind = switch )
            
            # index correct
            rhs = self.cost_data.weights[0]*C.T@Z[:,k]
            LHS = (M.T.dot(P[:,k]) - M_old.T.dot(P[:,k+1]))/dt + A.T.dot(P[:,k])
            Res_k = rhs - LHS
            Res.append(Res_k)
            
            # Res switchpoints
            if not abs(switch-switching_profile[k+1])<1e-14:
                # print(k)
                if gaps[k] is not None:
                    R_switch.append(M_old.T.dot(P[:,k+1]) - M.T.dot(gaps[k]))
                    if compute_norm:
                        if self.options.energy_prod:
                            Res_switchpoints_norm.append(self.space_product(R_switch[-1], R_switch[-1], space_norm = 'M_switch', switch = switch ))
                        else:
                            Res_switchpoints_norm.append(self.space_product(R_switch[-1], R_switch[-1], space_norm = 'M_switch', switch = switch ))
            else:
                R_switch.append(0)
                Res_switchpoints_norm.append(0)
                
            M_old = M
            if compute_norm:
                Res_norm.append(self.space_product(Res[-1], Res[-1], space_norm = norm_type, switch = switch))
        
        if 1:
            Res_norm.reverse()        
            Res.reverse()
            Res_switchpoints_norm.reverse()
            R_switch.reverse()
        return Res, Res_norm, R_switch, Res_switchpoints_norm
    
    def get_adjoint_est_matrix(self, basis, old_data = None, norm_type = 'H1dual', state_data = None, PetrovGalerkin = False, basis_state = None):
        start_time = perf_counter()
        
        assert self.isSwitchModel() , ' ... modify this to account for affine time-varying models...'
        
        if self.options.energy_prod:
            norm_type = 'switch_energy'
            
        if 0:
            C_dual_sigma = []
            C_rietz_sigma = []
            
            M_dual_sigma = []
            M_rietz_sigma = []
            A_dual_sigma = []
            A_rietz_sigma = []
            
            for i in range(len(self.pde.M)): # for all switches
                # rhs
                if old_data is None:
                    C = self.pde.C[i]
                    CDUAL = 1*C.T+ 0
                    CRIETZ = self.compute_rietzrepresentant(CDUAL, norm_type = norm_type, switch = i)
                    
                else:
                    CDUAL = old_data.adjoint_est_stuff.C_dual_sigma[i]
                    CRIETZ = old_data.adjoint_est_stuff.C_rietz_sigma[i]
                
                C_dual_sigma.append(CDUAL)
                C_rietz_sigma.append(CRIETZ)
                
                # M
                if state_data.M_dual_sigma is None or state_data.M_dual_sigma is None or PetrovGalerkin:
                    M = self.pde.M[i]
                    MDUAL = M@basis
                    MRIETZ = self.compute_rietzrepresentant(MDUAL, norm_type = norm_type, switch = i)
                    
                else:
                    MDUAL = state_data.M_dual_sigma[i]
                    MRIETZ = state_data.M_rietz_sigma[i]
                
                M_dual_sigma.append(MDUAL)
                M_rietz_sigma.append(MRIETZ)
                
                # A
                if True:
                    A = self.pde.A[i].T
                    ADUAL = A@basis
                    ARIETZ = self.compute_rietzrepresentant(ADUAL, norm_type = norm_type, switch = i)
                A_dual_sigma.append(ADUAL)
                A_rietz_sigma.append(ARIETZ)
                
            # compute all cases for est mat with time delay
            est_mat_sigma = []
            Rietz_sigma = []
            Dual_sigma = []
            for i in range(len(self.pde.M)): # for all switches
                est_i = []
                est_Rietz_i = []
                est_Dual_i = []
                for j in range(len(self.pde.M)):
                    
                    # read correct C, M, Mold, A
                    CDUAL = C_dual_sigma[i]
                    CRIETZ = C_rietz_sigma[i]
                    ADUAL = A_dual_sigma[i]
                    ARIETZ = A_rietz_sigma[i]
                    
                    # read M with index change i, j 
                    MDUAL_i = M_dual_sigma[i]
                    MRIETZ_i = M_rietz_sigma[i]
                    MDUAL_j = M_dual_sigma[j]
                    MRIETZ_j = M_rietz_sigma[j]
                    
                    # concatenate
                    RDUAL = np.concatenate((CDUAL, MDUAL_i, MDUAL_j, ADUAL), axis = 1)
                    RRIETZ = np.concatenate((CRIETZ, MRIETZ_i, MRIETZ_j, ARIETZ), axis = 1)             
                
                    # get error est mat
                    est_mat = RDUAL.T@RRIETZ
                    
                    # append
                    est_i.append(est_mat)
                    est_Rietz_i.append(RRIETZ)
                    est_Dual_i.append(RDUAL)
                    
                est_mat_sigma.append(est_i)
                Rietz_sigma.append(est_Rietz_i)
                Dual_sigma.append(est_Dual_i)
                
        else: # standard
            C_dual_sigma = []
            C_rietz_sigma = []
            
            M_dual_sigma = []
            M_rietz_sigma = []
            A_dual_sigma = []
            A_rietz_sigma = []
            
            est_mat_sigma = []
            Rietz_sigma = []
            Dual_sigma = []
            
            # compute and save rietz repr
            for i in range(len(self.pde.M)): # for all switches
                # rhs
                if old_data is None:
                    C = self.pde.C[i]
                    CDUAL = 1*C.T+ 0
                    CRIETZ = self.compute_rietzrepresentant(CDUAL, norm_type = norm_type, switch = i)
                    
                else:
                    CDUAL = old_data.adjoint_est_stuff.C_dual_sigma[i]
                    CRIETZ = old_data.adjoint_est_stuff.C_rietz_sigma[i]
                
                C_dual_sigma.append(CDUAL)
                C_rietz_sigma.append(CRIETZ)
                
                # M
                # TODO
                if state_data.M_dual_sigma is None or state_data.M_dual_sigma is None or PetrovGalerkin:
                    M = self.pde.M[i]
                    MDUAL = M@basis
                    MRIETZ = self.compute_rietzrepresentant(MDUAL, norm_type = norm_type, switch = i)
                    
                else:
                    MDUAL = state_data.M_dual_sigma[i]
                    MRIETZ = state_data.M_rietz_sigma[i]
                
                M_dual_sigma.append(MDUAL)
                M_rietz_sigma.append(MRIETZ)
                
                # A
                if True:
                    A = self.pde.A[i].T
                    ADUAL = A@basis
                    ARIETZ = self.compute_rietzrepresentant(ADUAL, norm_type = norm_type, switch = i)
                A_dual_sigma.append(ADUAL)
                A_rietz_sigma.append(ARIETZ)
            
                # concatenate
                RDUAL = np.concatenate((CDUAL, MDUAL, ADUAL), axis = 1)
                RRIETZ = np.concatenate((CRIETZ, MRIETZ, ARIETZ), axis = 1)             
            
                # get error est mat
                est_mat = RDUAL.T@RRIETZ
                
                # append
                est_mat_sigma.append(est_mat)
                Rietz_sigma.append(RRIETZ)
                Dual_sigma.append(RDUAL)
        
        # Mdual_mixed = []
        Mrietz_mixed = []
        for i in range(len(self.pde.M)): 
            for j in range(len(self.pde.M)): 
                if i == j:
                    pass
                else:
                    assert len(self.pde.M) == 2, 'generalize below for more switching modes ... '
                    M = self.pde.M[j]
                    MDUAL = M@basis
                    MRIETZ = self.compute_rietzrepresentant(MDUAL, norm_type = norm_type, switch = i)
                    # Mdual_mixed
                    Mrietz_mixed.append(MRIETZ)
                    
        # build switched estimator matrices
        est_mat_switched = []
        for i in range(len(self.pde.M)): 
            for j in range(len(self.pde.M)): 
                if i == j:
                    pass
                else:      
                   # read correct C, M, Mold, A
                   CDUAL = C_dual_sigma[i]
                   CRIETZ = C_rietz_sigma[i]
                   ADUAL = A_dual_sigma[i]
                   ARIETZ = A_rietz_sigma[i]
                   
                   # read M with index change i, j 
                   MDUAL_i = M_dual_sigma[i]
                   MRIETZ_i = M_rietz_sigma[i]
                   
                   # geswitchte
                   MDUAL_j = M_dual_sigma[j]
                   MRIETZ_j = Mrietz_mixed[i]
                   
                   # concatenate
                   RDUAL = np.concatenate((CDUAL, MDUAL_i, MDUAL_j, ADUAL), axis = 1)
                   RRIETZ = np.concatenate((CRIETZ, MRIETZ_i, MRIETZ_j, ARIETZ), axis = 1)             
               
                   # get error est mat
                   est_mat_switched.append(RDUAL.T@RRIETZ)
                   
        print(f'Adjoint error est assembly done in {perf_counter()-start_time} ')
        adjoint_est_stuff = collection()
        adjoint_est_stuff.est_mat_sigma = est_mat_sigma
        adjoint_est_stuff.Rietz_sigma = Rietz_sigma
        adjoint_est_stuff.Dual_sigma = Dual_sigma
        adjoint_est_stuff.C_dual_sigma = C_dual_sigma
        adjoint_est_stuff.C_rietz_sigma = C_rietz_sigma
        adjoint_est_stuff.M_dual_sigma = M_dual_sigma
        adjoint_est_stuff.M_rietz_sigma = M_rietz_sigma
        adjoint_est_stuff.A_dual_sigma = A_dual_sigma
        adjoint_est_stuff.A_rietz_sigma = A_rietz_sigma
        
        # switched stuff
        adjoint_est_stuff.Mrietz_mixed = Mrietz_mixed
        adjoint_est_stuff.est_mat_switched = est_mat_switched
        
        return adjoint_est_stuff

#%% cost function
    
    def get_snapshots(self, U, y0 = None, test_residual = False):
        
        if y0 is None:
            y0 = self.pde.y0
            
        if len(U.shape) == 1:
            U = self.vector_to_matrix(U, self.input_dim)
        Y, output, time, sigma_profile = self.solve_state(U = U, theta = self.theta, y0 = y0)
        # Z = output - self.cost_data.Yd
        # ZT = output[:,-1]-self.cost_data.YT
        Z = self.cost_data.output_product.dot(output) - self.cost_data.Mc_Yd
        ZT = self.cost_data.output_product.dot(output[:,-1]) - self.cost_data.Mc_YT
        
        P, B_listTP, gaps = self.solve_adjoint(Z, ZT, switching_profil = sigma_profile)
        
        if test_residual: # test residual
            R, R_norm, Rswitch, Rswitchnorm = self.adjoint_residual(P, ZT, Z, switching_profile = sigma_profile, compute_norm = True, gaps = gaps)
            print(f'adjoint residual {max(R_norm)}, max switch point norm {max(Rswitchnorm)} min {min(Rswitchnorm)}')
            _, R_norm_state = self.state_residual(U = U, Y = Y, compute_norm = True, switching_profile = sigma_profile)
            print(f'state residual {max(R_norm_state)}')
        
        return Y, P, output, sigma_profile, gaps

    def J(self, u, Y = None, output = None):
        U = self.vector_to_matrix(u, self.input_dim)
        if Y is None or output is None:
           Y, output, time, sigma_profile = self.solve_state(U = U, theta = self.theta)
        W = U - self.cost_data.Ud
        
        if 0:
            assert self.cost_data.Yd.shape[0] != self.pde.state_dim, 'this is only valid for the ROM if yd is not stat diemsnional'
            Z = output - self.cost_data.Yd
            ZT = output[:,-1]-self.cost_data.YT
            J1 = 0.5 * self.cost_data.weights[0] * self.space_time_product(Z, Z, 'output')
            J2 = 0.5 * self.cost_data.weights[1] * self.space_time_product(W, W, 'control')
            J3 = 0.5 * self.cost_data.weights[2] * self.space_product(ZT, ZT, space_norm = 'output')
            J = J1+J2+J3
        else:
          
            ################
            # quadratic stuff trajectory
            J1_2 = 0.5 * self.cost_data.weights[0] * self.space_time_product(output, output, 'output')
            J1_2 -= self.cost_data.weights[0] * self.space_time_product(output, self.cost_data.Mc_Yd, 'identity') ##change
            J1_2 += 0.5* self.cost_data.weights[0] * self.time_norm_scalar(self.cost_data.Yd_Mc_Yd)
            
            # quadratic stuff end time
            J3_2 = 0.5 * self.cost_data.weights[2] * self.space_product(output[:,-1], output[:,-1], space_norm = 'output')
            J3_2 -= self.cost_data.weights[2] * self.space_product(output[:,-1], self.cost_data.Mc_YT, space_norm = 'identity') ###change
            J3_2 += 0.5 * self.cost_data.weights[2] * self.cost_data.YT_Mc_YT
            
            # quadratic stuff control
            J2_2 = 0.5 * self.cost_data.weights[1] * self.space_time_product(W, W, 'control')
            
            # set together
            J_2 = J1_2+J2_2+J3_2
            J = J_2
       
        if self.isNonSmoothlyRegularized():
            J += self.cost_data.g_reg.weight*self.cost_data.g_reg.g(u)
        
        return J
    
    def J_smooth(self, u, Y = None, output = None):
        g = 0
        if self.isNonSmoothlyRegularized():
            g += self.cost_data.g_reg.weight*self.cost_data.g_reg.g(u)
        return self.J(u, Y = Y, output = output)-g
        
        
    def Jtracking_trajectory(self, u, Y = None, output = None):
        U = self.vector_to_matrix(u, self.input_dim)
        if Y is None or output is None:
           Y, output, time, sigma_profile = self.solve_state(U = U, theta = self.theta)
        W = U - self.cost_data.Ud
        
        # get tracking trajectory
        J1_2 = 0.5 * self.cost_data.weights[0] * self.space_time_product(output, output, 'output', return_trajectory = True)
        J1_2 -= self.cost_data.weights[0] * self.space_time_product(output, self.cost_data.Mc_Yd, 'identity', return_trajectory = True) ##change
        J1_2 += 0.5* self.cost_data.weights[0] * self.cost_data.Yd_Mc_Yd
        
        # get control term trajectory
        J2_2 = 0.5 * self.cost_data.weights[1] * self.space_time_product(W, W, 'control', return_trajectory = True)
        
        if self.isNonSmoothlyRegularized():
            J2_2 += self.cost_data.g_reg.weight*self.cost_data.g_reg.g(u)
        
        return J1_2, J2_2, J1_2 + J2_2
    
    def stage_cost(self, Y, output, U, m1, m2 = None):
        # stage cost from k1 to k2 ...
        if m2 is None:
            m2 = m1+1
        assert 0<= m1 <= m2 <= self.time_disc.K+1
        
        W = (U - self.cost_data.Ud)[:,m1+1:m2+1]
        
        if 0:
            assert self.cost_data.Yd.shape[0] != self.pde.state_dim, 'this is only valid for the ROM if yd is not stat diemsnional'
            Z = (output - self.cost_data.Yd)[:,m1:m2]#[:,m1:m2]
            ZT = output[:,-1]-self.cost_data.YT
            J1 = 0.5 * self.cost_data.weights[0] * self.space_time_product(Z, Z, 'output', time_norm = self.time_disc.D[m1:m2])
        
            J2 = 0.5 * self.cost_data.weights[1] * self.space_time_product(W, W, 'control', time_norm = self.time_disc.D[m1+1:m2+1])
            
            J = J1 + J2
            
            if m2 == self.time_disc.K: # add terminal cost
                J3 = 0.5 * self.cost_data.weights[2]* self.space_product(ZT, ZT, space_norm = 'output') 
                
                J += J3
        
        else:
            
            ################
            out_window = output[:,m1:m2]
            MC_Yd_window = self.cost_data.Mc_Yd[:,m1:m2]
            Yd_Mc_Yd_window = self.cost_data.Yd_Mc_Yd[m1:m2]
            D_window = self.time_disc.D[m1:m2]
            # quadratic stuff trajectory
            J1_2 = 0.5 * self.cost_data.weights[0] * self.space_time_product(out_window, out_window, 'output', time_norm = D_window)
            J1_2 -= self.cost_data.weights[0] * self.space_time_product(out_window, MC_Yd_window , 'identity', time_norm = D_window) 
            J1_2 += 0.5* self.cost_data.weights[0] * self.time_norm_scalar(Yd_Mc_Yd_window, time_norm = D_window)
            
            # quadratic stuff control
            J2_2 = 0.5 * self.cost_data.weights[1] * self.space_time_product(W, W, 'control', time_norm = self.time_disc.D[m1+1:m2+1])
            
            J_2 = J1_2+J2_2
            
            # quadratic stuff end time
            if m2 == self.time_disc.K:
                J3_2 = 0.5 * self.cost_data.weights[2] * self.space_product(output[:,-1], output[:,-1], space_norm = 'output')
                J3_2 -= self.cost_data.weights[2] * self.space_product(output[:,-1], self.cost_data.Mc_YT, space_norm = 'identity') 
                J3_2 += 0.5 * self.cost_data.weights[2] * self.cost_data.YT_Mc_YT
                
                J_2 += J3_2
            J = J_2
            
         
        if self.isNonSmoothlyRegularized():
            u_ = U[:,m1:m2].flatten()
            J2_2 += self.cost_data.g_reg.weight*self.cost_data.g_reg.g(u_)
            
        return J
    
    def gradJ_OBD(self, u, Y = None, return_switch_profil = False, P = None, B_listTP = None):                                         
        U = self.vector_to_matrix(u, self.input_dim) 
        if Y is None:
            Y, output, time, sigma_profile = self.solve_state(U = U, theta = self.theta) 
        
        W = U - self.cost_data.Ud
        if P is None or B_listTP is None:
            Z = self.cost_data.output_product.dot(output) - self.cost_data.Mc_Yd
            ZT = self.cost_data.output_product.dot(output[:,-1]) - self.cost_data.Mc_YT
            P, B_listTP, gaps = self.solve_adjoint(Z, ZT, sigma_profile)
        dJ1 = B_listTP
        dJ2 = self.cost_data.weights[1] * self.cost_data.input_product.dot(W)
        dJ = dJ1 + dJ2
        if not return_switch_profil:
            return dJ.flatten(), output, Y, P
        else:
            return dJ.flatten(), output, Y, P, sigma_profile, B_listTP, gaps
    
    def gradJ_DBO(self):
        pass
    
#%% products
    
    def compute_rietzrepresentant(self, y, norm_type = 'H1dual', switch = None):
        if norm_type == 'H1dual':
            # compute rietzrepresentative 
            if self.isFOM() and self.space_disc.DirichletBC is not None and 'Dirichlet' in self.model_type:
                _, y = self.space_disc.DirichletClearFun(LHS = None, rhs = y)
            y = self.pde.factorizedV1(y)
            return  y
        
        elif norm_type == 'H10dual':
            assert'Dirichlet' in self.model_type, 'this is no norm for this model'
            # compute rietzrepresentative 
            if self.isFOM() and self.space_disc.DirichletBC is not None and 'Dirichlet' in self.model_type:
                _, y = self.space_disc.DirichletClearFun(LHS = None, rhs = y)
            y = self.pde.factorizedV2(y)
            return y
        
        elif norm_type == 'switch_energy':
            
            if self.isFOM() and self.space_disc.DirichletBC is not None and 'Dirichlet' in self.model_type:
                # TODO add clearing of LHS
                assert 0, 'no Dirichlet here'
                _, y = self.space_disc.DirichletClearFun(LHS = None, rhs = y)
            y = self.pde.energy_products.factorized_energy_products[switch](y)
            
            return y
        
    def space_product_trajectory(self, v, w, norm = 'L2'):
        out = []
        for i in range(v.shape[1]):
            out.append(self.space_product(v[:,i], w[:,i], norm))
        return np.array(out).T
    
    def rel_space_norm_trajectory(self, vfom, vrom, norm = 'L2'):
        out = []
        for i in range(vfom.shape[1]):
            out.append(self.space_norm(vfom[:,i]-vrom[:,i], norm))
        return out
    
    def rel_scalar_trajectory(self, vfom, vrom):
        out = []
        for i in range(len(vfom)):
            out.append(abs(vfom[i]-vrom[i]))
        return out
    
    def space_norm_trajectory(self, v, norm = 'L2'):
        out = []
        for i in range(v.shape[1]):
            out.append(self.space_norm(v[:,i], norm))
        return out
    
    def space_norm_trajectory2(self, v, norm = 'L2'):
        return np.sqrt(self.space_time_product(v, v, space_norm = norm, return_trajectory = True))
        
    def time_norm_scalar(self, V, time_norm = None):
        if time_norm is None:
            time_norm = np.concatenate(([0], self.time_disc.D[1:])) 
        return np.vdot(time_norm , V)
    
    def L2_scalar_norm(self, v):
        return np.sqrt(v.T.dot(self.time_disc.D*v))
    
    def space_product(self, y1, y2, space_norm = 'L2', switch = None):
        if space_norm == 'L2':
            return y1.T.dot(self.products['L2'].dot(y2))
        elif space_norm == 'H1':
            return  y1.T.dot(self.products['H1'].dot(y2))
        elif space_norm == 'H10':
            return  y1.T.dot(self.products['H10'].dot(y2))
        elif space_norm == 'output':
            return y1.T.dot(self.cost_data.output_product.dot(y2))
        elif space_norm == 'control': 
            return  y1.T.dot(self.cost_data.input_product.dot(y2))
        elif space_norm == 'identity': 
            return  y1.T.dot(y2)
        elif space_norm == 'H1dual':
            y2 = self.compute_rietzrepresentant(y2, norm_type = space_norm)
            return  y1.T.dot(y2)
        elif space_norm == 'H10dual':
            y2 = self.compute_rietzrepresentant(y2, norm_type = space_norm)
            return  y1.T.dot(y2)
        elif space_norm == 'switch_energy_dual':
            y2 = self.compute_rietzrepresentant(y2, norm_type = 'switch_energy', switch = switch-1)
            return y1.T.dot(y2)
        elif space_norm == 'energy_product':
            return  y1.T.dot(self.pde.products['energy'][switch-1].dot(y2))
        elif space_norm == 'M_switch':
            return  y1.T.dot(self.pde.M[switch-1].dot(y2))
            
    def space_norm(self, y1, space_norm = 'L2', switch = None):
        return np.sqrt(self.space_product(y1, y1,space_norm, switch))
    
    def space_time_product(self, v, w, space_norm = 'L2', time_norm = None, return_trajectory = False, space_mat = None, switch_profile = None):
        if time_norm is None:
            time_norm = np.concatenate(([0], self.time_disc.D[1:]))
        
        if space_mat is not None:
            if len(v.shape)<2:
               v = self.vector_to_matrix(v, self.state_dim)
            if len(w.shape)<2:
               w = self.vector_to_matrix(w, self.state_dim)  
           
            if return_trajectory:
                return np.diag(v.T.dot(space_mat.dot(w)))
            else: 
                return np.vdot(time_norm , np.diag(v.T.dot(space_mat.dot(w))) )
        
        if space_norm == 'L2':
            if len(v.shape)<2:
               v = self.vector_to_matrix(v, self.state_dim)
            if len(w.shape)<2:
               w = self.vector_to_matrix(w, self.state_dim)  
           
            if return_trajectory: # return time traj w.r.t product
                return np.diag(v.T.dot(self.products['L2'].dot(w)))
            else: # space time product
                return np.vdot(time_norm , np.diag(v.T.dot(self.products['L2'].dot(w))) )
        
        elif space_norm == 'H1':
            if len(v.shape)<2:
               v = self.vector_to_matrix(v, self.state_dim)
            if len(w.shape)<2:
               w = self.vector_to_matrix(w, self.state_dim) 
            if return_trajectory: # return time traj w.r.t product
                return np.diag(v.T.dot(self.products['H1'].dot(w)))
            else: # space time product
                return np.vdot(time_norm , np.diag(v.T.dot(self.products['H1'].dot(w))) )
            
        elif space_norm == 'H10':
            if len(v.shape)<2:
               v = self.vector_to_matrix(v, self.state_dim)
            if len(w.shape)<2:
               w = self.vector_to_matrix(w, self.state_dim) 
            if return_trajectory: # return time traj w.r.t product
                return np.diag(v.T.dot(self.products['H10'].dot(w)))
            else: # space time product
                return np.vdot(time_norm , np.diag(v.T.dot(self.products['H10'].dot(w))) )
          
        elif space_norm == 'output':
            if len(v.shape)<2:
               v = self.vector_to_matrix(v, self.output_dim)
            if len(w.shape)<2:
               w = self.vector_to_matrix(w, self.output_dim)  
            if return_trajectory: # return time traj w.r.t product
                return np.diag(v.T.dot(self.cost_data.output_product.dot(w)))
            else: # space time product
                return np.vdot(time_norm , np.diag(v.T.dot(self.cost_data.output_product.dot(w))) )
        
        elif space_norm == 'control': 
            if len(v.shape)<2:
               v = self.vector_to_matrix(v, self.input_dim)
            if len(w.shape)<2:
               w = self.vector_to_matrix(w, self.input_dim)
               
            if return_trajectory: # return time traj w.r.t product
                return np.diag(v.T.dot(self.cost_data.input_product.dot(w)))
            else: # space time product
                return np.vdot(time_norm , np.diag(v.T.dot(self.cost_data.input_product.dot(w))) )
        
        elif space_norm == 'identity': 
            if len(v.shape)<2:
               v = self.vector_to_matrix(v, self.output_dim)
            if len(w.shape)<2:
               w = self.vector_to_matrix(w, self.output_dim) 
               
            if return_trajectory: # return time traj w.r.t product
                return np.diag(v.T.dot(w))
            else: # space time product
                return np.vdot(time_norm , np.diag(v.T.dot(w)))
        
        elif space_norm == 'energy_product':
            if len(v.shape)<2:
               v = self.vector_to_matrix(v, self.state_dim)
            if len(w.shape)<2:
               w = self.vector_to_matrix(w, self.state_dim) 
            out = []
            for i in range(v.shape[1]):
                out.append(self.space_product(v[:,i], w[:,i], 'energy_product', switch = switch_profile[i]))
            if return_trajectory:
                return np.array(out).T
            else: 
                return np.vdot(time_norm , np.array(out).T)
            
        elif space_norm == 'M_switch':
            if len(v.shape)<2:
               v = self.vector_to_matrix(v, self.state_dim)
            if len(w.shape)<2:
               w = self.vector_to_matrix(w, self.state_dim) 
            out = []
            for i in range(v.shape[1]):
                out.append(self.space_product(v[:,i], w[:,i], 'M_switch', switch = switch_profile[i]))
            if return_trajectory:
                return np.array(out).T
            else: 
                return np.vdot(time_norm , np.array(out).T)
        
    def space_time_norm( self, v, space_norm = 'L2', time_norm = None, switch_profile = None):
        return np.sqrt(self.space_time_product(v, v, space_norm, time_norm = time_norm, switch_profile = switch_profile))
    
    def rel_error_norm(self, U1, U2, space_norm = 'L2'):
        return self.space_time_norm(U1-U2, space_norm = space_norm)/self.space_time_norm(U1, space_norm = space_norm)
    
#%% optimization algorithms
    
    def set_default_options(self, tol = 1e-6, maxit = 200, save = False, plot = True, print_info = True):
        
        options = {'print_info': print_info,
                   'print_final': True,
                   'plot': plot,
                   'save': save,
                   'prec': 'id',
                   'path': None,
                   'tol': tol,
                   'maxit': maxit,
                   }
        
        return options
    
    def solve_ocp(self, U_0, method = 'BB', options = None, checkstuff = None, solve_unconstrained = None):
        

        if options is None:
            options = self.set_default_options()
        
        options['solve_unconstrained'] = solve_unconstrained
        u_opt, history = self.solve_BB(U_0.flatten(), options, checkstuff)
        
        return u_opt, history
    
    def solve_BB(self, u_0, options, checkstuff = None):
    
        if options['print_info']:
            print('#############################################################')
            print("Starting BB ")
            if self.isNonSmoothlyRegularized():
                print("Proximal Gradient method ")
            
        #use optimize the discretize 
        BBnorm = lambda x: self.space_time_norm(x, 'control')
        BBproduct = lambda x, y: self.space_time_product(x, y, 'control')
        BBgrad = lambda x, return_switch_profil: self.gradJ_OBD(x, return_switch_profil = return_switch_profil)
        
        if self.isNonSmoothlyRegularized() and not options['solve_unconstrained']:
            BBprox_grad = lambda alpha, u, grad_u: alpha*(u-self.cost_data.g_reg.prox_operator(1/alpha, u-1/alpha*grad_u))
        else:
            BBprox_grad = lambda alpha, u, grad_u: grad_u
        
        
        # initialize
        t = time()
        k = 0
        u_km1 = u_0
        
        grad_km1_uncons, out, Y, P = BBgrad(u_km1, return_switch_profil = False) 
        grad_km1 =  BBprox_grad(1, u_km1, grad_km1_uncons)
        u_k = grad_km1

        grad_k_uncons, out, Y, P, switch_profil, B_listTP, gaps = BBgrad(u_k, return_switch_profil = True)
        grad_k =  BBprox_grad(1, u_k, grad_k_uncons)
        grad_norm = BBnorm(grad_k)
        grad_norm0 = grad_norm
        history = {'grad_norm': [grad_norm],
                   'time_stages': [],
                   'u_list': [u_k]
                    }
        
        if options['print_info']:
            print(f"k: {k:2}, grad_norm = {grad_norm: 2.4e}, rel grad_norm = {grad_norm/grad_norm0: 2.4e}")
            
        # BB loop
        while grad_norm > max(options['tol'],options['tol']*grad_norm0) and k<options['maxit']:
         
            sk = u_k - u_km1
            dk = grad_k - grad_km1
            skdk = BBproduct(sk,dk)
            if  BBnorm(dk) <= max(options['tol'],options['tol']*grad_norm0):
                 _, out, Y, P, switch_profil, B_listTP, gaps = BBgrad(dk, return_switch_profil = True)
                 u_k = dk 
                 break
            if  BBnorm(sk) <= max(options['tol'],options['tol']*grad_norm0):
                 _, out, Y, P, switch_profil, B_listTP,gaps = BBgrad(sk, return_switch_profil = True)
                 u_k = sk 
                 break 
            if k%2==0: 
                alpha_k = BBproduct(dk,dk)/skdk
            else: 
                alpha_k = skdk / BBproduct(sk,sk)
            
            # get search direction depending on alpha
            grad_k =  BBprox_grad(alpha_k, u_k, grad_k_uncons)
            
            # update
            u_km1 = u_k
            u_k = u_k - grad_k/alpha_k
            grad_km1 = grad_k
            
            # compute new gradient and its norm
            grad_k_uncons, out, Y, P, switch_profil, B_listTP, gaps = BBgrad(u_k, return_switch_profil = True)
            grad_k = BBprox_grad(alpha_k, u_k, grad_k_uncons) #G_alpha_k(alph_k+1)
            grad_norm = BBnorm(grad_k)
            
            # update history
            history['grad_norm'].append(grad_norm)
            history['u_list'].append(u_k)
            k += 1
            if options['print_info']:
                print(f"k: {k:2}, grad_norm = {grad_norm: 2.4e}, rel grad_norm = {grad_norm/grad_norm0: 2.4e}")
        
        # finalize
        history['Y_opt'] = Y
        history['P_opt'] = P
        history['B_listTP_opt'] = B_listTP
        history['switch_profil_opt'] = switch_profil
        history['out_opt'] = out
        history['time'] = time() - t 
        history['k'] = k
        history['gaps'] = gaps
        methodd = 'BBgrad'
        if self.isNonSmoothlyRegularized():
            methodd += 'PROX'
            
        if k == options['maxit']:
            history['flag'] = methodd + f' reached maxit of k = {k:2} iterations in {history["time"]: .3f} seconds with gradient norm of {grad_norm:2.4e}, rel grad_norm = {grad_norm/grad_norm0: 2.4e}.'
        else:
            history['flag'] = methodd + f' converged in k = {k:2} iterations in {history["time"]: .3f} seconds with gradient norm of {grad_norm:2.4e}, rel grad_norm = {grad_norm/grad_norm0: 2.4e}.'
        # history['flag'] += 
            
        if options['print_final']:
            print( history['flag'])
            print('#############################################################')            
        if options['plot']:
            plt.figure()
            plt.semilogy(history['grad_norm'])
            plt.title(r'BB convergence of $\|\nabla F(u_k)\|_U$')
            plt.xlabel(r'$k$')
            if options['save']:
                plt.savefig( options['path'] )
                
        U_opt = self.vector_to_matrix(u_k, self.input_dim)
        history['U_opt'] = U_opt
        return U_opt, history

#%% error estimation

    def state_est(self, U, k = None, Y = None, Yr = None, switching_profile = None, norm_type = 'LinfH', computationtype = 'online', return_res = False):
        
        state_error_est = self.error_estimator['state']
        
        if computationtype == 'online':
            est, Res_squarednorm = state_error_est.est_online(U, k, Yr, switching_profile, norm_type)
        elif computationtype == 'offline_online':
            est, Res_squarednorm = state_error_est.offline_online_est(U, k, Yr, switching_profile, norm_type)
        elif computationtype == 'true':
            est, Res_squarednorm = state_error_est.est_true(U, Yr, Y, type_ = norm_type, k = k, switching_profile = switching_profile)
        
        if return_res:
            return est, Res_squarednorm
        else:
            return est
    
    def adjoint_est(self, U = None, Yr = None, Y = None, Pr = None , P = None, type_ = 'LinfH',
                 k = None, switching_profile_r = None, switching_profile = None, 
                 out = None, out_r = None, state_H_est_list = None, computationtype = 'online', option_adjoint_with_state = False, gaps_r = None):
        
        adjoint_error_est = self.error_estimator['adjoint']
        
        if computationtype == 'online':
            est = adjoint_error_est.est_online(U = U, Yr = Yr, Pr = Pr, type_ = type_,
                         k = k, switching_profile_r = switching_profile_r,  out_r = out_r, state_H_est_list = state_H_est_list, option_adjoint_with_state = option_adjoint_with_state, gaps_r = gaps_r)
        elif computationtype == 'offline_online':
            est = adjoint_error_est.offline_online_est(U = U, Yr = Yr, Pr = Pr, type_ = type_,
                         k = k, switching_profile_r = switching_profile_r,  out_r = out_r, state_H_est_list = state_H_est_list, option_adjoint_with_state = option_adjoint_with_state, gaps_r = gaps_r)
        elif computationtype == 'true':
            est = adjoint_error_est.est_true(U = U, Yr = Yr, Y = Y, Pr = Pr , P = P, type_ = type_,
                         k = k, 
                         switching_profile_r = switching_profile_r, switching_profile = switching_profile, 
                         out = out, out_r = out_r)
        else:
            assert 0, 'wrong input...'
        return est
        
    def cheap_optimal_control_error_est(self, U = None, Yr = None, Pr = None , type_ = 'new_split_up_cheap', k = None, switching_profile_r = None, out_r = None, state_H_est_list = None, computationtype = 'online', Bound_init_n = 0, return_init_bound = False, gaps_r = None):
    
        weight = 1
        type_state_adjoint = 'L2V_list'
        
        ### A':
        if type_ == 'new_cheap':
        
            # get L2V state bound
            list_state, Res_squarednorm = self.state_est(U = U, k = None, Y = None, 
                                        Yr = Yr, switching_profile = switching_profile_r, 
                                        norm_type = type_state_adjoint, 
                                        computationtype = computationtype,
                                        return_res = True)
                      
            # get L2V adjoint bound with state influence
            list_adjoint_pure = self.adjoint_est(U = U, Yr = Yr, Y = None, Pr = Pr , P = None, type_ = type_state_adjoint,
                         k = None, switching_profile_r = switching_profile_r, switching_profile = None, 
                         out = None, out_r = out_r, 
                         state_H_est_list = list_state, computationtype = computationtype, option_adjoint_with_state = True, gaps_r=gaps_r)
            
            # adjoint part
            adjoint_l2_V_bound = list_adjoint_pure[0]**2
            c_adjoint = max(self.pde.error_est_constants['BcontA'])**2/self.cost_data.weights[1]**2
            
            # assemble parts
            adjoint_part = c_adjoint*adjoint_l2_V_bound
            
            # est
            est = np.sqrt(adjoint_part)
            
            # with init
            est_coll = collection()
            
            # # compute control bound with init
            cons_squared = max((max(self.pde.error_est_constants['CcontA'])**2)/2, self.cost_data.weights[-1]*max(self.pde.error_est_constants['CcontM'])**2)
            initial_previous_part = Bound_init_n**2*cons_squared/(2*self.cost_data.weights[1])
            est_coll.est_with_init = np.sqrt(weight*adjoint_part + initial_previous_part)
            
            # output est
            adjoint_part_out = adjoint_l2_V_bound*max(self.pde.error_est_constants['BcontA'])**2/(self.cost_data.weights[1]*2)
            est_coll.est_output_current = np.sqrt(weight*adjoint_part_out)
            initial_previous_part = Bound_init_n**2*cons_squared/2
            est_coll.est_with_init_output = np.sqrt(weight*adjoint_part_out + initial_previous_part)   
            
            # return state res
            est_coll.Res_squarednorm = Res_squarednorm
            
            compare = False
            if compare:
                
                # get L2V state bound
                list_state, Res_squarednorm = self.state_est(U = U, k = None, Y = None, 
                                            Yr = Yr, switching_profile = switching_profile_r, 
                                            norm_type = type_state_adjoint, 
                                            computationtype = computationtype,
                                            return_res = True)
                zero_state_list = [0*i for i in list_state]
                
                # get L2V adjoint bound without state influence
                list_adjoint_pure = self.adjoint_est(U = U, Yr = Yr, Y = None, Pr = Pr , P = None, type_ = type_state_adjoint,
                             k = None, switching_profile_r = switching_profile_r, switching_profile = None, 
                             out = None, out_r = out_r, 
                             state_H_est_list = zero_state_list, computationtype = computationtype)
                
                # terminal part
                assert self.cost_data.weights[-1] == 0 , 'MODIFY THIS BOUND HERE to account for terminal costs! ... '
                terminal_part_ = 0
                c_terminal_state_ = 0* max(self.pde.error_est_constants['CcontM'])**2*self.cost_data.weights[-1]/self.cost_data.weights[1]
                
                # state part
                state_l2_H_bound_ = list_state[-1]**2
                c_state_ = max(self.pde.error_est_constants['CcontA'])**2/self.cost_data.weights[1]
                c_state_ = max(c_state_,c_terminal_state_)
                
                # adjoint part
                adjoint_l2_V_bound_ = list_adjoint_pure[0]**2
                c_adjoint_ = max(self.pde.error_est_constants['BcontA'])**2/self.cost_data.weights[1]**2
                
                # assemble parts
                state_part_ = weight*c_state_*state_l2_H_bound_
                adjoint_part_ = weight*c_adjoint_*adjoint_l2_V_bound_
                
                est_ = np.sqrt(state_part_ + adjoint_part_)
                
                # control with init
                est_coll_DEBUG = collection()
                c_const_ = max((max(self.pde.error_est_constants['CcontA'])**2)/2, self.cost_data.weights[-1]*max(self.pde.error_est_constants['CcontM'])**2)
                cons_squared_ = c_const_/self.cost_data.weights[1]
                initial_previous_part_ = Bound_init_n**2*cons_squared_
                est_coll_DEBUG.est_with_init =  np.sqrt(weight*state_part_ + weight*adjoint_part_ + initial_previous_part_)
                
                # return state res
                est_coll_DEBUG.Res_squarednorm = Res_squarednorm
                
                print(f'Test init bound B cheap: {adjoint_part_} {state_part_} {initial_previous_part_}')
                print(f'Test init bound A cheap: {adjoint_part} {0} {initial_previous_part}')
            
        elif type_ == 'new_split_up_cheap':
            
            # get L2V state bound
            list_state, Res_squarednorm = self.state_est(U = U, k = None, Y = None, 
                                        Yr = Yr, switching_profile = switching_profile_r, 
                                        norm_type = type_state_adjoint, 
                                        computationtype = computationtype,
                                        return_res = True)
            zero_state_list = [0*i for i in list_state]
            
            # get L2V adjoint bound wöithout state influence
            list_adjoint_pure = self.adjoint_est(U = U, Yr = Yr, Y = None, Pr = Pr , P = None, type_ = type_state_adjoint,
                         k = None, switching_profile_r = switching_profile_r, switching_profile = None, 
                         out = None, out_r = out_r, 
                         state_H_est_list = zero_state_list, computationtype = computationtype, gaps_r=  gaps_r)
            
            # terminal part
            assert self.cost_data.weights[-1] == 0 , 'MODIFY THIS BOUND HERE to account for terminal costs! ... '
            terminal_part = 0
            c_terminal_state = 0* max(self.pde.error_est_constants['CcontM'])**2*self.cost_data.weights[-1]/self.cost_data.weights[1]
            
            # state part
            state_l2_H_bound = list_state[-1]**2
            c_state = max(self.pde.error_est_constants['CcontA'])**2/self.cost_data.weights[1]
            c_state = max(c_state,c_terminal_state)
            
            # adjoint part
            adjoint_l2_V_bound = list_adjoint_pure[0]**2
            c_adjoint = max(self.pde.error_est_constants['BcontA'])**2/self.cost_data.weights[1]**2
            
            # assemble parts
            state_part = weight*c_state*state_l2_H_bound
            adjoint_part = weight*c_adjoint*adjoint_l2_V_bound
            
            est = np.sqrt(state_part + adjoint_part)
            
            # control with init
            est_coll = collection()
            c_const = max((max(self.pde.error_est_constants['CcontA'])**2)/2, self.cost_data.weights[-1]*max(self.pde.error_est_constants['CcontM'])**2)
            cons_squared = c_const/self.cost_data.weights[1]
            initial_previous_part = Bound_init_n**2*cons_squared
            est_coll.est_with_init =  np.sqrt(state_part + adjoint_part + initial_previous_part)
            
            # output bounds
            assert self.cost_data.weights[-1] == 0 , 'MODIFY THIS BOUND HERE to account for terminal costs! ... '
            terminal_part = 0
            initial_previous_part = c_const*Bound_init_n**2
            B_part = max(self.pde.error_est_constants['BcontA'])**2/self.cost_data.weights[1] * adjoint_l2_V_bound
            state_part = 2*max(self.pde.error_est_constants['CcontA'])**2*state_l2_H_bound
            est_coll.est_output_current =  np.sqrt(weight*B_part + weight*state_part + terminal_part )
            est_coll.est_with_init_output = np.sqrt(weight*B_part + weight*state_part + terminal_part+  initial_previous_part )
            
            # return state res
            est_coll.Res_squarednorm = Res_squarednorm
            
        else:
            assert 0, 'Wrong type given...'
            
        return est, est_coll
            
    def optimal_control_est(self, Ur, Y= None, P = None, switching_profile= None, B_listTP = None, out = None, Pr = None, B_listTPr = None, Yr = None, switch_profile_r = None, out_r = None, type_ = 'new', Bound_init_n = 0, return_init_bound = False):

        rest_est = collection()
        
        if type_ == 'perturbation_standard': # unconstrained error est (perturbation/gradient residual)
            
            # if control is reduced reconstruct optimal control
            if Y is not None and P is not None and B_listTP is not None:
                dJ, output, Y, P, gaps = self.gradJ_OBD(Ur, Y = Y, P = P, B_listTP = B_listTP)
                # grad_norm = self.space_time_norm(dJ, 'control')
                # est = grad_norm/self.cost_data.weights[1]
                # return est, Y, P, None
            else:
                # solve forward and backward and evaluate gard norm
                dJ, output, Y, P, gaps = self.gradJ_OBD(Ur)
            
            if self.cost_data.g_reg.type == 'box':
                # project on active and inactive sets
                xi = self.project_on_active_inactive_sets(dJ, Ur)
            else:
                xi = dJ
            
            # compute bound
            norm = self.space_time_norm(xi, 'control')
            est = norm/self.cost_data.weights[1]
            assert return_init_bound is False, 'modify this compute here init bound and output bound also ...'
            
        if type_ == 'new':
            if P is None or  B_listTP is None:
                dJ, output, Y, P, profile, B_listTP, gaps = self.gradJ_OBD(Ur, Y = Y, P = P, B_listTP = B_listTP, return_switch_profil=True)
            BTp_BTpr = B_listTP - B_listTPr
            norm = self.space_time_norm(BTp_BTpr, 'control')
            
            # control
            c_const = max((max(self.pde.error_est_constants['CcontA'])**2)/2, self.cost_data.weights[-1]*max(self.pde.error_est_constants['CcontM'])**2)
            cons_squared = c_const/(2*self.cost_data.weights[1])
            initial_previous_part = Bound_init_n**2*cons_squared
            
            B_part = norm**2/self.cost_data.weights[1]**2  
            
            rest_est.est_with_init = np.sqrt(B_part + initial_previous_part)
            est = norm/self.cost_data.weights[1]
            
            # compute output bound
            cons_squared = max((max(self.pde.error_est_constants['CcontA'])**2)/2, self.cost_data.weights[-1]*max(self.pde.error_est_constants['CcontM'])**2)
            initial_previous_part = Bound_init_n**2*cons_squared
            B_part = norm**2/(2*self.cost_data.weights[1])
            
            rest_est.est_with_init_output = np.sqrt(B_part + initial_previous_part)
            rest_est.est_output_current = np.sqrt(B_part)
            
            rest_est.Res_squarednorm = None
            
        if type_ == 'new_split_up':
        
            # compute Y(ur)
            if Y is None or out is None:
                Y, out, _ , switch_profile = self.solve_state(Ur)
            CY_CYr = out_r- out
            
            # compute P(yr)
            if P is None or B_listTP is None:
                Z = self.cost_data.output_product.dot(out_r) - self.cost_data.Mc_Yd
                ZT = self.cost_data.output_product.dot(out_r[:,-1]) - self.cost_data.Mc_YT
                P, B_listTP, gaps = self.solve_adjoint(Z, ZT, switch_profile_r)
            BTp_BTpr = B_listTPr - B_listTP
            
            assert self.cost_data.weights[-1] == 0 , 'MODIFY THIS BOUND HERE to account for terminal costs! ... '
            terminal_part_sqaured = (self.cost_data.weights[-1]/self.cost_data.weights[1]*0)**2
            adjoint_part_squared = (1/self.cost_data.weights[1] * self.space_time_norm(BTp_BTpr, 'control'))**2
            state_part_squared = 1/self.cost_data.weights[1]*self.space_time_norm(CY_CYr, 'output')**2
            
            c_const = max((max(self.pde.error_est_constants['CcontA'])**2)/2, self.cost_data.weights[-1]*max(self.pde.error_est_constants['CcontM'])**2)
            cons_squared = c_const/self.cost_data.weights[1]
            initial_previous_part = Bound_init_n**2*cons_squared
            rest_est.est_with_init = np.sqrt(adjoint_part_squared + state_part_squared + terminal_part_sqaured+ initial_previous_part)
            est = np.sqrt(adjoint_part_squared + state_part_squared + terminal_part_sqaured)
            
            # TODO output bound ...
            # add output current bound and output init bound
            initial_previous_part = c_const*Bound_init_n**2
            B_part = 1/self.cost_data.weights[1] * (self.space_time_norm(BTp_BTpr, 'control'))**2
            state_part = 2*self.space_time_norm(CY_CYr, 'output')**2
            assert self.cost_data.weights[-1] == 0 , 'MODIFY THIS BOUND HERE to account for terminal costs! ... '
            terminal_part = 0
            rest_est.est_with_init_output = np.sqrt(B_part + state_part + terminal_part+  initial_previous_part )
            rest_est.est_output_current = np.sqrt(B_part + state_part + terminal_part )
            
            rest_est.Res_squarednorm = None
            
        if return_init_bound:
            return est, Y, P, rest_est
        else:
            return est, Y, P, None
    
    # error est for control with initial guess perturbation for FOM ROM
    def state_control_perturbation(self, rho = None, control_bound = None, init_bound = None, switch_profile = None, kf = None, mpc_type = 'FOMROM', ResStateDualNormSquared = None):
        if rho is None:
            rho, info = self.compute_rhos(switch_profile)
            
            # get switchintervals of kf, do we switch up to kf?
            for i in range(len(info.interval_inds)):
                int_ = info.interval_inds[i]
                if kf <= int_[1] and kf>= int_[0]:
                    index_kf = i
            # get rho depending on kf
            rho_kf =  rho[index_kf]
        b_const = max(self.pde.error_est_constants['BcontA'])
        maxrho = max(rho)
        # print(f'max rho kf: {maxrho}')
        if mpc_type == 'FOMROM':
            init_bound_new = np.sqrt(rho_kf*init_bound**2 + maxrho*0.5*b_const**2*control_bound**2) 
        elif mpc_type == 'ROMROM':
            # compute integral up to kf
            RES_state_up_to_kf = self.time_disc.dt*sum(info.rho_time_mat[1:kf+1]*ResStateDualNormSquared[1:kf+1])
            init_bound_new = np.sqrt(rho_kf*init_bound**2 + maxrho*b_const**2*control_bound**2 + RES_state_up_to_kf)
            
        else:
            assert 0, 'MPC option not valid ...'
        
        return init_bound_new
    
    def compute_rhos(self, switch_profile, state = True):
        info = self.get_switch_intervall_info(switch_profile)
       
        if 1:
            if state:
                rho = [1]
                rho_k = 1
                for k in range(1,info.number_intervals):
                    cij = self.pde.error_est_constants['cij']
                    rho_k *= cij[info.switch_in_interval[k]-1, info.switch_in_interval[k-1]-1]
                    rho.append(rho_k)
            else:
                rho = [1]
                rho_k = 1
                for k in range(1,info.number_intervals):
                    cij = 2*self.pde.error_est_constants['cij']
                    rho_k *= cij[info.switch_in_interval[k]-1, info.switch_in_interval[k-1]-1]
                    rho.append(rho_k)
                rho.reverse()
        else:
            rho = np.ones((info.number_intervals,))
            
        info.rho_time_mat = []
        switch_end = info.switchpoints_index_extended
        for i in range(len(rho)):
             info.rho_time_mat.extend([rho[i]] * (switch_end[i+1] - switch_end[i]))
        info.rho_time_mat = np.array(info.rho_time_mat)   
        return rho, info
    
    def get_switch_intervall_info(self, switch_profile):
        assert switch_profile is not None
        kp = len(switch_profile)
        # get number of switching intervals
        number_intervals = 1
        switchpoints_index_extended = [0]
        switch_in_interval = [switch_profile[0]]
        for i in range(1,len(switch_profile)):
            if switch_profile[i] != switch_profile[i-1]:
                switch_in_interval.append(switch_profile[i])
                number_intervals += 1
                switchpoints_index_extended.append(i)
        switchpoints_index_extended.append(kp)
        switchpoints_index = switchpoints_index_extended[1:-1]
        len_switch_intervals = []
        for i in range(1,len(switchpoints_index_extended)):
            len_switch_intervals.append(switchpoints_index_extended[i]-switchpoints_index_extended[i-1])
        
        interval_inds = []
        left = 0
        for i in range(len(len_switch_intervals)):
            right = left+len_switch_intervals[i]-1
            int_ = [left, right]
            interval_inds.append(int_)
            left = right+1
            
        info = collection()
        info.number_intervals = number_intervals
        info.switchpoints_index = switchpoints_index
        info.switchpoints_index_extended = switchpoints_index_extended
        info.len_switch_intervals = len_switch_intervals
        info.switch_in_interval = switch_in_interval
        info.interval_inds = interval_inds
        return info
        
    def project_on_active_inactive_sets(self, xi, u):
        u = u.flatten()
        lb = self.cost_data.g_reg.u_low
        ub = self.cost_data.g_reg.u_up
        out = -1*xi
        for i in range(len(u)):
            if u[i] == lb:
                out[i] = -min(0, xi[i])
            elif u[i] == ub:
                out[i] = -max(0, xi[i])
            else:
                pass
        return out
    
    def compare_state_adjoint(self):
        pass
        
#%% plot

    def visualize_outputs(self, outputs, only_room2 = True, title = None, labels = None):
        plt.figure()
        if title is not None:
            plt.title(title)
        for outs, label in zip(outputs, labels):
            if not only_room2:
                plt.plot(self.time_disc.t_v, outs[0,:], label = label+' Room 1' )
            if 'FOM' in label:
                plt.plot(self.time_disc.t_v, outs[1,:], '--', label = label+' Room 2' )
            else:
                plt.plot(self.time_disc.t_v, outs[1,:], label = label+' Room 2' )
        plt.legend()
        plt.xlabel(r'$t$')
        plt.show()
    
    def visualize_output(self, output, only_room2 = True, title = None, semi = False):
         plt.figure()
         if title is not None:
             plt.title(title)
         if not semi:
             if not only_room2:
                 plt.plot(self.time_disc.t_v, output[0,:], label = 'Room 1' )
             plt.plot(self.time_disc.t_v, output[1,:], label = 'Room 2' )
         else:
             if not only_room2:
                 plt.semilogy(self.time_disc.t_v, output[0,:], label = 'Room 1' )
             plt.semilogy(self.time_disc.t_v, output[1,:], label = 'Room 2' )
         plt.legend()
         plt.xlabel(r'$t$')
         plt.show()
         
    def visualize_1d(self, output, title = None, semi = False, time = None):
         plt.figure()
         if time is None:
             timeint = self.time_disc.t_v
         else:
             timeint = self.time_disc.t_v[:time]
             
         if title is not None:
             plt.title(title)
             
         if semi:
             plt.semilogy(timeint, output)
         else:
             plt.plot(timeint, output)
         # plt.legend()
         plt.xlabel(r'$t$')
         plt.show()
         
    def visualize_1d_many(self, outputs, strings, title = None, semi = False, time = None):
         assert len(outputs) == len(strings), 'this has to be the same'
         plt.figure()
         if time is None:
             timeint = self.time_disc.t_v
         else:
             timeint = self.time_disc.t_v[:time]
             
         if title is not None:
             plt.title(title)
             
         for i in range(len(outputs)):
             if semi:
                 plt.semilogy(timeint, outputs[i], label = strings[i])
             else:
                 plt.plot(timeint,  outputs[i], label = strings[i])
                 
         plt.legend()
         plt.xlabel(r'$t$')
         plt.show()
     
    def visualize_trajectory(self, Y):
         for k in range( 0, self.time_disc.K, int(self.time_disc.K/10) ):
             yy = Y[:,k]
             t = self.time_disc.t_v[k]
             switch = self.pde.sigma(t, yy)
             self.plot_3d(yy, title = f"t = {t:2.3f}, switch = {switch}")
             
    def fenics_plot_solution(self, y, title='' ):
        yf = fenics.Function(self.space_disc.V)
        yf.vector()[:] = y
        
        plt.figure()
        # c =fenics.plot(yf, title=title, mode='color', vmin=-3, vmax=3)
        c =fenics.plot(yf, title=title, mode='color')
        plt.colorbar(c)
        
    def plot_3d(self, y, title=None, save_png=False, path=None, dpi='figure'):
        
        # read model
        Nx = self.space_disc.Nx
        Ny = self.space_disc.Ny
        mesh = self.space_disc.mesh
        V = self.space_disc.V
        
        # get plot data
        dims = ( Ny+1 , Nx+1 )
        X = np.reshape( mesh.coordinates()[:,0], dims )
        Y = np.reshape( mesh.coordinates()[:,1], dims )
        Z = np.reshape( y[fenics.vertex_to_dof_map(V)], dims )
        
        # plot
        fig = plt.figure()
        ax = fig.add_subplot( projection='3d' )
        if title is not None:
            ax.set_title(title)
        surf = ax.plot_surface( X, Y, Z, cmap=plt.cm.coolwarm )
        fig.colorbar(surf, shrink=0.5, aspect=5)
        # ax.view_init(elev=90, azim=-90, roll=0)
        if save_png:
            plt.savefig( path, dpi=dpi )
        plt.figure()
        
#%% helpers
    def update_parameters(self, **kwargs):
        for k, v in kwargs.items():
            if k in self.__dict__ and v is not None:
                setattr(self, k, v)
         
    def matrix_to_vector(self, V ):
         return V.flatten()
     
    def vector_to_matrix(self, v, dim ):
         return v.reshape(dim, self.time_disc.K)
     
    def derivative_check(self, mode = 1, unconstrained = True):
        print(f'derivative check for unconstrained={unconstrained} cost fun .......')
        
        if unconstrained:
            f = self.J_smooth
            df = self.gradJ_OBD
        else:
            f = self.J
            assert 0, 'whats the nonsmooth grad here? prox operator?...'
            df = None
        Eps = np.array([1,1e-1,1e-2,1e-3,1e-4,1e-5,1e-6])
        u  = np.random.random((self.input_dim, self.time_disc.K))
        du = np.random.random((self.input_dim, self.time_disc.K))
        T = np.zeros(np.shape(Eps))
        T2 = T
        ff = f(u)
        
        # Compute central & right-side difference quotient
        for i in range(len(Eps)):
            #print(Eps[i])
            f_plus = f(u+Eps[i]*du)
            f_minus = f(u-Eps[i]*du)
            if mode == 1:
                ddd = self.space_time_product(df(u)[0], du, 'control')
                T[i] = abs( ( (f_plus - f_minus)/(2*Eps[i]) ) - ddd )
                T2[i] =  abs( ( (f_plus - ff)/(Eps[i]) ) - ddd )
            else:
                T[i] = abs( ( (f_plus - f_minus)/(2*Eps[i]) ) - df(u,du) )
                T2[i] =  abs( ( (f_plus - ff)/(Eps[i]) ) - df(u,du) )
            
        #Plot
        # plt.figure()
        # plt.xlabel('$eps$')
        # plt.ylabel('$J$')
        # plt.loglog(Eps, Eps**2, label='O(eps^2)')
        # plt.loglog(Eps, T,'ro--', label='Test')
        # plt.legend(loc='upper left')
        # plt.grid()
        # plt.title("Central difference quotient")
        plt.figure()
        plt.xlabel('$eps$')
        plt.ylabel('$J$')
        plt.loglog(Eps, Eps, label='O(eps)')
        plt.loglog(Eps, T2, 'ro--',label='Test')
        plt.legend(loc='upper left')
        plt.grid()
        plt.title("Rightside difference quotient")
        # print(T)
        # print(Eps)
        # print(Eps**2)
    
#%% write read csv

    def write_dict_csv(data_dict, name):
       with open(name, 'w', newline='') as file:
           # Get the fieldnames from the keys of the first dictionary
           fieldnames = data_dict[0].keys()
           
           # Create a DictWriter object
           writer = csv.DictWriter(file, fieldnames=fieldnames)
           
           # Write the header
           writer.writeheader()
           
           # Write the data rows
           writer.writerows(data_dict)
   
    def write_list_csv(data_list, name):
        with open(name, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(data_list)

    def plot_to_csv(self, ax, name):
        # Extract data from the plot
        data = {}
        for i, line in enumerate(ax.get_lines()):
            data[f'x{i+1}'] = line.get_xdata()
            data[f'y{i+1}'] = line.get_ydata()

        # Convert the data to a DataFrame
        df = pd.DataFrame(data)

        # Save the DataFrame to a CSV file
        df.to_csv(name, index=False)
        
    def to_csv(self, data, name):
        df = pd.DataFrame({
                'x': data
                })
        df.to_csv(name, index=False)

        