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
# Description: this file contains the model reduction classes.

import scipy.sparse as sps
from scipy import linalg
from scipy.sparse.linalg import spsolve
import numpy as np
from model import model
from methods import collection
import matplotlib.pyplot as plt
from time import perf_counter
from pymor.algorithms.gram_schmidt import gram_schmidt
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.vectorarrays.numpy import NumpyVectorSpace

def project_model(model, U, V = None, product = None, H_prod = None):
    '''
    
    Parameters
    ----------
    model_ : TYPE
        DESCRIPTION.
    U : r x N
        Left projection matrix
    V : r x N, optional
        Right Projection matrix

    Returns
    -------
    projected_pde : TYPE
        DESCRIPTION.

    '''
    
    # init
    pde, cost = model.pde, model.cost_data
    if V is None:
        V = U
    
    # pde
    projected_pde = collection()
    if model.isSwitchModel():
        projected_pde.type = 'SwitchROM'
        projected_pde.sigma = pde.sigma
        if 'StateDep' in model.pde.type:
            projected_pde.type += 'StateDep'
        projected_pde.error_est_constants = pde.error_est_constants
        
    elif model.isTimeVaryingModel():
        projected_pde.type = 'TimeVaryingROM'
        projected_pde.A_time_coefficient = pde.A_time_coefficient
        projected_pde.M_time_coefficient = pde.M_time_coefficient
        projected_pde.B_time_coefficient = pde.B_time_coefficient
        projected_pde.C_time_coefficient = pde.C_time_coefficient

    if model.isNonSmoothlyRegularized():
        projected_pde.type += 'nonsmooth_g'
        
    projected_pde.A = []
    for AA in pde.A:
        projected_pde.A.append(U.T@(AA.dot(V)))
    projected_pde.M = []
    for MM in pde.M:
        projected_pde.M.append(U.T@(MM.dot(V)))
    projected_pde.B = []    
    
    for BB in pde.B:
        projected_pde.B.append(U.T@BB)    
        
    projected_pde.C = []    
    for CC in pde.C:
        if CC.shape[0] == CC.shape[1] == pde.state_dim: 
            assert CC.trace() == pde.state_dim, 'C ist not identity and it has to be projected'
            CC_proj = np.eye(np.shape(projected_pde.A[0])[0])
            projected_pde.C.append(CC_proj)
            
        else:
            projected_pde.C.append(CC@V)
            
    projected_pde.products = {}    
    for PP in pde.products.keys():
        
        MAT = pde.products[PP]
        if type(MAT) is list:
            projected_pde.products[PP] = []
            for iM in MAT:
                projected_pde.products[PP].append( (V.T@(iM.dot(V))) )
                
        else:
            projected_pde.products[PP] = (V.T@(MAT.dot(V))) 
        
    projected_pde.F = U.T@(pde.F)
    if product is not None:
        projected_pde.y0 =  U.T@product@(pde.y0) 
    else:
        projected_pde.y0 =  U.T@(pde.y0) 
    
    projected_pde.state_dim = np.shape(projected_pde.A[0])[0]
    projected_pde.input_dim = pde.input_dim
    
    # project cost
    projected_cost = collection()
    projected_cost.Ud = cost.Ud
    projected_cost.weights = cost.weights
    projected_cost.input_product = cost.input_product
    projected_cost.g_reg = cost.g_reg
    
    assert cost.Ud.shape[0] != pde.state_dim, 'in this case Ud etc have to be projected also pde.input_dim and input product has to eb modified ...'
    
    if cost.Yd.shape[0] == pde.state_dim: 
        projected_cost.Yd = None #V.T@cost.Yd
        projected_cost.YT = None #V.T@cost.YT
        projected_cost.Mc_Yd = V.T@cost.Mc_Yd
        projected_cost.Mc_YT = V.T@cost.Mc_YT
        projected_cost.output_product = V.T@(cost.output_product.dot(V))
        
        projected_cost.Yd_Mc_Yd = cost.Yd_Mc_Yd
        projected_cost.YT_Mc_YT = cost.YT_Mc_YT
        
        projected_pde.output_dim = np.shape(projected_pde.C[0])[0]
        
    else:
        # maybe copy
        projected_cost.Yd = cost.Yd
        projected_cost.YT = cost.YT
        projected_cost.Mc_Yd = cost.Mc_Yd
        projected_cost.Mc_YT = cost.Mc_YT
        projected_cost.output_product = cost.output_product
        
        projected_cost.Yd_Mc_Yd = cost.Yd_Mc_Yd
        projected_cost.YT_Mc_YT = cost.YT_Mc_YT
        
        projected_pde.output_dim = pde.output_dim
        
    return projected_pde, projected_cost

#%% POD

class pod_reductor():
     
    def __init__(self, model, model_toproject = None, H_prod = None, space_product = None, errorest_assembled = True, est_options = None):
        self.space_product = space_product
        self.old_data = None
        self.est_options = est_options
        self.errorest_assembled = errorest_assembled
        self.update_type = 'redo_svd'
        self.incremental_tol = 1e-14
        self.truncation_tol = 1e-14
        
        if space_product is not None:
            start_time = perf_counter()
            self.Wchol = linalg.cholesky(space_product.todense())
            end_time = perf_counter()
            print(f'Cholesky {end_time-start_time}')
            
        else:
            self.Wchol = None
        
        self.total_energy = None
        
        self.model = model
        if model_toproject is None:
            self.model_toproject = model
        else:
            self.model_toproject = model_toproject
            
        if H_prod is not None:
            self.H_prod = H_prod
        
        # track snapshots data
        self.Snapshots = []
        self.snapshot_energy = []
        self.Ds = []
        self.XisMax = []
        self.XisMin = []
    
    def sqrt_mat(self, W):
        # calculate W^(1/2), check if W is a diagonal matrix
        if 1:
            A,B = linalg.schur(W.todense())
            A = np.diag(np.sqrt(np.diag(A)))
            Whalf = B.dot(A.dot(B.T))
        return Whalf
    
    def assemble_error_est(self):
        pass
    
    
    def ROMtoFOM(self, u, adj = False):
        return self.V_right@u
    
    def FOMtoROM(self, U):
        # return self.POD_Basis.T@self.space_product@U
        if self.projection_product is None:
            return self.U_left.T@U
        else:
            return self.U_left.T@ self.projection_product@U
    
    def check_orthogonality(self):
        print(self.POD_Basis.T@self.space_product@self.POD_Basis)
    
    def incremental_svd_update(self, Snapshots, init = False):
        
        print('ROM POD constructing ...')
        start_time = perf_counter()
        
        if 0: 
            for snapshot in Snapshots:
                
                idx = np.argwhere(np.all(snapshot[..., :] == 0, axis=0))
                snapshot_zero = np.delete(snapshot, idx, axis=1)
                
                for ind in range(snapshot_zero.shape[1]):
                    
                    c = snapshot_zero[:, ind]
                    
                    if init and ind == 0: 
                        # init SVD
                        val = np.sqrt(c.T@self.space_product@c)
                        self.Singular_values = val
                        self.POD_Basis = c/val
                        self.POD_Basis = self.POD_Basis.reshape((c.shape[0],1))
                        self.POD_values = self.Singular_values**2
                        self.U_left = self.POD_Basis
                        self.V_right = self.POD_Basis
                        init = False
                        
                    else:
                        # project into POD space and measure how good the snapshot is already approximated
                        d = self.FOMtoROM(c)
                        diff = c-self.ROMtoFOM(d)
                        p = np.sqrt(diff.T@self.space_product@diff)
                        
                        if p < self.incremental_tol:
                            p = 0
                        if isinstance(self.Singular_values, float):
                            Q = np.array([[self.Singular_values]])
                            d = np.array([d])
                            Q = np.append(Q, d, axis = 1)
                            new_row = np.zeros((1, Q.shape[1])); new_row[0,-1] = p
                            Q = np.append(Q, new_row, axis = 0)
                        else:
                            Q = np.diag(self.Singular_values)
                            Q = np.hstack((Q, np.array([d]).T))
                            new_row = np.zeros((1, Q.shape[1])); new_row[0,-1] = p 
                            Q = np.vstack((Q, new_row))
                        
                        # perform svd
                        U, S, V = linalg.svd(Q, full_matrices=False)
                        indices = S > self.truncation_tol
                        S = S[indices]
                        U = U[:,indices]
                        
                        # extend svd
                        if isinstance(self.Singular_values, float):
                           k = 1
                        else:
                            k = self.Singular_values.shape[0]
                            
                        if p < self.incremental_tol or k >= self.POD_Basis.shape[0]:
                            self.POD_Basis = self.POD_Basis @ U[:k,:k]
                            self.Singular_values = S[:k]
                         
                        else:
                            j = diff/p
                            self.POD_Basis = np.append(self.POD_Basis, np.array([j]).T, axis = 1)
                            self.POD_Basis = self.POD_Basis @ U
                            self.Singular_values = S
                        
                        # orthonormalize if necessary
                        if abs(self.POD_Basis[:,0].T.dot(self.space_product.dot(self.POD_Basis[:,-1]))) > min(self.incremental_tol, self.incremental_tol*len(S)):
                            print('Reorthogonalize using Gram schmidt:')
                            VS = NumpyVectorSpace(dim = self.POD_Basis.shape[0], id = 'STATE')
    
                            TMP = VS.from_numpy(self.POD_Basis.T)
                            proj = NumpyMatrixOperator(self.space_product, source_id = 'STATE', range_id = 'STATE')
                            POD_basis = gram_schmidt(TMP, product=proj)
                            self.POD_Basis = POD_basis.to_numpy().T
                
                        # update everything
                        self.POD_Basis = self.POD_Basis
                        self.POD_values = self.Singular_values**2
                        self.U_left = self.POD_Basis
                        self.V_right = self.POD_Basis
        else:
            assert 0, 'to implement ..'
        
        end_time = perf_counter()  
        print(f'Basis incrementally constructed in {end_time-start_time}')
    
    def update_rom(self, l, Snapshots, space_product, time_product, PODmethod, plot = True, old_rom = None, model_to_project = None, n = 1, build_error_est = True):
        
        
        if self.update_type == 'redo_svd' or n == 0: # Possibility 1 perform svd on previously selected snapshots
            return self.get_rom(l, Snapshots, space_product, time_product, PODmethod, plot, model_to_project, build_error_est)
        
        elif self.update_type == 'incremental_svd': # perform incremental svd one colum at a time
            init = False
            if n == 0:
                init = True
            self.incremental_svd_update(Snapshots, init)
            self.project_and_build_error_est(model_to_project)
            return self.rom_pod
        
        else:
            pass
    
    def project_and_build_error_est(self, model_to_project = None):
        
        # project and update basis
        rom_pod = self.project(U = self.U_left, V = self.V_right, product = self.projection_product, H_prod = self.space_product, model_to_project = model_to_project)#self.project(pod_basis, self.space_product@pod_basis)
        
        # construct error est 
        rom_pod.error_estimator = get_error_estimator(model_to_project, rom_pod, self, error_est_options = None, old_data = None)
        self.rom_pod = rom_pod
        
        return self.rom_pod
    
    def coarse_rom(self, current_state_adjoint_r, tolerance = 1e-12, model_to_project = None, build_error_est = True):
        
        old_size = self.POD_Basis.shape[1]
        print(f'ROM POD coarsening ... before RB size {old_size}')
        start_time = perf_counter()
        
        # Determine removal indices
        if 1: # technique based on Fourier coefficients
            state_norms = []
            for state in current_state_adjoint_r:
                state_norms.append(self.rom_pod.space_time_product(state, state, space_norm = 'H1'))
            
            # compute xi_n
            Xi_i = []
            for i in range(current_state_adjoint_r[0].shape[0]):
                energy_content_in_coeff_n = []
                for j in range(len(current_state_adjoint_r)):
                    state = current_state_adjoint_r[j]
                    norm_coeff = self.rom_pod.time_norm_scalar(state[i,:])**2
                    energy_content_in_coeff_n.append(norm_coeff/state_norms[j])
                Xi_i.append(max(energy_content_in_coeff_n))
            maxXi = max(Xi_i); minXi = min(Xi_i)
            self.XisMax.append(maxXi); self.XisMin.append(minXi)
            print(f'Max Xi {maxXi}, min Xi {minXi}.')    
            # get indices that stay
            indices = np.array(Xi_i) > tolerance
        
        else: ### Technique based on relaxing energy in bases
            pass
        
        end_time_selecting_indices = perf_counter() -start_time
        
        if False in indices:
            
            start_time_rom_proj = perf_counter()
            
            # save error est matrix
            self.error_est_data_lmax = collection()
            self.error_est_data_lmax.state_est_stuff = self.old_data.state_est_stuff
            self.error_est_data_lmax.adjoint_est_stuff = self.old_data.adjoint_est_stuff
            
            coarse_flag = True
            # Remove indices
            pod_basis = self.POD_Basis[:, indices]
            self.POD_Basis = pod_basis
            
            if 0:
                self.U_left = self.space_product@pod_basis
                self.V_right = pod_basis
                self.projection_product = None
            else:
                self.U_left = pod_basis
                self.V_right = pod_basis
                self.projection_product = self.space_product
                
            # 2. project the model
            rom_pod = self.project(U = self.U_left, V = self.V_right, product = self.projection_product, H_prod = self.space_product, model_to_project = model_to_project)#self.project(pod_basis, self.space_product@pod_basis)
            
            # 3. get error estimators
            if build_error_est:
                
                # coarse error est matrix
                est2 = get_error_est_offline(model_to_project, rom_pod, self, l_cut = None, error_est_options = None, indices = indices)
                
                # rebuild error est matrix
                # est = get_error_estimator(model_to_project, rom_pod, self, error_est_options = None, old_data = None)
                # print(np.max(est['adjoint'].adjoint_est_stuff.est_mat_sigma[0]-est2['adjoint'].adjoint_est_stuff.est_mat_sigma[0]))
                rom_pod.error_estimator = est2
            self.rom_pod = rom_pod
            
            end_time = perf_counter()  
            time_rom_proj = end_time-start_time_rom_proj
            print(f'ROM coarsed in total {end_time-start_time}, new RB size {self.POD_Basis.shape[1]}, projection time {time_rom_proj}, selecting indices time {end_time_selecting_indices}.')
        
        else:
            rom_pod = self.rom_pod
            coarse_flag = False
            end_time = perf_counter()  
            time_rom_proj = 0
            print(f'ROM not coarsed in {end_time-start_time}.')
            
        return rom_pod, coarse_flag, time_rom_proj, end_time_selecting_indices
    
    def get_sub_rom(self, l):
        
        
        assert l <= self.POD_Basis.shape[1], 'you want to much here :D'
        # coarse basis
        basis = self.POD_Basis[:, :l]
        
        self.error_est_data_lmax = collection()
        self.error_est_data_lmax.state_est_stuff = self.old_data.state_est_stuff
        self.error_est_data_lmax.adjoint_est_stuff = self.old_data.adjoint_est_stuff
        
        # 2. project the model
        rom_pod = self.project(U = basis, V = basis, product = self.space_product, H_prod = self.space_product, model_to_project = self.model_toproject)
        
        # 3. get error estimators
        est2 = get_error_est_offline(self.model_toproject, rom_pod, self, l_cut = l, error_est_options = None, indices = None)
        rom_pod.error_estimator = est2
        return rom_pod
    
    def get_rom(self, l, Snapshots, space_product, time_product, PODmethod, plot = True, model_to_project = None, build_error_est = True, use_energy_content = True):
        
        print('ROM POD constructing ...')
        start_time = perf_counter()
        
        if  self.space_product is None:
            self.space_product = space_product
        
        if model_to_project is None:
            model_to_project = self.model_toproject
            
        # 1. get pod basis
        if self.snapshot_energy and 0:
            assert len(self.snapshot_energy) == len(Snapshots), 'each snapshots should have its energy ...'
            self.total_energy = sum(self.snapshot_energy)      
        else:
            # compute energy for snapshots
            self.snapshot_energy = []
            len_last = Snapshots[-1].shape[1]
            Snapshots_short = []
            for s in Snapshots:
                s = s[:,:len_last]
                Snapshots_short.append(s)
                self.snapshot_energy.append(self.model.space_time_product(s, s, time_norm = time_product.diagonal(), space_mat = self.space_product))       
            self.total_energy = sum(self.snapshot_energy)
            
            
        pod_basis, pod_values = self.pod_basis(Y = Snapshots_short, 
                                                l = l, 
                                                W = self.space_product, 
                                                D = time_product, 
                                                flag = PODmethod, 
                                                plot = plot,
                                                use_energy_content = use_energy_content)
        if 0:
            self.plot_pod_values()
         
        if 1: 
            self.Snapshots.append(Snapshots)
            self.Ds.append(time_product)
        
        if 0:
            self.U_left = self.space_product@pod_basis
            self.V_right = pod_basis
            self.projection_product = None
        else:
            self.U_left = pod_basis
            self.V_right = pod_basis
            self.projection_product = self.space_product
            
        # 2. project the model
        rom_pod = self.project(U = self.U_left, V = self.V_right, product = self.projection_product, H_prod = self.space_product, model_to_project = model_to_project)#self.project(pod_basis, self.space_product@pod_basis)
        
        # 3. get error estimators
        if build_error_est:
            rom_pod.error_estimator = get_error_estimator( model_to_project, rom_pod, self, error_est_options = None, old_data = None)
        self.rom_pod = rom_pod
        end_time = perf_counter()  
        print(f'ROM constructed in {end_time-start_time}')
        return rom_pod
    
    def project(self, U, V = None, product = None, H_prod = None, model_to_project = None):
        
        if model_to_project is None:
            model_to_project = self.model_toproject
            
        projected_pde, projected_cost = project_model(model_to_project, U, V, product , H_prod)
        rom = model(projected_pde, projected_cost, model_to_project.time_disc, model_to_project.space_disc, model_to_project.options)
        rom.type = projected_pde.type+'POD'
        return rom
        
    def pod_basis(self, Y, l, W = None, D = None, flag = 0, plot = False, energy_tolerance = None, use_energy_content = True):
        """
        #     Compute POD basis

        #     Parameters
        #     ----------
        #     Y: list of/or ndarray shape (n_x,n_t),
        #         Matrix containing the vectors {y^k} (Y = [y^1,...,y^nt]),
        #         or a list containing different snapshot matrices
        #     l: int,
        #         Length of the POD-basis.
        #     W: ndarray, shape (n_x,n_x)
        #         Gramian of the Hilbert space X, that containts the snapshots.
        #     D: list of/or ndarray of shape (n_t,n_t)
        #         Matrix containing the weights of the time discretization.
        #     flag: int
        #         parameter deciding which method to use for computing the POD-basis
        #         (if flag==0 svd, flag == 1 eig of YY', flag == 2 eig of Y'Y (snapshot method).

        #     Returns
        #     -------
        #     POD_Basis: ndarray, shape (n_x,l)
        #                 matrix containing the POD-basis vectors
        #     POD_Values: ndarray, shape (l,)
        #            vector containing the eigenvalues of Yhat (see below)

        """
        
        # set truncation tol
        tol = self.truncation_tol
        truncate_normalized_POD_values = False
        if energy_tolerance is None:
            energy_tolerance = 1-1e-13
        
        if W is None: 
            pass 
        if D is None:
            pass 
        
        # compute square root of the diagonal matrix D
        if type(D) == list and 0:   
            Dsqrt = [d.sqrt() for d in D]
        else:
            Dsqrt = D.sqrt() 
            
        # check if list of snapshots is given and determine nx, nt
        if type(Y) == list:
                K = len(Y)
                Dsqrt = [Dsqrt]*K
                Dsqrt = sps.block_diag(Dsqrt)
                nx, nt = Y[0].shape
                Y = np.concatenate( Y, axis=1 )      
        elif isinstance(Y, np.ndarray):
                nx, nt = Y.shape
        
        ### COMPUTE POD BASIS
        if flag == 0:
        # SVD
        # advantages: stable
        # disadvantages: one needs Wsqrt and to solve l linear systems with Wsqrt, gets expensive if Y is large
            
            # scale matrix
            Yhat = self.Wchol@Y@Dsqrt
            l_min = min(l,min(Yhat.shape)-1)
            print(f'Basissize dropped from {l} to {l_min} due to rank condition of snapshot matrix.')
            
            # perform svd
            U, S, V = sps.linalg.svds(Yhat, k=l_min)
            
            # get pod values
            POD_values = S**2
            if 0:
                print(f'PODvals {POD_values}')
            # sort from biggest to lowest
            U = np.fliplr(U)
            POD_values = np.flipud(POD_values)
            
            # truncate w.r.t. the normalized singular values
            if truncate_normalized_POD_values:
                normalized_values = POD_values/POD_values[0]
            else:
                normalized_values = POD_values
            print(f'Smallest singular value {normalized_values[-1]} and biggest {normalized_values[0]}.')
            indices = normalized_values > tol
            POD_values = POD_values[indices]
            U = U[:,indices]
            print(f'Basissize dropped from {l_min} to {U.shape[1]} due to truncation of small modes.')
            
            # get POD basis
            if 1:
    
                POD_Basis = linalg.solve_triangular(self.Wchol, U, lower = False)
            else:
                POD_Basis = U
        
        elif flag == 1: 
            # Compute eigenvalues of YY' with size (n_x, n_x):
            # advantages: cheap if n_x is small w.r.t K*n_t
            # disadvantages: one needs Wsqrt and to solve l linear systems with Wsqrt, might be inaccurate, additional GramSchmidt
            
            # scale matrix
            Yhat = self.Wchol@Y@Dsqrt
            Y_YT = Yhat@Yhat.T
            
            POD_values, U = sps.linalg.eigsh(Y_YT, k = l, which = 'LM')
            
            U = np.fliplr(U)
            POD_values = np.flipud(POD_values)
            
            if truncate_normalized_POD_values:
                normalized_values = POD_values/POD_values[0]
            else:
                normalized_values = POD_values   
            indices = POD_values > tol
            POD_values = POD_values[indices]
            U = U[:,indices]
            print(f'Basissize dropped from {l} to {U.shape[1]} due to truncation of small modes.')
            
            # get POD basis
            if 1:
                
               POD_Basis = linalg.solve_triangular(self.Wchol, U, lower = False)
            
            else:
                POD_Basis = U
            
        elif flag == 2: 
            # Method of snapshots: eigs of Y'Y with size (n_t,n_t)
            # advantages: cheap if K*n_t is small w.r.t n_x, only Dsqrt is required
            # disadvantages: might be inaccurate, additional GramSchmidt
          
            YT_Y = Dsqrt@Y.T@W@Y@Dsqrt

            if 1:
                POD_values, U = sps.linalg.eigsh(YT_Y, which = 'LM', k = l)
            else:
                pass
            
            U = np.fliplr(U)
            POD_values = np.flipud(POD_values)
            

            if truncate_normalized_POD_values:
                normalized_values = POD_values/POD_values[0]
            else:
                normalized_values = POD_values
            indices = normalized_values > tol
            POD_values = POD_values[indices]
            U = U[:,indices]
            print(f'Basissize dropped from {l} to {U.shape[1]} due to truncation of small modes.')
            
            # get POD basis
            POD_Basis = Y@Dsqrt@U*1/(np.sqrt(POD_values))
            
        else:
            assert 0, 'wrong flag input ...'
        
        # cut basis based on energy
        if energy_tolerance is not None and use_energy_content:
            size_before = len(POD_values)
            l_energy = 1
            local_energy = sum(POD_values[:l_energy])
            while local_energy/self.total_energy <= energy_tolerance and l_energy<size_before :
                # print(f'Energy is {local_energy/self.total_energy}<={energy_tolerance}')
                l_energy += 1
                local_energy = sum(POD_values[:l_energy])
            print(f'Basissize dropped due to energy criterion from {size_before} to {l_energy}.')
            POD_Basis  = POD_Basis[:,:l_energy]
            POD_values = POD_values[:l_energy]
            
        self.POD_Basis = POD_Basis
        self.POD_values = POD_values
        self.Singular_values = np.sqrt(POD_values)
        
        return POD_Basis, POD_values

    
    def select_snapshots(self):
        pass
    
    def plot_pod_values(self):
        plt.figure()
        plt.title('POD Eigenvalues decay')
        plt.semilogy(self.POD_values)
        plt.show()
        
#%% error estimator

# for BT
def get_error_est_offline(fom_toproject, rom, reductor, l_cut, error_est_options = None, indices = None ):
    
    assert np.linalg.norm(fom_toproject.pde.F,2)<1.14, 'error F, we need zero rhs or space and time separated rhs: THE ORDER TO PICK THE SUBEST MAT NEEDS TO BE CHECKED!'
    
    # load big error est stuff
    state_stuff_big = reductor.error_est_data_lmax.state_est_stuff
    adjoint_stuff_big = reductor.error_est_data_lmax.adjoint_est_stuff
    
    ### construct small error est stuff
    # state
    est_mat_sigma_state = []
    for i in range(len(rom.pde.A)):
        
        BDUAL = 1*state_stuff_big.B_dual_sigma[i]
        BRIETZ = 1*state_stuff_big.B_rietz_sigma[i]
        
        if indices is None:
        # read big
            MDUAL = 1*state_stuff_big.M_dual_sigma[i][:,:l_cut]
            MRIETZ = 1*state_stuff_big.M_rietz_sigma[i][:,:l_cut]
            ADUAL = 1*state_stuff_big.A_dual_sigma[i][:,:l_cut]
            ARIETZ = 1*state_stuff_big.A_rietz_sigma[i][:,:l_cut]
        else:
            MDUAL = 1*state_stuff_big.M_dual_sigma[i][:,indices]
            MRIETZ = 1*state_stuff_big.M_rietz_sigma[i][:,indices]
            ADUAL = 1*state_stuff_big.A_dual_sigma[i][:,indices]
            ARIETZ = 1*state_stuff_big.A_rietz_sigma[i][:,indices]
        
        # construct est mat
        RDUAL = np.concatenate((BDUAL, MDUAL, ADUAL), axis = 1)
        RRIETZ = np.concatenate((BRIETZ, MRIETZ, ARIETZ), axis = 1)
        est_mat = RDUAL.T@RRIETZ
        
        # save est mat
        est_mat_sigma_state.append(est_mat)
    
    state_est_stuff = collection()
    state_est_stuff.est_mat_sigma= est_mat_sigma_state
    
    # adjoint
    est_mat_sigma_adjoint = []
    for i in range(len(rom.pde.A)):
        
        CDUAL = 1*adjoint_stuff_big.C_dual_sigma[i]
        CRIETZ = 1*adjoint_stuff_big.C_rietz_sigma[i]
        
        # read big
        if indices is None:
            MDUAL = 1*adjoint_stuff_big.M_dual_sigma[i][:,:l_cut]
            MRIETZ = 1*adjoint_stuff_big.M_rietz_sigma[i][:,:l_cut]
            ADUAL = 1*adjoint_stuff_big.A_dual_sigma[i][:,:l_cut]
            ARIETZ = 1*adjoint_stuff_big.A_rietz_sigma[i][:,:l_cut]
        else:
            MDUAL = 1*adjoint_stuff_big.M_dual_sigma[i][:,indices]
            MRIETZ = 1*adjoint_stuff_big.M_rietz_sigma[i][:,indices]
            ADUAL = 1*adjoint_stuff_big.A_dual_sigma[i][:,indices]
            ARIETZ = 1*adjoint_stuff_big.A_rietz_sigma[i][:,indices]
            
        # construct est mat
        RDUAL = np.concatenate((CDUAL, MDUAL, ADUAL), axis = 1)
        RRIETZ = np.concatenate((CRIETZ, MRIETZ, ARIETZ), axis = 1)
        est_mat = RDUAL.T@RRIETZ
        
        # save est mat
        est_mat_sigma_adjoint.append(est_mat)
    
    adjoint_est_stuff = collection()
    adjoint_est_stuff.est_mat_sigma = est_mat_sigma_adjoint
    
    # adjoint switched matrices
    est_mat_switched_adjoint = []
    for i in range(len(rom.pde.M)): 
        for j in range(len(rom.pde.M)): 
            if i == j:
                pass
            else:      
               # read correct C, M, Mold, A
               CDUAL = adjoint_stuff_big.C_dual_sigma[i]
               CRIETZ = adjoint_stuff_big.C_rietz_sigma[i]
               
               
               if indices is None:
                   ADUAL = adjoint_stuff_big.A_dual_sigma[i][:,:l_cut]
                   ARIETZ = adjoint_stuff_big.A_rietz_sigma[i][:,:l_cut]
                   
                   # read M with index change i, j 
                   MDUAL_i = adjoint_stuff_big.M_dual_sigma[i][:,:l_cut]
                   MRIETZ_i = adjoint_stuff_big.M_rietz_sigma[i][:,:l_cut]
                   
                   # geswitchte
                   MDUAL_j = adjoint_stuff_big.M_dual_sigma[j][:,:l_cut]
                   MRIETZ_j = adjoint_stuff_big.Mrietz_mixed[i][:,:l_cut]
               else:
                   ADUAL = adjoint_stuff_big.A_dual_sigma[i][:,indices]
                   ARIETZ = adjoint_stuff_big.A_rietz_sigma[i][:,indices]
                   
                   # read M with index change i, j 
                   MDUAL_i = adjoint_stuff_big.M_dual_sigma[i][:,indices]
                   MRIETZ_i = adjoint_stuff_big.M_rietz_sigma[i][:,indices]
                   
                   # geswitchte
                   MDUAL_j = adjoint_stuff_big.M_dual_sigma[j][:,indices]
                   MRIETZ_j = adjoint_stuff_big.Mrietz_mixed[i][:,indices]
                   
               
               # concatenate
               RDUAL = np.concatenate((CDUAL, MDUAL_i, MDUAL_j, ADUAL), axis = 1)
               RRIETZ = np.concatenate((CRIETZ, MRIETZ_i, MRIETZ_j, ARIETZ), axis = 1)             
           
               # get error est mat
               est_mat_switched_adjoint.append(RDUAL.T@RRIETZ)
               
    adjoint_est_stuff.est_mat_switched = est_mat_switched_adjoint
     
    # save in rom
    reductor.error_est_data_l_local = collection()
    reductor.error_est_data_l_local.state_est_stuff = state_est_stuff
    reductor.error_est_data_l_local.adjoint_est_stuff = adjoint_est_stuff
    
    # collect
    error_estimator = {'state': state_error_est(fom_toproject, rom, reductor, error_est_options, state_est_stuff),
                       'adjoint': adjoint_error_est(fom_toproject, rom, reductor, error_est_options, adjoint_est_stuff)
                       # 'optimal_control':
                       # 'optimal_state':
                       # 'optimal_adjoint':
                       #  'optimal_value_fun':          
                       }
    return error_estimator

# for POD
def get_error_estimator(fom_toproject, rom, reductor, error_est_options, old_data = None):
    
    # compute rietztrepresentatives based on old data
    
    # get new error estimators
    
    if reductor.errorest_assembled:
        
        assert np.linalg.norm(fom_toproject.pde.F,2)<1.14, 'error F, we need zero rhs or space and time separated rhs'
        
        # state est mat
        state_est_stuff = fom_toproject.get_state_est_matrix(reductor.POD_Basis, old_data = reductor.old_data, norm_type = 'H1dual')
        
        # adjoint_est_mat
        adjoint_est_stuff = fom_toproject.get_adjoint_est_matrix(reductor.POD_Basis, old_data = reductor.old_data, norm_type = 'H1dual', state_data = state_est_stuff)
    else:
        state_est_stuff = None
        adjoint_est_stuff = None
    
    reductor.old_data = collection()
    reductor.old_data.state_est_stuff = state_est_stuff
    reductor.old_data.adjoint_est_stuff = adjoint_est_stuff
        
    # collect
    error_estimator = {'state': state_error_est(fom_toproject, rom, reductor, error_est_options, state_est_stuff),
                       'adjoint': adjoint_error_est(fom_toproject, rom, reductor, error_est_options, adjoint_est_stuff)
                       # 'optimal_control':
                       # 'optimal_state':
                       # 'optimal_adjoint':
                       #  'optimal_value_fun':          
                       }
    return error_estimator

def tune_young_constants(R, e):
    '''
    

    Parameters
    ----------
    R : float
        weighted sum of the squared residuals (thei reresents the disc of the L^2(V') integral).
    e : float
        the initial or terminal squared term in the H norm.

    Returns
    -------
    epsilon_bar: float
                  this is the epsilon minimizing the bound in Youngs inequality, 
                  it is computed analytically as the solution of a quadratic zero problem using the a,b,c-formula      

    '''
    
    # compute epsilon_bar using abc formula
    if 0:
        epsilon_bar = 2*R + np.sqrt(4*R**2+8*R*e)
        epsilon_bar /= 4*R
        if epsilon_bar == complex() or epsilon_bar <= 0.5+1e-15:
            print( 'error.., epsilon in Youngs estimate got complex or negative, set it canonically to 1')
            epsilon_bar = 1
    else:
        epsilon_bar = 1
    if 0: 
        print(epsilon_bar)
    # compute coeffs and total_bound
    coeff_R = epsilon_bar**2/(2*epsilon_bar-1)
    coeff_e = epsilon_bar/(2*epsilon_bar-1)
    total_bound = coeff_R*R+ coeff_e*e

    return epsilon_bar, coeff_R, coeff_e, total_bound

# three modes: true, online, offline
def assemble_switch_constants(switching_profile, ETA, t, k):
    space_time_constant = 1
    return space_time_constant

def threshold_sort(Res_dual_normBefore, tol = 1e-14):
    Res_dual_norm = [0 if abs(num) < tol else num for num in Res_dual_normBefore]
    return Res_dual_norm

class state_error_est():  
    def __init__(self, fom, rom, reductor, options = None, state_est_stuff = None):
        self.fom = fom
        self.rom = rom
        self.reductor = reductor
        self.options = options
        self.state_est_stuff = state_est_stuff
        self.c_init = 1
        self.old_est_flag = False
        self.tol_threshold = 1e-17
        
        # self.product = reductor.space_product
    def build_constants(self, t):
        
        eta2 = self.fom.pde.error_est_constants['eta2']
        cm = self.fom.pde.error_est_constants['cm']
        eta1 = self.fom.pde.error_est_constants['eta1']
        
        c1 = np.exp(2*eta2*t/cm)
        c2 = 1/(cm*eta1)
        
        return eta1, eta2, cm, c1, c2
    
    def est_online(self, U, k = None, Yr = None, switching_profile = None, type_ = 'LinfH'):
        
        if k is None:
           k = self.fom.time_disc.K-1
      
        if Yr is None or switching_profile is None:
            Yr, out, _, switching_profile = self.rom.solve_state(U = U)
            
        # reconstruct Y_U_r
        Y = self.reductor.ROMtoFOM(Yr)
        
        # error est constants
        t = self.fom.time_disc.t_v[k]
        assert abs(t-self.rom.time_disc.t_v[k])<1e-15, 'rom and fom do not have the same time scales ....'
        
        
        # get state residuals
        # Res_H_0 = self.fom.space_norm(self.fom.pde.y0-Y[:,0], space_norm = 'L2')**2
        Res, Res_dual_normBefore = self.fom.state_residual(U, Y, out = None, theta = 1, switching_profile = switching_profile, 
                                                     compute_norm = True, norm_type = 'H1dual')
        
        Res_dual_norm = threshold_sort(Res_dual_normBefore, tol = self.tol_threshold)
        
        Res_H_0 = Res_dual_norm[0]
        if 0: 
            print(Res_dual_normBefore)
            print(Res_dual_norm)
        
        # get error est constants
        eta1, eta2, cm, c1, c2 = self.build_constants(t)
        coercA = self.fom.pde.error_est_constants['eta1']
        
        if type_ == 'LinfH':
            type_ = 'L02V'
        if  type_ == 'LinfH_list':
            type_ = 'L2V_list'
        
        # get rho weights
        rhos, info = self.fom.compute_rhos(switching_profile, state = True)
        
        # if 1:
        #     initmat = info.rho_time_mat
        # else:
        #     initmat = np.ones(self.fom.time_disc.K)
        
        if type_ == 'L2V':
         
            if self.old_est_flag:
                est = np.sqrt(self.c_init*c1*cm/eta1 * Res_H_0 + (2*c1-1)/(eta1**2)*self.fom.time_disc.dt*sum(Res_dual_norm[1:k+1]))
            else:
                est = np.sqrt(self.c_init*info.rho_time_mat[k]*Res_H_0 + self.fom.time_disc.dt*sum(info.rho_time_mat[1:k+1]*np.array(Res_dual_norm[1:k+1])))
            if not self.fom.options.energy_prod:
                est =  est/coercA
        elif type_ == 'L2Vsharp':
            assert 0, 'implement this...'
        elif type_ == 'L2V_list':
            est = []
            if self.old_est_flag:
                for k in list(range(self.fom.time_disc.K)):
                    # get time
                    t = self.fom.time_disc.t_v[k]; 
                    eta1, eta2, cm, c1, c2 = self.build_constants(t)
                    est.append(np.sqrt(self.c_init*c1*cm/eta1 * Res_H_0 + (2*c1-1)/(eta1**2)*self.fom.time_disc.dt*sum(Res_dual_norm[1:k+1]))) 
            else:
                for k in list(range(self.fom.time_disc.K)):
                    est.append(np.sqrt(self.c_init* info.rho_time_mat[k]*Res_H_0 + self.fom.time_disc.dt*sum(info.rho_time_mat[1:k+1]*np.array(Res_dual_norm[1:k+1])))) 
            if not self.fom.options.energy_prod:
                 est = [i/coercA for i in est]
            
        return est, Res_dual_norm
    
    def offline_online_est(self, U, k = None, Yr = None, switching_profile = None, type_ = 'LinfH'):
        
        if type_ == 'LinfH':
           type_ = 'L2V'
        if  type_ == 'LinfH_list':
           type_ = 'L2V_list'
           
        if k is None:
           k = self.fom.time_disc.K-1
     
        if Yr is None or switching_profile is None:
            Yr, out, _, switching_profile = self.rom.solve_state(U = U)
        # reconstruct Y_U_r an 0 nur
        
        # error est constants
        t = self.fom.time_disc.t_v[k]
        if not abs(t-self.rom.time_disc.t_v[k])<1e-15:
            assert abs(t-self.rom.time_disc.t_v[k])<1e-15, 'rom and fom do not have the same time scales ....'
        
        # get state residuals
        Y0 = self.reductor.ROMtoFOM(Yr[:,0])
        if self.fom.options.energy_prod:           
            Res_H_0 = self.fom.space_norm(self.fom.pde.y0-Y0, space_norm = 'M_switch', switch = switching_profile[0] )**2
        else:
            Res_H_0 = self.fom.space_norm(self.fom.pde.y0-Y0, space_norm = 'L2')**2
        
        # multiply with est mat
        Res_dual_norm = [abs(Res_H_0)]
        for k_ind in range(k):
            
            # get sigma for k
            ind_sigma = switching_profile[k_ind+1]-1
            # print(ind_sigma)
            # get coeff k
            UUU = U[:,k_ind+1] 
            MYYY = - (1/self.fom.time_disc.dt)*(Yr[:,k_ind+1]-Yr[:,k_ind])
            AYYY = - Yr[:,k_ind+1]
            coeff_k = np.concatenate((UUU, MYYY, AYYY))
            dual_norm = abs(coeff_k.T@self.state_est_stuff.est_mat_sigma[ind_sigma]@coeff_k)
            Res_dual_norm.append(dual_norm)

        
        Res_dual_norm = threshold_sort(Res_dual_norm, tol = self.tol_threshold)
        # get error est constants
        eta1, eta2, cm, c1, c2 = self.build_constants(t)
        coercA = self.fom.pde.error_est_constants['eta1']
        
        # get rho weights
        rhos, info = self.fom.compute_rhos(switching_profile, state = True)
        
        if 1:
            initmat = info.rho_time_mat
        else:
            initmat = np.ones(self.fom.time_disc.K)
            
        if type_ == 'L2V':
            if self.old_est_flag:
                est = np.sqrt(self.c_init*c1*cm/eta1 * Res_H_0 + (2*c1-1)/(eta1**2)*self.fom.time_disc.dt*sum(Res_dual_norm[1:k+1]))
            else:
               est = np.sqrt(self.c_init* info.rho_time_mat[k]*Res_H_0 + self.fom.time_disc.dt*sum(info.rho_time_mat[1:k+1]*np.array(Res_dual_norm[1:k+1])))
      
            if not self.fom.options.energy_prod:
                est =  est/coercA
        elif type_ == 'L2Vsharp':
            assert 0, 'implement this...'
        elif type_ == 'L2V_list':
            est = []
            if self.old_est_flag:
                for k in list(range(self.fom.time_disc.K)):
                    # get time
                    t = self.fom.time_disc.t_v[k]; 
                    eta1, eta2, cm, c1, c2 = self.build_constants(t)
                    est.append(np.sqrt(self.c_init*c1*cm/eta1 * Res_H_0 + (2*c1-1)/(eta1**2)*self.fom.time_disc.dt*sum(Res_dual_norm[1:k+1]))) 
            else:
                for k in list(range(self.fom.time_disc.K)):
                    est.append(np.sqrt(self.c_init* info.rho_time_mat[k]*Res_H_0 + self.fom.time_disc.dt*sum(info.rho_time_mat[1:k+1]*np.array(Res_dual_norm[1:k+1])))) 
            if not self.fom.options.energy_prod:
                 est = [i/coercA for i in est]
            
        # compare with state res
        if 0:
            print(f'STATE {Res_dual_norm}')
        bugfix = False
        if bugfix:
            
            Y = self.reductor.ROMtoFOM(Yr)
            ResONLINE, Res_dual_normONLINE = self.fom.state_residual(U, Y, out = None, theta = 1, switching_profile = switching_profile, 
                                                         compute_norm = True, norm_type = 'H1dual')
            Res_dual_normONLINE = threshold_sort(Res_dual_normONLINE, tol = self.tol_threshold)
            for i, j in zip(Res_dual_norm, Res_dual_normONLINE):
                print(f'{i}, {j}, {i-j}')
            
            if type_ == 'L2V':
                if self.old_est_flag:
                    est_online = np.sqrt(self.c_init*c1*cm/eta1 * Res_dual_normONLINE[0] + (2*c1-1)/(eta1**2)*self.fom.time_disc.dt*sum(Res_dual_normONLINE[1:k+1]))
                else:
                    est_online = np.sqrt(self.c_init*Res_dual_normONLINE[0] + self.fom.time_disc.dt*sum(Res_dual_normONLINE[1:k+1]))
                if not self.fom.options.energy_prod:
                    est_online =  est_online/coercA
            elif type_ == 'L2Vsharp':
                assert 0, 'implement this...'
            elif type_ == 'L2V_list':
                est_online = []
                if self.old_est_flag:
                    for k in list(range(self.fom.time_disc.K)):
                        # get time
                        t = self.fom.time_disc.t_v[k]; 
                        eta1, eta2, cm, c1, c2 = self.build_constants(t)
                        est_online.append(np.sqrt(self.c_init*c1*cm/eta1 * Res_dual_normONLINE[0] + (2*c1-1)/(eta1**2)*self.fom.time_disc.dt*sum(Res_dual_normONLINE[1:k+1]))) 
                else:
                    for k in list(range(self.fom.time_disc.K)):
                        est_online.append(np.sqrt(self.c_init* Res_dual_normONLINE[0] + self.fom.time_disc.dt*sum(Res_dual_normONLINE[1:k+1]))) 
                if not self.fom.options.energy_prod:
                     est_online = [i/coercA for i in est_online]
            for i, j in zip(est, est_online):
                print(f'{i}, {j}, {i-j}')
                
        return est, Res_dual_norm
    
    def est_true(self, U, Yr = None , Y = None, type_ = 'LinfH', k = None, switching_profile = None):
        
        if type_ == 'LinfH':
            type_ = 'L2V'
        if  type_ == 'LinfH_list':
            type_ = 'L2V_list'
            
        if k is None:
           k = self.fom.time_disc.K-1
        assert abs(self.fom.time_disc.t_v[k]-self.rom.time_disc.t_v[k])<1e-15, 'rom and fom do not have the same time scales ....'
        # print(k)
            
        if Yr is None:
            Yr, out, _, switching_profile = self.rom.solve_state(U = U)
        
        # reconstruct Y_U_r
        Yr = self.reductor.ROMtoFOM(Yr)
        
        # get rho weights
        rhos, info = self.fom.compute_rhos(switching_profile, state = True)
        
        if 1:
            initmat = info.rho_time_mat
        else:
            initmat = np.ones(self.fom.time_disc.K)
            
        if Y is None:
            Y, out, _, switching_profile = self.fom.solve_state(U = U)
        
        if type_ == 'LinfH':    
            return self.fom.space_norm(Yr[:,k]-Y[:,k], space_norm = 'L2')
        elif type_ == 'L2V':
            if not self.fom.options.energy_prod:
                tmp = self.fom.time_disc.dt * np.ones(self.fom.time_disc.K)
                tmp[0] = 1*self.c_init*self.fom.time_disc.dt; tmp[k:] = 0;
                return self.fom.space_time_norm(Y-Yr, space_norm = 'H1')
            else:
                tmp = self.fom.time_disc.dt * initmat
                tmp[0] = 1*self.c_init*self.fom.time_disc.dt; tmp[k:] = 0;
                return self.fom.space_time_norm(Y-Yr, space_norm = 'energy_product', time_norm = tmp, switch_profile = switching_profile)
        elif type_ == 'L2V_list':
            est = []
            if not self.fom.options.energy_prod:
                for k in list(range(self.fom.time_disc.K)):
                    tmp = self.fom.time_disc.dt * np.ones(self.fom.time_disc.K)
                    tmp[0] =1*self.c_init*self.fom.time_disc.dt; tmp[k:] = 0;
                    est.append(self.fom.space_time_norm(Y-Yr, space_norm = 'H1', time_norm = tmp)) 
            else:
                for k in list(range(self.fom.time_disc.K)):
                    tmp = self.fom.time_disc.dt * initmat
                    tmp[0] =1*self.c_init*self.fom.time_disc.dt; tmp[k:] = 0;
                    est.append(self.fom.space_time_norm(Y-Yr, space_norm = 'energy_product', time_norm = tmp, switch_profile = switching_profile)) 
        elif type_ == 'LinfH_list':
            est = []
            for k in list(range(self.fom.time_disc.K)):
                est.append(self.fom.space_norm(Yr[:,k]-Y[:,k], space_norm = 'L2')) 
        
        return est, None
    
class adjoint_error_est():
    
    def __init__(self, fom, rom, reductor, options = None, adjoint_est_stuff = None):
        self.fom = fom
        self.rom = rom
        self.reductor = reductor
        self.options = options
        self.adjoint_est_stuff = adjoint_est_stuff
        self.c_init = 1
        self.old_est_flag = False
        self.tol_threshold = 1e-17
        
    def build_constants(self, t, T):
        
        eta2 = self.fom.pde.error_est_constants['eta2']
        cM = self.fom.pde.error_est_constants['cM']
        eta1 = self.fom.pde.error_est_constants['eta1']
        c_C = self.fom.pde.error_est_constants['c_C']
        
        c1 = 2*eta2+c_C
        c1_t = np.exp((c1/cM)*(T-t))
        c2_t = 1/(cM*eta1)
        c3_t = c_C/cM
        
        return eta1, eta2, c1_t, c2_t, c3_t, c_C
    
    def offline_online_est(self, U = None, Yr = None, Pr = None, type_ = 'LinfH',
                 k = None, switching_profile_r = None,  out_r = None, state_H_est_list = None, option_adjoint_with_state = False, gaps_r = None):
        
        if k is None:
           k = 0
      
        if Pr is None or switching_profile_r is None or out_r is None:
            if Yr is None or switching_profile_r is None or out_r is None:
                Yr, out, _, switching_profile_r = self.rom.solve_state(U = U)
            Z = self.rom.cost_data.output_product.dot(out_r) - self.rom.cost_data.Mc_Yd
            ZT = self.rom.cost_data.output_product.dot(out_r[:,-1]) - self.rom.cost_data.Mc_YT         
            Pr,_, gaps_r = self.rom.solve_adjoint(Z, ZT, switching_profile_r)
        else:
            Z = self.rom.cost_data.output_product.dot(out_r) - self.rom.cost_data.Mc_Yd
            ZT = self.rom.cost_data.output_product.dot(out_r[:,-1]) - self.rom.cost_data.Mc_YT 
            
        # error est constants
        t = self.fom.time_disc.t_v[k]
        T = self.fom.time_disc.t_v[-1]
        assert abs(t-self.rom.time_disc.t_v[k])<1e-15, 'rom and fom do not have the same time scales ....'
        
        # get error est constants
        eta1, eta2, c1_t, c2_t, c3_t, c_C = self.build_constants(t, T)
        coercA = self.fom.pde.error_est_constants['eta1']
        Res_dual_normAdj = []
        
        # reconstruct gaps
        P = self.reductor.ROMtoFOM(Pr, adj = True)
        Gaps_r = []
        for gap in gaps_r:
            if gap is None:
                Gaps_r.append(None)
            else:
                Gaps_r.append(self.reductor.ROMtoFOM(gap, adj = True))
        Rswitchnorm = [] 
        Rswitch = []
        
        ZZZ = (self.fom.cost_data.weights[0]+self.fom.cost_data.weights[2]/self.fom.time_disc.dt)*ZT
        MYYY = - (1/self.fom.time_disc.dt)*Pr[:,-1]
        AYYY = - Pr[:,-1]
        coeff_k = np.concatenate((ZZZ, MYYY, AYYY))
        ind_sigma = switching_profile_r[-1]-1
        dual_norm = abs(coeff_k.T@self.adjoint_est_stuff.est_mat_sigma[ind_sigma]@coeff_k)
        Res_dual_normAdj.append(dual_norm)
        ind_sigma_old = ind_sigma
        
        for k_ind in range(self.fom.time_disc.K-2, -1, -1 ):
            
            ind_sigma = switching_profile_r[k_ind]-1
            
            if abs(ind_sigma - ind_sigma_old)  >0 and 1:
                # now we switch
                ZZZ = self.fom.cost_data.weights[0]*Z[:,k_ind] 
                MYYY = - (1/self.fom.time_disc.dt)*(Pr[:,k_ind])
                MYYY_old = (1/self.fom.time_disc.dt)*(Pr[:,k_ind+1])
                AYYY = - Pr[:,k_ind]
                coeff_k = np.concatenate((ZZZ, MYYY, MYYY_old, AYYY))
                
                if ind_sigma == 0 and ind_sigma_old == 1:
                    ind_switch = 0
                else:
                    ind_switch = 1
                
                est_mat = self.adjoint_est_stuff.est_mat_switched[ind_switch]
                dual_norm = abs(coeff_k.T@est_mat@coeff_k)
                Res_dual_normAdj.append(dual_norm)
                
                # gaps
                Mold = self.fom.pde.M[ind_sigma_old]
                M = self.fom.pde.M[ind_sigma]
                res_switchpoint = Mold.T.dot(P[:,k_ind+1]) - M.T.dot(Gaps_r[k_ind])
                Rswitch.append(res_switchpoint)
                if self.fom.options.energy_prod:
                    Rswitchnorm.append(self.fom.space_product(res_switchpoint, res_switchpoint, space_norm = 'M_switch', switch = ind_sigma+1 ))
                else:
                    Rswitchnorm.append(self.fom.space_product(res_switchpoint, res_switchpoint, space_norm = 'L2', switch = ind_sigma+1))
  
            else:
                # get coefficients
                ZZZ = self.fom.cost_data.weights[0]*Z[:,k_ind] 
                MYYY = - (1/self.fom.time_disc.dt)*(Pr[:,k_ind]-Pr[:,k_ind+1])
                AYYY = - Pr[:,k_ind]
                coeff_k = np.concatenate((ZZZ, MYYY, AYYY))
            
                est_mat = self.adjoint_est_stuff.est_mat_sigma[ind_sigma]
                # multiply with est mat
                dual_norm = abs(coeff_k.T@est_mat@coeff_k)
                Res_dual_normAdj.append(dual_norm)
                Rswitchnorm.append(0)
            ind_sigma_old = ind_sigma
        
        Rswitchnorm.reverse()
        Res_dual_normAdj.reverse()
        Res_dual_normAdj[0] = 0
        
        if 0:
            print(f'ADJOINT  {Res_dual_normAdj}')    
          # get adjoint online residuals
        if 0:
              P = self.reductor.ROMtoFOM(Pr)
              # Y = self.reductor.ROMtoFOM(Pr)
              _, Res_dual_normAdjONLINE, Rswitch_online, RswitchnormONLINE = self.fom.adjoint_residual(P, ZT, Z, 
                                                                   out = out_r, theta = 1, switching_profile = switching_profile_r,
                                                                  compute_norm = True, norm_type = 'H1dual', gaps = Gaps_r)
              # delete from implicit time grid
              Res_dual_normAdjONLINE[0] = 0 
              diffs = []
              for i, j in zip(Res_dual_normAdj, Res_dual_normAdjONLINE):
                 print(f'offline {i}, online: {j}, diff: {i-j}')
                 diffs.append(abs(i-j))
              print(f' maximal difference between offline and online res norm {max(diffs)}')
              
              for i, j in zip(Rswitchnorm, RswitchnormONLINE):
                 print(f'offline {i}, online: {j}, diff: {i-j}')
                 diffs.append(abs(i-j))
              print(f' switching points maximal difference between offline and online res norm {max(diffs)}')
        
        if type_ == 'LinfH':
            type_ = 'L2V'
        if  type_ == 'LinfH_list':
            type_ = 'L2V_list'
        
        # get rho weights
        rhos, info = self.fom.compute_rhos(switching_profile_r, state = False)
        
        if 1:
            initmat = info.rho_time_mat
        else:
            initmat = np.ones(self.fom.time_disc.K)
            
        if not option_adjoint_with_state:
    
            if type_ == 'L2V':
                
                est = np.sqrt(Res_dual_normAdj[-1]*initmat[k] + self.fom.time_disc.dt*sum(Res_dual_normAdj[k:]*initmat[k:]) + 2*sum(Rswitchnorm[k:]*initmat[k+1:]))
                est= est/coercA    
            elif type_ == 'L2V_list':
                est = []
                for k in list(range(self.fom.time_disc.K)):
                        est.append(np.sqrt(Res_dual_normAdj[-1]*initmat[k] + self.fom.time_disc.dt*sum(Res_dual_normAdj[k:]*initmat[k:]) + 2*sum(Rswitchnorm[k:]*initmat[k+1:]))) 
                # est.reverse()
                if not self.fom.options.energy_prod:
                             est = [i/coercA for i in est]  
                             
        else:
            # with state error estimator
            if type_ == 'L2V':
                CA = max(self.fom.pde.error_est_constants['CcontA'])**4
                est = np.sqrt(Res_dual_normAdj[-1]*initmat[k] + 2*self.fom.time_disc.dt*sum(Res_dual_normAdj[k:]*initmat[k:])+2*CA*
                    state_H_est_list[k]**2 + 2*sum(Rswitchnorm[k:]*initmat[k+1:]))
                est= est/coercA    
            elif type_ == 'L2V_list':
                est = []
                CA = max(self.fom.pde.error_est_constants['CcontA'])**4
                for k in list(range(self.fom.time_disc.K)):
                        est.append(np.sqrt(Res_dual_normAdj[-1]*initmat[k] + 2*self.fom.time_disc.dt*sum(Res_dual_normAdj[k:]*initmat[k:])+2*CA*
                            state_H_est_list[k]**2 + 2*sum(Rswitchnorm[k:]*initmat[k+1:])))
                # est.reverse()
                if not self.fom.options.energy_prod:
                             est = [i/coercA for i in est]  
            
        return est
    
    def est_online(self, U = None, Yr = None, Pr = None, type_ = 'LinfH',
                 k = None, switching_profile_r = None,  out_r = None, state_H_est_list = None, option_adjoint_with_state = False, gaps_r = []):
    
        if k is None:
           k = 0#self.fom.time_disc.K-1
      
        if Pr is None or switching_profile_r is None or out_r is None:
            if Yr is None or switching_profile_r is None or out_r is None:
                Yr, out, _, switching_profile_r = self.rom.solve_state(U = U)
            Z = self.rom.cost_data.output_product.dot(out_r) - self.rom.cost_data.Mc_Yd
            ZT = self.rom.cost_data.output_product.dot(out_r[:,-1]) - self.rom.cost_data.Mc_YT         
            Pr,_ = self.rom.solve_adjoint(Z, ZT, switching_profile_r)
        else:
            Z = self.rom.cost_data.output_product.dot(out_r) - self.rom.cost_data.Mc_Yd
            ZT = self.rom.cost_data.output_product.dot(out_r[:,-1]) - self.rom.cost_data.Mc_YT 
            
        # reconstruct Y_U_r
        P = self.reductor.ROMtoFOM(Pr, adj = True)
        Gaps_r = []
        for gap in gaps_r:
            if gap is None:
                Gaps_r.append(None)
            else:
                Gaps_r.append(self.reductor.ROMtoFOM(gap, adj = True))
        
        # error est constants
        t = self.fom.time_disc.t_v[k]
        T = self.fom.time_disc.t_v[-1]
        assert abs(t-self.rom.time_disc.t_v[k])<1e-15, 'rom and fom do not have the same time scales ....'
        
        # get error est constants
        eta1, eta2, c1_t, c2_t, c3_t, c_C = self.build_constants(t, T)
        
        # get adjoint residuals
        ResAdj, Res_dual_normAdj, Rswitch, ResnormSwitch = self.fom.adjoint_residual(P, ZT, Z, 
                                                             out = out_r, theta = 1, switching_profile = switching_profile_r,
                                                            compute_norm = True, norm_type = 'H1dual', gaps = Gaps_r)
        # delete from implicit time grid
        Res_dual_normAdj[0] = 0 
        ResAdj[0] = 0*ResAdj[0]
        coercA = self.fom.pde.error_est_constants['eta1']
        
        if type_ == 'LinfH':
            type_ = 'L2V'
        if  type_ == 'LinfH_list':
            type_ = 'L2V_list'
            
        # get rho weights
        rhos, info = self.fom.compute_rhos(switching_profile_r, state = False)
        
        if 1:
            initmat = info.rho_time_mat
        else:
            initmat = np.ones(self.fom.time_disc.K)
                
        if not option_adjoint_with_state:
            if type_ == 'L2V':
                
                est = np.sqrt(Res_dual_normAdj[-1]*initmat[k] + self.fom.time_disc.dt*sum(Res_dual_normAdj[k:]*initmat[k:]) + 2*sum(ResnormSwitch[k:]*initmat[k+1:]))
                est= est/coercA    
            elif type_ == 'L2V_list':
                est = []
                for k in list(range(self.fom.time_disc.K)):
                        est.append(np.sqrt(Res_dual_normAdj[-1]*initmat[k] + self.fom.time_disc.dt*sum(Res_dual_normAdj[k:]*initmat[k:]) + 2*sum(ResnormSwitch[k:]*initmat[k+1:]))) 
                # est.reverse()
                if not self.fom.options.energy_prod:
                             est = [i/coercA for i in est]  
                             
        else:
            # with state error estimator
            if type_ == 'L2V':
                CA = max(self.fom.pde.error_est_constants['CcontA'])**4
                est = np.sqrt(Res_dual_normAdj[-1]*initmat[k] + 2*self.fom.time_disc.dt*sum(Res_dual_normAdj[k:]*initmat[k:])+2*CA*
                    state_H_est_list[k]**2 + 2*sum(ResnormSwitch[k:]*initmat[k+1:]))
                est= est/coercA    
            elif type_ == 'L2V_list':
                est = []
                CA = max(self.fom.pde.error_est_constants['CcontA'])**4
                for k in list(range(self.fom.time_disc.K)):
                        est.append(np.sqrt(Res_dual_normAdj[-1]*initmat[k] + 2*self.fom.time_disc.dt*sum(Res_dual_normAdj[k:]*initmat[k:])+2*CA*
                            state_H_est_list[k]**2 + 2*sum(ResnormSwitch[k:]*initmat[k+1:])))
                # est.reverse()
                if not self.fom.options.energy_prod:
                             est = [i/coercA for i in est]  
        return est
        
    def est_true(self, U = None, Yr = None, Y = None, Pr = None , P = None, type_ = 'LinfH',
                 k = None, switching_profile_r = None, switching_profile = None, out = None, out_r = None):
            
        if k is None: # k ist der index
           k = self.fom.time_disc.K-1
        assert abs(self.fom.time_disc.t_v[k]-self.rom.time_disc.t_v[k])<1e-15, 'rom and fom do not have the same time scales ....'
        
        if switching_profile_r is None or out_r is None:
            if Yr is None or switching_profile_r is None or out_r is None:
                Yr, out_r, _, switching_profile_r = self.rom.solve_state(U = U)
                Zr = self.rom.cost_data.output_product.dot(out_r) - self.rom.cost_data.Mc_Yd
                ZTr = self.rom.cost_data.output_product.dot(out_r[:,-1]) - self.rom.cost_data.Mc_YT
                
        Zr = self.rom.cost_data.output_product.dot(out_r) - self.rom.cost_data.Mc_Yd
        ZTr = self.rom.cost_data.output_product.dot(out_r[:,-1]) - self.rom.cost_data.Mc_YT       
        if Pr is None:
            Pr,_ = self.rom.solve_adjoint(Zr, ZTr, switching_profile_r)
            
        if P is None or switching_profile is None or out is None: 
            
            measure_u_to_p = False
            if measure_u_to_p:
                if Y is None or switching_profile is None or out is None:
                    Y, out, _, switching_profile = self.fom.solve_state(U = U)
                    Z = self.fom.cost_data.output_product.dot(out) - self.fom.cost_data.Mc_Yd
                    ZT = self.fom.cost_data.output_product.dot(out[:,-1]) - self.fom.cost_data.Mc_YT  
                    P,_ = self.fom.solve_adjoint(Z, ZT, switching_profile)
            else:
                P,_, _= self.fom.solve_adjoint(Zr, ZTr, switching_profile_r)
            
        # reconstruct Y_U_r
        Pr = self.reductor.ROMtoFOM(Pr, adj = True)
        
        if type_ == 'LinfH':
            type_ = 'L2V'
        if  type_ == 'LinfH_list':
            type_ = 'L2V_list'
        
        # get rho weights
        rhos, info = self.fom.compute_rhos(switching_profile_r, state = False)
        
        if 1:
            initmat = info.rho_time_mat
        else:
            initmat = np.ones(self.fom.time_disc.K)
            
        if type_ == 'L2V':
            if not self.fom.options.energy_prod:
                tmp = self.fom.time_disc.dt * np.ones(self.fom.time_disc.K)
                tmp[0] = 0*self.c_init*self.fom.time_disc.dt; tmp[:k] = 0;
                return self.fom.space_time_norm(P-Pr, space_norm = 'H1', time_norm = tmp)
            else:
                tmp = self.fom.time_disc.dt * initmat
                tmp[0] = 0*self.c_init*self.fom.time_disc.dt; tmp[:k] = 0;
                return self.fom.space_time_norm(P-Pr, space_norm = 'energy_product', time_norm = tmp, switch_profile = switching_profile_r)
        elif type_ == 'L2V_list':
            est = []
            if not self.fom.options.energy_prod:
                for k in list(range(self.fom.time_disc.K)):
                    tmp = self.fom.time_disc.dt * np.ones(self.fom.time_disc.K)
                    tmp[0] = 0*self.c_init*self.fom.time_disc.dt; tmp[:k] = 0;
                    est.append(self.fom.space_time_norm(P-Pr, space_norm = 'H1', time_norm = tmp))
            else:
                for k in list(range(self.fom.time_disc.K)):
                    tmp = self.fom.time_disc.dt * initmat
                    tmp[0] = 0*self.c_init*self.fom.time_disc.dt; tmp[:k] = 0;
                    est.append(self.fom.space_time_norm(P-Pr, space_norm = 'energy_product', time_norm = tmp, switch_profile = switching_profile_r))
        return est

class optimal_control_error_est():
    
    def __init__(self, fom, rom, reductor, options = None, data = None):
        self.fom = fom
        self.rom = rom
        self.reductor = reductor
        self.options = options

class optimal_state_est():
    pass

class optimal_adjoint_est():
    pass

class value_function_est():
    pass
    
#%% BT

from scipy.sparse.csgraph import structural_rank
from scipy.sparse import csr_matrix
from pymor.models.iosys import LTIModel
from pymor.reductors.bt import BTReductor
from pymor.algorithms.timestepping import ImplicitEulerTimeStepper
from pymor.core.defaults import set_defaults

# set_defaults({'pymor.bindings.scipy.apply_inverse.default_solver': 'scipy_bicgstab_spilu'})
# set_defaults({
#     'pymor.algorithms.lyapunov.solve_cont_lyap_lrcf.default_sparse_solver_backend':
#     'lradi',
#     # 'pymor.algorithms.lradi.lyap_lrcf_solver_options.projection_shifts_init_seed':
#     # 0
# })

class bt_reductor():
    
    def __init__(self, model, model_toproject = None, Wbig = None, Vbig = None, lmax = None, projection_product = None, options = None,  errorest_assembled = True):
        
        
        self.errorest_assembled = errorest_assembled
        self.projection_product = projection_product
        
        set_defaults({'pymor.operators.interface.as_array_max_length.value': model.state_dim})
        self.model = model
        if model_toproject is None:
            self.model_toproject = model
        else:
            self.model_toproject = model_toproject
    
        self.Wbig = Wbig
        self.Vbig = Vbig
        self.lmax = lmax
        
    def ROMtoFOM(self, u,  adj = False):
        if adj:
           return self.W.T@u
        else:
           return self.V.T@u
    
    def FOMtoROM(self, U):
        if self.projection_product is None:
            return self.W@U 
        else:
            return self.W@ self.projection_product@U
    
    def get_rom(self, l):
        if self.lmax is None:
            self.lmax = l
        
        assert 0 <= l <= self.lmax, 'insert valid rb size l'
        
        start_time = perf_counter()
        # if 1:# and not self.model.space_discpde.state_dim == 5304:
        if self.Wbig is None or self.Vbig is None:
            
            if not self.model.pde.state_dim == 5304 or 0:
                print('Construct Envelope system ...')
                _ = self.envelope_system(self.model)
                print('BT on envelope system ...')
                Wbig, Vbig = self.compute_bt_basis(l_max = self.lmax)
                
                # save mats
                with open('W.npy', 'wb') as f:
                    np.save(f, Wbig)
                with open('V.npy', 'wb') as f:
                    np.save(f, Vbig)
                
            else:
                # read proj mats
                with open('W.npy', 'rb') as f:
                    Wbig = np.load(f)
                with open('V.npy', 'rb') as f:
                    Vbig = np.load(f)
                self.Wbig = Wbig
                self.Vbig = Vbig
                
                if self.errorest_assembled:
                    self.error_est_data_lmax = collection()
                    self.error_est_data_lmax.state_est_stuff = self.model_toproject.get_state_est_matrix(self.Vbig.T, old_data = None, norm_type = 'H1dual')
                        
                    # adjoint_est_mat
                    # self.error_est_data_lmax.adjoint_est_stuff = self.model_toproject.get_adjoint_est_matrix(self.Wbig.T, old_data = None, norm_type = 'H1dual', state_data = self.error_est_data_lmax.state_est_stuff, PetrovGalerkin = True)
                    self.error_est_data_lmax.adjoint_est_stuff = self.model_toproject.get_adjoint_est_matrix(self.Wbig.T, old_data = None, norm_type = 'H1dual', state_data = self.error_est_data_lmax.state_est_stuff, PetrovGalerkin = True)
               
                else:
                    self.error_est_data_lmax = None
                
        else:
            Wbig, Vbig = self.Wbig, self.Vbig
        
        rom = self.project(Wbig[0:l,:].T, Vbig[0:l,:].T)
        self.W = self.Wbig[0:l,:]
        self.V = self.Vbig[0:l,:]
        self.rom = rom
        end_time = perf_counter()  
        self.offline_time = end_time-start_time
        print(f'BT ROM constructed in {end_time-start_time}')
        return rom
    
    def update_rom(self, l_new):
        
        # assert l_new <= self.lmax, 'BT reductor: extend lmax ...'
        l_new  = min (l_new, self.lmax)
        if l_new >= self.lmax:
            print(f' skipped from {l_new} to {self.lmax}')
        
        start_time = perf_counter()
        rom = self.project(self.Wbig[0:l_new,:].T, self.Vbig[0:l_new,:].T)
        self.W = self.Wbig[0:l_new,:]
        self.V = self.Vbig[0:l_new,:]
        self.rom = rom
        end_time = perf_counter()  
        print(f'BT ROM updated in {end_time-start_time}, size is now {l_new}')
        # get more dim projection matrices and 
        # increase the size 
        return rom
        
    def project(self, U, V = None):
        red_pde, red_cost = project_model(self.model_toproject, U, V = None, product = self.projection_product)
        rom = model(red_pde, red_cost, self.model_toproject.time_disc, self.model_toproject.space_disc, self.model_toproject.options)
        rom.type = red_pde.type+'BT'
        if self.errorest_assembled:
            l_cut = U.shape[1]
            rom.error_estimator = get_error_est_offline(fom_toproject = self.model_toproject, rom = rom, reductor = self, l_cut = l_cut, error_est_options = None)
        
        return rom 
    
    def est_svd_compute_rank(self, M):
        estDA = self.estimate_rank_sparse(M)
        SiDA, EiDA, ViTDA = sps.linalg.svds(M, k = estDA)
        rDA = self.compute_rank(EiDA) 
        return estDA, rDA

    def estimate_rank_sparse(self, M):
        rank_est2 = structural_rank(M)
        rank_est3 = rank_est1= 1e30
        est = min(rank_est1,rank_est2, rank_est3)
        print(f' rank esimate is {est}')
        return est

    def compute_rank(self, M):
        tol = 1e-15
        scaled = M/M[-1]
        rank = len([i for i in scaled if i>tol ])
        print(f' rank is {rank}')
        return rank
    
    def envelope_system(self, model):
        
        switch_model = model.pde
        
        # read switch model
        A = switch_model.A
        B = switch_model.B
        C = switch_model.C
        M = switch_model.M
        n_sys = len(switch_model.A)
            
        # init  envelope system
        envelope_system = collection()
        envelope_system.M = switch_model.M[0]
        envelope_system.A = switch_model.A[0]
        envelope_system.y0 = switch_model.y0
        envelope_system.state_dim = switch_model.state_dim
        
        
        ##### Strategy 1: extended output and input
        if 0:
            # init lists
            B_new = [B[0]]
            C_T_new = [C[0].T]
            ranks_M = []
            ranks_A = []
            U_M = []
            S_M = []
            U_A = []
            S_A = []
            R_C  = []
            
            for i in range(len(switch_model.A[1:])):
                
                i += 1
                # build Deltamatrices
                deltaA = A[0] - A[i]
                deltaM = M[0] - M[i]
                deltaB = B[0] - B[i]
                deltaC = C[0] - C[i]
                
                # compute low rank of DA
                estA = self.estimate_rank_sparse(deltaA)
                SiA, EiA, ViTA = sps.linalg.svds(deltaA, k = estA)
                # np.allclose(deltaA.todense(), SiA@np.diag(EiA)@ViTA)
                rA = self.compute_rank(EiA) # compute rank
                ranks_A.append(rA)
                r_init = estA - rA
                S_A.append(SiA[:,r_init:]) # save factors
                U_A.append((np.diag(EiA[r_init:])@ViTA[r_init:,:]).T)
                np.allclose(deltaA.todense(), S_A[-1]@(U_A[-1]).T)
                
                # compute low rank of DM
                estM = self.estimate_rank_sparse(deltaM)
                SiM, EiM, ViTM = sps.linalg.svds(deltaM, k = estM)
                rM = self.compute_rank(EiM) # compute rank
                ranks_M.append(rM)
                r_init = estM - rM
                S_M.append(SiM[:,r_init:]) # save factors
                U_M.append((np.diag(EiM[r_init:])@ViTM[r_init:,:]).T)
                np.allclose(deltaM.todense(), S_M[-1]@U_M[-1].T)
                      
                if 1:
                    plt.figure()
                    plt.title('singular values decay')
                    plt.semilogy(EiA, label = 'deltaA')
                    plt.semilogy(EiM, label = 'deltaM')
                    plt.legend()
                    plt.show()
                
                # compute R
                R_T = spsolve(M[i].T,U_M[-1])##
                R_C.append(A[i].T@R_T)
                
                ######### second option input kleiner machen
                if 1:
                    deltaA_tilde = deltaA - csr_matrix(S_M[-1]@R_C[-1].T)
                    estDA = self.estimate_rank_sparse(deltaM)
                    SiDA, EiDA, ViTDA = sps.linalg.svds(deltaA_tilde, k = estDA)
                    rDA = self.compute_rank(EiDA) # compute rank
                             
                # baue terme füe B
                B_term = deltaB-S_M[-1]@R_T.T@B[i] #deltaB
                # baue terme für C
                         
                # append B, C_T
                B_new.append(B_term) 
                C_T_new.append(deltaC.T)
                
            # set dimensions
            envelope_system.output_dim = n_sys*switch_model.output_dim + sum(ranks_M) + sum(ranks_A)
            envelope_system.input_dim = n_sys*switch_model.input_dim + sum(ranks_M) + sum(ranks_A)
        
            # get new matrices
            B_new.extend(S_A); B_new.extend(S_M)
            C_T_new.extend(U_A); C_T_new.extend(R_C)
                
            # transform into matrices and transpose
            envelope_system.B = np.concatenate(B_new, axis = 1)
            envelope_system.C = np.concatenate(C_T_new, axis = 1).T
           
        ##### Strategy 2: extended output and input
        else:
            
            # init lists
            B_new = [B[0]]
            C_T_new = [C[0].T]
            ranks_M = []
            ranks_A = []
            U_M = []
            S_M = []
            U_A = []
            S_A = []
            R_C  = []
            
            for i in range(len(switch_model.A[1:])):
                
                i += 1
                
                # build Deltamatrices
                deltaA = A[0] - A[i]
                deltaM = M[0] - M[i]
                deltaB = B[0] - B[i]
                deltaC = C[0] - C[i]
                
                # compute low rank of DM
                estM = self.estimate_rank_sparse(deltaM)
                SiM, EiM, ViTM = sps.linalg.svds(deltaM, k = estM)
                rM = self.compute_rank(EiM) # compute rank
                ranks_M.append(rM)
                r_init = estM - rM
                S_M.append(SiM[:,r_init:]) # save factors
                U_M.append((np.diag(EiM[r_init:])@ViTM[r_init:,:]).T)
                np.allclose(deltaM.todense(), S_M[-1]@U_M[-1].T)
                
                # compute R
                R_T = spsolve(M[i].T,U_M[-1])##
                R_C.append(A[i].T@R_T)
                
                # A tilde
                deltaA_tilde = deltaA - csr_matrix(S_M[-1]@R_C[-1].T)
                estDA = self.estimate_rank_sparse(deltaA_tilde)
                SiDA, EiDA, ViTDA = sps.linalg.svds(deltaA_tilde, k = estDA)
                rDA = self.compute_rank(EiDA) # compute rank
                S_A.append(SiDA[:,r_init:]) # save factors
                U_A.append((np.diag(EiDA[r_init:])@ViTDA[r_init:,:]).T)
                np.allclose(deltaA_tilde.todense(), S_A[-1]@(U_A[-1]).T)
                ranks_A.append(rDA)    
                
                if 1:
                    plt.figure()
                    plt.title('singular values decay')
                    plt.semilogy(EiDA, label = 'deltaA')
                    plt.semilogy(EiM, label = 'deltaM')
                    plt.legend()
                    plt.show()
                
                # baue terme füe B
                B_term = deltaB-S_M[-1]@R_T.T@B[i] #deltaB
                # baue terme für C
                         
                # append B, C_T
                B_new.append(B_term) 
                C_T_new.append(deltaC.T)
                
            # set dimensions
            envelope_system.output_dim = n_sys*switch_model.output_dim + sum(ranks_A)
            envelope_system.input_dim = n_sys*switch_model.input_dim + sum(ranks_A)
        
            # get new matrices
            B_new.extend(S_A)
            C_T_new.extend(U_A)
                
            # transform into matrices and transpose
            envelope_system.B = np.concatenate(B_new, axis = 1)
            envelope_system.C = np.concatenate(C_T_new, axis = 1).T
        
        # add data
        envelope_system.F = switch_model.F
        envelope_system.time_disc = model.time_disc
        envelope_system.sigma = switch_model.sigma
        envelope_system.type = 'FOM'
        
        self.envelope_system = envelope_system
        
        return envelope_system
    
    def compute_bt_basis(self, l_max):
        
        # create 
        time_stepper = ImplicitEulerTimeStepper(self.model.time_disc.dt)
        fom_env = LTIModel.from_matrices(A = -self.envelope_system.A, 
                                        B = self.envelope_system.B, 
                                        C = self.envelope_system.C, 
                                        E = self.envelope_system.M,
                                        sampling_time = 0,
                                        initial_data = self.envelope_system.y0,
                                        T = self.model.time_disc.T,
                                        time_stepper = time_stepper)
        
        # reduce sys
        bt_env = BTReductor(fom_env)
        
        if 1: # visualize eisngular values of all systems
            switch_model = self.model.pde
            # einzelne systeme
            fom0 = LTIModel.from_matrices(A = -switch_model.A[0], 
                                        B = switch_model.B[0], 
                                        C = switch_model.C[0], 
                                        E = switch_model.M[0],
                                        sampling_time = 0,
                                        initial_data = switch_model.y0,
                                        T = self.model.time_disc.T,
                                        time_stepper = time_stepper)
            fom1 = LTIModel.from_matrices(A = -switch_model.A[1], 
                                        B = switch_model.B[1], 
                                        C = switch_model.C[1], 
                                        E = switch_model.M[1],
                                        sampling_time = 0,
                                        initial_data = switch_model.y0,
                                        T = self.model.time_disc.T,
                                        time_stepper = time_stepper)
            
            # visualize singular values
            fig, ax = plt.subplots()
            ax.semilogy(fom0.hsv()[:500], '.-', label = r'System $\sigma = 1$')
            ax.semilogy(fom1.hsv()[:500], '.-', label = r'System $\sigma = 2$')
            ax.semilogy(fom_env.hsv()[:500], '.-',label = r'Envelope system')
            ax.set_xlabel(r'$l$')
            plt.legend()
            _ = ax.set_title(r'Hankel singular values')
            
            # visualize 
            error_bounds0 = bt_env.error_bounds()
            hsv = fom_env.hsv()
            fig, ax = plt.subplots()
            ax.semilogy(range(1, len(error_bounds0) + 1), error_bounds0, '.-')
            ax.semilogy(range(1, len(hsv)), hsv[1:], '.-')
            ax.set_xlabel('Reduced order l')
            _ = ax.set_title(r'Upper and lower $\mathcal{H}_\infty$ error bounds, system 1')
                            
        # construct rom
        # rom_env = bt_env.reduce(r = l_max)
        
        # get projection matrices and project 
        self.hsv_env = fom_env.hsv()
        
        # northonomrlaize w.r.t. space product..
        if self.projection_product is not None:
            assert self.projection_product is not None, 'insert projection product ...'
            # map self.projection_productntp operator
            from pymor.operators.numpy import NumpyMatrixOperator
            proj = NumpyMatrixOperator(self.projection_product, source_id = 'STATE', range_id = 'STATE')
            W = gram_schmidt(bt_env.W, product=proj)
            V = gram_schmidt(bt_env.V, product=proj)
        else:
            W = bt_env.W
            V = bt_env.V
        
        W, V = W.to_numpy(), V.to_numpy()
        self.Wbig = W # W.T left
        self.Vbig = V # V.T right
        
        return W, V

    def check_orthogonality(self):
        if self.projection_product is None:
            print((self.Vbig@self.Vbig.T).T)
            print((self.Wbig@self.Wbig.T).T)
        else:
             print((self.Vbig @ self.projection_product@ self.Vbig.T).T)
             print((self.Wbig @ self.projection_product@ self.Wbig.T).T)
class trivial_reductor():
    
    def __init__(self, model):
        self.model = model
        
    def ROMtoFOM(self, u):
        return u
    def FOMtoROM(self, U):
        return U
