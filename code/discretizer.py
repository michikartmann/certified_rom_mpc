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
# Description: this file contains the discretization routine to get the FOM.

import fenics as fenics
import numpy as np
from scipy.sparse import csr_matrix, diags, identity
from scipy.sparse.linalg import factorized, eigs
from methods import collection
import scipy as sp
from model import model

def discretize1dproblem(T = 6, dx = 1, K = 200 , debug = False, control_type = 'discrete', model_options = None, reaction_parameter = 3):
    
    ######### Time discretization
    time_disc = collection()
    time_disc.t0 = 0
    time_disc.T = T
    time_disc.K = K
    time_disc.dt = (time_disc.T-time_disc.t0)/(time_disc.K-1)
    time_disc.t_v = np.linspace(time_disc.t0, time_disc.T, num=time_disc.K )
    time_disc.D = time_disc.dt * np.ones(time_disc.K)
    time_disc.D_diag = diags(time_disc.D)
    
    tmp = time_disc.dt * np.ones(time_disc.K)
    tmp[0] = 1
    time_disc.D_diag_1 = diags(tmp)
    
    # domain and mesh
    x1 = 0; x2 = 1
    disc_density = 10*dx
    mesh = fenics.IntervalMesh(disc_density, x1, x2)
    V = fenics.FunctionSpace(mesh, "P", 1)
    
    def DirichletBoundary(x, on_boundary):
         tol = 1e-14
         return on_boundary and abs(x[0]-x1) < tol
    tol = 1e-14
    class BoundaryL(fenics.SubDomain): #left
        def inside(self, x, on_boundary):
            return on_boundary and fenics.near(x[0], x1, tol) #x1

    class BoundaryR(fenics.SubDomain): #right
        def inside(self, x, on_boundary):
            return on_boundary and fenics.near(x[0], x2, tol) #x2
        
    boundary_markers = fenics.MeshFunction('size_t', mesh, mesh.topology().dim()-1)
    boundary_markers.set_all(0)
    bx0 = BoundaryL()
    bx1 = BoundaryR()
    bx0.mark(boundary_markers, 0)
    bx1.mark(boundary_markers, 1)

    # Redefine boundary integration measure
    ds = fenics.Measure('ds', domain=mesh, subdomain_data=boundary_markers)
    
    # Enforcing u = u0 at x = 0
    # dirichlet boundary data
    u0 = fenics.Constant(0)
    bc = fenics.DirichletBC(V, u0, DirichletBoundary)
    bc_vals = bc.get_boundary_values()
    
    def apply_dirichlet_data(LHS, rhs):
        # dirichlet boundary data
        rhs = [bc_vals[k] if k in bc_vals else rhs[k] for k in range(rhs.size)]
        rhs = np.array(rhs).T
        return LHS, rhs
    
    # set up problem
    y = fenics.TrialFunction(V)
    v = fenics.TestFunction(V)
    f = fenics.Constant(0)
    y0 = fenics.Expression('0.2*sin(pi*x[0])', degree = 1) #fenics.Constant(0.0)
    
    # M
    M = fenics.assemble(fenics.inner(y,v)*fenics.dx)                              # mass matrix (time operator)
    M = csr_matrix(fenics.as_backend_type(M).mat().getValuesCSR()[::-1])
    M_time_coefficient = lambda t: 1
    
    # get coefficients for A and A
    A_time_coefficient = [ lambda t: 1,  # diff
                           lambda t: 0,  # adv
                           lambda t: -reaction_parameter-0*t**2 - 0*np.exp(t)+ 0*np.cos(10*np.pi*t) ] # reac#
    
    a_diff = fenics.assemble(fenics.dot(fenics.grad(y), fenics.grad(v))*fenics.dx)
    a_reac = fenics.assemble(y*v*fenics.dx)
    
    a_adv = 0*fenics.assemble(y*v*fenics.dx)
    
    
    bc.apply(a_reac); bc.apply(a_diff); bc.apply(a_adv)
    A1_diff = csr_matrix(fenics.as_backend_type(a_diff).mat().getValuesCSR()[::-1])    
    A2_adv = csr_matrix(fenics.as_backend_type(a_adv).mat().getValuesCSR()[::-1]) 
    A3_reac = csr_matrix(fenics.as_backend_type(a_reac).mat().getValuesCSR()[::-1]) 
    A = [A1_diff, A2_adv, A3_reac]
    
    ##### rhs F and initial data y0
    L = fenics.assemble( f*v*fenics.dx)
    F = L.get_local()
    F = np.tile(F, [time_disc.K,1]).T
    assert np.linalg.norm(F,2)<1.14, 'error F, we need zero rhs'
    y0 = fenics.interpolate(y0, V).vector().get_local()
    B = np.zeros((M.shape[0],))
    B[0] = 1 
    B = csr_matrix(B).T
    control_dofs = 1#B.shape[1]
    B_time_coefficient = lambda t: 1
    
    # C
    C = identity(len(y0))
    C_time_coefficient = lambda t: 1
    
    # get products
    L2_ = fenics.assemble(y*v * fenics.dx )   
    H10_ = fenics.assemble(fenics.dot(fenics.nabla_grad(y),fenics.nabla_grad(v)) * fenics.dx ) 
    H1_ = L2_ + H10_
    
    # get standard products
    L2 = csr_matrix(fenics.as_backend_type(L2_).mat().getValuesCSR()[::-1])
    H10 = csr_matrix(fenics.as_backend_type(H10_).mat().getValuesCSR()[::-1])
    H1 = csr_matrix(fenics.as_backend_type(H1_).mat().getValuesCSR()[::-1])
    
    # get dirichlet cleared products
    bc.apply(L2_); bc.apply(H10_); bc.apply(H1_)
    L2_0 = csr_matrix(fenics.as_backend_type(L2_).mat().getValuesCSR()[::-1])
    H10_0 = csr_matrix(fenics.as_backend_type(H10_).mat().getValuesCSR()[::-1])
    H1_0 = csr_matrix(fenics.as_backend_type(H1_).mat().getValuesCSR()[::-1])
    
    ### collect pde
    pde = collection()
    pde.A =  A 
    pde.A_time_coefficient = A_time_coefficient
    pde.M = [M]  
    pde.M_time_coefficient = M_time_coefficient
    pde.F = F
    pde.B = [B]
    pde.B_time_coefficient = B_time_coefficient
    pde.C = [C]
    pde.C_time_coefficient = C_time_coefficient
    pde.y0 = y0
    pde.state_dim = len(y0)
    pde.input_dim = control_dofs
    pde.output_dim = np.shape(C)[0]
    pde.products = {'H1': H1, 'L2': L2, 'H10': H10, 'H10_0': H10_0, 'H1_0': H1_0, 'L2_0': L2_0}
    pde.type = 'TimeVaryingFOMDirichlet'
    # for rietzrepresentatives
    pde.factorizedV1 = factorized(H1_0)
    pde.factorizedV2 = factorized(H10_0)
    pde.error_est_constants = {'cm': 1,
                               'eta1': 1,
                               'eta2': reaction_parameter
                               }
    
    ### space disc
    space_disc = collection()
    space_disc.V = V
    space_disc.mesh = mesh
    space_disc.Nx = disc_density
    space_disc.DirichletBC = bc
    space_disc.DirichletClearFun = apply_dirichlet_data
    
    ### cost data
    cost_data = collection()
    g_reg = collection()   
    # set special type
    if 0:
        
        g_reg.weight = 1e-1
        g_reg.type = 'box_cons'
        g_reg.g = lambda u: 0
        g_reg.u_low = -2
        g_reg.u_up = 3.5
        g_reg.prox_operator = lambda alpha_inv, u: np.clip(u, g_reg.u_low, g_reg.u_up)
    
    elif 0:
        
        g_reg.weight = 1e-1
        g_reg.type = 'l1_space_time' 
        g_reg.g = lambda u: None
        g_reg.prox_operator = lambda alpha_inv, u: prox_l1_space_time(alpha_inv, u, g_reg.weight)
        
    elif 0:
        
        g_reg.weight = 1e-1
        g_reg.type = 'l1_space_time_box_conx' 
        g_reg.g = lambda u: None
        g_reg.u_low = -2
        g_reg.u_up = 3.5
        g_reg.prox_operator = lambda alpha_inv, u: np.clip(prox_l1_space_time(alpha_inv, u, g_reg.weight), g_reg.u_low, g_reg.u_up)
    
    else:
        g_reg.weight = 0
        
    # add it to the cost data
    cost_data.g_reg = g_reg
    if not g_reg.weight == 0:
       pde.type += 'nonsmooth_g'
       
    cost_data.weights = [1, 1, 0]
    cost_data.Yd = None
    cost_data.Ud = None
    cost_data.YT = None
    cost_data.output_product = L2#H1 #L2 
    if control_type == 'discrete':
        cost_data.input_product = identity(control_dofs)
    elif control_type == 'continuous': 
        # continuous distributed
        assert 0, 'implement this...'
    else:
        # continuous boundary
        assert 0, 'implement this...'
    
    # create fom
    fom = model(pde, cost_data, time_disc, space_disc, model_options)

    return fom

def discretize_stability_problem(T = 6, dx = 1, K = 200 , debug = False, control_type = 'discrete', model_options = None):
    
    
    assert 0, 'when dealing with M not constant then modify in time-stepping the discretization of M'
    ######### Time discretization
    time_disc = collection()
    time_disc.t0 = 0
    time_disc.T = T
    time_disc.K = K
    time_disc.dt = (time_disc.T-time_disc.t0)/(time_disc.K-1)
    time_disc.t_v = np.linspace(time_disc.t0, time_disc.T, num=time_disc.K )
    time_disc.D = time_disc.dt * np.ones(time_disc.K)
    time_disc.D_diag = diags(time_disc.D)
    tmp = time_disc.dt * np.ones(time_disc.K)
    tmp[0] = 1
    time_disc.D_diag_1 = diags(tmp)
    Lx = 1
    Ly = 1
    dx = 1
    Nx = 50*dx
    Ny = 50*dx
    x1 = 0
    y1 = 0
    x2 = Lx
    y2 = Ly
    lower_left = fenics.Point(x1,y1)
    upper_right = fenics.Point(x2,y2)
    mesh = fenics.RectangleMesh(lower_left, upper_right, Nx, Ny)
    
    tol = 1e-14
    class BoundaryL(fenics.SubDomain): #left
        def inside(self, x, on_boundary):
            return on_boundary and fenics.near(x[0], x1, tol) #x1

    class BoundaryR(fenics.SubDomain): #right
        def inside(self, x, on_boundary):
            return on_boundary and fenics.near(x[0], x2, tol) #x2
           
    class BoundaryLow(fenics.SubDomain): #low
        def inside(self, x, on_boundary):
            return on_boundary and fenics.near(x[1], y1, tol) #y1
        
    class BoundaryUp(fenics.SubDomain): #up
        def inside(self, x, on_boundary):
            return on_boundary and fenics.near(x[1], y2, tol) #y2
        
    boundary_markers = fenics.MeshFunction('size_t', mesh, mesh.topology().dim()-1)
    boundary_markers.set_all(0)
    bx0 = BoundaryL()
    bx1 = BoundaryR()
    by0 = BoundaryLow()
    by1 = BoundaryUp()
    bx0.mark(boundary_markers, 0)
    bx1.mark(boundary_markers, 1)
    by0.mark(boundary_markers, 2)
    by1.mark(boundary_markers, 3)

    # Redefine boundary integration measure
    ds = fenics.Measure('ds', domain=mesh, subdomain_data=boundary_markers)
    
    ###### variational problem
    V = fenics.FunctionSpace(mesh, 'P', 1) 
    y = fenics.TrialFunction(V)
    v = fenics.TestFunction(V)
    
    # data
    # gamma_c = fenics.Constant(0.0)
    gamma_out = fenics.Constant(1) 
    y_out = fenics.Constant(0.0) 
    # y_c = fenics.Constant(1.0)
    f = fenics.Constant(0.0)
    y0 = fenics.Expression('0.2*sin(pi*x[0])*sin(pi*x[1])', degree = 1) #fenics.Constant(0.0)
    
    #### B control operator: distributed, discrete control
    # number of inputs
    n_u_x = 2
    n_u_y = 2
    shift = 0 
    ints_y = np.linspace(0+shift,Ly-shift,n_u_y+1)
    ints_x =  np.linspace(0+shift,Lx-shift,n_u_x+1)
    chars = []
    integrals_R_b = []
    for i_y in range(len(ints_y)-1):
        for i_x in range(len(ints_x)-1):
            chars.append(fenics.Expression('l0- tol<= x[0] && x[0] <= u0 + tol && l1 - tol<= x[1] && x[1] <= u1 + tol ? v1 : v2', 
                                      degree=0, tol=tol, v1=1, v2=0, 
                                      l0 = ints_x[i_x], u0 = ints_x[i_x+1], 
                                      l1 = ints_y[i_y], u1 = ints_y[i_y+1]))
            integrals_R_b.append((chars[-1]*v*fenics.dx))
    
    B = []
    for i in integrals_R_b:
        B.append(fenics.assemble(i).get_local()) 
    B = np.array(B).T
    control_dofs = B.shape[1]
    B_time_coefficient = lambda t: 1

    #### M
    M = fenics.assemble( y * v * fenics.dx )                                    # mass matrix (time operator)
    M = csr_matrix(fenics.as_backend_type(M).mat().getValuesCSR()[::-1])
    M_time_coefficient = lambda t: 1
    
    #### A  
    ## boundary conditions
    boundary_conditions = {0: {'Robin': (gamma_out, y_out, 'no_control')},       # x = 0, left
                           1: {'Robin': (gamma_out, y_out,'no_control')},       # x = 1, right
                           2: {'Robin': (gamma_out, y_out,'no_control')},       # y = 0, lower
                           3: {'Robin': (gamma_out, y_out,'no_control')}}       # y = 1, upper
    integrals_R_a = []
    integrals_R_L = []
    # integrals_R_b = []
    for i in boundary_conditions:
        if 'Robin' in boundary_conditions[i]:
            gamma_, y_, string = boundary_conditions[i]['Robin']
            integrals_R_a.append(gamma_*y*v*ds(i))
            integrals_R_L.append(y_*v*ds(i))
        else:
            assert 0, 'implement'
    
    # get coefficients for A and A
    A_time_coefficient = [      lambda t: 1, # diff
                                lambda t: 0,  # adv
                                lambda t: 1+np.sin(t)]      # reac
    a_diff = fenics.Constant(1)
    a_adv = fenics.Expression(("-0.01*(x[0]+x[1])","(x[1]*x[0])/2"), degree = 2)
    a_reac = fenics.Constant(1.0)
    A1_diff = fenics.assemble( a_diff * fenics.dot(fenics.nabla_grad(y),
            fenics.nabla_grad(v)) * fenics.dx )  
    A1_diff = csr_matrix(fenics.as_backend_type(A1_diff).mat().getValuesCSR()[::-1])    
    A2_adv = fenics.assemble( fenics.dot(fenics.nabla_grad(y), a_adv) * v * fenics.dx)
    A2_adv = csr_matrix(fenics.as_backend_type(A2_adv).mat().getValuesCSR()[::-1]) 
    A3_reac = fenics.assemble( a_reac*y * v * fenics.dx )
    A3_reac = csr_matrix(fenics.as_backend_type(A3_reac).mat().getValuesCSR()[::-1]) 
    A = [A1_diff, A2_adv, A3_reac]
    
    ##### rhs F and initial data y0
    L = fenics.assemble( f*v*fenics.dx + sum(integrals_R_L))
    F = L.get_local()
    F = np.tile(F, [time_disc.K,1]).T
    assert np.linalg.norm(F,2)<1.14, 'error F, we need zero rhs'
    y0 = fenics.interpolate(y0, V).vector().get_local()
    # assert np.linalg.norm(F,2)<1.14, 'error y0, we need zero initial data'
           
    #### C
    C = identity(len(y0))
    C_time_coefficient = lambda t: 1
    
    # get products
    L22 = fenics.assemble(y*v * fenics.dx )   
    H10 = fenics.assemble(fenics.dot(fenics.nabla_grad(y),fenics.nabla_grad(v)) * fenics.dx ) 
    L2 = csr_matrix(fenics.as_backend_type(L22).mat().getValuesCSR()[::-1])
    H10 = csr_matrix(fenics.as_backend_type(H10).mat().getValuesCSR()[::-1])
    H1 = H10 + L2
   
    ### collect pde
    pde = collection()
    pde.A =  A 
    pde.A_time_coefficient = A_time_coefficient
    pde.M = [M]  
    pde.M_time_coefficient = M_time_coefficient
    pde.F = F
    pde.B = [B]
    pde.B_time_coefficient = B_time_coefficient
    pde.C = [C]
    pde.C_time_coefficient = C_time_coefficient
    pde.y0 = y0
    pde.state_dim = len(y0)
    pde.input_dim = control_dofs
    pde.output_dim = np.shape(C)[0]
    pde.products = {'H1': H1, 'L2': L2, 'H10': H10}
    ### TODO
    pde.type = 'TimeVaryingFOM'
    pde.factorizedV1 = factorized(H1)
    
    # TODO modify this
    pde.error_est_constants = {'cm': None,
                               'eta1': None,
                               'eta2': None
                               }
    
    ### space disc
    space_disc = collection()
    space_disc.V = V
    space_disc.mesh = mesh
    space_disc.Nx = Nx
    space_disc.Ny = Ny
    space_disc.DirichletBC = None
    
    ### cost data
    cost_data = collection()
    g_reg = collection()   
    # set special type
    if 0:

        g_reg.weight = 1e-1
        g_reg.type = 'box_cons' 
        g_reg.g = lambda u: 0
        g_reg.u_low = -2
        g_reg.u_up = 3.5
        g_reg.prox_operator = lambda alpha_inv, u: np.clip(u, g_reg.u_low, g_reg.u_up)
    
    elif 0:
        
        g_reg.weight = 1e-1
        g_reg.type = 'l1_space_time' 
        g_reg.g = lambda u: None
        g_reg.prox_operator = lambda alpha_inv, u: prox_l1_space_time(alpha_inv, u, g_reg.weight)
        
    elif 0:
        
        g_reg.weight = 1e-1
        g_reg.type = 'l1_space_time_box_conx' 
        g_reg.g = lambda u: None
        g_reg.u_low = -2
        g_reg.u_up = 3.5
        g_reg.prox_operator = lambda alpha_inv, u: np.clip(prox_l1_space_time(alpha_inv, u, g_reg.weight), g_reg.u_low, g_reg.u_up)
    
    else:
        g_reg.weight = 0
        
    # add it to the cost data
    cost_data.g_reg = g_reg
    if not g_reg.weight == 0:
       pde.type += 'nonsmooth_g'
       
    cost_data.weights = [1, 1e-3, 0]
    cost_data.Yd = None
    cost_data.Ud = None
    cost_data.YT = None
    cost_data.output_product = L2
    if control_type == 'discrete':
        cost_data.input_product = identity(control_dofs)
    elif control_type == 'continuous': 
        # continuous distributed
        assert 0, 'implement this...'
    else:
        # continuous boundary
        assert 0, 'implement this...'
    
    ### create fom
    fom = model(pde, cost_data, time_disc, space_disc, model_options)
    return fom

def discretize(T = 6, dx = 1, K = 200 , debug = False, state_dependent_switching = False, control_type = 'discrete', model_options = None, reac_cons = 0.01, nonsmooth = False, use_energy_products = False):
     
    ######### Time discretization
    time_disc = collection()
    time_disc.t0 = 0
    time_disc.T = T
    time_disc.K = K
    time_disc.dt = (time_disc.T-time_disc.t0)/(time_disc.K-1)
    time_disc.t_v = np.linspace(time_disc.t0, time_disc.T, num=time_disc.K )
    time_disc.D = time_disc.dt * np.ones(time_disc.K)
    time_disc.D_diag = diags(time_disc.D)
    tmp = time_disc.dt * np.ones(time_disc.K)
    tmp[0] = 1
    time_disc.D_diag_1 = diags(tmp)
    
    ########## ROOM GEOMETRY and MESHING
    # room geometry in meteter
    L_room1 = 5
    L_room2 = 5
    L_door = 0.3
    L_door_y = 0.4
    Ly = 5
    door_shift = -1 # negativ, nach unten, positiv nach oben...
    L_wall_y = (Ly-L_door_y)/2
    assert abs(L_wall_y-door_shift)>0, 'door gets shifted too strong ....'
    Lx = L_room1 + L_room2 + L_door
    room1_area = (L_room1*Ly)
    room2_area = (L_room1*Ly)
    Nx = (2*50+3)*dx
    Ny = (50)*dx
    x1 = 0
    y1 = 0
    x2 = Lx
    y2 = Ly
    lower_left = fenics.Point(x1,y1)
    upper_right = fenics.Point(x2,y2)
    mesh = fenics.RectangleMesh(lower_left, upper_right, Nx, Ny)

    if debug:
        fenics.plot(mesh)

    tol = 1e-14
    class BoundaryL(fenics.SubDomain): #left
        def inside(self, x, on_boundary):
            return on_boundary and fenics.near(x[0], x1, tol) #x1

    class BoundaryR(fenics.SubDomain): #right
        def inside(self, x, on_boundary):
            return on_boundary and fenics.near(x[0], x2, tol) #x2
           
    class BoundaryLow(fenics.SubDomain): #low
        def inside(self, x, on_boundary):
            return on_boundary and fenics.near(x[1], y1, tol) #y1
        
    class BoundaryUp(fenics.SubDomain): #up
        def inside(self, x, on_boundary):
            return on_boundary and fenics.near(x[1], y2, tol) #y2
        
    boundary_markers = fenics.MeshFunction('size_t', mesh, mesh.topology().dim()-1)
    boundary_markers.set_all(0)
    bx0 = BoundaryL()
    bx1 = BoundaryR()
    by0 = BoundaryLow()
    by1 = BoundaryUp()
    bx0.mark(boundary_markers, 0)
    bx1.mark(boundary_markers, 1)
    by0.mark(boundary_markers, 2)
    by1.mark(boundary_markers, 3)

    # Redefine boundary integration measure
    ds = fenics.Measure('ds', domain=mesh, subdomain_data=boundary_markers)

    if debug:
        # Print all vertices that belong to the boundary parts
        for x in mesh.coordinates():   
            # print(x)
            if x[0]==L_room1 and x[1]==0: print('left door point on grid')
            if abs((x[0]-L_room1+L_door))<tol and x[1]==0: print('right door point on grid')
            if bx0.inside(x, True): print('%s is on x = 0' % x)
            if bx1.inside(x, True): print('%s is on x = 10.3' % x)
            if by0.inside(x, True): print('%s is on y = 0' % x)
            if by1.inside(x, True): print('%s is on y = 5' % x)
    
    
    ######### variational problem ######################
    V = fenics.FunctionSpace(mesh, 'P', 1) 
    y = fenics.TrialFunction(V)
    v = fenics.TestFunction(V)
    
    # switching parameter
    k1 = 0.01                  
    k2 = 10
    xi1 = 1                    
    xi2 = 0.5
    k_wall = k1
    xi_wall = xi1
    
    # get charateristics for walls, rooms and windows
    char_room1 = fenics.Expression('(l0- tol<= x[0]  && x[0] < u0 - tol )? v1 : v2', degree=0, tol=tol, v1=1, v2=0, l0 = 0, u0 = L_room1) 
    char_room2 = fenics.Expression('l0+tol< x[0]  && x[0] <= u0 + tol? v1 : v2', degree=0, tol=tol, v1=1, v2=0, l0 =  L_room1 + L_door, u0 = Lx) 
    char_door = fenics.Expression('l0- tol<= x[0] && x[0] <= u0 + tol && l1 - tol<= x[1] && x[1] <= u1 + tol ? v1 : v2', degree=0, tol=tol, v1=1, v2=0, l0 = L_room1, u0 = L_room1 + L_door, l1 = L_wall_y+door_shift, u1 = L_wall_y+L_door_y+door_shift) 
    char_wall_low = fenics.Expression('l0- tol<= x[0] && x[0] <= u0 + tol && l1 - tol<= x[1] && x[1] < u1 - tol ? v1 : v2', degree=0, tol=tol, v1=1, v2=0, l0 = L_room1, u0 = L_room1 + L_door, l1 = 0, u1 = L_wall_y+door_shift) 
    char_wall_up = fenics.Expression('l0- tol<= x[0] && x[0] <= u0 + tol && l1 + tol< x[1] && x[1] <= u1 + tol ? v1 : v2', degree=0, tol=tol, v1=1, v2=0, l0 = L_room1, u0 = L_room1 + L_door, l1 = L_wall_y+L_door_y+door_shift, u1 = Ly) 
    domain_list = [char_room1, char_room2,char_door, char_wall_low, char_wall_up ]
    
    # split control boundary 
    nu_y = 10 
    uuu = np.linspace(0, Ly, nu_y+1)
    tol = 1e-14
    expres = []
    ints = []
    for i in range(len(uuu)-1):
        l00 = uuu[i]
        u00 = uuu[i+1]
        ints.append([l00,u00])
        expres.append(fenics.Expression('(l0 <= x[1]  && x[1] < u0 - tol )? v1 : v2', degree=0, tol=tol, v1=1, v2=0, l0 = l00, u0 = u00))
    # print(ints)
    
    if debug:
        print('debugging domain splitting in wall etc ...')
        f_p = fenics.Function(V)
        f_p2 = fenics.Function(V)
        f_p3 = fenics.Function(V)
        f_p4 = fenics.Function(V)
        f_p5 = fenics.Function(V)
        f_p.interpolate(char_room1)
        f_p2.interpolate(char_room2)
        f_p3.interpolate(char_door)
        f_p4.interpolate(char_wall_low)
        f_p5.interpolate(char_wall_up)
        
        a = np.argwhere(f_p.vector().get_local())
        print(a.shape)
        a2 = np.argwhere(f_p2.vector().get_local())
        print(a2.shape)
        a3 = np.argwhere(f_p3.vector().get_local())
        print(a3.shape)
        a4 = np.argwhere(f_p4.vector().get_local())
        print(a4.shape)
        a5 = np.argwhere(f_p5.vector().get_local())
        print(a5.shape)
        print(f'sum: {a.shape[0]+a2.shape[0]+a3.shape[0]+a4.shape[0]+a5.shape[0]}')
        
        c = list(a[:,0]) + list(a2[:,0]) + list(a3[:,0]) +list(a4[:,0] )+list(a5[:,0] )
        c.sort()

#%% inputs and outputs  

    # data
    gamma_c = fenics.Constant(0.0)
    gamma_out = fenics.Constant(0.1) 
    y_out = fenics.Constant(0.0) 
    y_c = fenics.Constant(1.0)
    f = fenics.Constant(0.0)
    y0 = fenics.Constant(0.0)
    
    if 1: # boundary control 
        # number of inputs
        n_u = 20
        ints = np.linspace(0,Ly,n_u+1)
        intervalls = [[i,j] for i, j in zip(ints[0:-1], ints[1:])]
        chars_leftbndry = []
        for interval in intervalls:
            chars_leftbndry.append(fenics.Expression('(l0- tol<= x[1]  && x[1] < u0 - tol )? v1 : v2', degree=0, tol=tol, v1=1, v2=0, l0 = interval[0], u0 = interval[1]))
    
        # BC
        boundary_conditions = {0: {'Robin': (gamma_c, y_c, control_type)},   # x = 0, left
                               1: {'Robin': (gamma_out, y_out,'no_control')},   # x = 1, right
                               2: {'Robin': (gamma_out, y_out,'no_control')}, # y = 0, lower
                               3: {'Robin': (gamma_out, y_out,'no_control')}}      # y = 1, upper
     
        # Collect Robin integrals
        integrals_R_a = []
        integrals_R_L = []
        integrals_R_b = []
        for i in boundary_conditions:
            if 'Robin' in boundary_conditions[i]:
                gamma_, y_, string = boundary_conditions[i]['Robin']
                integrals_R_a.append(gamma_*y*v*ds(i))
                if string == 'continuous': #then it is a matrix, this means we control on the dofs of gamma c
                   assert control_type == 'continuous', 'change control type'
                   integrals_R_b.append(y_*y*v*ds(i))
                elif string == 'discrete': #this means we control only a few parameters on the dofs of gamma c
                    assert control_type == 'discrete', 'change control type'
                    
                    for characteristic in chars_leftbndry:
                        integrals_R_b.append(characteristic*y_*v*ds(i))

                else:
                    integrals_R_L.append(y_*v*ds(i))
         
        # B        
        if control_type == 'continuous':         
            B = fenics.assemble(sum(integrals_R_b))
            B = csr_matrix(fenics.as_backend_type(B).mat().getValuesCSR()[::-1])
        elif control_type == 'discrete': 
            B = []
            for i in  integrals_R_b:
                B.append(fenics.assemble(i).get_local()) 
            B = np.array(B).T
        
        else:
            assert 0, 'pls insert correct control type'
        control_dofs = B.shape[1]
        B2 = np.zeros(B.shape)#csc_matrix(B.shape)
        
    else: # distributed control
        
        assert 0, 'implement distributed control'
        # BC
        boundary_conditions = {0: {'Robin': (gamma_out, y_out, 'no_control')},   # x = 0, left
                               1: {'Robin': (gamma_out, y_out,'no_control')},   # x = 1, right
                               2: {'Robin': (gamma_out, y_out,'no_control')}, # y = 0, lower
                               3: {'Robin': (gamma_out, y_out,'no_control')}}      # y = 1, upper
     
        # Collect Robin integrals
        integrals_R_a = []
        integrals_R_L = []
        integrals_R_b = []
        for i in boundary_conditions:
            if 'Robin' in boundary_conditions[i]:
                gamma_, y_, string = boundary_conditions[i]['Robin']
                integrals_R_a.append(gamma_*y*v*ds(i))
                integrals_R_L.append(y_*v*ds(i))
            else:
                assert 0, 'implement'
        
    C1 = fenics.assemble( char_room1*v*fenics.dx ).get_local()/room1_area
    C2 = fenics.assemble( char_room2*v*fenics.dx ).get_local()/room2_area
    C = np.array([C1, C2])
    
#%% rest
    Massembled_list = []
    Dassembled_list = []
    for char in domain_list:
        tmp = fenics.assemble( char*y * v * fenics.dx )
        Massembled_list.append( csr_matrix(fenics.as_backend_type(tmp).mat().getValuesCSR()[::-1]) ) 
        tmpd = fenics.assemble(char * fenics.dot(fenics.nabla_grad(y),fenics.nabla_grad(v)) * fenics.dx)
        Dassembled_list.append( csr_matrix(fenics.as_backend_type(tmpd).mat().getValuesCSR()[::-1]))

    A_integrals = fenics.assemble(sum(integrals_R_a)   )
    A_integrals = csr_matrix(fenics.as_backend_type(A_integrals).mat().getValuesCSR()[::-1])

    L = fenics.assemble( f*v*fenics.dx + sum(integrals_R_L))
    F = L.get_local()
    F = np.tile(F, [time_disc.K,1]).T
    assert np.linalg.norm(F,2)<1.14, 'error F, we need zero rhs'

    y0 = fenics.interpolate(y0, V).vector().get_local()
    assert np.linalg.norm(F,2)<1.14, 'error y0, we need zero initial data'
    
    # get products
    L22 = fenics.assemble(y*v * fenics.dx )   
    H10 = fenics.assemble(fenics.dot(fenics.nabla_grad(y),fenics.nabla_grad(v)) * fenics.dx ) 
    L2 = csr_matrix(fenics.as_backend_type(L22).mat().getValuesCSR()[::-1])
    H1 = csr_matrix(fenics.as_backend_type(H10).mat().getValuesCSR()[::-1]) + L2

    # advection: kein low rank switching......
    adv_constant = 0.01#0.01
    adv_fun = fenics.Expression(("1","0"), degree = 1)
    adv_tomp = fenics.assemble( fenics.dot(fenics.nabla_grad(y), adv_fun) * v * fenics.dx)
    A_adv = adv_constant*csr_matrix(fenics.as_backend_type(adv_tomp).mat().getValuesCSR()[::-1]) 
    
    reac_const = reac_cons
    A_reac_tmp = fenics.assemble( y * v * fenics.dx)
    A_reac = reac_const*csr_matrix(fenics.as_backend_type(A_reac_tmp).mat().getValuesCSR()[::-1]) 
    
    # construct mass and operator matrices
    if 1:
        M1_ = xi2*Massembled_list[0]+xi2*Massembled_list[1]+xi1*Massembled_list[2]+xi_wall*(Massembled_list[3]+Massembled_list[4])
        M2_ = xi2*Massembled_list[0]+xi2*Massembled_list[1]+xi2*Massembled_list[2]+xi_wall*(Massembled_list[3]+Massembled_list[4])
        A1_ = k2*Dassembled_list[0]+k2*Dassembled_list[1]+k1*Dassembled_list[2]+k_wall*(Dassembled_list[3]+Dassembled_list[4])+A_integrals + A_adv + A_reac
        A2_ = k2*Dassembled_list[0]+k2*Dassembled_list[1]+k2*Dassembled_list[2]+k_wall*(Dassembled_list[3]+Dassembled_list[4])+A_integrals + A_adv + A_reac
        
    # define switching law: 2: open door, 1: closed door
    if state_dependent_switching:
        def sigma(t, y, sigma):
            v1 = 0.05
            v2 = 0.15
            if t == 0:
                return 1
            if  y < v1 and sigma == 1:
                return 2 #open
            elif y > v2 and sigma == 2:
                return 1 #closed
            else:
                return sigma  
    else:
        def sigma(t, y, sigma = None): 
            if time_disc.t0 <= t <= 1.1 or 1.6<t<=3:
                return 2 
            elif 1.1< t <= 1.6 or 3<t<= time_disc.T:
                return 1
            
    # collect space_disk
    space_disc = collection()
    space_disc.V = V
    space_disc.mesh = mesh
    space_disc.Nx = Nx
    space_disc.Ny = Ny
    space_disc.DirichletBC = None
    
    # collect forward equation in model
    switch_model = collection()
    switch_model.A =  [A1_, A2_] 
    switch_model.M = [M1_, M2_]  
    switch_model.F = F
    switch_model.B = [B,B]
    switch_model.C = [C, C]
    switch_model.sigma = sigma
    switch_model.y0 = y0
    switch_model.state_dim = len(y0)
    switch_model.input_dim = control_dofs
    switch_model.output_dim = np.shape(C)[0]
    switch_model.products = {'H1': H1, 'L2': L2}
    switch_model.type = 'SwitchFOM'
    switch_model.nummber_modes = len(switch_model.C)
    
    if state_dependent_switching:
        switch_model.type += 'StateDep'
    if 0:
        switch_model.type += 'SymmetricA'
    switch_model.factorizedV1 = factorized(H1)
    
    # create energy products
    switch_model.energy_products = collection()
    switch_model.products['energy'] = [0.5*A1_ + 0.5*A1_.T, 0.5*A2_ + 0.5*A2_.T]
    # switch_model.energy_products.products = [0.5*A1_ + 0.5*A1_.T, 0.5*A2_ + 0.5*A2_.T]
    if model_options.energy_prod:
        switch_model.energy_products.factorized_energy_products = []
        for i in switch_model.products['energy']:
            switch_model.energy_products.factorized_energy_products.append(factorized(i))
    
    # compute coercivty constants of FOM
    if 1:
        # build cholesky of H1
        
        Acoerc = []
        for Ai in switch_model.products['energy']:#switch_model.A:
            
            if 0:
                vals, _ = eigs(A = Ai, M = H1, k = 1, which = 'SM')
                val = np.real(vals[0])
            elif 0:
                # perfrom cholesky of H1
                H1_chol = sp.linalg.cholesky(H1.todense())
                A_hat = H1_chol@Ai
                A_hat = np.linalg.solve( H1_chol, A_hat.T)
                _, S, _ = sp.linalg.svd(A_hat.T, full_matrices=False)
                ev = S**2
                val = min(ev)
                print(val)
            else:
                val = 0.015
            Acoerc.append(val)
    
    # compute weighted continuity constants
    BcontA = []
    CcontA = []
    CcontM = []
    if model_options.energy_prod or 1: # compute continuity constants w.r.t. energy product
        
        for i in range(len(switch_model.products['energy'])):
            
            # get energy product
            Ai = switch_model.products['energy'][i]
            Ai_chol = sp.linalg.cholesky(Ai.todense(), lower = True)#sp.linalg.sqrtm(Ai.todense())#sp.linalg.cholesky(Ai.todense())
            
            # BA
            bb = switch_model.B[i]
            Bhat = Ai_chol.T@bb
            _, S, _ = sp.linalg.svd(Bhat, full_matrices=False)
            BcontA.append(max(S))
            
            # CA
            cc = switch_model.C[i]
            Chat = cc@Ai_chol
            _, S, _ = sp.linalg.svd(Chat, full_matrices=False)
            CcontA.append(max(S))
            
            # CM
            Mi = switch_model.M[i]
            Mi_chol = sp.linalg.cholesky(Mi.todense(), lower = True)
            Chat = cc@Mi_chol
            _, S, _ = sp.linalg.svd(Chat, full_matrices=False)
            CcontM.append(max(S))
     
    Bcont = []
    Ccont = [] 
    H1_chol = sp.linalg.cholesky(H1.todense(), lower = True)
    for i in range(len(switch_model.products['energy'])):
        # BcontV
        Bhat = H1_chol.T@switch_model.B[i]
        _, S, _ = sp.linalg.svd(Bhat, full_matrices=False)
        Bcont.append(max(S))
        
        # CcontV
        Chat = switch_model.C[i]@H1_chol
        _, S, _ = sp.linalg.svd(Chat, full_matrices=False)
        Ccont.append(max(S))
    
    # compute switch_product constants
    c_ij = np.zeros((switch_model.nummber_modes,switch_model.nummber_modes))
    for i in range(len(switch_model.products['energy'])):
        for j in range(len(switch_model.products['energy'])):
            if i == j:
                c_ij[i,j] = 1
            else:
                vals, _ = eigs(A = switch_model.M[i], M = switch_model.M[j], k = 1, which = 'LM')
                val = np.real(vals[0])
                c_ij[i,j] = val
                
        
    # TODO modify this
    if reac_const>0:
        eta2 = 0
        eta1 =  min(k1, k2, reac_const)
    elif reac_const < 0:
        eta2 = min(k1, k2)+reac_const
        eta1 = min(k1, k2)
    else:
        eta2 = min(k1, k2)+reac_const
        eta1 = min(k1, k2)
        
    if model_options.energy_prod and 1:
        BC_scale = eta1
        switch_model.error_est_constants = {'cm': 1,
                                            'cM': 1,
                                            'eta1': 1,
                                            'eta2': 1,
                                            'A_coerc_modes':[],
                                            'c_C': max(Ccont),#np.sqrt(2), 
                                            'c_B': max(Bcont),#np.sqrt(5)
                                            'BcontA': BcontA,
                                            'CcontA': CcontA,
                                            'CcontM': CcontM,
                                            'cij': c_ij
                                            }
    else:
        switch_model.error_est_constants = {'cm': min(xi1,xi2),
                                            'cM': max(xi1,xi2),
                                            'eta1': min(Acoerc),#eta1,
                                            'eta2': eta2,
                                            'c_C': max(Ccont),#np.sqrt(2), 
                                            'c_B': max(Bcont),#np.sqrt(5)
                                            'BcontA': BcontA,
                                            'CcontA': CcontA,
                                            'CcontM': CcontM,
                                            'cij': c_ij
                                            }
    
    ##### cost function data
    cost_data = collection()
    g_reg = collection()   
    # set special type
    if nonsmooth:
        
        u_low = -20
        u_up = 20
        
        if nonsmooth == 'box':
            
            g_reg.weight = 1e-3
            g_reg.type = 'box' 
            g_reg.g = lambda u: 0
            g_reg.u_low = u_low
            g_reg.u_up = u_up
            g_reg.prox_operator = lambda alpha_inv, u: np.clip(u, g_reg.u_low, g_reg.u_up)
        
        elif nonsmooth == 'l1':
            
            g_reg.weight = 1e-3
            g_reg.type = 'l1_space_time' 
            g_reg.g = lambda u: np.linalg.norm(u,1)
            g_reg.prox_operator = lambda alpha_inv, u: prox_l1_space_time(alpha_inv, u, g_reg.weight)
            
            
        elif nonsmooth == 'l1box':
            
            g_reg.weight = 1e-3
            g_reg.type = 'l1_space_time_box_cons' 
            g_reg.g = lambda u: np.linalg.norm(u,1)
            g_reg.u_low = u_low
            g_reg.u_up = u_up
            g_reg.prox_operator = lambda alpha_inv, u: np.clip(prox_l1_space_time(alpha_inv, u, g_reg.weight), g_reg.u_low, g_reg.u_up)
        
        else:
            g_reg.weight = 0
            assert 0, 'invalid nonsmoothness choice'
            
    else:
        g_reg.weight = 0
        g_reg.type = None
    # add it to the cost data
    cost_data.g_reg = g_reg
    if not g_reg.weight == 0:
       switch_model.type += 'nonsmooth_g'
    cost_data.weights = [1, 1e-3, 1]
    cost_data.Ud = None
    cost_data.YT = None
    cost_data.Yd = None
    cost_data.output_product = identity(switch_model.output_dim)
    if control_type == 'discrete':
        cost_data.input_product = identity(control_dofs)
    elif control_type == 'continuous': 
        # continuous distributed
        assert 0, 'implement this...'
    else:
        # continuous boundary
        assert 0, 'implement this...'

    #### collect in model
    fom = model(switch_model, cost_data, time_disc, space_disc, model_options)

    return fom

def get_y0(V, fenics_expression):
    y0 = fenics.interpolate(fenics_expression, V).vector().get_local()
    return y0

def prox_l1_space_time(alpha_inv, u, weight_parameter):
    C2 = alpha_inv*weight_parameter
    out = 1*u
    out[u > C2] -= C2
    out[u < -C2] += C2
    out[(u <= C2) & (u >= -C2)] = 0
    return out
