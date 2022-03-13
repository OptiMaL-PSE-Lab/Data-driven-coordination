# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 23:52:04 2021

@author: dv516
"""

import pyomo.environ as pyo
from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition
import numpy as np
import cvxpy as cp
import scipy.linalg as LA

def nearest_PD(A):
    """Find the nearest positive-definite matrix to input

    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """

    B = (A + A.T) / 2
    _, s, V = LA.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        return A3

    spacing = np.spacing(LA.norm(A))

    I = np.eye(A.shape[0])
    k = 1
    while not isPD(A3):
        mineig = np.min(np.real(LA.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

    return A3


def isPD(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = LA.cholesky(B)
        return True
    except LA.LinAlgError:
        return False
  
def quadratic_fitting(X_mat, y_mat, discr = False):
    N, M = X_mat.shape[0], X_mat.shape[1]
    P = cp.Variable((M, M), PSD = True)
    q = cp.Variable((M, 1))
    # r = cp.Variable()
    X = cp.Parameter(X_mat.shape)
    y = cp.Parameter(y_mat.shape)
    X.value = X_mat
    y.value = y_mat
    quadratic = cp.bmat([cp.quad_form(X.value[i].reshape(-1,1), P) + \
                        q.T @ X.value[i].reshape(-1,1) - y.value[i] for i in range(N)])
    # quadratic = cp.quad_form(X, P) + q.T @ X 
    # quadratic = cp.quad_form(X, P) + q.T @ X + r
    obj = cp.Minimize(cp.norm(quadratic))
    if not discr:
        prob = cp.Problem(obj)
    else:
        const_P = [P >> np.eye(M)*1e-9]
        prob = cp.Problem(obj, constraints = const_P)
    if not prob.is_dcp():
        print("Problem is not disciplined convex. No global certificate")
    prob.solve()
    if prob.status not in ['unbounded', 'infeasible']:
        return P.value, q.value
    else:
        print(prob.status, ' CVX objective fitting call at: ')
        print('X matrix', X_mat)
        print('y array', y_mat)
        raise ValueError
        
def quadratic_discrimination(x_inside, y_outside):
    N, M, D = x_inside.shape[0], y_outside.shape[0], x_inside.shape[1]
    u = cp.Variable(N, pos = True)
    v = cp.Variable(M, pos = True)
    P = cp.Variable((D,D), PSD = True)
    q = cp.Variable((D, 1))
    r = cp.Variable()
    X = cp.Parameter(x_inside.shape, value = x_inside)
    Y = cp.Parameter(y_outside.shape)
    X.value = x_inside ; Y.value = y_outside
    const_u = [cp.quad_form(X.value[i].reshape(-1,1), P) + \
                        q.T @ X.value[i].reshape(-1,1) + r <= -(1 - u[i]) for i in range(N)]
    const_v = [cp.quad_form(Y.value[i].reshape(-1,1), P) + \
                        q.T @ Y.value[i].reshape(-1,1) + r >= (1 - v[i]) for i in range(M)]
    const_P = [P >> np.eye(D)*1e-9]
    # const_P = [P >> np.eye(D)*1]
    # const_P = [P >> 0]
    prob = cp.Problem(cp.Minimize(cp.sum(u) + cp.sum(v)), \
                      constraints = const_u + const_v + const_P)
    if not prob.is_dcp():
        print("Problem is not disciplined convex. No global certificate")
    prob.solve()
    if prob.status not in ['unbounded', 'infeasible']:
        return P.value, q.value, r.value
    else:
        print(prob.status, ' CVX ineq. classification call at: ')
        print('x_inside', x_inside)
        print('x_outside', y_outside)
        raise ValueError

def SQP_data(H_big, g_big, C_big, y_big, A_big, N_global, radius, mu = 1000, 
             constr_list = None, bounds = None):
    
    dy_list = []
    for i in y_big.keys():
        dy_list += [cp.Variable(y_big[i].shape)]
    quad_expr = []
    consensus = []
    constraints = []

    keys = list(y_big.keys())
    for i in range(len(y_big)):
        y_ = cp.Parameter(y_big[keys[i]].shape, value = y_big[keys[i]])
        H_ = cp.Parameter(H_big[keys[i]].shape, value = H_big[keys[i]], PSD = True)
        A_ = cp.Parameter(A_big[keys[i]].shape, value = A_big[keys[i]])
        g_ = cp.Parameter(g_big[keys[i]].shape, value = g_big[keys[i]])
        quad_expr += [cp.quad_form(dy_list[i], H_) + g_.T @ dy_list[i]]
        consensus += [A_ @ (y_ + dy_list[i])]
        #print(consensus[0].shape, A_.shape, y_.shape, dy_list[0].shape, y_big[1].shape)
        if C_big[keys[i]] is not None:
            P_ineq = cp.Parameter(C_big[keys[i]][0].shape, value = C_big[keys[i]][0], PSD = True)
            q_ineq = cp.Parameter(C_big[keys[i]][1].shape, value = C_big[keys[i]][1])
            r_ineq = cp.Parameter(value = C_big[keys[i]][2])
            # constraints += [cp.quad_form(y_ + dy_list[i], P_ineq) + q_ineq.T @ (y_ + dy_list[i]) + r_ineq << 0 ]
            constraints += [cp.quad_form(dy_list[i], P_ineq) + q_ineq.T @ (dy_list[i]) + r_ineq <= 0 ]
        constraints += [cp.norm(dy_list[i]) <= radius]
    
    if bounds is not None:
        for i in range(len(dy_list)):
            N_var = len(y_big[keys[i]]) ; y_ = y_big[keys[i]]
            constraints += [bounds[j,0] <=  y_[j] + dy_list[i][j] for j in range(N_var)]
            constraints += [y_[j] + dy_list[i][j] <= bounds[j,1]  for j in range(N_var)]
    
    # for i in range(len(y_big) - 1):
    #     constraints += [dy_list[i] == dy_list[i+1]]
        
    if N_global >= 1:
        s = cp.Variable((N_global, 1))   
        
        objective = cp.Minimize(cp.sum(quad_expr) + mu/2*cp.sum_squares(s))
    #mu/2*cp.square(s))
        constraints += [cp.sum(consensus) == s]
    else:
        objective = cp.Minimize(cp.sum(quad_expr))
    
    prob = cp.Problem(objective, constraints)
    if not prob.is_dcp():
        print("Problem is not disciplined convex. No global certificate")
    prob.solve()

    if prob.status not in ['unbounded', 'infeasible']:
        out_dy = [dy.value for dy in dy_list]
        return out_dy
    
    else:
        print(prob.status, ' CVX min. call error: ')
        print("y_big: ", y_big)
        print("C_big: ", C_big)
        print("H_big: ", H_big)
        print("A_big: ", A_big)
        print("g_big: ", g_big)
        raise ValueError
    
def constr_creation(x, g):
    # if g is None:
    #     if any(isinstance(item, float) for item in x) or any(isinstance(item, int) for item in x):
    #         feas = 1
    #     else:
    #         feas = np.ones(len(np.array(x)))
    # elif any(isinstance(item, float) for item in x) or any(isinstance(item, int) for item in x):
    #     feas = np.product((np.array(g) <= 0).astype(int))
    # else:
    #     feas = np.product( (np.array(g) <= 0).astype(int), axis = 1)
    return np.product((np.array(g) <= 0).astype(int))
    
def samples_in_trust(center, radius, \
                     X_samples_list, y_samples_list, g_list = None):
    X = np.array(X_samples_list) 
    y = np.array(y_samples_list) 
    if g_list is not None:
        g = np.array(g_list)
    ind = np.where(np.linalg.norm(X - np.array(center), axis = 1,\
                                  keepdims = True, ord = np.inf) < radius)[0]
    X_in_trust = X[ind] - np.array(center)
    # print(X_in_trust)
    y_center = y[np.where(X == np.array(center))[0]]
    if len(y_center) > 1:
        y_center = y_center[0]
    y_in_trust = y[ind] - float(y_center)
    # print(y_in_trust)
    if g_list is not None:
        g_in_trust = g[ind]
    
    if g_list is not None:
        return X_in_trust, y_in_trust, g_in_trust
    else:
        return X_in_trust, y_in_trust
        
class System:
    def __init__(self, N_agents, N_var, index_agents, global_ind):
        """
        :N_agents: Number of subsystems
        :index_agents: Dictionary where index_agents[1] gives the list of 
                      agent 1 variable indices as e.g. [1, 3]
        :global_ind:   list of indices for global variables e.g. [3]
        """
        
        self.N = N_agents
        
        if len(index_agents) != N_agents:
            raise ValueError('index_agents list should have as many items as N_agents')
        
        self.idx_agents = index_agents
        self.global_ind = global_ind
        self.N_var = N_var
    
    def initialize(self, rho, N_it, z, 
                          fi_list, A_dict, rho_inc = 1, seed=0): 
        self.rho = rho
        self.rho_inc = rho_inc
        self.N_it = N_it
        self.f_list = fi_list
        self.H_big = {}
        self.g_big = {}
        self.C_big = {}
        self.y_big = {}
        self.A_big = A_dict
        self.seed = seed
        
        self.systems = {}
        self.prim_r = []
        self.dual_r = []
        
        # self.systems = {}
        # self.SQP_out = {}
        #self.u_list = {}
        self.obj = {}
        self.obj_global = []
        self.obj_total = []
        self.z_temp = {} 
        self.radius_list = []
        self.local_feas = {ag+1: [] for ag in range(self.N)}
        
        
        self.z_list = {ag+1: {i: [z[i]] for i in self.global_ind} for ag in range(self.N)}
        self.SQP_list = {ag+1: {i: [] for i in self.global_ind} for ag in range(self.N)}
        self.z_array = []
        self.z_temp = {i: [] for i in self.global_ind} 
        
        # self.systems, self.SQP_list = {}, {}
        for i in range(self.N):
            self.systems[i+1] = {} # ; self.SQP_list[i+1] = {} 
            self.obj[i+1] = []
            for j in range(self.N_var):
                if j+1 in self.idx_agents[i+1]:
                    self.systems[i+1][j+1] = []
        
        if len(z) != len(self.global_ind):
            raise ValueError('z should have as many elements as global_ind')
        
    # def compute_residuals(self):
    #     for idx in self.global_ind:
    #         self.z_temp[idx] = [self.systems[i+1][idx][-1] for i in range(self.N)]
    #     self.prim_r += [ np.linalg.norm([np.linalg.norm([self.systems[i+1][idx][-1] - \
    #                      np.mean(self.z_temp[idx]) for i in range(self.N)]) for idx in self.global_ind])]
    #     old_z = {idx: np.mean([self.z_list[ag+1][idx][-1] for ag in range(self.N)]) for idx in self.global_ind}
    #     self.dual_r += [ self.rho*np.linalg.norm([np.linalg.norm(np.mean(self.z_temp[idx]) - \
    #                      old_z[idx]) for idx in self.global_ind])]
    
    def update_lists(self, res):
        for ag in range(self.N):
            k = 0
            for j in self.global_ind:
                self.SQP_list[ag+1][j] += [np.array(self.center[k])+ float(res[ag][k])]
                #self.z_list[ag+1][j] += [self.SQP_list[ag+1][j][-1]]
                k += 1
        for ag in range(self.N):
            for j in self.global_ind:
                self.z_list[ag+1][j] += [np.mean([self.SQP_list[idx+1][j][-1] for idx in range(self.N)])]
        self.z_array += [[np.mean([self.z_list[ag+1][i][-1] for ag in range(self.N)]) for i in self.global_ind]]

    
    
    def initialize_trust(self, N_s, solver, H = None, bounds = None):
        if H is None:
            H = np.ones(len(self.center))
        dim = len(self.center)
        for k in range(N_s):
            center = np.array(self.center)
            uniform_sampling = np.zeros(dim)

            for i in range(dim):
                lower_bound = - self.radius ; upper_bound = self.radius
                if bounds is not None:
                    if center[i] - self.radius < bounds[i,0]:
                        lower_bound =  center[i] - bounds[i,0]
                    if center[i] + self.radius > bounds[i,1]:
                        upper_bound = bounds[i,1] - center[i]
                np.random.seed(self.seed)
                self.seed += 1
                uniform_sampling[i] = np.random.uniform(lower_bound, upper_bound)
            # print(uniform_sampling)
            x = center + uniform_sampling
            # print(x[:10])
            self.z_array += [x.tolist()]
            
            self.solve_subproblems_trust(x, solver)
            
    def sample_trust(self, N_s, solver, H = None, bounds = None):
        if H is None:
            H = np.ones(len(self.center))
        dim = len(self.center)
        for k in range(N_s):
            center = np.array(self.center)
            uniform_sampling = np.zeros(dim)

            for i in range(dim):
                lower_bound = - self.radius ; upper_bound = self.radius
                if bounds is not None:
                    if center[i] - self.radius < bounds[i,0]:
                        lower_bound = bounds[i,0] - center[i]
                    if center[i] + self.radius > bounds[i,1]:
                        upper_bound = bounds[i,1] - center[i]
                np.random.seed(self.seed)
                self.seed += 1
                uniform_sampling[i] = np.random.uniform(lower_bound, upper_bound)
            
            x = center + uniform_sampling
            self.z_array += [x.tolist()]
            
            self.solve_subproblems_trust(x, solver)
            
    def solve_subproblems(self, solver):
        for ag in range(self.N):
            #print(z_list, u_list, rho)
            out = self.f_list[ag](self.z_list[ag+1], self.rho, self.global_ind, 
                                  self.idx_agents[ag+1], solver = solver)
            if type(out) != tuple:
                ins = out
                self.local_feas[ag+1] += [1]
                # print('All good')
            else:
                ins = out[0]
                s = out[1]
                if (s.solver.status != SolverStatus.ok) or (s.solver.termination_condition != TerminationCondition.optimal):
                    print('Infeasible for agent: ', ag+1)
                    # print(self.z_list[ag+1])
                    self.local_feas[ag+1] += [0]
                else:
                    self.local_feas[ag+1] += [1]
                    
            self.obj[ag+1] += [pyo.value(ins.obj)]
            for j in ins.x:
                self.systems[ag+1][j] += [pyo.value(ins.x[j])]
        
        self.compute_residuals() 
        self.obj_global += [np.sum([self.obj[ag+1][-1] for ag in range(self.N)])]
        
    def compute_residuals(self):
        for idx in self.global_ind:
            self.z_temp[idx] = [self.systems[i+1][idx][-1] for i in range(self.N)]
        self.prim_r += [ np.linalg.norm([np.linalg.norm([self.systems[i+1][idx][-1] - \
                         np.mean(self.z_temp[idx]) for i in range(self.N)]) for idx in self.global_ind])]
        self.dual_r += [ self.rho*np.linalg.norm([np.linalg.norm(np.mean(self.z_temp[idx]) - \
                         self.z_list[1][idx][-1]) for idx in self.global_ind])]
        
    def solve_subproblems_trust(self, x, solver):
        for ag in range(self.N):
            #print(z_list, u_list, rho)
            for j in range(len(self.global_ind)):
                self.z_list[ag+1][self.global_ind[j]] += [x[j]]
            out = self.f_list[ag](self.z_list[ag+1], self.rho, self.global_ind, 
                                  self.idx_agents[ag+1], solver = solver)
            if type(out) != tuple:
                ins = out
                self.local_feas[ag+1] += [1]
                # print('All good')
            else:
                ins = out[0]
                s = out[1]
                if (s.solver.status != SolverStatus.ok) or (s.solver.termination_condition != TerminationCondition.optimal):
                    print('Infeasible for agent: ', ag+1)
                    # print(self.z_list[ag+1])
                    self.local_feas[ag+1] += [0]
                else:
                    self.local_feas[ag+1] += [1]
            self.obj[ag+1] += [pyo.value(ins.obj)]
            for j in ins.x:
                self.systems[ag+1][j] += [pyo.value(ins.x[j])]
        
        self.compute_residuals()   
        self.obj_global += [np.sum([self.obj[ag+1][-1] for ag in range(self.N)])]
    
    def trust_fitting(self, N_samples, solver, bounds = None):
        
        for idx in range(1, self.N+1):
            X_trust, y_trust, feas_trust = samples_in_trust(self.center, self.radius, 
                                                        self.z_array, self.obj[idx], 
                                                        g_list = self.local_feas[idx])
            # print('Trust region shapes: ', X_trust.shape[0], y_trust.shape[0], 'Overall dataset: ', len(self.z_array))
        # print(feas_trust)
        
            if len(y_trust) < N_samples:
                self.sample_trust(N_samples - len(y_trust), solver, bounds = bounds)
                X_trust, y_trust, feas_trust = samples_in_trust(self.center, self.radius, 
                                                        self.z_array, self.obj[idx], 
                                                        g_list = self.local_feas[idx])
                # print('Trust region shapes revised: ', X_trust.shape, y_trust.shape, 'Overall dataset: ', len(self.z_array))
            if len(y_trust) < N_samples:
                print('Only ', len(y_trust), ' instead of ', N_samples, ' samples in trust for agent ', idx)
            P, q = quadratic_fitting(X_trust, y_trust)
            feas_X = X_trust.copy()[feas_trust == 1]
            infeas_X = X_trust.copy()[feas_trust != 1]
        # print(P,q,r)
            self.H_big[idx] = nearest_PD(P)
            self.g_big[idx] = q
            self.y_big[idx] = np.array(self.center).reshape(-1,1)
            if len(feas_X) == len(X_trust):
            # print('All feasible')
                self.C_big[idx] = None
            else:
                P_ineq, q_ineq, r_ineq = quadratic_discrimination(feas_X, infeas_X)
                P_ineq = nearest_PD(P_ineq)
                self.C_big[idx] = (P_ineq, q_ineq, r_ineq)
            # print(P_ineq, q_ineq, r_ineq)
            
            
    def solve(self, N_samples, init_trust, mu = 1000, beta_inc = 1.2, 
              bounds = None, constr_list = None,
              beta_red = 0.8, eta1 = 0.2, eta2 = 0.8, infeas_start = False):
        
        if not infeas_start:
            solver = True
        else:
            solver = False
        
        self.radius = init_trust
        self.radius_list += [init_trust]
        
        self.center = np.array([self.z_list[1][i][0] for i in self.global_ind])
        self.z_array += [[np.mean([self.z_list[ag+1][i][0] for ag in range(self.N)]) for i in self.global_ind]]
        
        self.solve_subproblems(solver)
        
        # self.N_samples = N_samples-1
        self.initialize_trust(N_samples-1, solver, bounds = bounds)
        self.best_obj = [self.obj_global[0]] ; self.n_eval = [1]
        self.center_list = [self.center.tolist()]
        
        ## CHECK THIS
        
        self.trust_fitting(N_samples, solver, bounds = bounds)
        
        ##
        
        # while len(self.obj_global) < self.N_it:
        while len(self.obj_global) < self.N_it:
            
            self.rho *= self.rho_inc
            
            res = SQP_data(self.H_big, self.g_big, self.C_big, 
                                  self.y_big, self.A_big,
                                  len(self.global_ind)*(self.N-1), 
                                  self.radius, mu = mu,
                                  bounds = bounds, constr_list = constr_list)
            # print(res, type(res))
            self.update_lists(res)
            
            # solve_subproblem at new z
            self.solve_subproblems(solver)
            
            
            # is_feas = bool(np.product([self.local_feas[ag+1][-1] for ag in range(self.N)]))
            # is_obj_improved = self.obj_global[-1] < self.best_obj[-1]
            # if is_obj_improved and is_feas:
            #     self.best_obj += [self.obj_global[-1]]
            #     self.center = self.z_array[-1]
            #     self.center_list += [self.center]
            # else:
            #     self.best_obj += [self.best_obj[-1]]
            #     self.center_list += [self.center]
            # self.radius *= 0.95
            # self.radius_list += [self.radius]
            
            actual_dec = - self.obj_global[-1] + self.best_obj[-1]
            dx = (np.array(self.z_array[-1]) - np.array(self.center)).reshape(-1,1)
            pred_dec = -float(np.sum([dx.T @ self.H_big[ag+1] @ dx + self.g_big[ag+1].T @ dx for ag in range(self.N)]))
            #print(dx, actual_dec, pred_dec)
              
            # update radius and center
            is_feas = bool(np.product([self.local_feas[ag+1][-1] for ag in range(self.N)]))
            is_obj_improved = self.obj_global[-1] < self.best_obj[-1]
            if is_obj_improved and is_feas:
                if (actual_dec >= eta2*pred_dec) and (np.abs(np.linalg.norm(dx) - self.radius) < 1e-6):
                    self.best_obj += [self.obj_global[-1]]
                    self.n_eval += [len(self.obj_global)]
                    self.center = self.z_array[-1]
                    self.radius *= beta_inc
                    #self.update_lmbda_dict(lmbda)
                elif actual_dec <= eta1*pred_dec:
                    self.best_obj += [self.best_obj[-1]]
                    self.n_eval += [len(self.obj_global)]
                    self.radius *= beta_red
                else:
                    self.best_obj += [self.obj_global[-1]]
                    self.n_eval += [len(self.obj_global)]
                    self.center = self.z_array[-1]
                    #self.update_lmbda_dict(lmbda)
            else:
                self.best_obj += [self.best_obj[-1]]
                self.n_eval += [len(self.obj_global)]
                self.radius *= beta_red
            self.center_list += [list(self.center)]
            self.radius_list += [self.radius]
            
            # trust_fitting
            self.trust_fitting(N_samples, solver, bounds = bounds)
            
            # raise NotImplementedError
  
# from Problems.ToyProblem1 import f1, f2       
  
# rho = 50
# N_it = 50

# N = 2
# N_var = 3
# list_fi = [f1, f2]

# global_ind = [3]
# index_agents = {1: [1, 3], 2: [2, 3]}
# z = {3: 4.5}

# actual_f = 13.864179350870021
# actual_x = 0.398

# A_dict = {1: np.array([[1]]), 2: np.array([[-1]])}

# System_dataAL = System(N, N_var, index_agents, global_ind)
# System_dataAL.initialize(rho, N_it, z, list_fi, A_dict)
# System_dataAL.solve(6, 1, mu = 1e7)
    
    
# x_list = []
# for k in range(100):
#     temp = np.array([0,0])
#     temp = temp + np.random.uniform(low = -1, high = 1, size = (len(temp),))*np.ones(len(temp))
#     x_list += [temp.tolist()]
# plt.scatter(np.array(x_list)[:,0], np.array(x_list)[:,1])
# circle_x = np.linspace(-1,1,100)
# circle_upper_y = np.sqrt(1 - circle_x**2)
# plt.plot(circle_x, circle_upper_y, c = 'k')
# plt.plot(circle_x, -circle_upper_y, c = 'k')
# plt.plot([-1,1], [-1, -1], c = 'k')
# plt.plot([-1,1], [1, 1], c = 'k')
# plt.plot([-1,-1], [-1, 1], c = 'k')
# plt.plot([1,1], [-1, 1], c = 'k')


    
    
    




