# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 22:27:06 2021

@author: dv516
"""

import pyomo.environ as pyo
import numpy as np

def centralised(data):
    model = pyo.AbstractModel()
    # model.n_x = pyo.RangeSet(1, 4)
    # model.x_init = pyo.Param(model.i)
    
    model.N = pyo.Param()
    model.N_i = pyo.Param()
    model.N_j = pyo.Param()
    # model.N_k = pyo.Param()
    model.k = pyo.RangeSet(1, model.N)
    model.i = pyo.RangeSet(1, model.N_i)
    model.j = pyo.RangeSet(1, model.N_j)
    model.x_k_s = pyo.RangeSet(1, 2*model.N)
    model.x = pyo.Var(model.x_k_s, bounds = (0,5))
    model.x_i = pyo.Param(model.i)
    model.x_j = pyo.Param(model.j)
    model.z_i = pyo.Param(model.i)
    model.z_j = pyo.Param(model.j)
    
    model.C_k  = pyo.Var(model.k, within = pyo.NonNegativeReals)
    model.C_ik = pyo.Var(model.i, model.k, within = pyo.NonNegativeReals)
    model.C_kj = pyo.Var(model.k, model.j, within = pyo.NonNegativeReals)
    
    
    model.y_k   = pyo.Var(model.k, within = pyo.Binary)
    model.y_ik  = pyo.Var(model.i, model.k, within = pyo.Binary)
    model.y_kj  = pyo.Var(model.k, model.j, within = pyo.Binary)
    model.y_ixks = pyo.Var(model.i, model.x_k_s, within = pyo.Binary)
    model.y_jxks = pyo.Var(model.j, model.x_k_s, within = pyo.Binary)
    model.y_x = pyo.Var(within = pyo.Binary)
    model.y_y = pyo.Var(within = pyo.Binary)
    model.alpha_plus_i  = pyo.Var(model.i, model.x_k_s, within = pyo.NonNegativeReals)
    model.alpha_minus_i = pyo.Var(model.i, model.x_k_s, within = pyo.NonNegativeReals)
    model.alpha_plus_j  = pyo.Var(model.j, model.x_k_s, within = pyo.NonNegativeReals)
    model.alpha_minus_j = pyo.Var(model.j, model.x_k_s, within = pyo.NonNegativeReals)
    model.alpha_plusx = pyo.Var()
    model.alpha_plusy = pyo.Var()
    model.alpha_minusx = pyo.Var()
    model.alpha_minusy = pyo.Var()
    
    model.f_k = pyo.Var(model.k, within = pyo.NonNegativeReals)
    model.f_ik = pyo.Var(model.i, model.k, within = pyo.NonNegativeReals)
    model.f_kj = pyo.Var(model.k, model.j, within = pyo.NonNegativeReals)
    model.D_ik = pyo.Var(model.i, model.k, within = pyo.NonNegativeReals)
    model.D_kj = pyo.Var(model.k, model.j, within = pyo.NonNegativeReals)
    # model.dummy_ik = pyo.Var(model.i, model.k, within = pyo.NonNegativeReals)
    # model.dummy_kj = pyo.Var(model.k, model.j, within = pyo.NonNegativeReals)
    
    model.ff_k   = pyo.Param(model.k)
    model.vf_k   = pyo.Param(model.k)
    model.mc_k   = pyo.Param(model.k)
    model.cs_i   = pyo.Param(model.i)
    model.cv_k   = pyo.Param(model.k)
    model.ft_ik  = pyo.Param(model.i, model.k)
    model.vt_ik  = pyo.Param(model.i, model.k)
    model.ft_kj  = pyo.Param(model.k, model.j)
    model.vt_kj  = pyo.Param(model.k, model.j)
    model.a_i    = pyo.Param(model.i)
    model.d_j    = pyo.Param(model.j)
    model.f_ik_U = pyo.Param(model.i, model.k)
    model.f_kj_U = pyo.Param(model.k, model.j)
    model.f_ik_L = pyo.Param(model.i, model.k)
    model.f_kj_L = pyo.Param(model.k, model.j)
    model.D_ik_U = pyo.Param(model.i, model.k)
    model.D_kj_U = pyo.Param(model.k, model.j)
    model.D_ik_L = pyo.Param(model.i, model.k)
    model.D_kj_L = pyo.Param(model.k, model.j)
    
    def o(m):
        return sum(m.C_k[k] for k in m.k) + \
               sum(m.C_ik[i,k] for i in m.i for k in m.k) + \
               sum(m.C_kj[k,j] for k in m.k for j in m.j)
    model.obj = pyo.Objective(rule = o)
    
    # if model.N > 1:
    #     def alpha_k_bounds(m):
    #         return m.alpha_plusx + m.alpha_plusy + m.alpha_minusx + m.alpha_minusy >= 0.5
    #     model.ineq_alpha_k_bounds = pyo.Constraint(rule = alpha_k_bounds)
    
    #     def alpha_x_plus(m):
    #         return m.alpha_plusx - m.alpha_minusx == m.x[1] - m.x[2]
    #     model.eq_alpha_x_plus = pyo.Constraint(rule = alpha_x_plus)
    
    #     def alpha_y_plus(m):
    #         return m.alpha_plusy - m.alpha_minusy == m.x[3] - m.x[4]
    #     model.eq_alpha_y_plus = pyo.Constraint(rule = alpha_y_plus)
    
    #     def apx_bound(m):
    #         return m.alpha_plusx <= m.y_x*10
    #     model.eq_apx_bound = pyo.Constraint(rule=apx_bound)
    
    #     def amx_bound(m):
    #         return m.alpha_minusx <= (1-m.y_x)*10
    #     model.eq_amx_bound = pyo.Constraint(rule=amx_bound)
    
    #     def apy_bound(m):
    #         return m.alpha_plusy <= m.y_y*10
    #     model.eq_apy_bound = pyo.Constraint(rule=apy_bound)
    
    #     def amy_bound(m):
    #         return m.alpha_minusy <= (1-m.y_y)*10
    #     model.eq_amy_bound = pyo.Constraint(rule=amy_bound)
    
    def cost_k(m, k):
        return m.C_k[k] == (m.ff_k[k]*m.y_k[k] + m.vf_k[k]*m.f_k[k])
    model.eq_cost_k = pyo.Constraint(model.k, rule = cost_k)
    
    def bound_f_k(m, k):
        return  m.f_k[k] <= m.mc_k[k]*m.y_k[k] # 0 <=
    model.ineq_bound_f_k = pyo.Constraint(model.k, rule = bound_f_k)
    
    def cost_ik(m, i, k):
        return m.C_ik[i,k] == (m.cs_i[i]*m.f_ik[i,k] + m.ft_ik[i,k]*m.y_ik[i,k] + m.vt_ik[i,k]*5*(m.D_ik[i,k]*m.f_ik[i,k])) # *m.dummy_ik[i,k])
    model.eq_cost_ik = pyo.Constraint(model.i, model.k, rule = cost_ik)
   
    # def dummy_ik_MC1(m, i, k):
    #     return m.dummy_ik[i,k] >=  m.f_ik_L[i,k]*m.D_ik[i,k] + m.f_ik[i,k]*m.D_ik_L[i,k] - m.f_ik_L[i,k]*m.D_ik_L[i,k] - 1e4*(1-m.y_ik[i,k])
    # model.ineq_dummy_ik_MC1 = pyo.Constraint(model.i, model.k, rule = dummy_ik_MC1) 
   
    # def dummy_ik_MC2(m, i, k):
    #     return m.dummy_ik[i,k] >=  m.f_ik_U[i,k]*m.D_ik[i,k] + m.f_ik[i,k]*m.D_ik_U[i,k] - m.f_ik_U[i,k]*m.D_ik_U[i,k] - 1e4*(1-m.y_ik[i,k])
    # model.ineq_dummy_ik_MC2 = pyo.Constraint(model.i, model.k, rule = dummy_ik_MC2)  
   
    # def dummy_ik_MC3(m, i, k):
    #     return m.dummy_ik[i,k] <=  m.f_ik_L[i,k]*m.D_ik[i,k] + m.f_ik[i,k]*m.D_ik_U[i,k] - m.f_ik_L[i,k]*m.D_ik_U[i,k] + 1e4*(1-m.y_ik[i,k])
    # model.ineq_dummy_ik_MC3 = pyo.Constraint(model.i, model.k, rule = dummy_ik_MC3) 
   
    # def dummy_ik_MC4(m, i, k):
    #     return m.dummy_ik[i,k] <=  m.f_ik_U[i,k]*m.D_ik[i,k] + m.f_ik[i,k]*m.D_ik_L[i,k] - m.f_ik_U[i,k]*m.D_ik_L[i,k] + 1e4*(1-m.y_ik[i,k])
    # model.ineq_dummy_ik_MC4 = pyo.Constraint(model.i, model.k, rule = dummy_ik_MC4)   
   
    # def dummy_ik_bound(m, i, k):
    #     return m.dummy_ik[i,k] <= 1e4*m.y_ik[i,k]
    # model.ineq_dummy_ik_bound = pyo.Constraint(model.i, model.k, rule = dummy_ik_bound)
   
    def bound_f_ik(m, i, k):
        return m.f_ik[i,k] <= m.f_ik_U[i,k]*m.y_ik[i,k] # 0 <=
    model.ineq_bound_f_ik = pyo.Constraint(model.i, model.k, rule = bound_f_ik)
    
    def bound_D_ik_1(m, i, k):
        return m.D_ik_L[i,k]*m.y_ik[i,k] <= m.D_ik[i,k] 
    model.ineq_bound_D_ik_1 = pyo.Constraint(model.i, model.k, rule = bound_D_ik_1)
    
    def bound_D_ik_2(m, i, k):
        return  m.D_ik[i,k] <= m.D_ik_U[i,k]*m.y_ik[i,k]
    model.ineq_bound_D_ik_2 = pyo.Constraint(model.i, model.k, rule = bound_D_ik_2)
    
    def cost_kj(m, k, j):
        return m.C_kj[k,j] == (m.ft_kj[k,j]*m.y_kj[k,j] + m.vt_kj[k,j]*5*(m.D_kj[k,j]*m.f_kj[k,j])) # *m.dummy_kj[k,j])
    model.eq_cost_kj = pyo.Constraint(model.k, model.j, rule = cost_kj)
    
    # def dummy_kj_bound(m, k, j):
    #     return m.dummy_kj[k,j] <= 1e4*m.y_kj[k,j]
    # model.ineq_dummy_kj_bound = pyo.Constraint(model.k, model.j, rule = dummy_kj_bound)
    
    # def dummy_kj_MC1(m, k, j):
    #     return m.dummy_kj[k,j] >=  m.f_kj_L[k,j]*m.D_kj[k,j] + m.f_kj[k,j]*m.D_kj_L[k,j] - m.f_kj_L[k,j]*m.D_kj_L[k,j] - 1e4*(1-m.y_kj[k,j])
    # model.ineq_dummy_kj_MC1 = pyo.Constraint(model.k, model.j, rule = dummy_kj_MC1) 
   
    # def dummy_kj_MC2(m, k, j):
    #     return m.dummy_kj[k,j] >=  m.f_kj_U[k,j]*m.D_kj[k,j] + m.f_kj[k,j]*m.D_kj_U[k,j] - m.f_kj_U[k,j]*m.D_kj_U[k,j] - 1e4*(1-m.y_kj[k,j])
    # model.ineq_dummy_kj_MC2 = pyo.Constraint(model.k, model.j, rule = dummy_kj_MC2)  
   
    # def dummy_kj_MC3(m, k, j):
    #     return m.dummy_kj[k,j] <=  m.f_kj_L[k,j]*m.D_kj[k,j] + m.f_kj[k,j]*m.D_kj_U[k,j] - m.f_kj_L[k,j]*m.D_kj_U[k,j] + 1e4*(1-m.y_kj[k,j])
    # model.ineq_dummy_kj_MC3 = pyo.Constraint(model.k, model.j, rule = dummy_kj_MC3) 
   
    # def dummy_kj_MC4(m, k, j):
    #     return m.dummy_kj[k,j] <=  m.f_kj_U[k,j]*m.D_kj[k,j] + m.f_kj[k,j]*m.D_kj_L[k,j] - m.f_kj_U[k,j]*m.D_kj_L[k,j] + 1e4*(1-m.y_kj[k,j])
    # model.ineq_dummy_kj_MC4 = pyo.Constraint(model.k, model.j, rule = dummy_kj_MC4)   
    
    def bound_f_kj(m, k, j):
        return m.f_kj[k,j] <= m.f_kj_U[k,j]*m.y_kj[k,j] # 0 <= 
    model.ineq_bound_f_kj = pyo.Constraint(model.k, model.j, rule = bound_f_kj)
    
    def bound_D_kj_1(m, k, j):
        return m.D_kj_L[k,j]*m.y_kj[k,j] <= m.D_kj[k,j] # <= m.D_kj_U[k,j]*m.y_kj[k,j]
    model.ineq_bound_D_kj_1 = pyo.Constraint(model.k, model.j, rule = bound_D_kj_1)
    
    def bound_D_kj_2(m, k, j):
        return m.D_kj_U[k,j]*m.y_kj[k,j] >= m.D_kj[k,j] # <= m.D_kj_U[k,j]*m.y_kj[k,j]
    model.ineq_bound_D_kj_2 = pyo.Constraint(model.k, model.j, rule = bound_D_kj_2)
    
    # def D_ik(m, i, k):
    #     return m.D_ik[i,k]**2 >= (m.x[k] - m.x_i[i])**2 + (m.x[k+m.N] - m.z_i[i])**2
    # model.ineq_D_ik = pyo.Constraint(model.i, model.k, rule = D_ik)
    
    # def D_kj(m, k, j):
    #     return m.D_kj[k,j]**2 >= (m.x[k] - m.x_j[j])**2 + (m.x[k+m.N] - m.z_j[j])**2
    # model.ineq_D_kj = pyo.Constraint(model.i, model.k, rule = D_kj)
    
    def D_ik(m, i, k):
        return m.D_ik[i,k] == m.alpha_plus_i[i, k] + m.alpha_minus_i[i, k] + m.alpha_plus_i[i, k+m.N] + m.alpha_minus_i[i, k+m.N]
    model.ineq_D_ik = pyo.Constraint(model.i, model.k, rule = D_ik)
    
    def D_kj(m, k, j):
        return m.D_kj[k,j] == m.alpha_plus_j[j, k] + m.alpha_minus_j[j, k] + m.alpha_plus_j[j, k+m.N] + m.alpha_minus_j[j, k+m.N]
    model.ineq_D_kj = pyo.Constraint(model.k, model.j, rule = D_kj)
    
    def alpha_i_x(m, i, k):
        return m.alpha_plus_i[i, k] - m.alpha_minus_i[i, k] == m.x[k] - m.x_i[i]
    model.i_x = pyo.Constraint(model.i, model.k, rule = alpha_i_x)
    
    def alpha_i_y(m, i, k):
        return m.alpha_plus_i[i, k+m.N] - m.alpha_minus_i[i, k+m.N] == m.x[k+m.N] - m.z_i[i]
    model.eq_alpha_i_x = pyo.Constraint(model.i, model.k, rule = alpha_i_y)
    
    def alpha_j_x(m, j, k):
        return m.alpha_plus_j[j, k] - m.alpha_minus_j[j, k] == m.x[k] - m.x_j[j]
    model.eq_alpha_j_x = pyo.Constraint(model.j, model.k, rule = alpha_j_x)
    
    def alpha_j_y(m, j, k):
        return m.alpha_plus_j[j, k+m.N] - m.alpha_minus_j[j, k+m.N] == m.x[k+m.N] - m.z_j[j]
    model.eq_alpha_j_y = pyo.Constraint(model.j, model.k, rule = alpha_j_y)
    
    def api_bound(m, i, x_k_s):
        return m.alpha_plus_i[i, x_k_s] <= m.y_ixks[i, x_k_s]*10
    model.eq_api_bound = pyo.Constraint(model.i, model.x_k_s, rule=api_bound)
    
    def ami_bound(m, i, x_k_s):
        return m.alpha_minus_i[i, x_k_s] <= (1-m.y_ixks[i, x_k_s])*10
    model.eq_ami_bound = pyo.Constraint(model.i, model.x_k_s, rule=ami_bound)
    
    def apj_bound(m, j, x_k_s):
        return m.alpha_plus_j[j, x_k_s] <= m.y_jxks[j, x_k_s]*10
    model.eq_apj_bound = pyo.Constraint(model.j, model.x_k_s, rule=apj_bound)
    
    def amj_bound(m, j, x_k_s):
        return m.alpha_minus_j[j, x_k_s] <= (1-m.y_jxks[j, x_k_s])*10
    model.eq_amj_bound = pyo.Constraint(model.j, model.x_k_s, rule=amj_bound)
    
    def W_k_1(m, i, k):
        return m.y_k[k] >= m.y_ik[i,k]
    model.ineq_W_k_1 = pyo.Constraint(model.i, model.k, rule = W_k_1)
 
    def W_k_2(m, k, j):
        return m.y_k[k] >= m.y_kj[k,j]
    model.ineq_W_k_2 = pyo.Constraint(model.k, model.j, rule = W_k_2)
    
    def max_avail(m, i):
        return sum(m.f_ik[i,k] for k in m.k) <= m.a_i[i]
    model.ineq_max_avail = pyo.Constraint(model.i, rule = max_avail)
    
    def link_ik(m, k):
        return sum(m.f_ik[i,k]*m.cv_k[k] for i in m.i) == m.f_k[k]
    model.eq_link_ik = pyo.Constraint(model.k, rule = link_ik)
    
    def link_kj(m, k):
        return sum(m.f_kj[k,j] for j in m.j) == m.f_k[k]
    model.eq_link_kj = pyo.Constraint(model.k, rule = link_kj)
    
    def link_demand(m, j):
        return sum(m.f_kj[k,j] for k in m.k) == m.d_j[j]
    model.eq_link_demand = pyo.Constraint(model.j, rule = link_demand)
    
    ins = model.create_instance(data)
    
    # print(ins.ineq_W_k_2[1,1].pprint())
    
    # for i in ins.i:
    #     ins.pos_x[i] = ins.mu_x[i]
    #     ins.pos_x_est[i] = ins.mu_x[i]
    #     ins.pos_y[i] = ins.mu_y[i]
    #     ins.pos_y_est[i] = ins.mu_y[i]
    
    # for j in ins.i:
    #     ins.x[j] = ins.x_init[j]
    
    #solver = pyo.SolverFactory('cbc')
    # solver = pyo.SolverFactory('mosek')
    
    # solver = pyo.SolverFactory('mosek')
    solver = pyo.SolverFactory('gurobi_direct')
    solver.options['NonConvex'] = 2
    # solver.options['max_iter'] = int(1e4)
    solver.solve(ins)
    
    return ins



def SP_4N(data, return_solver = False, node_type = 'Supplier', idx = 1):
    model = pyo.AbstractModel()

    
    model.N = pyo.Param()
    model.k = pyo.RangeSet(1, model.N)
    model.x_k_s = pyo.RangeSet(1, 2*model.N)
    model.I = pyo.Set()
    model.z_I = pyo.Set()
    model.x = pyo.Var(model.I)
    model.z = pyo.Param(model.z_I)
    model.jdx = pyo.Param()
    model.idx = pyo.Param()
    model.N_i = pyo.Param()
    model.N_j = pyo.Param()
    # model.node_type = pyo.Param()
    
    model.x_coord = pyo.Param()
    model.y_coord = pyo.Param()
    model.rho = pyo.Param()
    
    model.C_k = pyo.Var(model.k, within = pyo.NonNegativeReals)
    model.C_local  = pyo.Var(model.k, within = pyo.NonNegativeReals)
    
    model.y_local  = pyo.Var(model.k, within = pyo.Binary)
    model.y_xks    = pyo.Var(model.x_k_s, within = pyo.Binary)
    model.y_x = pyo.Var(within = pyo.Binary)
    model.y_y = pyo.Var(within = pyo.Binary)
    model.alpha_plus   = pyo.Var(model.x_k_s, within = pyo.NonNegativeReals)
    model.alpha_minus  = pyo.Var(model.x_k_s, within = pyo.NonNegativeReals)
    model.alpha_plusx  = pyo.Var()
    model.alpha_plusy  = pyo.Var()
    model.alpha_minusx = pyo.Var()
    model.alpha_minusy = pyo.Var()
    
    # model.f_k = pyo.Var(model.k, within = pyo.NonNegativeReals)
    # model.f_local = pyo.Var(model.k, within = pyo.NonNegativeReals)
    model.D_local = pyo.Var(model.k, within = pyo.NonNegativeReals, bounds = (0.5,20))
    # model.dummy_ik = pyo.Var(model.i, model.k, within = pyo.NonNegativeReals)
    # model.dummy_kj = pyo.Var(model.k, model.j, within = pyo.NonNegativeReals)
    
    model.ff_k   = pyo.Param(model.k)
    model.vf_k   = pyo.Param(model.k)
    model.mc_k   = pyo.Param(model.k)
    model.cv_k   = pyo.Param(model.k)
    
    model.f_k_U = pyo.Param(model.k)
    model.f_k_L = pyo.Param(model.k)
    model.D_k_U = pyo.Param(model.k)
    model.D_k_L = pyo.Param(model.k)  
    model.ft_k  = pyo.Param(model.k)
    model.vt_k  = pyo.Param(model.k)
    
    
    def bound_D_k_1(m, k):
        return m.D_k_L[k]*m.y_local[k] <= m.D_local[k] 
    model.ineq_bound_D_k_1 = pyo.Constraint(model.k, rule = bound_D_k_1)
    
    def bound_D_k_2(m, k):
        return  m.D_local[k] <= m.D_k_U[k]*m.y_local[k]
    model.ineq_bound_D_k_2 = pyo.Constraint(model.k, rule = bound_D_k_2)
    def bound_f_local(m, k):
        if m.idx == 1:
            return m.x[3*m.N + m.jdx + k] <= m.f_k_U[k]*m.y_local[k]
        elif m.jdx == 0:
            return (m.x[2*m.N+k] - m.x[3*m.N + m.jdx + k]*m.cv_k[k])/m.cv_k[k] <= m.f_k_U[k]*m.y_local[k]
        else:
            return (m.x[2*m.N+k] - m.x[3*m.N + m.jdx + k]) <= m.f_k_U[k]*m.y_local[k]
    model.ineq_bound_f_k = pyo.Constraint(model.k, rule = bound_f_local)
    
    if node_type == 'Supplier':
        model.a     = pyo.Param()
        model.cs    = pyo.Param()
        
        def cost_local(m, k):
            if m.idx == 1:
                x = m.x[3*m.N + k]
            else:
                x = (m.x[2*m.N+k] - m.x[3*m.N + k]*m.cv_k[k])/m.cv_k[k]
            return m.C_local[k] == (m.cs*x + m.ft_k[k]*m.y_local[k] + m.vt_k[k]*5*(m.D_local[k]*x)) # *m.dummy_ik[i,k])
        model.eq_cost = pyo.Constraint(model.k, rule = cost_local)
        
        def max_avail(m):
            if m.idx == 1:
                return sum(m.x[3*m.N+k] for k in m.k) <= m.a
            else:
                return sum((m.x[2*m.N+k] - m.x[3*m.N + k]*m.cv_k[k])/m.cv_k[k] for k in m.k) <= m.a
        model.ineq_max_avail = pyo.Constraint(rule = max_avail)
        
            
    else:
        model.d = pyo.Param()
        def cost_local(m, k):
             if m.idx == 1:
                x = m.x[3*m.N + m.jdx + k]
             else:
                x = (m.x[2*m.N+k] - m.x[3*m.N + m.jdx + k]) 
             return m.C_local[k] == (m.ft_k[k]*m.y_local[k] + m.vt_k[k]*5*(m.D_local[k]*x)) # *m.dummy_kj[k,j])
        model.eq_cost = pyo.Constraint(model.k, rule = cost_local)
    
        def link_demand(m):
            if m.idx == 1:
                return sum(m.x[3*m.N + m.jdx + k] for k in m.k) == m.d
            else:
                return sum((m.x[2*m.N+k] - m.x[3*m.N + m.jdx + k]) for k in m.k) == m.d
        model.eq_link_demand = pyo.Constraint(rule = link_demand)  
    
    def o(m):
        return sum(m.C_k[k] + m.C_local[k] for k in m.k) + \
               m.rho*sum((m.x[i] - m.z[i])**2 for i in m.z_I)
    model.obj = pyo.Objective(rule = o)
    
    
    # def alpha_k_bounds(m):
    #     return m.alpha_plusx + m.alpha_plusy + m.alpha_minusx + m.alpha_minusy >= 0.5
    # model.ineq_alpha_k_bounds = pyo.Constraint(rule = alpha_k_bounds)
    
    # def alpha_x_plus(m):
    #     return m.alpha_plusx - m.alpha_minusx == m.x[1] - m.x[2]
    # model.eq_alpha_x_plus = pyo.Constraint(rule = alpha_x_plus)
    
    # def alpha_y_plus(m):
    #     return m.alpha_plusy - m.alpha_minusy == m.x[3] - m.x[4]
    # model.eq_alpha_y_plus = pyo.Constraint(rule = alpha_y_plus)
    
    # def apx_bound(m):
    #     return m.alpha_plusx <= m.y_x*10
    # model.eq_apx_bound = pyo.Constraint(rule=apx_bound)
    
    # def amx_bound(m):
    #     return m.alpha_minusx <= (1-m.y_x)*10
    # model.eq_amx_bound = pyo.Constraint(rule=amx_bound)
    
    # def apy_bound(m):
    #     return m.alpha_plusy <= m.y_y*10
    # model.eq_apy_bound = pyo.Constraint(rule=apy_bound)
    
    # def amy_bound(m):
    #     return m.alpha_minusy <= (1-m.y_y)*10
    # model.eq_amy_bound = pyo.Constraint(rule=amy_bound)


    def cost_k(m, k):
        return m.C_k[k] == (m.ff_k[k] + m.vf_k[k]*m.x[2*m.N+k])/(m.N_i+m.N_j)
    model.eq_cost_k = pyo.Constraint(model.k, rule = cost_k)
    

    def D_local(m, k):
        return m.D_local[k] == m.alpha_plus[k] + m.alpha_minus[k] + m.alpha_plus[k+m.N] + m.alpha_minus[k+m.N]
    model.ineq_D_local = pyo.Constraint(model.k, rule = D_local)
    

    def alpha_x(m, k):
        return m.alpha_plus[k] - m.alpha_minus[k] == m.x[k] - m.x_coord
    model.eq_alpha_x = pyo.Constraint(model.k, rule = alpha_x)
    
    def alpha_y(m, k):
        return m.alpha_plus[k+m.N] - m.alpha_minus[k+m.N] == m.x[k+m.N] - m.y_coord
    model.eq_alpha_y = pyo.Constraint(model.k, rule = alpha_y)
    
    
    def ap_bound(m, x_k_s):
        return m.alpha_plus[x_k_s] <= m.y_xks[x_k_s]*10
    model.eq_ap_bound = pyo.Constraint(model.x_k_s, rule=ap_bound)
    
    def am_bound(m, x_k_s):
        return m.alpha_minus[x_k_s] <= (1-m.y_xks[x_k_s])*10
    model.eq_am_bound = pyo.Constraint(model.x_k_s, rule=am_bound)
    
    def bound_f_k(m, k):
        return  m.x[2*m.N+k] <= m.mc_k[k] # 0 <=
    model.ineq_bound_processing_k = pyo.Constraint(model.k, rule = bound_f_k)
    
    # def W_k_1(m, i, k):
    #     return m.y_k[k] >= m.y_ik[i,k]
    # model.ineq_W_k_1 = pyo.Constraint(model.i, model.k, rule = W_k_1)
 
    # def W_k_2(m, k, j):
    #     return m.y_k[k] >= m.y_kj[k,j]
    # model.ineq_W_k_2 = pyo.Constraint(model.k, model.j, rule = W_k_2)
    
    

    ins = model.create_instance(data)
    
    #ins.x = ins.x_init
    
    #solver = pyo.SolverFactory('cbc')
    solver = pyo.SolverFactory('gurobi_direct')
    solver.options['NonConvex'] = 2
    res = solver.solve(ins)
    
    if return_solver:
        return ins, res
    else:
        return ins


def f_4N_lowD(z_list, rho, global_ind, index, node_type, idx, u_list = None, 
      solver = False, seed=2, bounds = None):
    
    N = 1
    np.random.seed(seed)
    
    def rescale(x, lower_upper_tuple):
        return lower_upper_tuple[0] + x*(lower_upper_tuple[1] - lower_upper_tuple[0])
        
    x_low = 0 ; x_high = 5 ; y_low = 0 ; y_high = 5
    dict_dummy = {
                  'x_i': {1: np.random.uniform(low=x_low, high=x_high), 
                          2: np.random.uniform(low=x_low, high=x_high)}, 
                  'x_j': {1: np.random.uniform(low=x_low, high=x_high), 
                          2: np.random.uniform(low=x_low, high=x_high)},
                  'z_i': {1: np.random.uniform(low=y_low, high=y_high), 
                          2: np.random.uniform(low=y_low, high=y_high)}, 
                  'z_j': {1: np.random.uniform(low=y_low, high=y_high), 
                          2: np.random.uniform(low=y_low, high=y_high)},
                  'cs_i':  {1: 20, 2: 22},
                  'a_i':   {1: 120, 2: 120},
                  'd_j':   {1: 100, 2: 100},
                  'ff_k':  {1: 7.18},
                  'vf_k':  {1: 0.087},
                  'mc_k':  {1: 250},
                  'cv_k':  {1: 0.9},
                  'ft_ik': {(1,1): 10, (2,1): 10,},
                  'ft_kj': {(1,1): 10, (1,2): 10,},
                  # 'vt_ik': {(1,1): 0.3, (1,2): 0.3, (2,1): 0.3, (2,2): 0.3},
                  # 'vt_kj': {(1,1): 0.3, (1,2): 0.3, (2,1): 0.3, (2,2): 0.3},
                  'vt_ik': {(1,1): 0.3, (2,1): 0.3,},
                  'vt_kj': {(1,1): 0.3, (1,2): 0.3,},
                  'D_ik_L': {(1,1): 0.5, (2,1): 0.5,},
                  'D_kj_L': {(1,1): 0.5, (1,2): 0.5,},
                  'D_ik_U': {(1,1): 10, (2,1): 10,},
                  'D_kj_U': {(1,1): 10, (1,2): 10,},
                  'f_ik_U': {(1,1): 200, (2,1): 200,},
                  'f_kj_U': {(1,1): 200, (1,2): 200,},
                  'f_ik_L': {(1,1): 0, (2,1): 0,},
                  'f_kj_L': {(1,1): 0, (1,2): 0,},
                  }
    N_i = len(dict_dummy['x_i'])
    N_j = len(dict_dummy['x_j'])
    jdx = N*(N_i-1)
    
    if node_type == 'Supplier': 
        data = {None: {
                  'x_coord': {None: dict_dummy['x_i'][idx]}, 
                  'y_coord': {None: dict_dummy['z_i'][idx]},
                  'node_type': {None: node_type},
                  'N': {None: N}, 'idx': {None: idx}, 
                  'z': {}, 'u': {}, 'z_I': {None: global_ind}, 
                  'rho': {None: rho}, 'I': {None: index},
                  
                  'cs':  {None: dict_dummy['cs_i'][idx]},
                  'a':   {None: dict_dummy['a_i'][idx]},
                  'ff_k':  {1: 7.18,},
                  'vf_k':  {1: 0.087,},
                  'mc_k':  {1: 250,},
                  'cv_k':  {1: 0.9,},
                  'ft_k': {1: dict_dummy['ft_ik'][(idx,1)],},
                  'vt_k': {1: dict_dummy['vt_ik'][(idx,1)],},
                  'D_k_L': {1: dict_dummy['D_ik_L'][(idx,1)],},
                  'D_k_U': {1: dict_dummy['D_ik_U'][(idx,1)],},
                  'f_k_U': {1: dict_dummy['f_ik_U'][(idx,1)],},
                  'f_k_L': {1: dict_dummy['f_ik_L'][(idx,1)],},
                  'jdx': {None: 0},
                  'N_i': {None: N_i}, 'N_j': {None: N_j},
                }
          }
    else:
        data = {None: {
                  'x_coord': {None: dict_dummy['x_i'][idx]}, 
                  'y_coord': {None: dict_dummy['z_i'][idx]},
                  'node_type': {None: node_type},
                  'N': {None: N}, 'idx': {None: idx}, 
                  'z': {}, 'u': {}, 'z_I': {None: global_ind}, 
                  'rho': {None: rho}, 'I': {None: index},
                  

                  'd':   {None: dict_dummy['d_j'][idx]},
                  'ff_k':  {1: 7.18, },
                  'vf_k':  {1: 0.087, },
                  'mc_k':  {1: 250,},
                  'cv_k':  {1: 0.9,},
                  
                  'ft_k': {1: dict_dummy['ft_kj'][(1,idx)],},
                  'vt_k': {1: dict_dummy['vt_kj'][(1,idx)],},
                  'D_k_L': {1: dict_dummy['D_kj_L'][(1,idx)],},
                  'D_k_U': {1: dict_dummy['D_kj_U'][(1,idx)],},
                  'f_k_U': {1: dict_dummy['f_kj_U'][(1,idx)],},
                  'f_k_L': {1: dict_dummy['f_kj_L'][(1,idx)],},
                  'jdx': {None: jdx},
                  'N_i': {None: N_i}, 'N_j': {None: N_j},
                }
          }
    
    for i in global_ind:
        if bounds is not None:
            data[None]['z'][i] = rescale(z_list[i][-1], bounds[i-1])
        else:
            data[None]['z'][i] = z_list[i][-1]
        if u_list is not None:
            data[None]['u'][i] = u_list[i][-1]
        else:
            data[None]['u'][i] = 0
    # print(data)
    return SP_4N(data, return_solver = solver, node_type = node_type, idx = idx)


def f_4N_highD(z_list, rho, global_ind, index, node_type, idx, u_list = None, 
      solver = False, seed=2, bounds = None):
    
    N = 2
    np.random.seed(seed)
    
    def rescale(x, lower_upper_tuple):
        return lower_upper_tuple[0] + x*(lower_upper_tuple[1] - lower_upper_tuple[0])
      
    x_low = 0 ; x_high = 5 ; y_low = 0 ; y_high = 5
    
    dict_dummy = {
                  'x_i': {1: np.random.uniform(low=x_low, high=x_high), 
                          2: np.random.uniform(low=x_low, high=x_high)}, 
                  'x_j': {1: np.random.uniform(low=x_low, high=x_high), 
                          2: np.random.uniform(low=x_low, high=x_high)},
                  'z_i': {1: np.random.uniform(low=y_low, high=y_high), 
                          2: np.random.uniform(low=y_low, high=y_high)}, 
                  'z_j': {1: np.random.uniform(low=y_low, high=y_high), 
                          2: np.random.uniform(low=y_low, high=y_high)},
                  'cs_i':  {1: 20, 2: 22},
                  'a_i':   {1: 120, 2: 120},
                  'd_j':   {1: 100, 2: 100},
                  'ff_k':  {1: 7.18, 2: 7.18},
                  'vf_k':  {1: 0.087, 2: 0.087},
                  'mc_k':  {1: 125, 2: 125},
                  'cv_k':  {1: 0.9, 2: 0.9},
                  'ft_ik': {(1,1): 10, (1,2): 10, (2,1): 10, (2,2): 10},
                  'ft_kj': {(1,1): 10, (1,2): 10, (2,1): 10, (2,2): 10},
                  # 'vt_ik': {(1,1): 0.3, (1,2): 0.3, (2,1): 0.3, (2,2): 0.3},
                  # 'vt_kj': {(1,1): 0.3, (1,2): 0.3, (2,1): 0.3, (2,2): 0.3},
                  'vt_ik': {(1,1): 0.3, (1,2): 0.3, (2,1): 0.3, (2,2): 0.3},
                  'vt_kj': {(1,1): 0.3, (1,2): 0.3, (2,1): 0.3, (2,2): 0.3},
                  'D_ik_L': {(1,1): 0.5, (1,2): 0.5, (2,1): 0.5, (2,2): 0.5},
                  'D_kj_L': {(1,1): 0.5, (1,2): 0.5, (2,1): 0.5, (2,2): 0.5},
                  'D_ik_U': {(1,1): 10, (1,2): 10, (2,1): 10, (2,2): 10},
                  'D_kj_U': {(1,1): 10, (1,2): 10, (2,1): 10, (2,2): 10},
                  'f_ik_U': {(1,1): 200, (1,2): 200, (2,1): 200, (2,2): 200},
                  'f_kj_U': {(1,1): 200, (1,2): 200, (2,1): 200, (2,2): 200},
                  'f_ik_L': {(1,1): 0, (1,2): 0, (2,1): 0, (2,2): 0},
                  'f_kj_L': {(1,1): 0, (1,2): 0, (2,1): 0, (2,2): 0},
                  }

    N_i = len(dict_dummy['x_i'])
    N_j = len(dict_dummy['x_j'])
    jdx = N*(N_i-1)

    if node_type == 'Supplier': 
        data = {None: {
                  'x_coord': {None: dict_dummy['x_i'][idx]}, 
                  'y_coord': {None: dict_dummy['z_i'][idx]},
                  'node_type': {None: node_type},
                  'N': {None: N}, 'idx': {None: idx}, 
                  'z': {}, 'u': {}, 'z_I': {None: global_ind}, 
                  'rho': {None: rho}, 'I': {None: index},
                  
                  'cs':  {None: dict_dummy['cs_i'][idx]},
                  'a':   {None: dict_dummy['a_i'][idx]},
                  'ff_k':  {1: 7.18, 2: 7.18},
                  'vf_k':  {1: 0.087, 2: 0.087},
                  'mc_k':  {1: 125, 2: 125},
                  'cv_k':  {1: 0.9, 2: 0.9},
                  'ft_k': {1: dict_dummy['ft_ik'][(idx,1)],
                            2: dict_dummy['ft_ik'][(idx,2)],},
                  'vt_k': {1: dict_dummy['vt_ik'][(idx,1)],
                            2: dict_dummy['vt_ik'][(idx,2)],},
                  'D_k_L': {1: dict_dummy['D_ik_L'][(idx,1)],
                            2: dict_dummy['D_ik_L'][(idx,2)],},
                  'D_k_U': {1: dict_dummy['D_ik_U'][(idx,1)],
                            2: dict_dummy['D_ik_U'][(idx,2)],},
                  'f_k_U': {1: dict_dummy['f_ik_U'][(idx,1)],
                            2: dict_dummy['f_ik_U'][(idx,2)],},
                  'f_k_L': {1: dict_dummy['f_ik_L'][(idx,1)],
                            2: dict_dummy['f_ik_L'][(idx,2)],},
                  'jdx': {None: 0},
                  'N_i': {None: N_i}, 'N_j': {None: N_j},
                }
          }
    else:
        data = {None: {
                  'x_coord': {None: dict_dummy['x_i'][idx]}, 
                  'y_coord': {None: dict_dummy['z_i'][idx]},
                  'node_type': {None: node_type},
                  'N': {None: N}, 'idx': {None: idx}, 
                  'z': {}, 'u': {}, 'z_I': {None: global_ind}, 
                  'rho': {None: rho}, 'I': {None: index},
                  

                  'd':   {None: dict_dummy['d_j'][idx]},
                  'ff_k':  {1: 7.18, 2: 7.18},
                  'vf_k':  {1: 0.087, 2: 0.087},
                  'mc_k':  {1: 125, 2: 125},
                  'cv_k':  {1: 0.9, 2: 0.9},
                  
                  'ft_k': {1: dict_dummy['ft_kj'][(1,idx)],
                            2: dict_dummy['ft_kj'][(2,idx)],},
                  'vt_k': {1: dict_dummy['vt_kj'][(1,idx)],
                            2: dict_dummy['vt_kj'][(2,idx)],},
                  'D_k_L': {1: dict_dummy['D_kj_L'][(1,idx)],
                            2: dict_dummy['D_kj_L'][(2,idx)],},
                  'D_k_U': {1: dict_dummy['D_kj_U'][(1,idx)],
                            2: dict_dummy['D_kj_U'][(2,idx)],},
                  'f_k_U': {1: dict_dummy['f_kj_U'][(1,idx)],
                            2: dict_dummy['f_kj_U'][(2,idx)],},
                  'f_k_L': {1: dict_dummy['f_kj_L'][(1,idx)],
                            2: dict_dummy['f_kj_L'][(2,idx)],},
                  'jdx': {None: jdx},
                  'N_i': {None: N_i}, 'N_j': {None: N_j},
                 
                }
          }
    
    # print(idx, jdx)
    
    for i in global_ind:
        if bounds is not None:
            data[None]['z'][i] = rescale(z_list[i][-1], bounds[i-1])
        else:
            data[None]['z'][i] = z_list[i][-1]
        if u_list is not None:
            data[None]['u'][i] = u_list[i][-1]
        else:
            data[None]['u'][i] = 0
    # print(data)
    return SP_4N(data, return_solver = solver, node_type = node_type, idx = idx)


def SP_2N(data, return_solver = False, node_type = 'Supplier'):
    model = pyo.AbstractModel()
    # model.n_x = pyo.RangeSet(1, 4)
    # model.x_init = pyo.Param(model.i)
    
    model.I = pyo.Set()
    model.z_I = pyo.Set()
    model.z = pyo.Param(model.z_I)
    model.x = pyo.Var(model.I)
    model.N = pyo.Param()
    model.N_i = pyo.Param()
    model.N_j = pyo.Param()
    model.i = pyo.RangeSet(1, model.N_i)
    model.j = pyo.RangeSet(1, model.N_j)
    model.k = pyo.RangeSet(1, model.N)
    model.x_k_s = pyo.RangeSet(1, 2*model.N)
    model.rho = pyo.Param()
    
    model.C_k  = pyo.Var(model.k, within = pyo.NonNegativeReals)
    model.y_k   = pyo.Var(model.k, within = pyo.Binary)

    model.f_k = pyo.Var(model.k, within = pyo.NonNegativeReals)
    
    model.ff_k   = pyo.Param(model.k)
    model.vf_k   = pyo.Param(model.k)
    model.mc_k   = pyo.Param(model.k)

    model.cv_k   = pyo.Param(model.k)
    
    if node_type == 'Supplier':
        model.x_i = pyo.Param(model.i)
        model.z_i = pyo.Param(model.i)
        model.C_ik = pyo.Var(model.i, model.k, within = pyo.NonNegativeReals)
        model.y_ixks = pyo.Var(model.i, model.x_k_s, within = pyo.Binary)
        model.y_ik  = pyo.Var(model.i, model.k, within = pyo.Binary)
        model.alpha_plus_i  = pyo.Var(model.i, model.x_k_s, within = pyo.NonNegativeReals)
        model.alpha_minus_i = pyo.Var(model.i, model.x_k_s, within = pyo.NonNegativeReals)
        model.f_ik = pyo.Var(model.i, model.k, within = pyo.NonNegativeReals)    
        model.D_ik = pyo.Var(model.i, model.k, within = pyo.NonNegativeReals)
        model.cs_i   = pyo.Param(model.i)
        model.a_i    = pyo.Param(model.i)
        model.ft_ik  = pyo.Param(model.i, model.k)
        model.vt_ik  = pyo.Param(model.i, model.k)
        model.f_ik_U = pyo.Param(model.i, model.k)
        model.f_ik_L = pyo.Param(model.i, model.k)
        model.D_ik_U = pyo.Param(model.i, model.k)
        model.D_ik_L = pyo.Param(model.i, model.k)
        
        def o(m):
            return sum(m.C_k[k] for k in m.k)/2 + \
                    sum(m.C_ik[i,k] for i in m.i for k in m.k) + \
                        m.rho*sum((m.x[i] - m.z[i])**2 for i in m.z_I)
        model.obj = pyo.Objective(rule = o)
        
        def cost_ik(m, i, k):
            return m.C_ik[i,k] == (m.cs_i[i]*m.f_ik[i,k] + m.ft_ik[i,k]*m.y_ik[i,k] + m.vt_ik[i,k]*5*(m.D_ik[i,k]*m.f_ik[i,k])) # *m.dummy_ik[i,k])
        model.eq_cost_ik = pyo.Constraint(model.i, model.k, rule = cost_ik)
        
        def bound_f_ik(m, i, k):
            return m.f_ik[i,k] <= m.f_ik_U[i,k]*m.y_ik[i,k] # 0 <=
        model.ineq_bound_f_ik = pyo.Constraint(model.i, model.k, rule = bound_f_ik)
    
        def bound_D_ik_1(m, i, k):
            return m.D_ik_L[i,k]*m.y_ik[i,k] <= m.D_ik[i,k] 
        model.ineq_bound_D_ik_1 = pyo.Constraint(model.i, model.k, rule = bound_D_ik_1)
    
        def bound_D_ik_2(m, i, k):
            return  m.D_ik[i,k] <= m.D_ik_U[i,k]*m.y_ik[i,k]
        model.ineq_bound_D_ik_2 = pyo.Constraint(model.i, model.k, rule = bound_D_ik_2)
    
        def D_ik(m, i, k):
            return m.D_ik[i,k] == m.alpha_plus_i[i, k] + m.alpha_minus_i[i, k] + m.alpha_plus_i[i, k+m.N] + m.alpha_minus_i[i, k+m.N]
        model.ineq_D_ik = pyo.Constraint(model.i, model.k, rule = D_ik)
    
        def alpha_i_x(m, i, k):
            return m.alpha_plus_i[i, k] - m.alpha_minus_i[i, k] == m.x[k] - m.x_i[i]
        model.i_x = pyo.Constraint(model.i, model.k, rule = alpha_i_x)
        
        def alpha_i_y(m, i, k):
            return m.alpha_plus_i[i, k+m.N] - m.alpha_minus_i[i, k+m.N] == m.x[k+m.N] - m.z_i[i]
        model.eq_alpha_i_x = pyo.Constraint(model.i, model.k, rule = alpha_i_y)
    
        def api_bound(m, i, x_k_s):
            return m.alpha_plus_i[i, x_k_s] <= m.y_ixks[i, x_k_s]*10
        model.eq_api_bound = pyo.Constraint(model.i, model.x_k_s, rule=api_bound)
    
        def ami_bound(m, i, x_k_s):
            return m.alpha_minus_i[i, x_k_s] <= (1-m.y_ixks[i, x_k_s])*10
        model.eq_ami_bound = pyo.Constraint(model.i, model.x_k_s, rule=ami_bound)     
        
        def W_k_1(m, i, k):
            return m.y_k[k] >= m.y_ik[i,k]
        model.ineq_W_k_1 = pyo.Constraint(model.i, model.k, rule = W_k_1)
 
        def max_avail(m, i):
            return sum(m.f_ik[i,k] for k in m.k) <= m.a_i[i]
        model.ineq_max_avail = pyo.Constraint(model.i, rule = max_avail)    
 
        def link_ik(m, k):
            return sum(m.f_ik[i,k]*m.cv_k[k] for i in m.i) == m.x[k+2*m.N]
        model.eq_link_ik = pyo.Constraint(model.k, rule = link_ik)   
 
    else:
        model.x_j = pyo.Param(model.j)
        model.z_j = pyo.Param(model.j)
        model.C_kj = pyo.Var(model.k, model.j, within = pyo.NonNegativeReals)
        model.y_jxks = pyo.Var(model.j, model.x_k_s, within = pyo.Binary)
        model.y_kj  = pyo.Var(model.k, model.j, within = pyo.Binary)
        model.alpha_plus_j  = pyo.Var(model.j, model.x_k_s, within = pyo.NonNegativeReals)
        model.alpha_minus_j = pyo.Var(model.j, model.x_k_s, within = pyo.NonNegativeReals)
        model.f_kj = pyo.Var(model.k, model.j, within = pyo.NonNegativeReals)
        model.D_kj = pyo.Var(model.k, model.j, within = pyo.NonNegativeReals)
        model.ft_kj  = pyo.Param(model.k, model.j)
        model.vt_kj  = pyo.Param(model.k, model.j)
        model.d_j    = pyo.Param(model.j)
        model.f_kj_U = pyo.Param(model.k, model.j)
        model.f_kj_L = pyo.Param(model.k, model.j)
        model.D_kj_U = pyo.Param(model.k, model.j)
        model.D_kj_L = pyo.Param(model.k, model.j)
        
        def o(m):
            return sum(m.C_k[k] for k in m.k)/2 + \
                    sum(m.C_kj[k,j] for k in m.k for j in m.j) + \
                        m.rho*sum((m.x[i] - m.z[i])**2 for i in m.z_I)
        model.obj = pyo.Objective(rule = o)
        
        def cost_kj(m, k, j):
            return m.C_kj[k,j] == (m.ft_kj[k,j]*m.y_kj[k,j] + m.vt_kj[k,j]*5*(m.D_kj[k,j]*m.f_kj[k,j])) # *m.dummy_kj[k,j])
        model.eq_cost_kj = pyo.Constraint(model.k, model.j, rule = cost_kj)
        
        def bound_f_kj(m, k, j):
            return m.f_kj[k,j] <= m.f_kj_U[k,j]*m.y_kj[k,j] # 0 <= 
        model.ineq_bound_f_kj = pyo.Constraint(model.k, model.j, rule = bound_f_kj)
    
        def bound_D_kj_1(m, k, j):
            return m.D_kj_L[k,j]*m.y_kj[k,j] <= m.D_kj[k,j] # <= m.D_kj_U[k,j]*m.y_kj[k,j]
        model.ineq_bound_D_kj_1 = pyo.Constraint(model.k, model.j, rule = bound_D_kj_1)
    
        def bound_D_kj_2(m, k, j):
            return m.D_kj_U[k,j]*m.y_kj[k,j] >= m.D_kj[k,j] # <= m.D_kj_U[k,j]*m.y_kj[k,j]
        model.ineq_bound_D_kj_2 = pyo.Constraint(model.k, model.j, rule = bound_D_kj_2)

        def D_kj(m, k, j):
            return m.D_kj[k,j] == m.alpha_plus_j[j, k] + m.alpha_minus_j[j, k] + m.alpha_plus_j[j, k+m.N] + m.alpha_minus_j[j, k+m.N]
        model.ineq_D_kj = pyo.Constraint(model.k, model.j, rule = D_kj)
    
        def alpha_j_x(m, j, k):
            return m.alpha_plus_j[j, k] - m.alpha_minus_j[j, k] == m.x[k] - m.x_j[j]
        model.eq_alpha_j_x = pyo.Constraint(model.j, model.k, rule = alpha_j_x)
    
        def alpha_j_y(m, j, k):
            return m.alpha_plus_j[j, k+m.N] - m.alpha_minus_j[j, k+m.N] == m.x[k+m.N] - m.z_j[j]
        model.eq_alpha_j_y = pyo.Constraint(model.j, model.k, rule = alpha_j_y)
    
        def apj_bound(m, j, x_k_s):
            return m.alpha_plus_j[j, x_k_s] <= m.y_jxks[j, x_k_s]*10
        model.eq_apj_bound = pyo.Constraint(model.j, model.x_k_s, rule=apj_bound)
    
        def amj_bound(m, j, x_k_s):
            return m.alpha_minus_j[j, x_k_s] <= (1-m.y_jxks[j, x_k_s])*10
        model.eq_amj_bound = pyo.Constraint(model.j, model.x_k_s, rule=amj_bound)
    
        def W_k_2(m, k, j):
            return m.y_k[k] >= m.y_kj[k,j]
        model.ineq_W_k_2 = pyo.Constraint(model.k, model.j, rule = W_k_2)
    
        def link_kj(m, k):
            return sum(m.f_kj[k,j] for j in m.j) == m.x[k+2*m.N]
        model.eq_link_kj = pyo.Constraint(model.k, rule = link_kj)
    
        def link_demand(m, j):
            return sum(m.f_kj[k,j] for k in m.k) == m.d_j[j]
        model.eq_link_demand = pyo.Constraint(model.j, rule = link_demand)
    
    
    def cost_k(m, k):
        return m.C_k[k] == (m.ff_k[k]*m.y_k[k] + m.vf_k[k]*m.x[k+2*m.N])
    model.eq_cost_k = pyo.Constraint(model.k, rule = cost_k)
    
    
    ins = model.create_instance(data)
    
    #ins.x = ins.x_init
    
    #solver = pyo.SolverFactory('cbc')
    solver = pyo.SolverFactory('gurobi_direct')
    solver.options['NonConvex'] = 2
    res = solver.solve(ins)
    
    if return_solver:
        return ins, res
    else:
        return ins


def f_2N(z_list, rho, global_ind, index, node_type, data, u_list = None, 
      solver = False, seed=2, bounds = None):
    
    np.random.seed(seed)
    
    def rescale(x, lower_upper_tuple):
        return lower_upper_tuple[0] + x*(lower_upper_tuple[1] - lower_upper_tuple[0])
    
    data[None]['node_type'] = {}
    data[None]['z_I'] = {} 
    data[None]['rho'] = {}
    data[None]['I'] = {}
    data[None]['z'] = {}
    data[None]['u'] = {}
    
    data[None]['node_type'][None] = node_type
    data[None]['z_I'][None] = global_ind 
    data[None]['rho'][None] = rho
    data[None]['I'][None] = index
    
    for i in global_ind:
        if bounds is not None:
            data[None]['z'][i] = rescale(z_list[i][-1], bounds[i-1])
        else:
            data[None]['z'][i] = z_list[i][-1]
        if u_list is not None:
            data[None]['u'][i] = u_list[i][-1]
        else:
            data[None]['u'][i] = 0
    # print(data)
    return SP_2N(data, return_solver = solver, node_type = node_type)



