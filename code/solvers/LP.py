""" This file is taken from the MOM codebase"""
import sys
sys.path.append("./")
sys.path.append("../")

import numpy as np
import cvxpy as cp
from multiprocessing import Pool
from functools import partial
from cvxpy import OSQP

# Compute LP
def ComputeLP(A, b, c):
    (dim_constraints, dim_target) = A.shape
    x = cp.Variable(dim_target)
    constr = [A@x == b, x >= 0, x<=1]
    obj = cp.Minimize(c.T@x)
    prob = cp.Problem(obj, constr)
    solver_options = {'GUROBI': {'Presolve': 2, 'WarmStart': 2}}
    # prob.solve(solver = cp.GUROBI, verbose=False, solver_opts=solver_options)
    prob.solve(warm_start=True)
    val = prob.value
    del prob
    return x.value, val, constr[0].dual_value

def ComputeLP_dual(A, b, c, gamma=1):
    (dim_constraints, dim_target) = A.shape
    x = cp.Variable(dim_target)
    constr = [A@x == b, x >= 0, x<=1]
    obj = cp.Minimize(c.T@x+ gamma*cp.sum_squares(x))
    prob = cp.Problem(obj, constr)
    solver_options = {'GUROBI': {'Presolve': 2, 'WarmStart': 2}}
    # prob.solve(solver = cp.GUROBI, verbose=False, solver_opts=solver_options)
    prob.solve()
    val = prob.value
    del prob
    return x.value, val, constr[0].dual_value, constr[1].dual_value

class Solver:

    def __init__(self, A, b, c, gamma=None):
        self.A = A
        self.b = b
        self.c = c
        self.gamma= gamma

    def StoreLP(self, i):
        (dim_constraints, dim_target) = self.A.shape
        # print("shape", self.A[i].shape, self.b[i].shape, self.c[i].shape)
        x, _, dual = ComputeLP(self.A, self.b, self.c[i])
        
        reduced_cost = self.c[i] + (self.A).T @ dual
        idx = np.argpartition(reduced_cost, -(dim_target - dim_constraints))
        basic_tmp = idx[:dim_constraints]
        nonb_tmp = idx[dim_constraints:]
        basic = np.sort(basic_tmp)
        nonb = np.sort(nonb_tmp)
        return (i, basic, nonb, x)
    
    def ComputeLP(self, i):
        x, val, dual = ComputeLP(self.A, self.b, self.c[i])
        return (i, val, x)

    def ComputeLP_dual(self, i):
        x, val, dual_nu, dual_lambda = ComputeLP_dual(self.A, self.b, self.c[i], self.gamma)
        return (i, dual_lambda, x)


# Prepare the optimal solution
def ComputeBasis(A, b, c):
    if A.shape[-1] != c.shape[-1] or A.shape[0] != b.shape[0]:
        raise ValueError("input dimensions do not coincide")

    # print(A.shape, b.shape, c.shape)
    # exit()
    (dim_constraints, dim_target) = A.shape
    N_samples = c.shape[0]
    solver = Solver(A, b, c)
    basic = np.zeros((N_samples, dim_constraints), dtype = np.int_)
    nonb = np.zeros((N_samples, dim_target - dim_constraints), dtype = np.int_)
    solution = np.zeros((N_samples, dim_target), dtype = np.float64)
    
    for j in range (N_samples):
        i, base_get, nonb_get, solu_get = solver.StoreLP(j)
        basic[i] = base_get
        nonb[i] = nonb_get
        solution[i] = solu_get
    return basic, nonb, solution

# Compute the Loss
def ComputeLoss(A, b, c, z, direct=False, Theta=None, hat_c=None, solved=False, solution=None, benchmark=1):
    if A.shape[0] != z.shape[0] or A.shape[0] != b.shape[0] or A.shape[0] != c.shape[0]:
        raise ValueError("input dimensions do not coincide")
    if A.shape[1] != b.shape[1]:
        raise ValueError("input dimensions do not coincide")
    (N_samples, dim_constraints, dim_target) = A.shape
    dim_features = z.shape[1]

    
    if direct == False:
        hat_c = z@Theta.T
    normalized_err = np.zeros(N_samples)
    error = np.zeros(N_samples)
    for i in range(N_samples):
        error[i] = np.linalg.norm(hat_c[i] - c[i] ,2) / np.linalg.norm(c[i], 2)
        normalized_err[i] = np.linalg.norm(hat_c[i] / np.linalg.norm(hat_c[i], 2) - c[i] / np.linalg.norm(c[i], 2), 2)

    Loss_true = []
    
    if solved == False:
        po1 = Pool()
        solution = np.zeros((N_samples, dim_target), dtype = np.float64)
        solver = Solver(A, b, c)
        for result in po1.map(solver.StoreLP, range(N_samples)):

            i, _, _, solu_get = result

            solution[i] = solu_get
        po1.close()
        po1.join()
    if direct == False:
        est_c = z @ Theta.T
    else:
        est_c = hat_c
    po2 = Pool()
    solution_est = np.zeros((N_samples, dim_target), dtype = np.float64)
    solver = Solver(A, b, est_c)
    for result in po2.map(solver.StoreLP, range(N_samples)):

        i, _, _, solu_get = result

        solution_est[i] = solu_get
    po2.close()
    po2.join()
    if benchmark == 1:
        for i in range(N_samples):
            Loss_true.append(c[i].T@(solution_est[i] - solution[i]) / np.linalg.norm(c[i], 2))
    elif benchmark == 2:
        for i in range(N_samples):
            Loss_true.append(c[i].T@(solution_est[i] - solution[i]) / c[i].T@solution[i])
    return Loss_true, error, normalized_err
    
    