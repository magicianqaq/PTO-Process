import sys
sys.path.append("./")
sys.path.append("../")

import cvxpy as cp
import numpy as np

def solve_qp(Q, c, G=None, h=None, A=None, b=None):
    """
    Use CVXPY to solve a quadratic programming problem:
    minimize (1/2) x.T @ Q @ x + c.T @ x
    subject to:
    G @ x <= h (inequality constraints)
    A @ x = b  (equality constraints)

    Parameters:
    Q: (n, n) symmetric positive definite matrix
    c: (n,) linear coefficient vector
    G: (m, n) inequality constraint matrix (optional)
    h: (m,) inequality constraint constant vector (optional)
    A: (p, n) equality constraint matrix (optional)
    b: (p,) equality constraint constant vector (optional)
    Returns:
    x_opt: optimal solution
    obj_val: optimal objective value
    """
    # Decision variable x
    n = Q.shape[0]
    x = cp.Variable(n)
    
    # Define the objective function (1/2) x.T @ Q @ x + c.T @ x
    objective = cp.Minimize((1/2) * cp.quad_form(x, Q) + c.T @ x)

    # Define constraints
    constraints = []
    if G is not None and h is not None:
        constraints.append(G @ x <= h)  # Inequality constraint Gx <= h
    if A is not None and b is not None:
        constraints.append(A @ x == b)  # Equality constraint Ax = b

    # Define the problem
    prob = cp.Problem(objective, constraints)

    # Solve the problem
    prob.solve()

    # Return the optimal solution and optimal objective value
    return x.value, prob.value

def obj(c, x, x_1=None):
    if x_1 is not None:
        return c.T @ x - pen(c, x, x_1)
    else:
        return c.T @ x

def pen(c, x, x_1):
    sigma = 1
    return sigma * c.T @ (x_1 - x)