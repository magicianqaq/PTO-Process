import cvxpy as cp
import numpy as np
import sys
sys.path.append("./")
sys.path.append("../")

def solve_mip_max(A, b, c, x_1=None, int_type=None, solver='GUROBI'):
    """
    Solve a mixed-integer programming problem using cvxpy.
    
    Parameters:
    A : numpy.ndarray
        The constraint matrix of shape (m, n).
    b : numpy.ndarray
        The constraint vector of shape (m,).
    c : numpy.ndarray
        The objective function vector of shape (n,).
    int_idx : list or None
        Specifies which variables should be integers. If None, all variables are continuous.
    solver : str
        The solver to use. Default is 'GUROBI', but it can be changed to other solvers as needed.

    Returns:
    prob.value : float
        The optimal objective value.
    x.value : numpy.ndarray
        The optimal solution.
    """
    num_vars = len(c)
    
    # Create decision variables
    if int_type == 1:
        x = cp.Variable(num_vars, integer=True) 
    elif int_type == 2:
        x = cp.Variable(num_vars, boolean=True)
    else:
        x = cp.Variable(num_vars)  

    # Objective function
    objective = cp.Maximize(obj(c, x, x_1))

    # Constraints
    constraints = [A @ x <= b]

    # Define the problem
    prob = cp.Problem(objective, constraints)

    # Solve the problem
    prob.solve(solver=solver)

    # Return the optimal objective value and solution
    return x.value, prob.value

def solve_mip(A, b, c, x_1=None, int_type=None, solver='GUROBI'):
    """
    Solve a mixed-integer programming problem using cvxpy.
    
    Parameters:
    A : numpy.ndarray
        The constraint matrix of shape (m, n).
    b : numpy.ndarray
        The constraint vector of shape (m,).
    c : numpy.ndarray
        The objective function vector of shape (n,).
    int_idx : list or None
        Specifies which variables should be integers. If None, all variables are continuous.
    solver : str
        The solver to use. Default is 'GUROBI', but it can be changed to other solvers as needed.

    Returns:
    prob.value : float
        The optimal objective value.
    x.value : numpy.ndarray
        The optimal solution.
    """
    num_vars = len(c)
    
    # Create decision variables
    if int_type == 1:
        x = cp.Variable(num_vars, integer=True) 
    elif int_type == 2:
        x = cp.Variable(num_vars, boolean=True)
    else:
        x = cp.Variable(num_vars)  

    # Objective function
    objective = cp.Minimize(obj(c, x, x_1))

    # Constraints
    constraints = [A @ x <= b]

    # Define the problem
    prob = cp.Problem(objective, constraints)

    # Solve the problem
    prob.solve(solver=solver)

    # Return the optimal objective value and solution
    return x.value, prob.value

def obj(c, x, x_1=None):
    if x_1 is not None:
        return c.T @ x - pen(c, x, x_1)
    else:
        return c.T @ x

def pen(c, x, x_1):
    sigma = 1
    return sigma * c.T @ (x_1 - x)
