import numpy as np
import cvxpy as cp

def calc_Jacobian(x, u, param):

    L_f = param["L_f"]
    L_r = param["L_r"]
    dt   = param["h"]

    psi = x[2]
    v   = x[3]
    delta = u[1]
    a   = u[0]

    # Jacobian of the system dynamics
    A = np.zeros((4, 4))
    B = np.zeros((4, 2))

    #############################################################################
    #                    TODO: Implement your code here                         #
    #############################################################################

    L = L_f + L_r
    tan_delta = np.arctan(delta)
    arg = (L_r * tan_delta) / L
    beta = np.arctan(arg)
    denom_sq = arg**2 + 1.0
    
    A[0, 0] = 1.0
    A[0, 2] = -dt * v * np.sin(psi + beta)
    A[0, 3] = dt * np.cos(psi + beta)
    
    A[1, 1] = 1.0
    A[1, 2] = dt * v * np.cos(psi + beta)
    A[1, 3] = dt * np.sin(psi + beta)
    
    A[2, 2] = 1.0
    A[2, 3] = (dt * tan_delta) / (np.sqrt(denom_sq) * L)
    
    A[3, 3] = 1.0
    
    delta_sq_1 = delta**2 + 1.0
    
    B[0, 1] = -(dt * L_r * v * np.sin(psi + beta)) / (delta_sq_1 * denom_sq * L)
    B[1, 1] = (dt * L_r * v * np.cos(psi + beta)) / (delta_sq_1 * denom_sq * L)
    B[2, 1] = (dt * v) / (delta_sq_1 * (denom_sq**1.5) * L)
    B[3, 0] = dt
    #############################################################################
    #                            END OF YOUR CODE                               #
    #############################################################################

    return [A, B]

def LQR_Controller(x_bar, u_bar, x0, param):
    len_state = x_bar.shape[0]
    len_ctrl  = u_bar.shape[0]
    dim_state = x_bar.shape[1]
    dim_ctrl  = u_bar.shape[1]

    n_u = len_ctrl * dim_ctrl
    n_x = len_state * dim_state
    n_var = n_u + n_x

    n_eq  = dim_state * len_ctrl # dynamics
    n_ieq = dim_ctrl * len_ctrl  # input constraints

    
    #############################################################################
    #                    TODO: Implement your code here                         #
    #############################################################################

    # define the parameters
    Q = np.diag([10.0, 10.0, 5.0, 2.0])
    R = np.diag([0.5, 10.0])
    Pt = np.diag([20.0, 20.0, 10.0, 5.0])

    # define the cost function
    P = np.zeros((n_var, n_var))
    for k in range(len_ctrl):
        P[k*dim_state : (k+1)*dim_state, k*dim_state : (k+1)*dim_state] = Q
        P[n_x + k*dim_ctrl : n_x + (k+1)*dim_ctrl, n_x + k*dim_ctrl : n_x + (k+1)*dim_ctrl] = R
    P[len_ctrl*dim_state : (len_ctrl+1)*dim_state, len_ctrl*dim_state : (len_ctrl+1)*dim_state] = Pt
    q = np.zeros(n_var)
    
    # define the constraints (initial condition and linearized dynamics)
    n_eq_total = dim_state * len_state
    A = np.zeros((n_eq_total, n_var))
    b = np.zeros(n_eq_total)

    A[0:dim_state, 0:dim_state] = np.eye(dim_state)
    b[0:dim_state] = x0 - x_bar[0, :]

    for k in range(len_ctrl):
        Ak, Bk = calc_Jacobian(x_bar[k, :], u_bar[k, :], param)
        row_start = (k + 1) * dim_state
        row_end = (k + 2) * dim_state
        
        A[row_start:row_end, (k+1)*dim_state : (k+2)*dim_state] = np.eye(dim_state)
        A[row_start:row_end, k*dim_state : (k+1)*dim_state] = -Ak
        A[row_start:row_end, n_x + k*dim_ctrl : n_x + (k+1)*dim_ctrl] = -Bk
    
    # Define and solve the CVXPY problem.
    x = cp.Variable(n_var)
    prob = cp.Problem(cp.Minimize(0.5 * cp.quad_form(x, P) + q @ x), [A @ x == b])
    prob.solve(verbose=False, max_iter=10000)

    #############################################################################
    #                            END OF YOUR CODE                               #
    #############################################################################

    u_act = x.value[n_x:n_x + dim_ctrl] + u_bar[0, :]
    return u_act

def CMPC_Controller(x_bar, u_bar, x0, param):
    len_state = x_bar.shape[0]
    len_ctrl  = u_bar.shape[0]
    dim_state = x_bar.shape[1]
    dim_ctrl  = u_bar.shape[1]
    
    n_u = len_ctrl * dim_ctrl
    n_x = len_state * dim_state
    n_var = n_u + n_x

    n_eq  = dim_state * len_ctrl # dynamics
    n_ieq = dim_ctrl * len_ctrl # input constraints

    a_limit = param["a_lim"]
    delta_limit = param["delta_lim"]
    
    #############################################################################
    #                    TODO: Implement your code here                         #
    #############################################################################
    
    # define the parameters
    Q = np.diag([20.0, 20.0, 10.0, 5.0])
    R = np.diag([0.1, 1.0])
    Pt = np.diag([30.0, 30.0, 15.0, 10.0])
    
    # define the cost function
    P = np.zeros((n_var, n_var))
    for k in range(len_ctrl):
        P[k*dim_state : (k+1)*dim_state, k*dim_state : (k+1)*dim_state] = Q
        P[n_x + k*dim_ctrl : n_x + (k+1)*dim_ctrl, n_x + k*dim_ctrl : n_x + (k+1)*dim_ctrl] = R
    P[len_ctrl*dim_state : (len_ctrl+1)*dim_state, len_ctrl*dim_state : (len_ctrl+1)*dim_state] = Pt
    q = np.zeros(n_var)
    
    # define the constraints
    n_eq_total = dim_state * len_state
    A = np.zeros((n_eq_total, n_var))
    b = np.zeros(n_eq_total)

    A[0:dim_state, 0:dim_state] = np.eye(dim_state)
    b[0:dim_state] = x0 - x_bar[0, :]

    for k in range(len_ctrl):
        Ak, Bk = calc_Jacobian(x_bar[k, :], u_bar[k, :], param)
        row_start = (k + 1) * dim_state
        row_end = (k + 2) * dim_state
        
        A[row_start:row_end, (k+1)*dim_state : (k+2)*dim_state] = np.eye(dim_state)
        A[row_start:row_end, k*dim_state : (k+1)*dim_state] = -Ak
        A[row_start:row_end, n_x + k*dim_ctrl : n_x + (k+1)*dim_ctrl] = -Bk

    lb = np.zeros(n_u)
    ub = np.zeros(n_u)
    for k in range(len_ctrl):
        lb[k*dim_ctrl] = a_limit[0] - u_bar[k, 0]
        lb[k*dim_ctrl + 1] = delta_limit[0] - u_bar[k, 1]
        ub[k*dim_ctrl] = a_limit[1] - u_bar[k, 0]
        ub[k*dim_ctrl + 1] = delta_limit[1] - u_bar[k, 1]

    # Define and solve the CVXPY problem.
    x = cp.Variable(n_var)
    constraints = [A @ x == b, x[n_x:] >= lb, x[n_x:] <= ub]
    prob = cp.Problem(cp.Minimize(0.5 * cp.quad_form(x, P) + q @ x), constraints)
    prob.solve(verbose=False, max_iter=10000)

    #############################################################################
    #                            END OF YOUR CODE                               #
    #############################################################################
    
    u_act = x.value[n_x:n_x + dim_ctrl] + u_bar[0, :]
    return u_act