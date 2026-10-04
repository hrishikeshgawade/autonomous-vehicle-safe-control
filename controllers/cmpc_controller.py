"""
Trajectory Tracking via Time-Varying LQR and Constrained Model Predictive Control (CMPC).

This module implements trajectory tracking controllers for an autonomous ground
vehicle using a nonlinear kinematic bicycle model. Linearization is performed
analytically along the nominal reference trajectory to obtain time-varying Jacobians
(A_k, B_k). Both unconstrained TV-LQR and input-constrained MPC (CMPC) are formulated
as convex Quadratic Programs (QP) and solved using CVXPY.

Part of the Autonomous Vehicles / Self-Driving Systems Coursework.
"""

from typing import Dict, List, Tuple, Any
import numpy as np
import cvxpy as cp


def calc_Jacobian(
    x: np.ndarray,
    u: np.ndarray,
    param: Dict[str, Any]
) -> List[np.ndarray]:
    """
    Computes analytical Jacobians (A, B) of discrete-time kinematic bicycle dynamics.

    Kinematic model:
        x_dot = v * cos(psi + beta)
        y_dot = v * sin(psi + beta)
        psi_dot = (v / L_r) * sin(beta)
        v_dot = a
    where side-slip angle beta is defined as:
        beta = arctan( (L_r / (L_f + L_r)) * tan(delta) )

    Discretized via forward Euler: s_{k+1} = s_k + dt * f(s_k, u_k).

    Args:
        x: State vector [x, y, psi, v]
            - x[0]: Vehicle X position (m)
            - x[1]: Vehicle Y position (m)
            - x[2]: Yaw angle psi (rad)
            - x[3]: Longitudinal velocity v (m/s)
        u: Control vector [a, delta]
            - u[0]: Longitudinal acceleration a (m/s^2)
            - u[1]: Front steering angle delta (rad)
        param: Dictionary containing vehicle parameters:
            - L_f: Distance from CG to front axle (m)
            - L_r: Distance from CG to rear axle (m)
            - h: Discretization timestep dt (s)

    Returns:
        [A, B]: List of analytical Jacobian matrices:
            - A: Discrete state transition Jacobian df/dx, shape (4, 4)
            - B: Discrete control input Jacobian df/du, shape (4, 2)
    """
    L_f = param["L_f"]
    L_r = param["L_r"]
    dt = param["h"]

    psi = x[2]
    v = x[3]
    a = u[0]
    delta = u[1]

    # Jacobian of the system dynamics
    A = np.zeros((4, 4))
    B = np.zeros((4, 2))

    L = L_f + L_r
    tan_delta = np.arctan(delta)
    arg = (L_r * tan_delta) / L
    beta = np.arctan(arg)
    denom_sq = arg**2 + 1.0

    # State Jacobian A = I + dt * df_c/dx
    A[0, 0] = 1.0
    A[0, 2] = -dt * v * np.sin(psi + beta)
    A[0, 3] = dt * np.cos(psi + beta)

    A[1, 1] = 1.0
    A[1, 2] = dt * v * np.cos(psi + beta)
    A[1, 3] = dt * np.sin(psi + beta)

    A[2, 2] = 1.0
    A[2, 3] = (dt * tan_delta) / (np.sqrt(denom_sq) * L)

    A[3, 3] = 1.0

    # Input Jacobian B = dt * df_c/du
    delta_sq_1 = delta**2 + 1.0

    B[0, 1] = -(dt * L_r * v * np.sin(psi + beta)) / (delta_sq_1 * denom_sq * L)
    B[1, 1] = (dt * L_r * v * np.cos(psi + beta)) / (delta_sq_1 * denom_sq * L)
    B[2, 1] = (dt * v) / (delta_sq_1 * (denom_sq**1.5) * L)
    B[3, 0] = dt

    return [A, B]


def LQR_Controller(
    x_bar: np.ndarray,
    u_bar: np.ndarray,
    x0: np.ndarray,
    param: Dict[str, Any]
) -> np.ndarray:
    """
    Solves finite-horizon Time-Varying LQR tracking over preview horizon N via CVXPY.

    Optimizes state and control deviations along reference trajectory (x_bar, u_bar):
        min_{delta_s, delta_u} delta_s_N^T Pt delta_s_N + sum_{k=0}^{N-1} (delta_s_k^T Q delta_s_k + delta_u_k^T R delta_u_k)
        subject to:
            delta_s_0 = x0 - x_bar[0]
            delta_s_{k+1} = A_k delta_s_k + B_k delta_u_k

    Args:
        x_bar: Reference state trajectory over horizon, shape (N+1, 4)
        u_bar: Reference control trajectory over horizon, shape (N, 2)
        x0: Current vehicle initial state, shape (4,)
        param: Vehicle and simulation parameter dictionary

    Returns:
        u_act: Absolute control input to apply at current step: u_bar[0] + delta_u[0], shape (2,)
    """
    len_state = x_bar.shape[0]
    len_ctrl = u_bar.shape[0]
    dim_state = x_bar.shape[1]
    dim_ctrl = u_bar.shape[1]

    n_u = len_ctrl * dim_ctrl
    n_x = len_state * dim_state
    n_var = n_u + n_x

    # Tuning weights: state error penalty Q, control effort penalty R, terminal penalty Pt
    Q = np.diag([10.0, 10.0, 5.0, 2.0])
    R = np.diag([0.5, 10.0])
    Pt = np.diag([20.0, 20.0, 10.0, 5.0])

    # Construct block-diagonal cost Hessian matrix P
    P = np.zeros((n_var, n_var))
    for k in range(len_ctrl):
        P[k * dim_state : (k + 1) * dim_state, k * dim_state : (k + 1) * dim_state] = Q
        P[n_x + k * dim_ctrl : n_x + (k + 1) * dim_ctrl, n_x + k * dim_ctrl : n_x + (k + 1) * dim_ctrl] = R
    P[len_ctrl * dim_state : (len_ctrl + 1) * dim_state, len_ctrl * dim_state : (len_ctrl + 1) * dim_state] = Pt
    q = np.zeros(n_var)

    # Equality constraints: initial condition & time-varying linearized dynamics
    n_eq_total = dim_state * len_state
    A_eq = np.zeros((n_eq_total, n_var))
    b_eq = np.zeros(n_eq_total)

    # Initial condition: delta_s_0 = x0 - x_bar[0]
    A_eq[0:dim_state, 0:dim_state] = np.eye(dim_state)
    b_eq[0:dim_state] = x0 - x_bar[0, :]

    # Linearized dynamics: delta_s_{k+1} - A_k delta_s_k - B_k delta_u_k = 0
    for k in range(len_ctrl):
        Ak, Bk = calc_Jacobian(x_bar[k, :], u_bar[k, :], param)
        row_start = (k + 1) * dim_state
        row_end = (k + 2) * dim_state

        A_eq[row_start:row_end, (k + 1) * dim_state : (k + 2) * dim_state] = np.eye(dim_state)
        A_eq[row_start:row_end, k * dim_state : (k + 1) * dim_state] = -Ak
        A_eq[row_start:row_end, n_x + k * dim_ctrl : n_x + (k + 1) * dim_ctrl] = -Bk

    # Solve batch QP with CVXPY
    x = cp.Variable(n_var)
    prob = cp.Problem(cp.Minimize(0.5 * cp.quad_form(x, P) + q @ x), [A_eq @ x == b_eq])
    prob.solve(verbose=False, max_iter=10000)

    # Extract first control deviation and compute actual actuator command
    u_act = x.value[n_x : n_x + dim_ctrl] + u_bar[0, :]
    return u_act


def CMPC_Controller(
    x_bar: np.ndarray,
    u_bar: np.ndarray,
    x0: np.ndarray,
    param: Dict[str, Any]
) -> np.ndarray:
    """
    Solves Constrained Model Predictive Control (CMPC) with hard actuator limits via CVXPY.

    Optimizes state and control deviations along reference trajectory (x_bar, u_bar):
        min_{delta_s, delta_u} delta_s_N^T Pt delta_s_N + sum_{k=0}^{N-1} (delta_s_k^T Q delta_s_k + delta_u_k^T R delta_u_k)
        subject to:
            delta_s_0 = x0 - x_bar[0]
            delta_s_{k+1} = A_k delta_s_k + B_k delta_u_k
            u_min - u_bar_k <= delta_u_k <= u_max - u_bar_k

    Args:
        x_bar: Reference state trajectory over horizon, shape (N+1, 4)
        u_bar: Reference control trajectory over horizon, shape (N, 2)
        x0: Current vehicle initial state, shape (4,)
        param: Vehicle and simulation parameter dictionary containing:
            - a_lim: Acceleration bounds [a_min, a_max] (m/s^2)
            - delta_lim: Steering angle bounds [delta_min, delta_max] (rad)

    Returns:
        u_act: Absolute control input to apply at current step: u_bar[0] + delta_u[0], shape (2,)
    """
    len_state = x_bar.shape[0]
    len_ctrl = u_bar.shape[0]
    dim_state = x_bar.shape[1]
    dim_ctrl = u_bar.shape[1]

    n_u = len_ctrl * dim_ctrl
    n_x = len_state * dim_state
    n_var = n_u + n_x

    a_limit = param["a_lim"]
    delta_limit = param["delta_lim"]

    # Tuning weights: aggressive path tracking Q, smooth control R, terminal penalty Pt
    Q = np.diag([20.0, 20.0, 10.0, 5.0])
    R = np.diag([0.1, 1.0])
    Pt = np.diag([30.0, 30.0, 15.0, 10.0])

    # Construct block-diagonal cost Hessian matrix P
    P = np.zeros((n_var, n_var))
    for k in range(len_ctrl):
        P[k * dim_state : (k + 1) * dim_state, k * dim_state : (k + 1) * dim_state] = Q
        P[n_x + k * dim_ctrl : n_x + (k + 1) * dim_ctrl, n_x + k * dim_ctrl : n_x + (k + 1) * dim_ctrl] = R
    P[len_ctrl * dim_state : (len_ctrl + 1) * dim_state, len_ctrl * dim_state : (len_ctrl + 1) * dim_state] = Pt
    q = np.zeros(n_var)

    # Equality constraints: initial condition & time-varying linearized dynamics
    n_eq_total = dim_state * len_state
    A_eq = np.zeros((n_eq_total, n_var))
    b_eq = np.zeros(n_eq_total)

    # Initial condition: delta_s_0 = x0 - x_bar[0]
    A_eq[0:dim_state, 0:dim_state] = np.eye(dim_state)
    b_eq[0:dim_state] = x0 - x_bar[0, :]

    # Linearized dynamics: delta_s_{k+1} - A_k delta_s_k - B_k delta_u_k = 0
    for k in range(len_ctrl):
        Ak, Bk = calc_Jacobian(x_bar[k, :], u_bar[k, :], param)
        row_start = (k + 1) * dim_state
        row_end = (k + 2) * dim_state

        A_eq[row_start:row_end, (k + 1) * dim_state : (k + 2) * dim_state] = np.eye(dim_state)
        A_eq[row_start:row_end, k * dim_state : (k + 1) * dim_state] = -Ak
        A_eq[row_start:row_end, n_x + k * dim_ctrl : n_x + (k + 1) * dim_ctrl] = -Bk

    # Actuator limits transformed to bounds on control deviations delta_u
    lb = np.zeros(n_u)
    ub = np.zeros(n_u)
    for k in range(len_ctrl):
        lb[k * dim_ctrl] = a_limit[0] - u_bar[k, 0]
        lb[k * dim_ctrl + 1] = delta_limit[0] - u_bar[k, 1]
        ub[k * dim_ctrl] = a_limit[1] - u_bar[k, 0]
        ub[k * dim_ctrl + 1] = delta_limit[1] - u_bar[k, 1]

    # Solve constrained QP with CVXPY
    x = cp.Variable(n_var)
    constraints = [A_eq @ x == b_eq, x[n_x:] >= lb, x[n_x:] <= ub]
    prob = cp.Problem(cp.Minimize(0.5 * cp.quad_form(x, P) + q @ x), constraints)
    prob.solve(verbose=False, max_iter=10000)

    # Extract first control deviation and compute actual actuator command
    u_act = x.value[n_x : n_x + dim_ctrl] + u_bar[0, :]
    return u_act