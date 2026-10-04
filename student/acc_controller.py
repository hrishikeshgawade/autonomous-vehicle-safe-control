"""
Safety-Critical Adaptive Cruise Control (ACC) via CLF-CBF-QP.

This module implements an optimization-based safety-critical controller for
autonomous vehicles, combining Control Lyapunov Functions (CLF) for desired velocity
tracking and Control Barrier Functions (CBF) for forward-invariant headway safety.
A slack variable relaxation is utilized to guarantee QP feasibility when tracking
and safety constraints conflict.

Part of the Autonomous Vehicles / Self-Driving Systems Coursework.
"""

from typing import Dict, Tuple, Any
import numpy as np


def ACC_Controller(
    t: float,
    x: np.ndarray,
    param: Dict[str, Any]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Constructs the Quadratic Program (QP) matrices for Safety-Critical ACC.

    Solves the optimization problem:
        min_{z} (1/2) * z^T P z + q^T z
        subject to: A z <= b
    where decision vector z = [F_w, delta]^T:
        - F_w: Wheel traction/braking force (N)
        - delta: Slack variable relaxing the velocity tracking constraint (delta >= 0)

    State vector x:
        - x[0]: Inter-vehicle headway distance D (m)
        - x[1]: Ego vehicle velocity v (m/s)

    Parameters in `param`:
        - vd: Desired cruise velocity (m/s)
        - v0: Lead vehicle velocity (m/s)
        - m: Ego vehicle mass (kg)
        - Cag: Maximum traction acceleration coefficient * g (m/s^2)
        - Cdg: Maximum braking deceleration coefficient * g (m/s^2)

    Returns:
        A: Linear inequality constraint matrix, shape (5, 2)
        b: Linear inequality constraint bound vector, shape (5,)
        P: Quadratic cost Hessian matrix, shape (2, 2)
        q: Linear cost gradient vector, shape (2, 1)
    """
    vd = param["vd"]
    v0 = param["v0"]
    m = param["m"]
    Cag = param["Cag"]
    Cdg = param["Cdg"]

    # Initialize QP matrices
    P = np.zeros((2, 2))
    q = np.zeros([2, 1])
    A = np.zeros([5, 2])
    b = np.zeros([5])

    # Controller tuning hyperparameters
    lam = 3.0     # CLF tracking convergence rate (h_dot <= -lam * h + delta)
    alpha = 1.0   # CBF class-K boundary compliance rate (B_dot >= -alpha * B)
    w = 1e8       # High quadratic penalty on slack violation to prioritize safety

    # Quadratic cost: min (F_w^2 + w * delta^2) => (1/2) * z^T P z with P = diag([2, 2w])
    P = np.array([[2.0, 0.0], [0.0, 2.0 * w]])
    q = np.zeros([2, 1])

    # Current vehicle state
    D = x[0]
    v = x[1]

    # Control Lyapunov Function (CLF) for speed tracking: h(v) = 0.5 * (v - vd)^2
    h = 0.5 * (v - vd) ** 2

    # Control Barrier Function (CBF) enforcing half-speedometer headway + dynamic braking:
    # B(D, v) = D - 1.8*v - (max(0, v - v0)^2) / (2 * c_d * g)
    v_rel = max(0.0, v - v0)
    B = D - 1.8 * v - 0.5 * (v_rel ** 2) / Cdg
    cbf_term = 1.8 + v_rel / Cdg

    # 1. CLF Velocity Tracking Constraint with Slack:
    #    (v - vd)/m * F_w - delta <= -lam * h
    A[0, 0] = (v - vd) / m
    A[0, 1] = -1.0
    b[0] = -lam * h

    # 2. CBF Safety Headway Constraint (ensuring forward invariance B >= 0):
    #    (cbf_term / m) * F_w <= (v0 - v) + alpha * B
    A[1, 0] = cbf_term / m
    A[1, 1] = 0.0
    b[1] = (v0 - v) + alpha * B

    # 3. Maximum Wheel Traction Force: F_w / m <= Cag
    A[2, 0] = 1.0 / m
    b[2] = Cag

    # 4. Maximum Braking Force: -F_w / m <= Cdg
    A[3, 0] = -1.0 / m
    b[3] = Cdg

    # 5. Slack Variable Non-Negativity: -delta <= 0
    A[4, 1] = -1.0
    b[4] = 0.0

    return A, b, P, q