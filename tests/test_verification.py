"""
Automated unit and integration test suite for autonomous vehicle controllers.
Verifies CLF-CBF-QP Adaptive Cruise Control, Jacobian accuracy, TV-LQR, and CMPC.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

from controllers.acc_controller import ACC_Controller
from controllers.cmpc_controller import calc_Jacobian, LQR_Controller, CMPC_Controller
from simulation.acc_utils import sim_vehicle
from simulation.cmpc_utils import BicycleSim


def test_acc_safety_and_tracking():
    """Verify ACC satisfies safety barrier invariance B(t) >= 0 and no collision."""
    param_acc = {
        "vd": 15,
        "m": 2000,
        "Cag": 0.3 * 9.81,
        "Cdg": 0.8 * 9.81,
        "v01": 5,
        "v02": 20,
        "switch_time": 30,
        "terminal_time": 50,
    }
    y0 = np.array([300, 20])
    t, B, y, u = sim_vehicle(ACC_Controller, param_acc, y0)

    min_B = np.min(B)
    min_D = np.min(y[:, 0])
    u_min_norm = np.min(u) / param_acc["m"]
    u_max_norm = np.max(u) / param_acc["m"]

    assert min_B >= -1e-4, f"Safety barrier violated: min B = {min_B}"
    assert min_D > 0, f"Collision detected: min D = {min_D}"
    assert u_min_norm >= -param_acc["Cdg"] - 1e-4, "Braking limit violated"
    assert u_max_norm <= param_acc["Cag"] + 1e-4, "Traction limit violated"


def test_bicycle_jacobian_accuracy():
    """Verify analytical Jacobians against finite difference numerical differentiation."""
    param_cmpc = {
        "h": 0.05,
        "T": 15.0,
        "L_f": 1.0,
        "L_r": 1.0,
        "a_lim": [-10.0, 4.0],
        "delta_lim": [-0.8, 0.8],
    }
    Sim = BicycleSim(param_cmpc)
    x_test = np.array([1.5, 2.5, 0.4, 3.2])
    u_test = np.array([2.0, 0.3])
    A_ana, B_ana = calc_Jacobian(x_test, u_test, param_cmpc)

    eps = 1e-6
    A_num = np.zeros((4, 4))
    B_num = np.zeros((4, 2))
    f0 = Sim.Fun_dynamics_dt(x_test, u_test)

    for i in range(4):
        xp = x_test.copy()
        xp[i] += eps
        A_num[:, i] = (Sim.Fun_dynamics_dt(xp, u_test) - f0) / eps
    for j in range(2):
        up = u_test.copy()
        up[j] += eps
        B_num[:, j] = (Sim.Fun_dynamics_dt(x_test, up) - f0) / eps

    err_A = np.max(np.abs(A_ana - A_num))
    err_B = np.max(np.abs(B_ana - B_num))

    assert err_A < 1e-5, f"Jacobian A error too high: {err_A}"
    assert err_B < 1e-5, f"Jacobian B error too high: {err_B}"


def test_lqr_and_cmpc_tracking():
    """Verify LQR and CMPC trajectory tracking performance and box constraint adherence."""
    param_cmpc = {
        "h": 0.05,
        "T": 15.0,
        "L_f": 1.0,
        "L_r": 1.0,
        "a_lim": [-10.0, 4.0],
        "delta_lim": [-0.8, 0.8],
    }
    Sim = BicycleSim(param_cmpc)
    u_bar, x_bar = Sim.GenRef(5, 5)
    preview = 20
    x0 = np.array([2, 2, 3.14 * 0.3, 2])

    # LQR
    ctrl_lqr = lambda x, u, xint: LQR_Controller(x, u, xint, param_cmpc)
    x_log_l, u_log_l = Sim.SimVehicle(x_bar, u_bar, preview, x0, ctrl_lqr)
    err_lqr = np.linalg.norm((x_log_l - x_bar)[-100:], "fro")
    assert err_lqr < 0.05, f"LQR error too high: {err_lqr}"

    # CMPC
    ctrl_cmpc = lambda x, u, xint: CMPC_Controller(x, u, xint, param_cmpc)
    x_log_c, u_log_c = Sim.SimVehicle(x_bar, u_bar, preview, x0, ctrl_cmpc)
    err_cmpc = np.linalg.norm((x_log_c - x_bar)[-100:], "fro")
    assert err_cmpc < 0.05, f"CMPC error too high: {err_cmpc}"

    # Bound checks
    a_min = np.min(u_log_c[:, 0])
    a_max = np.max(u_log_c[:, 0])
    d_min = np.min(u_log_c[:, 1])
    d_max = np.max(u_log_c[:, 1])

    assert a_min >= param_cmpc["a_lim"][0] - 1e-4, f"Accel lower bound violated: {a_min}"
    assert a_max <= param_cmpc["a_lim"][1] + 1e-4, f"Accel upper bound violated: {a_max}"
    assert d_min >= param_cmpc["delta_lim"][0] - 1e-4, f"Steering lower bound violated: {d_min}"
    assert d_max <= param_cmpc["delta_lim"][1] + 1e-4, f"Steering upper bound violated: {d_max}"


if __name__ == "__main__":
    print("Running test_acc_safety_and_tracking()...")
    test_acc_safety_and_tracking()
    print("PASS!")

    print("Running test_bicycle_jacobian_accuracy()...")
    test_bicycle_jacobian_accuracy()
    print("PASS!")

    print("Running test_lqr_and_cmpc_tracking()...")
    test_lqr_and_cmpc_tracking()
    print("PASS!")

    print("\nAll integration tests PASSED!")
