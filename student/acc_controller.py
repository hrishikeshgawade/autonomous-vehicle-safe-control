import numpy as np

def ACC_Controller(t, x, param):
    vd = param["vd"]
    v0 = param["v0"]
    m = param["m"]
    Cag = param["Cag"]
    Cdg = param["Cdg"]

    # cost function and constraints for the QP
    P = np.zeros((2,2))
    q = np.zeros([2, 1])
    A = np.zeros([5, 2])
    b = np.zeros([5])
    
    #############################################################################
    #                    TODO: Implement your code here                         #
    #############################################################################

    # set the parameters
    lam = 0.5
    alpha = 1.0
    w = 1e6

    # construct the cost function
    P = np.array([[2.0, 0.0], [0.0, 2.0 * w]])
    q = np.zeros([2, 1])
    
    # construct the constraints
    v = x[1]
    D = x[0]
    
    h = 0.5 * (v - vd)**2
    v_rel = max(0.0, v - v0)
    B = D - 1.8 * v - 0.5 * (v_rel**2) / Cdg
    cbf_term = 1.8 + v_rel / Cdg
    
    # 1. Tracking constraint (CLF with slack): (v - vd)/m * Fw - delta <= -lam * h
    A[0, 0] = (v - vd) / m
    A[0, 1] = -1.0
    b[0] = -lam * h
    
    # 2. Safety constraint (CBF): 1/m * cbf_term * Fw <= (v0 - v) + alpha * B
    A[1, 0] = cbf_term / m
    A[1, 1] = 0.0
    b[1] = (v0 - v) + alpha * B
    
    # 3. Wheel force upper bound: Fw / m <= Cag
    A[2, 0] = 1.0 / m
    b[2] = Cag
    
    # 4. Wheel force lower bound: -Fw / m <= Cdg
    A[3, 0] = -1.0 / m
    b[3] = Cdg
    
    # 5. Slack non-negativity: -delta <= 0
    A[4, 1] = -1.0
    b[4] = 0.0
    #############################################################################
    #                            END OF YOUR CODE                               #
    #############################################################################
    
    return A, b, P, q