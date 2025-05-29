import numpy as np
from vmdpy import VMD

def vmd_decompose(x: np.ndarray,
                  k,
                  alpha,
                  tau,
                  tol):
    # 强制转换，避免传入字符串
    k     = int(k)
    alpha = float(alpha)
    tau   = float(tau)
    tol   = float(tol)
    modes, _, _ = VMD(x, alpha, tau, k, False, 1, tol)
    return np.array(modes).T
