import numpy as np


def dist2R(x_i, xr, yr, pi_pos, pi_neg):
   
    m,n = xr.shape
    for i in range(m):
        dxr=np.linalg.norm(xr-x_i,axis=1)
        if yr[i] == 0:
            dxr = pow(pi_pos,(1/n))*dxr
        else:
            dxr = pow(pi_neg,(1/n))*dxr


