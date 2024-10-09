"""
- Drift and volatility function examples that take randomiser samples as an arg
- method to create rv_frozen instances to make normally distributed randomisers
@author: Felix L. Wolf
"""
import numpy as np
import scipy.stats as st


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Drift- and volatility coefficient functions that take samples from randomiser
# as input and bring those in the shape required by the MC simulation engine

def driftfunc_det(const, t, p):
    # returns array shape (p.dim0, p.dim1, len(T)) filled with const.
    return const * np.ones(np.shape(p) + np.shape(t))


def volfunc_rnd(t, p):
    # returns array shape (len(p), len(T)) where each column is p
    p_reshape = p[:, :, np.newaxis]  # shape becomes (p.dim0, p.dim1, 1)
    return np.broadcast_to(p_reshape, np.shape(p) + np.shape(t))
# note: this code does not handle p of shape (N,)

# create frozen_rv instances of normal randomisers
def make_randomiser(mean_low, mean_high, sd_low, sd_high):
    rv_low = st.norm(loc=mean_low, scale=sd_low)
    rv_high = st.norm(loc=mean_high, scale=sd_high)
    return [rv_low, rv_high]
