"""
Characteristic functions of the randomised switching processes
@author: Felix L. Wolf
"""

import numpy as np
import scipy.integrate as si
import scipy.stats as st
import itertools
from methods.quadrature_calc import \
    CollocationNormal, CollocationUniform, CollocationExponential, \
    CollocationTruncatedUniform, CollocationTruncatedExponential


# characteristic function of the component process (randomised ABM)
# Y^ϑ_j(t) = 0 + \int_0^t b_j(s, ϑ)ds + \int_0^t \sigma_j(s, ϑ)dW_j(t)
# assumes that ϑ is a normal random variable
def ChF_Component(u, t, RndDistr, DriftFunc, VolFunc,
                  N=5, integration_steps=100):
    # u- up to a vector of values for evaluation
    u = np.atleast_1d(u)
    i = 1j  # imaginary unit
    quad_dict = CollocationNormal(RndDistr, N)
    quadpoints = quad_dict["points"]  # θ_1, ..., θ_N
    quadweights = quad_dict["weights"].squeeze()  # w_1, ..., w_N
    # for each realisation θ, the conditional process is normally distributed
    # Y^θ_j(t) ~ N(mean, var), get parameters
    time = np.linspace(0, t, int(np.ceil(t)*integration_steps))  # [0,t]
    b_mat = DriftFunc(time, quadpoints).squeeze()  # squeeze to remove the
    sigma_mat = VolFunc(time, quadpoints).squeeze()  # "NoOfPaths" dimension
    # (implementation is written for MC sim where quadpoints will be a matrix)
    mean = si.trapezoid(b_mat, time)  # int_0^t b_j(s) ds
    var = si.trapezoid(sigma_mat**2, time)  # int_0^t sigj**2(s) ds

    ChF_output = np.zeros_like(u)
    for n in range(N):
        cf = np.exp(i * mean[n] * u) * np.exp(-0.5 * var[n] * u**2)
        ChF_output = ChF_output + quadweights[n] * cf
    return ChF_output


def ChF_Composite(u, t, SwitchingTimes, RndDistrList, DriftFuncList,
                  VolFuncList, x0=0., N=5, integration_steps=100):
    # u- can be a (real-valued) vector
    u = np.atleast_1d(u)
    i = 1j  # imaginary unit
    # Crop components to the involved ones up to time t
    SwitchingTimes = SwitchingTimes[SwitchingTimes <= t]
    K = len(SwitchingTimes)
    RndDistrList = RndDistrList[:K]
    DriftFuncList = DriftFuncList[:K]
    VolFuncList = VolFuncList[:K]
    #############################################
    # compute internal component process times s_j(t)
    sojourn_times = np.diff(np.append(SwitchingTimes, t))
    #############################################
    ChF_output = np.ones_like(u) * np.exp(1j * u * x0)
    for j in range(K):
        sj = sojourn_times[j]
        distj = RndDistrList[j]
        bj = DriftFuncList[j]
        sigj = VolFuncList[j]
        cf = ChF_Component(u, sj, distj, bj, sigj, N, integration_steps)
        ChF_output = ChF_output * cf
    return ChF_output


################################################################################
# Characteristic Function with stochastic switching, M (det.) switches
# this is for uniform switching and independent uniform sojourn times
# i.e. the sojourn times take care of no overlaps.
def ChF_StochSwitchUnifIndep(u, t, SojournDistrList, RndDistrList,
                             DriftFuncList, VolFuncList,
                             x0=0., N=5, L=5, integration_steps=100):
    u = np.atleast_1d(u)
    M = len(SojournDistrList) + 1
    if M != len(RndDistrList):
        print("Mismatch: number of switches and number of supplied components")
    # Produce collection of vectors SwitchingTimes, each of which demarks
    # a det. switching composite randomised process
    sojourn_pnts = np.zeros([M-1, N])
    sojourn_wgts = np.zeros_like(sojourn_pnts)
    for j, zeta_j in enumerate(SojournDistrList):
        if zeta_j.dist.name == "uniform":
            a, b = zeta_j.support()
            switchquad_dict = CollocationUniform(a, b, L, outtype="dict")
        elif zeta_j.dist.name == "expon":
            arrival_scale = zeta_j.kwds["scale"]
            switchquad_dict = CollocationExponential(arrival_scale, L)
        else:
            print("Unknown sojourn time distribution")
            return
        sojourn_pnts[j] = switchquad_dict["points"]
        sojourn_wgts[j] = switchquad_dict["weights"]
    # Ensure that at time t there has always been the full amount of switches
    tau_quadmax = np.sum(sojourn_pnts[:, -1])  # latest quadrature switching
    if tau_quadmax >= t:
        print(f"Time t:{t} <= tau_M: {tau_quadmax}, not all switches reached.")

    # Create all quadrature point combinations by "flattening the multiindex"
    sojourn_pnts_flat = np.array(list(itertools.product(*sojourn_pnts)))
    # compute the switching times [tau_0:=0, tau_1, ..., tau_M]
    SwitchingTimes = np.cumsum(sojourn_pnts_flat, axis=1)
    SwitchingTimes = np.insert(SwitchingTimes, 0, 0, axis=1)

    # quadrature weights v_{l_j} and their product V_|l|
    v_mat = np.array(list(itertools.product(*sojourn_wgts)))  # all combinations
    v_prod = np.prod(v_mat, axis=1)

    # sum up the sojourn quadratures
    ChF_output = np.zeros_like(u)  # no x0 scaling! that sits in the det. chf.
    for l, SwitchingTime in enumerate(SwitchingTimes):
        cf = ChF_Composite(u, t, SwitchingTime, RndDistrList, DriftFuncList,
                           VolFuncList, x0, N, integration_steps)
        ChF_output = ChF_output + v_prod[l] * cf
    return ChF_output


# Stochastic Switching with exponential
# sojourn times and truncation to encode sum of sojourn times less than t.
def ChF_StochSwitch(u, t, M, SojournDistrList, RndDistrList, DriftFuncList,
                    VolFuncList, x0=0., N=5, L=5, integration_steps=100):
    u = np.atleast_1d(u)
    if M == 0:
        print("ChF_StochSwitch with M switches needs at least M=1 switch!")
    SojournDistrList = SojournDistrList[:M]  # M entries
    RndDistrList = RndDistrList[:M+1]  # from M switches yield M+1 components
    DriftFuncList = DriftFuncList[:M+1]
    VolFuncList = VolFuncList[:M+1]

    # compute the nested sojourn points and weights
    if SojournDistrList[0].dist.name == "expon":
        sojourn_pnts, sojourn_wgts = ExponSojournNesting(t, M, L, SojournDistrList)
    elif SojournDistrList[0].dist.name == "uniform":
        sojourn_pnts, sojourn_wgts = UnifSojournNesting(M, L, SojournDistrList)
    else:
        print("Unknown sojourn time distribution")
        return
    # Produce collection of vectors SwitchingTimes, each of which demarks
    # a det. switching composite randomised process
    SwitchingTimes = np.zeros([L ** M, M + 1])  # DetSwitch expects tau_0=0 col
    if M == 1:
        V_prod = np.prod(sojourn_wgts.reshape([-1, 1]), axis=1)
        SwitchingTimes[:, 1] = np.cumsum(sojourn_pnts)
    else:
        V_prod = np.prod(sojourn_wgts, axis=1)
        SwitchingTimes[:, 1:] = np.cumsum(sojourn_pnts, axis=1)
    # quadrature weights v_{l_j} are needed for their product V_|l|
    # sum up the sojourn quadratures
    ChF_output = np.zeros_like(u)  # no x0 scaling! that sits in the det. chf.
    for l, SwitchingTime in enumerate(SwitchingTimes):
        cf = ChF_Composite(u, t, SwitchingTime, RndDistrList, DriftFuncList,
                           VolFuncList, x0, N, integration_steps)
        ChF_output = ChF_output + V_prod[l] * cf
    return ChF_output


# function to create the nested sojourn points for uniform sojourn times
def UnifSojournNesting(M, L, SojournDistrList):
    # M- number of sojourn times, L- number of quadrature nodes per sojourn time
    # create L**M "paths" corresponding to all point nestings
    # z_{l_0}, ..., z ^ {l_0, ..., l_{M - 1}}_{l_M}) with dynamic-depth loop

    # depth 0 is just the classic sojourn points of first sojourn time
    zeta_1 = SojournDistrList[0]
    a, b = zeta_1.support()
    switchquad_dict = CollocationUniform(a, b, L, outtype="dict")
    points = switchquad_dict["points"]
    weights = switchquad_dict["weights"]
    for j in range(1, M):  # depth j has L^(j+1) realisations of j+1 points each
        points_old = points
        weights_old = weights
        points = np.zeros([L**(j+1), j+1])
        weights = np.zeros_like(points)
        counter = 0
        zeta_j = SojournDistrList[j]  # in article this is zeta_{j+1}
        a, b = zeta_j.support()
        for k in range(L**j):  # iterate over previous realisations
            trunc_pt = np.sum(points_old[k])  # truncation with point sum
            switchquad_dict = CollocationTruncatedUniform(a, b, trunc_pt, L)
            quad_pts = switchquad_dict["points"]  # size (L,)
            quad_wgts = switchquad_dict["weights"]
            points_old_tiled = np.tile(points_old[k], (L, 1))  # repeat previous
            weights_old_tiled = np.tile(weights_old[k], (L, 1))
            points_new = np.concatenate((points_old_tiled,
                                         quad_pts.reshape(-1, 1)), axis=1)
            weights_new = np.concatenate((weights_old_tiled,
                                          quad_wgts.reshape(-1, 1)), axis=1)
            # *_new are arrays of shape (L, j+1)
            # write them into the updated storage of shape (L**(j+1), j+1)
            points[counter:counter+L] = points_new
            weights[counter:counter+L] = weights_new
            counter += L
    return points, weights


# sojourn nesting for exponential sojourn times
def ExponSojournNesting(T, M, L, SojournDistrList):
    # M- number of sojourn times, L- number of quadrature nodes per sojourn time
    # create L**M "paths" corresponding to all point nestings
    # z_{l_0}, ..., z ^ {l_0, ..., l_{M - 1}}_{l_M}) with dynamic-depth loop

    # depth 0 is just the classic sojourn points of first sojourn time
    zeta_1 = SojournDistrList[0]
    expon_scale = zeta_1.kwds["scale"]  # 1 / (arrivals / time)
    switchquad_dict = CollocationTruncatedExponential(expon_scale, T, L)
    points = switchquad_dict["points"]
    weights = switchquad_dict["weights"]
    for j in range(1, M):  # depth j has L^(j+1) realisations of j+1 points each
        points_old = points
        weights_old = weights
        points = np.zeros([L**(j+1), j+1])
        weights = np.zeros_like(points)
        counter = 0
        zeta_j = SojournDistrList[j]  # in article this is zeta_{j+1}
        expon_scale = zeta_j.kwds["scale"]
        for k in range(L**j):  # iterate over previous realisations
            trunc_pt = T - np.sum(points_old[k])  # truncation with point sum
            switchquad_dict = CollocationTruncatedExponential(expon_scale,
                                                              trunc_pt, L)
            quad_pts = switchquad_dict["points"]  # size (L,)
            quad_wgts = switchquad_dict["weights"]
            points_old_tiled = np.tile(points_old[k], (L, 1))  # repeat previous
            weights_old_tiled = np.tile(weights_old[k], (L, 1))
            points_new = np.concatenate((points_old_tiled,
                                         quad_pts.reshape(-1, 1)), axis=1)
            weights_new = np.concatenate((weights_old_tiled,
                                          quad_wgts.reshape(-1, 1)), axis=1)
            # *_new are arrays of shape (L, j+1)
            # write them into the updated storage of shape (L**(j+1), j+1)
            points[counter:counter+L] = points_new
            weights[counter:counter+L] = weights_new
            counter += L
    return points, weights


################################################################################
# Charac. Function with stochastic switching, stochastic number of switches
# assume iid exponential sojourn times
def ChF_FullStoch(u, t, expon_scale, RndDistrList, DriftFuncList,
                  VolFuncList, x0=0., MaxNumSwitches=9, N=5, L=5,
                  integration_steps=100):
    u = np.atleast_1d(u)
    # Step 1: Probability of 0, ..., MaxNumSwitches Switches:
    M_weights = np.zeros(MaxNumSwitches + 1)
    for m in range(MaxNumSwitches + 1):
        M_weights[m] = ProbabMExponSwitches(expon_scale, m, t)
    print(f"Stochastic switches truncation to {MaxNumSwitches} switches loses "
          f"{100*(1 - np.sum(M_weights)):.3f}% of the probability mass.")
    # rescale M_weights so that sum(M_weights) = 1, else cf(0) != 1
    M_weights = M_weights/M_weights.sum()

    ChF_output = np.zeros_like(u)
    # at m=0, no switch occurs, compute directly ChF of 1st component
    cf = ChF_Component(u, t, RndDistrList[0], DriftFuncList[0],
                       VolFuncList[0], N, integration_steps)
    ChF_output = ChF_output + M_weights[0] * cf
    zeta = st.expon(scale=expon_scale)  # iid sojourn distribution
    for m in range(1, MaxNumSwitches + 1):
        # create SojournDistrList
        SojournDistrList = [zeta] * m
        # for m switches we need m+1 distributions (fencepost!)
        cf = ChF_StochSwitch(u, t, m, SojournDistrList, RndDistrList[:m+1],
                             DriftFuncList[:m + 1], VolFuncList[:m+1],
                             x0, N, L, integration_steps)
        ChF_output = ChF_output + M_weights[m] * cf
    return ChF_output


# Compute the probability of m switches at t based on the sojourn times:
# P[M(t) > m] = P[\sum_{j=1}^{m+1} zeta_j <= t] =: P[Z <= t]
# for Z ~ Erlang(m+1, lambd) and \zeta_j ~ Exp(lambd) i.i.d.
def ProbabMExponSwitches(expon_scale, m, t):
    if m == 0:  # P[M(t)=0] = 1-P[M(t) > 0] = 1-P[zeta_1 <= t] (= P[zeta_1 > t])
        return 1 - st.expon(scale=expon_scale).cdf(t)
    # m >= 1:
    Z1 = st.erlang(a=m+1, scale=expon_scale)  # {Z1 <= t} = {M(t) > m}
    Z2 = st.erlang(a=m, scale=expon_scale)  # {Z2 <= t} = {M(t) > m-1}
    PE1 = Z1.cdf(t)  # probability of {M(t) > m}
    PE2 = Z2.cdf(t)  # probability of {M(t) > m-1}
    # P[M(t)=m] = 1-P(E1) - (1 - P(E2)) = P(E2) - P(E1)
    return PE2 - PE1


# Calibrate the number of stochastic switches to truncate at probability p
def M_truncate(p_threshold, expon_scale, t):
    M = 0
    p = 1 - ProbabMExponSwitches(expon_scale, 0, t)
    while p > p_threshold:
        M += 1
        p = p - ProbabMExponSwitches(expon_scale, M, t)
        if M > 100:
            break
    return M, p


# Characteristic function of Arithmetic Brownian Motion Local Volatility model
# i.e. time-dependent vola Black Scholes
# dX(t) = mu(t) dt + sigma(t) dW(t)
# with mu(t) = \sum_j \mu_j 1_{t\in[tau_j, tau_{j+1})}
# and sigma(t) = \sum_j \sigma_j 1_{t\in[tau_j, tau_{j+1})}
def ChF_ABM(u, t, SwitchingTimes, drift_vec, vol_vec,
            x0=0., integration_steps=100):
    u = np.atleast_1d(u)
    i = 1j
    # Truncation up to time t:
    SwitchingTimes = SwitchingTimes[SwitchingTimes<=t]
    K = len(SwitchingTimes)
    drift_vec = drift_vec[:K]
    vol_vec = vol_vec[:K]
    # Compute times between switches
    sojourn_times = np.diff(np.append(SwitchingTimes, t))
    # N(\int_0^t mu_s ds, \int_0^t sig^2_s ds) with piecew. const. integrands
    mean = np.sum(sojourn_times * drift_vec) + x0
    var = np.sum(sojourn_times * vol_vec**2)
    cf = np.exp(i * mean * u) * np.exp(-.5 * var * u**2)
    return cf
