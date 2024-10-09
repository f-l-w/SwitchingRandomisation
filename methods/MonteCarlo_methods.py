"""
Functions used in the MC path simulation for the switching, randomised processes
@author: Felix L. Wolf
"""
import numpy as np
import scipy.stats as st
import scipy.integrate as si
from methods.quadrature_calc import CollocationNormal
import itertools


# Generate Paths of a component process
# (Monte Carlo Simulation of the process defined in Definition 2.1)
def GenerateComponentPaths(NoOfPaths, NoOfSteps, T,
                           RndDistr, DriftFunc, VolFunc,
                           NoOfRealisations=100):
    # T - maturity (float)
    # NoOfPaths - number of Brownian Motion paths (int)
    # NoOfSteps - number of steps between 0 and maturity T (int)
    # NoOfRealisations - number of randomiser samples (int)
    # RndDistr - rv_frozen object from scipy
    #            (distribution of randomiser (e.g. st.norm(loc=0,scale=1))
    # DriftFunc, VolFunc - drift and volatility coefficient functions
    #    that take arguments (t, theta) = (time, randomiser realisation)

    time = np.linspace(0, T, NoOfSteps + 1)
    dt = T / NoOfSteps

    # Normal Samples for Brownian increments
    Z = st.norm.rvs(loc=0., scale=1., size=[NoOfPaths, NoOfSteps])  # N(0,1)
    # making sure that samples from normal have mean 0 and variance 1
    if NoOfPaths > 1:
        Z = (Z - np.mean(Z, axis=0, keepdims=True)) / np.std(Z, axis=0,
                                                             keepdims=True)
    dW = np.sqrt(dt) * Z  # Brownian motion increments
    # align dW with output shape (NoOfPaths, NoOfRealisations, NoOfSteps)
    # such that each bundle of randomiser realisations shares same BM increments
    dW = np.repeat(dW, repeats=NoOfRealisations, axis=0).reshape(
        NoOfPaths, NoOfRealisations, NoOfSteps)

    # evaluate the drift and volatility coefficients
    realisations = RndDistr.rvs(size=[NoOfPaths, NoOfRealisations])
    b_mat = DriftFunc(time, realisations)
    sigma_mat = VolFunc(time, realisations)

    # Construct Brownian motion from its increments
    W = np.cumsum(dW[:, 0, :], axis=1)
    W = np.insert(W, 0, 0, axis=1)

    # Construct the component process
    dY = b_mat[:, :, :-1] * dt + sigma_mat[:, :, :-1] * dW
    # evaluating b, sigma at t_k not at t_{k+1}
    Y = np.cumsum(dY, axis=2)
    Y = np.insert(Y, 0, 0, axis=2)

    paths = {"time": time, "Yj": Y, "Wj": W, "realisations": realisations}
    return paths


# # #
# Generate paths of a composite process (deterministic switching)
# (Monte Carlo simulation of the process defined in Definition 2.2)
def GenerateCompositePaths_det(PathDictList, SwitchingTimes, T):
    # PathDictList - [dictionary of 1st component paths,
    #                 dictionary of 2nd component paths, etc.]
    #                 as created with GenerateComponentPaths
    # SwitchingTimes - [0, tau_1, tau_2, ..., tau_M]  (array of floats)
    # T - final maturity of paths (float)

    # M is number of switches INCLUDING INITIAL SWITCH AT tau_0 = 0
    M = len(SwitchingTimes)  # number of switches
    if len(PathDictList) < M:
        print("not enough component processes supplied")
        return None

    # Ensure that the component processes are compatible with another:
    ShapeCheckArray = np.zeros([M, 2])
    for j in range(M):
        ShapeCheckArray[j] = PathDictList[j]["Yj"].shape[
                             :2]  # paths + realisats.
    if not np.all(ShapeCheckArray == ShapeCheckArray[0]):
        print("Number of paths / realisations varies between components")
        return None

    # Construct composite process by concatenating components at switching times
    # Each switching time tau corresponds to a unique component starting at tau
    for j, tau_j in enumerate(SwitchingTimes):
        paths = PathDictList[j]  # paths on [tau_j, tau_{j+1})
        Ytime = paths["time"]  # time [0, tau_{j+1} - tau_j)

        # construct sojourn times, accounting for stop at final maturity
        try:
            tau_nxt = np.minimum(SwitchingTimes[j + 1], T)
        except IndexError:  # when tau_j is the last SwitchingTime
            tau_nxt = T
        sj = tau_nxt - tau_j  # sojourn time

        # warning when the component process paths don't cover enough time
        # to reach the next switching time:
        # if Ytime[-1] < sj:
        #     print(f"Path simulation Y_{j} is too short:")
        #     print(f'    Path time ends at: {Ytime[-1]}')
        #     print(f'    Sojourn time is: {sj}')

        # Component processes can be longer than needed,
        # crop their time domain to correspond with [tau_j, tau_{j+1}):
        timej = Ytime[Ytime <= sj]
        idx_sj = len(timej)
        Yj = paths["Yj"][:, :, :idx_sj]  # no +1 on index needed
        Wj = paths["Wj"][:, :idx_sj]

        if j == 0:  # On [0, tau_1) component Y_0 and composite X are the same:
            TimeComposite = timej
            X = Yj
            W = Wj
        else:  # Concatenate components in a continuous way,
            # shifting their time and paths (Yj, Wj) by the last preceding value
            timej_shifted = timej[1:] + TimeComposite[-1]
            TimeComposite = np.append(TimeComposite, timej_shifted)
            NoPaths, NoReals = Yj.shape[0], Yj.shape[1]
            Yj_shifted = Yj[:, :, 1:] + X[:, :, -1].reshape(NoPaths, NoReals, 1)
            X = np.append(X, Yj_shifted, axis=2)
            Wj_shifted = Wj[:, 1:] + W[:, -1].reshape(-1, 1)
            W = np.append(W, Wj_shifted, axis=1)

    CompositePaths = {"time": TimeComposite, "X": X, "W": W}
    return CompositePaths
################################################################################


# Generate paths of the local volatility process (deterministic switching)
# (Monte Carlo simulation of the process defined in Theorem 3.1)
def GenerateLocalVol_Det(NoOfPaths, NoOfSteps, T, SwitchingTimes,
                         RndDistrList, DriftFuncList, VolFuncList,
                         N=5, x_init=0., Xdict=None):
    # NoOfPaths, T, SwitchingTimes - same as above
    # NoOfSteps - number of steps WITHIN EACH COMPONENT (int)
    # RndDistrList - list of rv_frozen as above (the randomisers involved)
    # DriftFuncList, VolFuncList - list of funcs as above (the coefficients)
    # N - number of points in quadrature (same for every component) (int)
    # x_init - initial value (float)
    # Xdict - output of GenerateCompositePaths to use the same driving BM

    if T < SwitchingTimes[-1]:
        print("Ensure final maturity is after last switch!")
        return None

    M = len(SwitchingTimes)  # contains initial switching time tau_0=0
    # crop randomisers and coefficient functions according to the
    # actual number of switches supplied
    RndDistrList = RndDistrList[:M]
    DriftFuncList = DriftFuncList[:M]
    VolFuncList = VolFuncList[:M]
    # Attention: In the code, there are M components in total.
    #            In the article, there are M+1 components in total.

    # # # # # # # #
    # 1) Prepare the local drift and volatility coefficient functions
    # # # # # # # #

    # 1a) Quadrature components:
    # Compute the quadrature weights and points for all randomisers
    quad_dict = CollocationNormal(RndDistrList, N)
    quadpoints = quad_dict["points"]
    quadweights = quad_dict["weights"]

    # unravel the sum over multi indices (n0, ..., nM) into a single sum:
    # evaluate \sum_n0 ... \sum_nM from right to left (nM, nM-1, ..., n0)
    # quadrature points:
    theta_mat = np.array(list(itertools.product(*quadpoints)))
    # quadrature weights (and their product)
    w_mat = np.array(list(itertools.product(*quadweights)))  # all combinations
    w_prod = np.prod(w_mat, axis=1)

    # 1b) Segment the time discretisation [0, T] into shifted component times
    #     [0, tau_1], [0, tau_2-tau_1], ..., each with NoOfSteps steps !
    sojourn_times = np.diff(np.append(SwitchingTimes, T))
    times_components = np.linspace(0, sojourn_times, NoOfSteps + 1, axis=1)
    # total time discretisation for component process:
    dt = np.repeat(sojourn_times / NoOfSteps, NoOfSteps)
    StepsTotal = M * NoOfSteps
    time_full = np.insert(np.cumsum(dt), 0, 0)

    # 1c) Prepare samples of beta(T, theta), gamma(T, theta)
    #     and parametrization of the normally distributed conditional process
    beta_mat = np.empty([N ** M, 0])  # relates to drift
    gamma_mat = np.empty_like(beta_mat)  # relates to volatility
    E_mat = np.empty_like(beta_mat)  # \int_0^{t_k} beta(T,theta) dt
    Var_mat = np.empty_like(beta_mat)  # \int_0^{t_k} sig^2(T,theta) dt
    # constructed by concatenation, final output is of shape [N**M, StepsTotal]
    for j in range(M):
        # for each component isolate the drift/vol func b_j, sig_j,
        # time discretization of [0, tau_{j+1} - tau_j),
        # and the corresponding realisation from the flattened multi-index
        bj, sigj = DriftFuncList[j], VolFuncList[j]
        timej = times_components[j]
        realisationsj = theta_mat[:, j].reshape(1, N ** M)
        # functions expect realisations to be of matrix shape [n, m]
        # where for each path 1...n the m realisations are repeated
        bj_sample = bj(timej, realisationsj).squeeze()
        sigj_sample = sigj(timej, realisationsj).squeeze()
        # beta is defined on [0, tau_{j+1} - tau_j), remove final sample value
        beta_mat = np.concatenate((beta_mat, bj_sample[:, :-1]), axis=1)
        gamma_mat = np.concatenate((gamma_mat, sigj_sample[:, :-1]), axis=1)

        # integrate components of beta individually to avoid errors from
        # discontinuities between switching regimes
        bj_integral = si.cumulative_trapezoid(bj_sample, timej)
        sigj_sq_integral = si.cumulative_trapezoid(sigj_sample ** 2, timej)
        # shift beta integral by initial or preceding value of the integral
        if j == 0:
            bj_integral = bj_integral + x_init
        else:
            bj_integral = bj_integral + E_mat[:, -1].reshape(N ** M, 1)
            sigj_sq_integral = sigj_sq_integral + Var_mat[:, -1].reshape(N ** M,
                                                                         1)
        E_mat = np.concatenate((E_mat, bj_integral), axis=1)
        Var_mat = np.concatenate((Var_mat, sigj_sq_integral), axis=1)
    SD_mat = np.sqrt(Var_mat)

    # # # # # # # #
    # 2) Monte Carlo path simulation
    # # # # # # # #

    # Create Brownian motion (or import from arguments)
    if Xdict is None:  # no Brownian motion submitted
        dt = np.repeat(sojourn_times / NoOfSteps, NoOfSteps)
        Z = st.norm.rvs(loc=0., scale=1., size=[NoOfPaths, StepsTotal])
        # making sure that samples from normal have mean 0 and variance 1
        Z = (Z - np.mean(Z, axis=0, keepdims=True)) / np.std(Z, axis=0,
                                                             keepdims=True)
        dW = np.sqrt(dt) * Z
    else:  # supplied Xdict from Composite X MC experiment
        dt = np.diff(Xdict["time"])
        dW = np.diff(Xdict["W"])
        NoOfPaths = dW.shape[0]

    # create loc vol paths
    Xlv = np.zeros([NoOfPaths, StepsTotal + 1])
    Xlv[:, 0] = x_init  # initial value
    for i in range(0, StepsTotal):
        if i == 0:  # density is point-mass reduces Lambda_n to weights only
            Lambd_vec = np.repeat(w_prod[:, np.newaxis], NoOfPaths, axis=1)
        else:
            Lambd_vec = LocVolLambda(Xlv[:, i], w_prod,
                                     E_mat[:, i], SD_mat[:, i])
        # Lambd_vec is the vector of Lambda_|n|s
        barmu = LocVolMu(beta_mat[:, i], Lambd_vec)  # \bar\mu is Xlv drift
        barsig = LocVolSig(gamma_mat[:, i], Lambd_vec)  # \bar\sigma is Xlv vol
        Xlv[:, i + 1] = Xlv[:, i] + barmu * dt[i] + barsig * dW[:, i]

    paths = {"time": time_full, "Xlv": Xlv}
    return paths

################################################################################
# Analytical functions
################################################################################
# Implementations of the coefficient functions of the loc vol model.
# The fractional structure in (3.2), (3.3) leads to numerical instability,
# thus we use a robust reformulation.

# All functions below assume the same flattened multi-index,
# |n| = (n0, ..., nM) to transform \sum_n0 ... \sum_nM into a single sum
# by evaluating from right to left, first elements of sum_nM, then sum_nM-1, ...

# Compute function kappa from softmax formulation.
# For all flattened multi indices |n| it holds that:
# (\prod_{k\in|n|} w_k) f(x; X^{\theta_|n|}(T))
# = exp(log(\prod_k w_k) - 1/2 log(2 \pi SD^2]) - (x-E)^2 / (2 SD^2)
# =: e[kappa(x; |n|)]
def LocVolKappa(x, weightprod, E, SD):
    # assumes flattened multiindex |n| = 1, ..., N**M =: Nflat
    # weightprod - quadrature weight product of multiindex,
    # E, SD - expectations and standard devs of randomisers in multiindex
    # x is argument vector (typically of same length as number MC paths)
    Nflat = weightprod.size
    out = np.zeros([Nflat, x.size])
    for n in range(Nflat):
        # out[n] = np.log(weightprod[n] * st.norm.pdf(x, loc=E[n], scale=SD[n]))
        out[n] = np.log(weightprod[n]) - 1 / 2 * np.log(
            2 * np.pi * SD[n] ** 2) - (x - E[n]) ** 2 / (2 * SD[n] ** 2)
    return out


# Robust computation of softmax density Lambda
# exp(kappa_|n| + log(C)) / sum_|m| exp(kappa_|m| + log(C))
def LocVolLambda(x, weightprod, E, SD):
    # assumes flattened multiindex |n| = 1, ..., N**M =: Nflat
    # E, SD - expectations and standard devs of randomisers in multiindex
    # x is argument vector (typically of same length as number MC paths)
    Nflat = weightprod.size
    kappas = LocVolKappa(x, weightprod, E, SD)
    logC = - np.max(kappas, axis=0)
    denominator = np.sum(np.exp(kappas + logC), axis=0)
    Lambdas = np.exp(kappas + logC) / denominator
    return Lambdas


# Drift coefficient function (3.2)
def LocVolMu(beta_n, Lambda_n):
    # beta_n is (2.8) for multiindex n, Lambda_n is output of LocVolLambda
    return np.sum(beta_n[:, np.newaxis] * Lambda_n, axis=0)


# Volatility coefficient function, square-root of (3.3)
def LocVolSig(gamma_n, Lambda_n):
    # gamma_n is (2.8) for multiindex n, Lambda_n is output of LocVolLambda
    # input gamma_n contains sigma_j unsquared,
    # returns local volatility sigma unsquared
    return np.sqrt(np.sum(gamma_n[:, np.newaxis] ** 2 * Lambda_n, axis=0))
