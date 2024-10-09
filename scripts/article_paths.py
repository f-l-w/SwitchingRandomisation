#%%
"""
Simulate trajectories of randomised processes and of the corresponding
local volatility model.
@author: Felix L. Wolf
"""
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from methods.MonteCarlo_methods import GenerateComponentPaths, \
    GenerateCompositePaths_det, GenerateLocalVol_Det
from methods.randomiser_util import make_randomiser, volfunc_rnd
from methods.char_funcs import ExponSojournNesting  # for quad. pnts sampling
################################################################################
# The randomised process with deterministic switching
################################################################################
# Simulation of a composite randomised process with det. switching,
# alternating between two regimes characterised by randomisers Theta_0, Theta_1,
# as well as the corresponding local volatility model
#
# For randomised, det. switching process see Definition 2.2.
# For local volatility model see Theorem 3.1.
# See equation (1.1) and Section 6 for details of the parametrisation and model.

# ## MC simulation parameters:
N_paths = 100  # Number of Brownian Motion paths
N_realisations = 100  # Number of randomiser realisations
N_steps = 100  # Number of time steps

# ## Process parameters
r = 0.05  # constant, deterministic interest rate
T = 2  # maturity
tau_list = np.arange(0, T, 0.5)  # list of switching times
T_component = np.diff(np.append(tau_list, T))  # sub-maturities per regime

# parametrise the normally distributed randomsiers Theta_0, Theta_1
mu0_det, sig0_det = 0.05, .1  # Theta_0 ~ N(mu0_det, sig0_det^2)
mu1_det, sig1_det = 0.8, .5  # Theta_1 ~ N(mu1_det, sig1_det^2)
# create sequence of random variables associated with the alternating regimes
randomisers_det = make_randomiser(mu0_det, mu1_det, sig0_det, sig1_det) * 10

# Create volatility and drift coefficient functions (see eq (1.1))
vol_func = lambda t, p: np.abs(volfunc_rnd(t, p))
# volfunc(t, p) = [p, p, ..., p] id.
# absolute value to ensure positive volatility
vols_list = [vol_func, vol_func] * 10

drift_funcGBM = lambda t, p: r - 0.5 * volfunc_rnd(t, p)**2
drifts_list = [drift_funcGBM, drift_funcGBM] * 10

# ## Begin simulation
# Seed:
reset = 1234573 #  1234567  for all but print-fig1 1234567
np.random.seed(reset)

# Simulation of component processes (see Definition 2.1)
components = []
for j, t in enumerate(T_component):
    S = j % 2  # state of the economy (0 = low, 1 = high)
    Yj = GenerateComponentPaths(N_paths, N_steps,
                                T_component[j], randomisers_det[S],
                                drifts_list[S], vols_list[S],
                                NoOfRealisations=N_realisations)
    components.append(Yj)  # {"time", "Yj", "Wj", "realisations"}
    if j == 0:
        # Save randomiser realisations of first BM path for graph generation
        theta0 = Yj['realisations']

# Creation of (deterministic) composite process (see Definition 2.2)
X_det = GenerateCompositePaths_det(components, tau_list, T)
Xtime = X_det["time"]  # Time discretization of the randomised composite process
Xpaths = X_det["X"]  # Paths of the randomised composite process

# Create local volatility realisations with the same underlying Brownian motions
# for the deterministic switching case, see Theorem 3.1
Xlv_det = GenerateLocalVol_Det(N_paths, N_steps, T, tau_list,
                               randomisers_det, drifts_list, vols_list,
                               N=5, x_init=0., Xdict=X_det)
Xlvtime = Xlv_det['time']  # Time discretization of the loc.vol. process
Xlvpaths = Xlv_det['Xlv']  # Paths of the loc.vol. process


#####################################
# Plot deterministic switching:

# Helper function to update and label the switching times in axis of graph
def update_ticks(x, pos):
    eps_ball = np.min(np.abs(x-tau_list))
    if eps_ball < 0.001:
        mylist = tau_list.tolist()
        i = mylist.index(x)
        return fr'$\tau_{i}$'
        i = [0., .5, 1., 1.5].index(x)
        return fr'$\tau_{i}$'

    else:
        return x


m_rnd = 15  # how many different randomiser realisations to show (fig1)
m_bm = 20  # how many different BM trajectories to show (fig2)

# color space for the plots
colors = plt.cm.twilight_shifted(np.linspace(0, 1, m_rnd))
# latex formatting in plot titles and legends
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')


# Figure 1: Randomised, deterministic switching example.
# One Brownian motion path, many randomiser realisations

# Xpaths has shape (N_paths, N_realisations, N_steps)
bmpath = 2  # choose path of Brownian motion
# select randomiser realisations, bring them in order for gradient color scheme
thetamask = theta0[bmpath].argsort()
sorted_realisations = Xpaths[bmpath, thetamask]
# take only the first m_rnd many paths:
sorted_realisations = sorted_realisations[:m_rnd]

#####################################
# The setup allows for randomisers in the 'high volatility' environment
# to sample a small volatility coefficient, these are outliers.
# Fig1 appears in Introduction, to avoid any confusion we remove the outliers.

# Function that computes the total variation along axis 1 of 2dim array
TV = lambda mymat: np.sum(np.abs(np.diff(mymat).squeeze()), axis=1)
# Find outlier path in second regime:
regime2 = np.where(np.logical_and(Xtime >= 0.5, Xtime <= 1.))
paths_regime2 = sorted_realisations[:, regime2]
tv_regime2 = TV(paths_regime2)
idx_regime2 = np.where(tv_regime2 == tv_regime2.min())[0][0]  # 82
# Find outlier path in fourth regime
regime4 = np.where(np.logical_and(Xtime >= 1.5, Xtime <= 2.))
paths_regime4 = sorted_realisations[:, regime4]
tv_regime4 = TV(paths_regime4)
idx_regime4 = np.where(tv_regime4 == tv_regime4.min())[0][0]  # 63
# Sanitize results for plotting by removing outliers:
sorted_realisations = np.delete(sorted_realisations,
                                (idx_regime2, idx_regime4), 0)
#####################################

# Plot Figure1: trajectories of deterministic switching randomised process
fig1, ax1 = plt.subplots()
for k, path in enumerate(sorted_realisations):
    if k == 0:
        ax1.plot(Xtime, path, alpha=1, color=colors[k])
                 # label=fr'$X(\bar\omega_{l+1}, \omega^*_k)$,' +
                 #      '$1\leq k\leq$' + f'${m_rnd}$')
    elif 0 < k < m_rnd:
        ax1.plot(Xtime, path, alpha=1, color=colors[k])
for tau in tau_list[1:]:  # vertical lines at switching times
    ax1.axvline(x=tau, color="grey", linewidth=.6, linestyle='dashed')

ax1.xaxis.set_major_formatter(mticker.FuncFormatter(update_ticks))
# plt.legend()
ax1.set_title(r'Trajectories of $X^{\boldsymbol{\vartheta}}(t)$ with deterministic switching')
ax1.set_xlabel('Time')
plt.show()

# Plot Figure 2a: Det. Switching vs Local Volatility Model
fig2, ax2 = plt.subplots()
myalpha = 1 / (m_bm * m_rnd) * 100  # transparency to avoid visual clutter
for l, rnd_realisations in enumerate(Xpaths):
    if l < m_bm:  # this limits the number of different Xdict paths
        # plot paths of X_det
        theta0l = theta0[l]  # the realisations of the first component
        thetamask = theta0l.argsort()
        sorted_realisations = rnd_realisations[thetamask]
        for k, path in enumerate(sorted_realisations):
            if k == 0:
                ax2.plot(Xtime, path, alpha=myalpha, color=colors[k])
            if (k == 1) and (l == 0):
                ax2.plot(Xtime, path, alpha=myalpha, color=colors[k],
                         label='Randomised model')
            elif 0 < k < m_rnd:
                ax2.plot(Xtime, path, alpha=myalpha, color=colors[k])
        # plot paths of Xlv_det
        if l == 0:
            ax2.plot(Xlvtime, Xlvpaths[l], color='lawngreen', zorder=3,
                     label='Local volatility model')
        else:
            ax2.plot(Xlvtime, Xlvpaths[l], color='lawngreen', zorder=3)

for tau in tau_list[1:]:  # vertical lines at switching times
    ax2.axvline(x=tau, color="grey", linewidth=.6, linestyle='dashed')

ax2.xaxis.set_major_formatter(mticker.FuncFormatter(update_ticks))
ax2.set_xlabel('Time')
ax2.set_title(r'Comparison with the local volatility model $\bar X(t)$')
plt.legend()
plt.show()

################################################################################
# The randomised process with stochastic switching
################################################################################
# Simulation of a composite randomised process with stochastic switching,
# alternating between two regimes characterised by randomisers Theta_0, Theta_1,
# switching at stochastic (exponentially distributed) times.
#
# For randomised, stoch. switching process see Definition 4.1.
# See Section 6 for details of the parametrisation and model.

# Create the randomiser random variables (same parametrisation as before)
randomisers_stoch = make_randomiser(mu0_det, mu1_det, sig0_det, sig1_det) * 10
T2 = 1.5  # maturity
N_sojourn_realisations = 4  # number of samples for randomised switching time
N_switches = 2  # number of switches

# Choose how the sojourn times (i.e. stochastic switching times) are selected.
# Options: expon : random samples from exponential distribution Exp(2)
#          uniform : random samples from uniform distribution Unif(.25, .75)
#          quad : quadrature nodes associated with Exp(2)

# To create Figure 3, choose 'expon'.

samples = 'expon'  # 'uniform' # 'quad'

if samples == 'quad':
    # compute quadrature nodes of EXP(2) switching:
    L = N_sojourn_realisations
    zeta = st.expon(scale=.5)
    soj_pts, soj_wgts = ExponSojournNesting(T, 2, L, [zeta] * 10)

elif samples == 'expon':  # sample paths randomly from Exp(2)
    np.random.seed(98765432)  # yields nice values for pretty visualisation
    soj_pts = st.expon.rvs(scale=.5, size=(N_sojourn_realisations, 2))
    # each row contains two sojourn times, reject if their sum is larger than T2
    for idx, row in enumerate(soj_pts):
        while np.sum(row) >= T2:
            soj_pts[idx] = st.expon.rvs(scale=.5, size=(2,))
    soj_wgts = np.ones_like(soj_pts) * 1 / N_sojourn_realisations

elif samples == 'uniform':
    soj_pts = st.uniform.rvs(loc=.25, scale=.5, size=(N_sojourn_realisations, 2))
    # each row contains two sojourn times, reject if their sum is larger than T2
    for idx, row in enumerate(soj_pts):
        while np.sum(row) >= T2:
            soj_pts[idx] = st.uniform.rvs(loc=.25, scale=.5, size=(2,))
    soj_wgts = np.ones_like(soj_pts) * 1 / N_sojourn_realisations


# Container for all X switching time realisations:
X_expon = []
# compute number of steps dynamically (more steps for longer times)
# this keeps stepsize uniform for regimes of vastly different durations
stepsize = 0.001

# Simulation of component process up to time given by sojourn time realisations
for soj_list in soj_pts:
    soj_sum = np.sum(soj_list)
    T_stoch = np.append(soj_list, T2 - soj_sum)
    comps_stoch = []
    np.random.seed(reset)
    for j, t in enumerate(T_stoch):
        S = j % 2  # state of the economy (0 = low, 1 = high)
        T_j = T_stoch[j]
        steps_j = int(np.ceil(T_j / stepsize))
        Yj = GenerateComponentPaths(1, steps_j,
                                    T_j, randomisers_stoch[S],
                                    drifts_list[S], vols_list[S],
                                    NoOfRealisations=N_realisations)
        comps_stoch.append(Yj)  # {"time", "Yj", "Wj", "realisations"}

    # Create composite paths (Definition 4.1)
    # concatenation is defined with explicit switching times starting in 0
    tau_stoch = np.append(0, np.cumsum(soj_list))
    X_real = GenerateCompositePaths_det(comps_stoch, tau_stoch, T)
    X_expon.append(X_real)

#####################################
# Plot stochastic switching example (Figure 2b):

colors2 = plt.cm.Dark2(np.linspace(0, 1, N_sojourn_realisations))
rnd_idx = 0
bm_idx = 0
fig3, ax3 = plt.subplots()
for k, X_dict in enumerate(X_expon):
    # X_expon contains the stoch.soj.time realisations
    # Xpaths = X_dict["X"]  # each element of Xpaths corresponds to one Xdict path
    Xpath = X_dict['X'][bm_idx]  # always take the first Xdict path
    Xtime = X_dict["time"]
    ax3.plot(Xtime, Xpath[rnd_idx], color=colors2[k])

# Insert vertical lines according to the stochastic switching times
ylo, yhi = ax3.get_ylim()
for k, X_dict in enumerate(X_expon):
    Xpath = X_dict['X'][bm_idx]
    Xtime = X_dict["time"]
    tau_reals = np.cumsum(soj_pts[k])
    tau_idcs = [np.where(Xtime <= tau)[0][-1] for tau in tau_reals]
    switchvals = Xpath[rnd_idx, tau_idcs]
    # ax3.vlines(x=tau_reals, ymin=ylo, ymax=switchvals, color=colors2[k],
    ax3.vlines(x=tau_reals, ymin=ylo, ymax=yhi, color=colors2[k],
              linestyle='dotted', linewidth=2)

# Dummy plots to insert minimalist legend labels
ax3.plot(0, 0, zorder=0, color='black', label=r'$X^{\boldsymbol{\vartheta}}(t; \bar\omega, \omega^*_1, \omega^*_2)$')
ax3.plot(0, 0, zorder=0, color='black', linestyle='dotted', linewidth=2,
         label=r'$\boldsymbol{\tau}(\omega^*_2)$')
ax3.set_xlabel('Time')
ax3.set_title(r'Trajectories of $X^{\boldsymbol{\vartheta}}(t)$ with stochastic switching')
plt.legend()
plt.show()


fig1.savefig(f'../figures/Figure1.png', bbox_inches='tight')
fig2.savefig(f'../figures/Figure2a.png', bbox_inches='tight')
fig3.savefig(f'../figures/Figure2b.png', bbox_inches='tight')

# fig1.savefig(f'article/Figure1.pdf', bbox_inches='tight')
# fig2.savefig(f'article/Figure2a.pdf', bbox_inches='tight')
# fig3.savefig(f'article/Figure2b.pdf', bbox_inches='tight')
