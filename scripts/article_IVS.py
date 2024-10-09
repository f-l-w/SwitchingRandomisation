# %%
"""
Compute implied volatility surfaces and their graphs as used in the article
# Implied volatility surface 1: Strike / Expiry / IV
#  for no switching, deterministic switching, stochastic switching (fixed M)

# Implied volatility surface 2: Strike / Randomiser Variance / IV
#  for deterministic switching, stochastic switching (fully and with fixed M)
#
@author: Felix L. Wolf
"""
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm
from methods.char_funcs import ChF_Component, ChF_Composite, \
    ChF_StochSwitch, ChF_FullStoch, ProbabMExponSwitches, M_truncate, ChF_ABM
from methods.COS_funcs import CallPutOptionPriceCOSMthd, ImpliedVolatility, \
    ImpliedVolRange
from methods.randomiser_util import volfunc_rnd, make_randomiser, \
    driftfunc_det

################################################################################
# Option parameters
################################################################################
r = 0.05  # interest rate
S0 = 1  # initial stock price (don't change this! We imply that X(0) = 0)
CP = 'c'
T = 1  # maturity
Nk = 31  # steps in domain of strike K
K = np.linspace(0.8, 1.4, Nk)  # Strike domain
idx_ATM = np.where(K == 1.)[0][0]
Lcos = 7  # truncation domain for COS method
Ncos = 5000  # expansion terms in COS method

################################################################################
# Process parameters
################################################################################
# randomiser for deterministic switching: N(.1, .1**2) and N(.25, .5**2)
# implied ATM vol is 0.4291405319263591
mu0_det, sig0_det = (0.15, 0.1)  # mean, sd in regime 1
mu1_det, sig1_det = (0.3, 1.)  # mean, sd in regime 2

#######################################
# the drift and volatility functions:
vol_func = lambda t, p: volfunc_rnd(t, p)  # i.e. \sigma_j := identity
drift_funcGBM = lambda t, p: r - 0.5 * volfunc_rnd(t,
                                                   p) ** 2
vols_list = [vol_func, vol_func]
drifts_list = [drift_funcGBM, drift_funcGBM]

#######################################
# Switching rules:
# deterministic switching rule (2/year)
tau_list = np.array([0., T / 2])  # np.array([0., T/4, 2*T/4, 3*T/4])
# stoch. switching with exponential sojourn times (Exp(2) mean is 2/year)
exp_scale = T / len(tau_list)  # scale = 1 / rate
zeta_expon = st.expon(scale=exp_scale)
zeta_list = [zeta_expon]
################################################################################
# Characteristic functions
N_j = 7
# a) deterministic switching
randomisers_det = make_randomiser(mu0_det, mu1_det, sig0_det, sig1_det) * 10
cfX_det = lambda u: ChF_Composite(u, T, tau_list, randomisers_det, drifts_list,
                                  vols_list, N=N_j)
# b) stochastic switching (fixed M, exponential sojourn)
randomisers_exp = randomisers_det
M_soj = 1
cfX_stoch_expon = lambda u: ChF_StochSwitch(u, T, M_soj, zeta_list,
                                            randomisers_exp,
                                            drifts_list, vols_list,
                                            N=N_j, L=N_j)
# c) fully stochastic switching
randomisers_fs = randomisers_det * 10
Mmax, ploss = M_truncate(0.1, exp_scale, T)
cfX_fullstoch = lambda u: ChF_FullStoch(u, T, exp_scale, randomisers_fs,
                                        drifts_list * 10, vols_list * 10,
                                        MaxNumSwitches=Mmax, N=N_j, L=N_j)

# d) no switching:
mu_ns = (mu0_det + mu1_det) / 2
sig_ns = (sig0_det + sig1_det) / 2
randomiser_ns = st.norm(loc=mu_ns, scale=sig_ns)
cfX_noswitch = lambda u: ChF_Component(u, T, randomiser_ns,
                                       drift_funcGBM, vol_func, N=N_j)
################################################################################
# Pricing
CallPrice = lambda cf: CallPutOptionPriceCOSMthd(cf, CP, S0, r, T, K, Ncos,
                                                 Lcos)
price_det = CallPrice(cfX_det)
price_stoch_expon = CallPrice(cfX_stoch_expon)
# Implied Volatility
siginit = 0.5
ImplVol = lambda price_arr: ImpliedVolRange(CP, price_arr, S0, K, T, r, siginit)
impl_det = ImplVol(price_det)
impl_stoch_expon = ImplVol(price_stoch_expon)
ATM = impl_det[K == 1.]
################################################################################
# Implied volatility surface IVS: Strike / Maturity / IV
Tgrid = np.linspace(.5, 1., 11)
IVS_det = np.zeros([Tgrid.size, K.size])
IVS_stoch_expon = np.zeros([Tgrid.size, K.size])
IVS_fullstoch = np.zeros([Tgrid.size, K.size])
IVS_noswitch = np.zeros([Tgrid.size, K.size])
for k, Tk in enumerate(Tgrid):
    Call_Tk = lambda cf: CallPutOptionPriceCOSMthd(cf, CP, S0, r, Tk, K,
                                                   Ncos, Lcos)
    IV_Tk = lambda price_arr: ImpliedVolRange(CP, price_arr, S0, K, Tk, r,
                                              sigma0=0.5)
    IVS_det[k] = IV_Tk(Call_Tk(cfX_det))
    IVS_stoch_expon[k] = IV_Tk(Call_Tk(cfX_stoch_expon))
    IVS_fullstoch[k] = IV_Tk(Call_Tk(cfX_fullstoch))
    IVS_noswitch[k] = IV_Tk(Call_Tk(cfX_noswitch))
################################################################################
# Implied volatility surface IVS2: Strike / Randomiser / IV
rndmsrs = lambda sig1: make_randomiser(mu0_det, mu1_det, sig0_det, sig1)
sig1_range = np.linspace(0.0001, 1, 11)
IVS2_det = np.zeros([sig1_range.size, K.size])
IVS2_stoch_expon = np.zeros([sig1_range.size, K.size])
IVS2_fullstoch = np.zeros([sig1_range.size, K.size])
IVS2_noswitch = np.zeros([sig1_range.size, K.size])
for j, s1 in enumerate(sig1_range):
    randomisers_j = rndmsrs(s1)
    cfj_det = lambda u: ChF_Composite(u, T, tau_list, randomisers_j,
                                      drifts_list, vols_list, N=N_j)
    cfj_stochexpon = lambda u: ChF_StochSwitch(u, T, M_soj, zeta_list,
                                               randomisers_j, drifts_list,
                                               vols_list, N=N_j, L=5)
    cfj_fullstoch = lambda u: ChF_FullStoch(u, T, exp_scale, randomisers_j * 10,
                                            drifts_list * 10, vols_list * 10,
                                            MaxNumSwitches=Mmax, N=N_j, L=N_j)
    cfj_noswitch = lambda u: ChF_Component(u, T, randomisers_j[1],
                                           drift_funcGBM, vol_func, N=N_j)
    IVS2_det[j] = ImplVol(CallPrice(cfj_det))
    IVS2_stoch_expon[j] = ImplVol(CallPrice(cfj_stochexpon))
    IVS2_fullstoch[j] = ImplVol(CallPrice(cfj_fullstoch))
    IVS2_noswitch[j] = ImplVol(CallPrice(cfj_noswitch))
################################################################################
# Setting up the plotting engine:
#
# latex formatting in plot titles and legends
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')


# helper function to obtain colormaps
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


blues2 = truncate_colormap(cm.Blues, 0.3, 1.0)
purples2 = truncate_colormap(cm.RdPu, 0.3, 1.0)
oranges2 = truncate_colormap(cm.OrRd, 0.3, 1.0)
greys2 = truncate_colormap(cm.Greys, 0.3, 1.0)
################################################################################
# Create the IVS figures of the article:

# fig1 corresponds to Figure 4 in the article
X, Y = np.meshgrid(K[::-1], Tgrid)
fig1 = plt.figure(figsize=(6.4, 3))
ax1 = fig1.add_subplot(1, 1, 1, projection='3d')
surf = ax1.plot_surface(X, Y, IVS_det, rstride=1, cstride=1, cmap=blues2,
                        linewidth=0, antialiased=False, alpha=0.5)
surf2 = ax1.plot_surface(X, Y, IVS_stoch_expon, rstride=1, cstride=1,
                         cmap=purples2,
                         linewidth=0, antialiased=False, alpha=0.5)
surf4 = ax1.plot_surface(X, Y, IVS_noswitch, rstride=1, cstride=1, cmap=greys2,
                         linewidth=0, antialiased=False, alpha=0.5)

fig1.colorbar(surf, shrink=0.5, aspect=10, label='det.\ switch', pad=0.)
fig1.colorbar(surf2, shrink=0.5, aspect=10, label='stoch.\ switch', pad=0.025)

fig1.colorbar(surf4, shrink=0.5, aspect=10, label='no switch', pad=0.2)
ax1.view_init(30, -60 + 10)  # view_init(30, 50)
ax1.set_xlabel('Strike')
ax1.set_ylabel('Expiry')
ax1.zaxis.set_rotate_label(False)
ax1.set_zlabel('Implied volatility', rotation=90)
fig1.tight_layout()
plt.show()

# fig3 corresponds to Figure 4 in article
X2, Y2 = np.meshgrid(K, sig1_range)
fig3 = plt.figure()  # plt.figure(figsize=(7, 4)) #  default is 6.4, 4.8
ax31 = fig3.add_subplot(1, 3, 1, projection='3d')
ax32 = fig3.add_subplot(1, 3, 2, projection='3d')
ax33 = fig3.add_subplot(1, 3, 3, projection='3d')
# ax1.plot_surface(X, Y, IVS_det, label="deterministic")
# ax1.plot_surface(X, Y, IVM_stoch, label="stochastic", alpha=0.2)
surf31 = ax31.plot_surface(X2, Y2, IVS2_det, rstride=1, cstride=1, cmap=blues2,
                           linewidth=0, antialiased=True)
surf32 = ax32.plot_surface(X2, Y2, IVS2_stoch_expon, rstride=1, cstride=1,
                           cmap=purples2,
                           linewidth=0, antialiased=True)
surf33 = ax33.plot_surface(X2, Y2, IVS2_fullstoch, rstride=1, cstride=1,
                           cmap=oranges2,
                           linewidth=0, antialiased=True)
ax32.scatter(X2[0], Y2[0], np.max(IVS2_det), alpha=0.)
ax33.scatter(X2[0], Y2[0], np.max(IVS2_det), alpha=0.)
ax32.scatter(X2[0], Y2[0], np.min(IVS2_det), alpha=0.)
ax33.scatter(X2[0], Y2[0], np.min(IVS2_det), alpha=0.)
fig3.colorbar(surf31, label='IV det.\ switch', location='bottom')
fig3.colorbar(surf32, label='IV stoch.\ switch', location='bottom')
fig3.colorbar(surf33, label='IV fully stoch.\ switch', location='bottom')
ax31.view_init(20, -60 + 30)  # view_init(30, -60-10)
ax32.view_init(20, -60 + 30)  # view_init(30, -60-10)
ax33.view_init(20, -60 + 30)  # view_init(30, -60-10)
# ax3.invert_yaxis()
ax31.set_xlabel('Strike')
ax31.set_ylabel('Parameter of \n Uncertainty')
ax31.set_zlabel('Implied volatility')
# fig3.suptitle('Implied volatility surface: Uncertainty')
plt.setp(ax31.get_zticklabels(), visible=False)
plt.setp(ax32.get_zticklabels(), visible=False)
fig3.tight_layout()
plt.show()
