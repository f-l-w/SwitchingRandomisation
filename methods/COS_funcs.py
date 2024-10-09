# COS method implementation for European option pricing
# @author: LECH A.GRZELAK
# with some adjustments by Felix L Wolf to ensure functionality with
# the characteristic functions provided in methods/char_funcs.py

import numpy as np
import scipy.stats as st


def CallPutOptionPriceCOSMthd(cf,CP,S0,r,tau,K,N,L):
    # cf   - characteristic function as a functon, in the book denoted as φ
    # CP   - C for call and P for put
    # S0   - Initial stock price
    # r    - interest rate (constant)
    # tau  - time to maturity
    # K    - list of strikes
    # N    - Number of expansion terms
    # Lcos    - size of truncation domain (typ.:Lcos=8 or Lcos=10)

    # reshape K to a column vector
    K = np.array(K).reshape([len(K), 1])

    # assigning i=sqrt(-1)
    i = 1j

    x0 = np.log(S0 / K)

    # truncation domain
    a = 0.0 - L * np.sqrt(tau)
    b = 0.0 + L * np.sqrt(tau)

    # sumation from k = 0 to k=N-1
    k = np.linspace(0, N-1, N).reshape([N, 1])
    u = k * np.pi / (b - a)

    # Determine coefficients for Put Prices  
    H_k = CallPutCoefficients(CP, a, b, k)

    mat = np.exp(i * np.outer((x0 - a), u))

    temp = cf(u) * H_k
    temp[0] = 0.5 * temp[0]

    value = np.exp(-r * tau) * K * np.real(mat.dot(temp))
    return value.squeeze()


# Determine coefficients for Put Prices
def CallPutCoefficients(CP,a,b,k):
    if str(CP).lower()=="c" or str(CP).lower()=="1":
        c = 0.0
        d = b
        coef = Chi_Psi(a,b,c,d,k)
        Chi_k = coef["chi"]
        Psi_k = coef["psi"]
        if a < b and b < 0.0:
            H_k = np.zeros([len(k),1])
        else:
            H_k      = 2.0 / (b - a) * (Chi_k - Psi_k)

    elif str(CP).lower()=="p" or str(CP).lower()=="-1":
        c = a
        d = 0.0
        coef = Chi_Psi(a,b,c,d,k)
        Chi_k = coef["chi"]
        Psi_k = coef["psi"]
        H_k      = 2.0 / (b - a) * (- Chi_k + Psi_k)

    return H_k


def Chi_Psi(a,b,c,d,k):
    psi = np.sin(k * np.pi * (d - a) / (b - a)) - np.sin(k * np.pi * (c - a)/(b - a))
    psi[1:] = psi[1:] * (b - a) / (k[1:] * np.pi)
    psi[0] = d - c

    chi = 1.0 / (1.0 + np.power((k * np.pi / (b - a)) , 2.0))
    expr1 = np.cos(k * np.pi * (d - a)/(b - a)) * np.exp(d)  - np.cos(k * np.pi
                  * (c - a) / (b - a)) * np.exp(c)
    expr2 = k * np.pi / (b - a) * np.sin(k * np.pi *
                        (d - a) / (b - a))   - k * np.pi / (b - a) * np.sin(k
                        * np.pi * (c - a) / (b - a)) * np.exp(c)
    chi = chi * (expr1 + expr2)

    value = {"chi":chi,"psi":psi }
    return value


def ImpliedVolatility(CP, V_market, S_0, K, tau, r, sigma0):
    error = 1e10  # initial error
    Nmax = 100
    # Handy lambda expressions
    optPrice = lambda sigma: BS_Call_Option_Price(CP, S_0, K, sigma, tau, r)
    vega = lambda sigma: dV_dsigma(S_0, K, sigma, tau, r)

    sigma = sigma0
    # While the difference between the model and the arket price is large
    # follow the iteration
    i = 0
    while error > 10e-10 and i < Nmax:
        f = V_market - optPrice(sigma)
        f_prim = -vega(sigma)
        sigma_new = sigma - f / f_prim

        error = abs(sigma_new - sigma)
        sigma = sigma_new
        i = i + 1
        # print(i)
    return sigma


def ImpliedVolRange(CP, V_market, S_0, K_arr, tau, r, sigma0):
    if V_market.size != K_arr.size:
        print("ImpliedVoLRange() expects market price and strike congruence.")
        return
    out = np.zeros_like(K_arr)
    for idx, K in enumerate(K_arr):
        out[idx] = ImpliedVolatility(CP, V_market[idx], S_0, K, tau, r, sigma0)
        sigma0 = out[idx]  # update initial value with previous result
    return out


# Vega, dV/dsigma
def dV_dsigma(S_0, K, sigma, tau, r):
    # parameters and value of Vega
    d2 = (np.log(S_0 / (K)) + (r - 0.5 * np.power(sigma, 2.0)) * tau) / float(
        sigma * np.sqrt(tau))
    value = K * np.exp(-r * tau) * st.norm.pdf(d2) * np.sqrt(tau)
    return value


def BS_Call_Option_Price(CP, S_0, K, sigma, tau, r):
    # Black-Scholes Call option price
    d1 = (np.log(S_0 / (K)) + (r + 0.5 * np.power(sigma, 2.0)) * tau) / (
                sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    if str(CP).lower() == "c" or str(CP).lower() == "1":
        value = st.norm.cdf(d1) * S_0 - st.norm.cdf(d2) * K * np.exp(-r * tau)
    elif str(CP).lower() == "p" or str(CP).lower() == "-1":
        value = st.norm.cdf(-d2) * K * np.exp(-r * tau) - st.norm.cdf(-d1) * S_0
    return value
