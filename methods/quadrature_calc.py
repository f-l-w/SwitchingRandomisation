"""
Compute quadrature points and weights for various distributions:
- uniform, exponential, normal
- truncated uniform, truncated exponential, truncated normal

@author: Felix L. Wolf
Partially based on an original implementation by https://github.com/LechGrzelak
"""

import math
import numpy as np
from scipy.linalg import cholesky as chol
import scipy.linalg as linalg
from scipy.stats import norm, expon, uniform
from scipy.integrate import quad
from scipy.special import comb  # N choose k
from methods.nearest_pd_matrix import nearestPD


def CollocationTruncatedUniform(a, b, rlimit, N):
    free_rv = uniform(loc=a, scale=b-a)
    # check if a truncation really takes place:
    if b < rlimit:  # can call standard uniform collocation
        x_i, w_i = CollocationUniform(a, b, N, outtype="tuple")
        return {"points": x_i, "weights": w_i}
    # otherwise procede with truncation:
    F_limit = free_rv.cdf(rlimit)
    f = lambda x: free_rv.pdf(x)/F_limit
    moments = np.zeros(2*N+1)
    moments[0] = 1
    for j in range(1, 2 * N + 1):
        # include 0th moment in
        integrand = lambda x: x ** j * f(x)  # \int_0^rlimit x^j f(x) dx
        moments[j] = quad(integrand, 0, rlimit)[0]
    # fill moment matrix
    M = np.zeros([N + 1, N + 1])
    for i in range(0, N + 1):
        for j in range(0, N + 1):
            M[i, j] = moments[i + j]
    # Once the moments are computed use generic code to calculate quad. pairs
    x_i, w_i = FindCollocation(M, makePD=False)
    return {"points": x_i, "weights": w_i}


def CollocationTruncatedExponential(expon_scale, rlimit, N):
    free_rv = expon(scale=expon_scale)
    F_limit = free_rv.cdf(rlimit)
    f = lambda x: free_rv.pdf(x)/F_limit  # truncated density
    # store the required 2N moments:
    moments = np.zeros(2*N+1)
    moments[0] = 1
    for j in range(1, 2*N+1):
        # include 0th moment in
        integrand = lambda x: x**j * f(x)  # \int_0^rlimit x^j f(x) dx
        moments[j] = quad(integrand, 0, rlimit)[0]
    # fill moment matrix
    M = np.zeros([N + 1, N + 1])
    for i in range(0, N + 1):
        for j in range(0, N + 1):
            M[i, j] = moments[i + j]
    # Once the moments are computed use generic code to calculate quad. pairs
    x_i, w_i = FindCollocation(M, makePD=False)
    return {"points": x_i, "weights": w_i}


def CollocationExponential(scale_var, N):
    # Exp(lambda) with mean 1/lambda characterised by scale_var = 1/lambda
    moment = lambda n: math.factorial(n) * scale_var**n
    # Creation of Matrix M, dimension N+1 x N+1
    M = np.zeros([N + 1, N + 1])
    for i in range(0, N + 1):
        for j in range(0, N + 1):
            M[i, j] = moment(i + j)
    # Once the moments are computed use generic code to calculate quad. pairs
    x_i, w_i = FindCollocation(M, makePD=False)
    return {"points": x_i, "weights": w_i}


# Function by Lech A. Grzelak for uniform collocation:
def CollocationUniform(a, b, N, outtype="tuple"):
    moment = lambda n: (b ** (n + 1) - a ** (n + 1)) / ((n + 1) * (b - a))

    # Creation of Matrix M, dimension N+1 x N+1
    M = np.zeros([N + 1, N + 1])
    for i in range(0, N + 1):
        for j in range(0, N + 1):
            M[i, j] = moment(i + j)

    # Once the moments are computed use generic code to calculate quad. pairs
    x_i, w_i = FindCollocation(M)
    if outtype == "tuple":
        return x_i, w_i
    else:
        return {"points": x_i, "weights": w_i}


def FindCollocation(M, makePD=True):
    # M - number of collocation points (int)
    # makePD - optional argument to run approximation algorithm for closest
    #          positive definite matrix

    # Creation of UPPER diagonal matrix, R, dimension N+1 x N+1
    # Since Matrix M also includes the 0 moment we adjust the size
    N = len(M) - 1
    if makePD:
        R = chol(nearestPD(M))
    if not makePD:
        R = chol(M)

    # Creation of vector alpha and beta
    alpha = np.zeros([N])
    beta = np.zeros([N - 1])
    alpha[0] = R[0, 1]
    beta[0] = (R[1, 1] / R[0, 0]) ** 2.0
    for i in range(1, N):
        alpha[i] = R[i, i + 1] / R[i, i] - R[i - 1, i] / R[i - 1, i - 1]
    for i in range(1, N - 1):
        beta[i] = (R[i + 1, i + 1] / R[i, i]) ** 2.0

    # Construction of matrix J
    J = np.diag(np.sqrt(beta), k=-1) + np.diag(alpha, k=0) + np.diag(
        np.sqrt(beta), k=1);
    # computation of the weights
    eigenValues, eigenVectors = linalg.eig(J)
    w_i = eigenVectors[0, :] ** 2.0
    x_i = np.real(eigenValues)
    # sorting the arguments
    idx = np.argsort(x_i)
    w_i = w_i[idx]
    x_i = x_i[idx]
    return x_i, w_i


# Quadrature Points for truncated normal distribution
def CollocationTruncatedNormal(N, mu, sigma, b):
    # get moments (not centred) of truncated normal
    # upper bound b, lower bound a = -infty
    # analytical formulas see
    # https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    beta = (b - mu)/sigma

    L = np.ones(2*N+1)
    L[0] = 1
    L[1] = - norm.pdf(beta) / norm.cdf(beta)
    for i in range(2, 2*N+1):
        L[i] = -(np.power(beta, i-1) * norm.pdf(beta)) / norm.cdf(beta) \
            + (i-1)*L[i-2]

    moment_vec = np.ones([2*N+1])
    for k in range(2*N+1):
        full_sum = 0
        for i in range(k+1):
            full_sum += comb(k, i) * np.power(sigma, i) * np.power(mu, k-i) * L[i]
        moment_vec[k] = full_sum
    # print(moment_vec)

    M = np.zeros([N+1, N+1])
    for i in range(0, N+1):
        for j in range(0, N+1):
            M[i, j] = moment_vec[i+j]
    # print(M)

    x_i, w_i = FindCollocation(M)
    return x_i


# Compute points and weights of normal distributions (based on the list of
# random distributions (rv_frozen objects from scipy.stats)
def CollocationNormal(dist_list, N):
    # N: number of collocation points, dist_list: list of rv_frozen
    dist_list = np.atleast_1d(dist_list)
    M = len(dist_list)

    # Construct standard normal moments and moment matrix
    Z = norm(loc=0., scale=1.)
    moment_vec = np.ones(2*N + 1)
    for k in range(2*N+1):
        moment_vec[k] = Z.moment(k)
    moment_mat = np.zeros([N+1, N+1])
    for i in range(0, N+1):
        for j in range(0, N+1):
            moment_mat[i, j] = moment_vec[i+j]
    # Compute quadrature points and weights with standard method
    x_i, w_i = FindCollocation(moment_mat, makePD=False)

    quadweights = np.tile(w_i, M).reshape(M, -1)
    quadpoints = np.zeros([M, N])
    for j, distr in enumerate(dist_list):
        if distr.dist.name != 'norm':
            print(f"{j}th distribution {distr} not a normal distribution "
                  "(invoked normal collocation points")
        mu = distr.mean()
        sigma = np.sqrt(distr.var())
        quadpoints[j] = x_i * sigma + mu

    out = {"points": quadpoints, "weights": quadweights}
    return out
