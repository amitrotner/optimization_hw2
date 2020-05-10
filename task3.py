import mcholmz
import numpy as np
import scipy
from scipy import io
import matplotlib.pyplot as plt

H_well = scipy.io.loadmat("h.mat")['H_well']
H_ill = scipy.io.loadmat("h.mat")['H_ill']


def plot_graph(f_name, iter, convergence_curve):
    plt.plot(range(iter), convergence_curve)
    plt.title(f_name)
    plt.ylabel(r"$f(x_{k})-f^*$")
    plt.xlabel("Iteration Number")
    plt.yscale('log')
    plt.show()


def rosen(x):
    return sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0)


def rosen_der(x):
    xm = x[1:-1]
    xm_m1 = x[:-2]
    xm_p1 = x[2:]
    der = np.zeros_like(x)
    der[1:-1] = 200 * (xm - xm_m1 ** 2) - 400 * (xm_p1 - xm ** 2) * xm - 2 * (1 - xm)
    der[0] = -400 * x[0] * (x[1] - x[0] ** 2) - 2 * (1 - x[0])
    der[-1] = 200 * (x[-1] - x[-2] ** 2)
    return der


def rosen_hess(x):
    x = np.asarray(x)
    H = np.diag(-400 * x[:-1], 1) - np.diag(400 * x[:-1], -1)
    diagonal = np.zeros_like(x)
    diagonal[0] = 1200 * x[0] ** 2 - 400 * x[1] + 2
    diagonal[-1] = 200
    diagonal[1:-1] = 202 + 1200 * x[1:-1] ** 2 - 400 * x[2:]
    H = H + np.diag(diagonal)
    return H


def f_well(x):
    H = H_well
    return 0.5 * np.matmul(x.T, np.matmul(H, x))


def f_well_grad(x):
    H = H_well
    return np.matmul(H, x)


def f_well_hess(x):
    H = H_well
    return H


def f_ill(x):
    H = H_ill
    return 0.5 * np.matmul(x.T, np.matmul(H, x))


def f_ill_grad(x):
    H = H_ill
    return 0.5 * np.matmul((H + H.T), x)


def f_ill_hess(x):
    H = H_ill
    return 0.5 * (H + H.T)


def gradient_decent(f, f_name, der_f, x0, alpha0, sigma, beta, epsilon):
    d = -der_f(x0)
    x = x0
    iter = 1
    convergence_curve = []
    convergence_curve += [f(x)]
    while np.linalg.norm(der_f(x)) >= epsilon:
        alpha = alpha0
        F_armijo = (f(x + alpha * d) - f(x))
        F_armijo_sigma = (sigma * alpha * np.matmul(der_f(x).T, d))
        while F_armijo > F_armijo_sigma:
            alpha = beta * alpha
            F_armijo = (f(x + alpha * d) - f(x))
            F_armijo_sigma = (sigma * alpha * np.matmul(der_f(x).T, d))
        x = x + alpha * d
        d = -der_f(x)
        convergence_curve += [float(f(x))]
        iter += 1
    plot_graph(f_name, iter, convergence_curve)
    return x


def newton(f, f_name, der_f, hes_f, x0, alpha0, sigma, beta, epsilon):
    L, D, e = mcholmz.modifiedChol(hes_f(x0))
    grad = der_f(x0)
    y = scipy.linalg.solve_triangular(-L, grad, lower=True)
    d = scipy.linalg.solve_triangular(np.matmul(np.diagflat(D), L.T), y, lower=False)
    x = x0
    iter = 1
    convergence_curve = []
    convergence_curve += [f(x)]
    while np.linalg.norm(der_f(x)) >= epsilon:
        alpha = alpha0
        F_armijo = (f(x + alpha * d) - f(x))
        F_armijo_sigma = (sigma * alpha * np.matmul(der_f(x).T, d))
        while float(F_armijo) > float(F_armijo_sigma):
            alpha = beta * alpha
            F_armijo = (f(x + alpha * d) - f(x))
            F_armijo_sigma = (sigma * alpha * np.matmul(der_f(x).T, d))
        x = x + alpha * d
        L, D, e = mcholmz.modifiedChol(hes_f(x))
        grad = der_f(x)
        y = scipy.linalg.solve_triangular(-L, grad, lower=True)
        d = scipy.linalg.solve_triangular(np.matmul(np.diagflat(D), L.T), y, lower=False)
        convergence_curve += [float(f(x))]
        iter += 1
    plot_graph(f_name, iter, convergence_curve)
    return x


if __name__ == '__main__':
    x0_quad = scipy.io.loadmat("h.mat")['x0']
    x0_rosen = np.zeros(10)
    alpha0 = 1
    sigma = 0.25
    beta = 0.5
    epsilon = 1e-5

    gradient_decent(rosen, "The Convergence Curve of Gradient Decent \n with the Rosenbrock Function", rosen_der, x0_rosen, alpha0, sigma, beta, epsilon)
    gradient_decent(f_well, "The Convergence Curve of Gradient Decent \n with the Quadratic Function using H_well", f_well_grad, x0_quad, alpha0, sigma, beta, epsilon)
    gradient_decent(f_ill, "The Convergence Curve of Gradient Decent \n with the Quadratic Function using H_ill", f_ill_grad, x0_quad, alpha0, sigma, beta, epsilon)

    newton(rosen, "The Convergence Curve of Newton's Method \n with the Rosenbrock Function", rosen_der, rosen_hess, x0_rosen, alpha0, sigma, beta, epsilon)
    newton(f_well, "The Convergence Curve of Newton's Method \n with the Quadratic Function using H_well", f_well_grad, f_well_hess, x0_quad, alpha0, sigma, beta, epsilon)
    newton(f_ill, "The Convergence Curve of Newton's Method \n with the Quadratic Function using H_ill", f_ill_grad, f_ill_hess, x0_quad, alpha0, sigma, beta, epsilon)

