# Author: Daniel Sharp
import mpart as mt
import numpy as np
import pickle
from matplotlib import pyplot as plt
from matplotlib import rc
rc('text',usetex=True)
# Class to represent a GMM
# For some reason, there's no good way to evaluate the logpdf and gradlogpdf of a GMM in python
class GMM:
    # Initialize the GMM
    def __init__(self, weights, centers, vars):
        self.weights = np.array(weights)
        self.centers = np.array(centers)
        self.vars = np.array(vars)

      # Evaluate the logpdf of the GMM at all points x (assumed vector)
    def logpdf(self, x):
        norm_diffs = np.subtract.outer(x, self.centers)
        single_logpdf = -(norm_diffs*norm_diffs/self.vars + np.log(2*np.pi*self.vars))/2
        eps = np.min(single_logpdf)
        single_pdf = np.exp(single_logpdf + eps)
        combined_pdfs = single_pdf @ self.weights
        logpdf = np.log(combined_pdfs) - eps
        return logpdf

      # Evaluate the derivative of the logpdf of the GMM at all points x (assumed vector)
    def gradlogpdf(self, x):
        norm_diffs = np.subtract.outer(x, self.centers)
        single_logpdf = -(norm_diffs*norm_diffs/self.vars + np.log(2*np.pi*self.vars))/2
        single_pdf = np.exp(single_logpdf)
        combined_pdfs = single_pdf @ self.weights
        grad_single_normpdf = -norm_diffs * single_pdf / self.vars
        denom = combined_pdfs
        numer = grad_single_normpdf @ self.weights
        return numer / denom

# Verify the GMM works as needed
def test_gmm():
    gmm = GMM([1/3,2/3], [-2, 2], [1, 1])
    x = np.arange(-5, 5, 0.001)
    orig = gmm.logpdf(x)
    delta = 1e-8
    fdiff = np.array([gmm.logpdf(np.array([y+delta]))[0] for y in x])
    fdiff = (fdiff - orig)/delta
    grad_logpdf = gmm.gradlogpdf(x)
    plt.plot(x, fdiff)
    plt.plot(x, grad_logpdf)
    plt.show()
    print(np.linalg.norm(grad_logpdf - fdiff)/len(x))

# KL divergence objective
def obj(coeffs, transport_map, x, target):
    num_points = x.shape[1]
    transport_map.SetCoeffs(coeffs)
    map_of_x = transport_map.Evaluate(x)
    logpdf = target.logpdf(map_of_x)
    log_det = transport_map.LogDeterminant(x)
    return -np.sum(logpdf + log_det)/num_points

# Gradient of KL divergence objective
def grad_obj(coeffs, transport_map, x, target):
    num_points = x.shape[1]
    transport_map.SetCoeffs(coeffs)
    map_of_x = transport_map.Evaluate(x)
    sens_vecs = target.gradlogpdf(map_of_x)
    grad_logpdf = transport_map.CoeffGrad(x, sens_vecs)
    grad_log_det = transport_map.LogDeterminantCoeffGrad(x)
    return -np.sum(grad_logpdf + grad_log_det, 1)/num_points

# Armijo backtracking on function fun(x, *args) at point x0 with eval fx0 and gradient dfx0
# alpha_0 is starting step size, beta is the scaling for the armijo-goldstein RHS, tau is the
# step truncation, and we take at most max_iter steps
def Armijo(fun, args, x0, fx0, dfx0, descent_dir, alpha_0 = 1.1, beta = 0.1, tau = 0.8, max_iter = 50):
    j = 0
    armijo_goldstein_met = False
    alpha = alpha_0
    rhs_dot_prod = dfx0 @ descent_dir
    while j < max_iter and not armijo_goldstein_met:
        alpha *= tau
        f_new = fun(x0 + alpha*descent_dir, *args)
        armijo_goldstein_met = f_new <= fx0 + alpha*beta*rhs_dot_prod
        j += 1
    return alpha

class GradientDescent:
    def __init__(self, fun, d_fun, args = (), callback = None, linesearch = Armijo):
        self.fun = fun
        self.d_fun = d_fun
        self.args = args
        self.callback = callback
        self.linesearch = linesearch
        self.status = 0
    
    # Perform a gradient descent step
    def step(self, x, f_x, df_x):
        grad = df_x
        descent_dir = -grad
        alpha = self.linesearch(self.fun, self.args, x, f_x, grad, descent_dir)
        return x + descent_dir*alpha

    # Catch-all minimization routine. Absolut tolerances on x and function value. Relative tolerance on gradient norm
    # Maximum of maxiter steps
    def minimize(self, x0, xtol = 1e-7, ftol = 1e-7, gtol = 1e-7, maxiter = 1000):
        x = x0
        f_x = self.fun(x, *self.args)
        df_x = self.d_fun(x, *self.args)
        met_xtol = met_ftol = met_gtol = exceeded_iters = False
        j = 0
        while not (met_xtol or met_ftol or met_gtol or exceeded_iters):
            x_new = self.step(x, f_x, df_x)
            # Check all the convergence criteria
            x_diff = np.linalg.norm(x_new - x)
            met_xtol = x_diff < xtol

            f_x_new = self.fun(x_new, *self.args)
            f_x_diff = abs(f_x_new - f_x)
            met_ftol = f_x_diff < ftol
            f_x = f_x_new

            df_x = self.d_fun(x_new, *self.args)
            met_gtol = np.linalg.norm(df_x) < gtol
            j += 1
            exceeded_iters = j > maxiter

            # Call the callback if needed
            if self.callback is not None:
                self.callback(x, x_new, f_x, df_x, j)
            x = x_new
        status = 0
        if exceeded_iters:
            status = -1
        elif met_xtol:
            status = 1
        elif met_ftol:
            status = 2
        elif met_gtol:
            status = 3
        self.status = status
        return x

    # Maps status codes to coherent messages
    def GetStatus(self):
        if self.status == 0:
            return "Optimization not yet run"
        elif self.status == -1:
            return "Failure: Hit maximum iterations"
        elif self.status == 1:
            return "Success: Met tolerance on x"
        elif self.status == 2:
            return "Success: Met tolerance on f"
        elif self.status == 3:
            return "Success: Met tolerance on ∇f"

# Testing suite verifying the gradient descent works
def Rosenbrock(pt, a=1, b=100):
    x,y = pt
    return (x-a)**2 + b*((y-x**2)**2)

def d_Rosenbrock(pt, a=1, b=100):
    x,y = pt
    d_1 = -2*(a-x) - 4*b*x*(y-(x**2))
    d_2 = 2*b*(y-(x**2))
    return np.array([d_1,d_2])

def test_gradDesc():
    fun = lambda x: (x-3)*(x-3)/2
    d_fun = lambda x: (x-3)
    gd = GradientDescent(fun, d_fun)
    x1_fun = gd.minimize(np.array([0.]))
    print(f"easy function converges to {x1_fun}, status {gd.GetStatus()}")
    gd = GradientDescent(Rosenbrock, d_Rosenbrock)
    x1_r = gd.minimize(np.array([-3,-4.]), maxiter=10000)
    print(f"rosenbrock converges to {x1_r}, status {gd.GetStatus()}")

# This is the default construction of the transport map
# I chose -5.5, 5.5 to be >3 std deviations from the mean of each GMM
def DefaultTransport(total_order):
    mset = mt.FixedMultiIndexSet(1, total_order)
    opts = mt.MapOptions()
    opts.basisLB, opts.basisUB = -5.5, 5.5
    opts.basisType = mt.BasisTypes.HermiteFunctions
    comp = mt.CreateComponent(mset, opts)
    return comp

# This is a simple callback for the optimization routine, showing how to use the callback functionality of optimization
# Use this (with these particular arguments) as the basis for any other callback routines
def progress_callback(x, x_new, f_x, df_x, j):
    print(f"Iteration {j}, f(x_j) = {f_x}, ||df_x|| = {np.linalg.norm(df_x)}")

# Heart of the script. Play with different arguments here if you want, or go deeper (e.g. change the callback)
def train_map(num_test_pts, total_order = 8, min_args = {}, GD_args = {'callback': progress_callback}):
    ref_test_pts = np.random.randn(1,num_test_pts)
    gmm = GMM([1/3,2/3], [-2., 2.], [1., 1.])
    tmap = DefaultTransport(total_order)
    args = (tmap, ref_test_pts, gmm)
    gd = GradientDescent(obj, grad_obj, args, **GD_args)
    x0 = np.zeros(tmap.numCoeffs)
    x1 = gd.minimize(x0, **min_args)
    print(f"Converges with status {gd.GetStatus()}")
    tmap.SetCoeffs(x1)
    return tmap

if __name__ == '__main__':
    # Where to save the coefficients of the map
    serialized_map_filename = 'my_gmm_map.pkl'
    # SET THIS BASED ON IF YOU'VE RUN THE SCRIPT BEFORE
    already_serialized = False
    # How high order map we want
    total_order = 10
    tmap = None
    if already_serialized:
        tmap = DefaultTransport(total_order)
        with open(serialized_map_filename, 'rb') as f:
            coeffs = pickle.load(f)
        tmap.SetCoeffs(coeffs)
    else:
        seed = 1928342
        # Ensure that the number of parameters * number of MC points for training <= 128*128
        num_test_pts = (512*512) // (total_order+1)
        optimization_args = {'maxiter': 500} # Optimization algorithm gets 100 steps
        tmap1 = train_map(num_test_pts, total_order, optimization_args)
        optimization_args = {'maxiter': 10} 
        tmap2 = train_map(num_test_pts, total_order, optimization_args)
        with open(serialized_map_filename, 'wb') as f:
            pickle.dump(tmap1.CoeffMap(), f)
            pickle.dump(tmap2.CoeffMap(), f)

    # Example use of `tmap` for generative modeling
    test_pts = np.random.randn(1, 20000)
    eval_samples1 = tmap1.Evaluate(test_pts).flatten()
    eval_samples2 = tmap2.Evaluate(test_pts).flatten()
    # Messy way to get the PDF, reinitializing the gmm object (change if you change the target)
    eval_samples1.sort()
    eval_samples2.sort()
    gmm = GMM([1/3,2/3], [-2., 2.], [1., 1.])
    fig, ax = plt.subplots()
    ax.set_xlabel("x", fontsize=30)
    ax.xaxis.set_tick_params(labelsize=30)
    ax.yaxis.set_tick_params(labelsize=30)
    ax.hist(eval_samples1, bins=128, lw=3.0, histtype="step", density=True, label="param trans, iter = 500")
    ax.hist(eval_samples2, bins=128, lw=3.0, histtype="step", density=True, label="param trans, iter = 10")
    ax.plot(eval_samples1, np.exp(gmm.logpdf(eval_samples1)), lw=3.0, label=R"target")
    ax.plot(eval_samples2, np.exp(gmm.logpdf(eval_samples2)), lw=3.0, label=R"target")
    ax.legend(fontsize=24,framealpha=0,loc='upper left')
    plt.show()
