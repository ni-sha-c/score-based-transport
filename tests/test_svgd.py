# Modified by Evan Montoya
import numpy as np
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text',usetex=True)
import seaborn as sns
sns.set_style("whitegrid")
class SVGD:
    def __init__(self):
        pass

    def svgd_kernel(self, theta, h=-1):
        # Pairwise euclidean distances between observations in n-dimensional space.
        sq_dist = pdist(theta)

        #Convert a vector-form distance vector to a square-form distance matrix, and vice-versa.
        pairwise_dists = squareform(sq_dist)**2

        if h < 0:  # if h < 0, using median trick
            h = np.median(pairwise_dists)
            h = np.sqrt(0.5 * h / np.log(theta.shape[0] + 1))

        # compute the rbf kernel
        Kxy = np.exp(-pairwise_dists / h**2 / 2)
        dxkxy = -np.matmul(Kxy, theta)
        sumkxy = np.sum(Kxy, axis=1)

        for i in range(theta.shape[1]):
            dxkxy[:, i] = dxkxy[:, i] + np.multiply(theta[:, i], sumkxy)
        dxkxy = dxkxy / (h**2)

        return Kxy, dxkxy

    def update(self, x0, lnprob, n_iter=1000, stepsize=1e-3, bandwidth=-1, alpha=1.0, debug=True):
        # Check input
        if x0 is None or lnprob is None:
            raise ValueError('x0 or lnprob cannot be None!')

        theta = np.copy(x0)

        # adagrad with momentum
        fudge_factor = 1e-6
        historical_grad = 0
        for iteration in range(n_iter):
            if debug and (iteration + 1) % 1000 == 0:
                print('iter ' + str(iteration + 1))
                print(theta.shape)
            lnpgrad = lnprob(theta)
            # calculating the kernel matrix
            kxy, dxkxy = self.svgd_kernel(theta, h=-1)
            grad_theta = (np.matmul(kxy, lnpgrad) + dxkxy) / x0.shape[0]


            # adagrad
            if iteration == 0:
                historical_grad = historical_grad + grad_theta ** 2
            else:
                historical_grad = alpha * historical_grad + (1 - alpha) * (grad_theta ** 2)
            adj_grad = np.divide(grad_theta, fudge_factor + np.sqrt(historical_grad))
            theta = theta + stepsize * adj_grad

        return theta

# Parameters for the bimodal Gaussian distribution
mu1 = -2
sigma1 = 1
w1 = 1/3
mu2 = 2
sigma2 = 1
w2 = 2/3

# Define the target distribution
def bimodal_p(x):
    pdf1 = w1 * (1.0 / (sigma1 * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu1) / sigma1)**2)
    pdf2 = w2 * (1.0 / (sigma2 * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu2) / sigma2)**2)
    return pdf1 + pdf2

# Define the gradient of the logarithm of the target distribution
def gradient_log_bimodal_p(x):
    t1 = w1/(sigma1 * np.sqrt(2 * np.pi)) * np.exp(-1/2 * ((x-mu1) / sigma1) ** 2) * -((x-mu1)/(sigma1**2))
    t2 = w2/(sigma2 * np.sqrt(2 * np.pi)) * np.exp(-1/2 * ((x-mu2) / sigma2) ** 2) * -((x-mu2)/(sigma2**2))
    t3 = (1 / (w1 * (1.0 / (sigma1 * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu1) / sigma1)**2) + w2 * (1.0 / (sigma2 * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu2) / sigma2)**2)))
    grad = t3 * (t1 + t2)
    return grad

# Initial samples
x0 = np.random.normal(0, 1, [512, 1])

# Use SVGD to update the samples
theta1 = SVGD().update(x0, gradient_log_bimodal_p, n_iter=5, stepsize=0.01)
theta2 = SVGD().update(x0, gradient_log_bimodal_p, n_iter=500, stepsize=0.01)
theta3 = SVGD().update(x0, gradient_log_bimodal_p, n_iter=1000, stepsize=0.01)
# Sanity Checking
print("mu1: ", mu1)
print("mu2: ", mu2)
print("bimodal mean", np.mean(theta1))
print("bimodal mean", np.mean(theta2))
print("bimodal mean", np.mean(theta3))
theta1 = theta1.reshape(-1)
theta2 = theta2.reshape(-1)
theta3 = theta3.reshape(-1)
# Plotting

x_range = np.linspace(-7, 7, 500)
true_pdf = bimodal_p(x_range)
fig, ax = plt.subplots()
ax.set_xlabel("x", fontsize=30)
ax.xaxis.set_tick_params(labelsize=30)
ax.yaxis.set_tick_params(labelsize=30)
ax.grid(True)

colors = sns.color_palette("Set2", 10)
sns.histplot(theta1, ax=ax, color=colors[0], kde=True, fill=False, stat='density', bins=35, element="step", kde_kws={'bw_adjust': 0.5}, line_kws = {'linewidth': 3}, label='SVGD, iter=5')

sns.histplot(theta2, ax=ax, color=colors[1], kde=True, fill=False, stat='density', bins=35, element="step", kde_kws={'bw_adjust':0.5}, line_kws = {'linewidth': 3}, label='SVGD, iter=500')

sns.histplot(theta3, ax=ax, color=colors[2], kde=True, fill=False, stat='density', bins=35, element="step", kde_kws={'bw_adjust':0.5}, line_kws = {'linewidth': 3}, label='SVGD, iter=1000')

ax.plot(x_range, true_pdf, label='target', color='red', lw=3)
ax.set_ylabel(" ")
# Add labels and  legend
ax.legend(fontsize=24)
plt.tight_layout()
plt.show()

