from neurodiffeq import diff
from neurodiffeq.solvers import Solver1D, Solver2D
from neurodiffeq.conditions import IVP, DirichletBVP2D
from neurodiffeq.networks import FCNN, SinActv
import torch
import numpy as np
from matplotlib.pyplot import *
def target_score(x):
    return [x**2, x**3]
def d_target_score(x):
    return array([2*x, 0, 0, 3*x**2]).reshape(2, 2)
def pde_x(u, x, y):
    return [diff(u[0], x, order=2) + diff(u[0], y, order=1)]
def pde_y(u, x, y):
    return [diff(u[0], x, order=1) + diff(u[0], y, order=2)]
conditions = [
    DirichletBVP2D(
        x_min=0, x_min_val=lambda y: [torch.sin(np.pi*y), 0],
        x_max=1, x_max_val=lambda y: [0,0],                   
        y_min=0, y_min_val=lambda x: [0,0],                   
        y_max=1, y_max_val=lambda x: [0,0]                   
    )
]
nets = [FCNN(n_input_units=2, n_output_units=2, hidden_units=(512,))]

solver = Solver2D(pde_system, conditions, xy_min=(0, 0), xy_max=(1, 1), nets=nets)
solver.fit(max_epochs=2000)
solution = solver.get_solution()
x, y = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
u = solution(x, y, to_numpy=True)
"""
fig, ax = subplots()
sol = ax.contourf(x, y, u)
cbar = fig.colorbar(sol)
cbar.ax.set_ylabel('$u(x, y)$',fontsize=20)
cbar.ax.tick_params(labelsize=20)
ax.set_xlabel('x',fontsize=20)
ax.set_ylabel('y',fontsize=20)
ax.set_title('Solution $u(x, y)$',fontsize=20)
ax.xaxis.set_tick_params(labelsize=20)
ax.yaxis.set_tick_params(labelsize=20)
fig.tight_layout()
show()
"""
