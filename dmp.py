import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import copy

class DMP:
    def __init__(self, config_init, theta_arr_lst = None): 
        """
        config in C-space
        """
        n_dim = config.size
        sim_lst = []
        theta_arr_lst = [None for i in range(n_dim)] if theta_arr_lst is None else theta_arr_lst
        for n in range(n_dim):
            state_init = np.array([1.0, config_init[n], 0.0])
            dyn = DynamicalSystem(state_init, theta_arr = theta_arr_lst[n])
            sim = DynamicalSimu(dyn)
            sim_lst.append(sim)

        self.config = config_init
        self.sim_lst = sim_lst

    def propagate(self, dt):
        [sim.propagate(dt) for sim in self.sim_lst]
        self.config = np.array([sim.state[1] for sim in self.sim_lst])

class DynamicalSystem:
    def __init__(self, state_init, N_basis = 10, a_x = 1.0, theta_arr = None):
        self.state_init = copy.deepcopy(state_init)
        self.N_basis = N_basis
        self.a_x = a_x
        self.theta_arr = np.zeros(N_basis) if theta_arr is None else theta_arr
        t_arr = np.linspace(0, 1, N_basis)
        self.c_arr = np.exp(-a_x * t_arr)

    def ds_dt(self, state):
        a_x = self.a_x
        a = 25
        b = 6
        x, y, z = state
        dx_dt = - a_x * x
        dy_dt = z 
        dz_dt = a * (b * (-y) - z) + self._force(x) 
        return np.array([dx_dt, dy_dt, dz_dt])

    def _force(self, x):
        h = 0.1
        w_arr = np.exp(-0.5 * h * (x - self.c_arr)**2)
        w_sum = np.sum(w_arr)
        y_diff = self.state_init[1]
        f = np.dot(self.theta_arr, w_arr) * x * y_diff / w_sum
        return f

class DynamicalSimu:
    def __init__(self, dyn):
        self.dyn = dyn
        self.state = copy.deepcopy(dyn.state_init)

    def propagate(self, dt):
        ds_dt = self.dyn.ds_dt(self.state)
        self.state = self.state + ds_dt * dt


if __name__=="__main__":
    theta_arr1 = np.ones(10) * 0
    theta_arr2 = np.ones(10) * 100.0
    theta_arr_lst = [theta_arr1, theta_arr2]
    config = np.array([1.0, 1.0])
    dmp = DMP(config, theta_arr_lst)
    dt = 0.001
    C_ = []
    for i in range(1000):
        dmp.propagate(dt)
        C_.append(dmp.config)
    C = np.vstack(C_)
    plt.plot(C[:, 0], C[:, 1])
    plt.show()

