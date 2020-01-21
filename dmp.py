import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import copy

class DynamicalSystem:
    def __init__(self, state, N_basis = 10, a_x = 1.0):
        self.state_init = copy.deepcopy(state)
        self.state = copy.deepcopy(state)
        self.N_basis = N_basis
        self.a_x = a_x
        t_arr = np.linspace(0, 1, N_basis)
        self.c_arr = np.exp(-a_x * t_arr)

    def _time_derivative(self):
        a_x = self.a_x
        a = 25
        b = 6
        x, y, z = self.state
        dx_dt = - a_x * x
        dy_dt = z 
        dz_dt = a * (b * (-y) - z) + self._force(x) 
        return np.array([dx_dt, dy_dt, dz_dt])

    def _force(self, x):
        h = 0.1
        theta_arr = np.ones(self.N_basis) * 100
        w_arr = np.exp(-0.5 * h * (x - self.c_arr)**2)
        w_sum = np.sum(w_arr)
        y_diff = self.state_init[1]
        f = np.dot(theta_arr, w_arr) * x * y_diff / w_sum
        return f

    def propagate(self, dt):
        ds_dt = self._time_derivative()
        self.state = self.state + ds_dt * dt

if __name__=="__main__":
    sys = DynamicalSystem(np.array([1.0, 1.0, 0.0]))
    S_ = []
    for i in range(1000):
        sys.propagate(0.001)
        S_.append(sys.state)
    S = np.array(S_)
    plt.scatter(S[:, 1], S[:, 2])
    plt.show()


