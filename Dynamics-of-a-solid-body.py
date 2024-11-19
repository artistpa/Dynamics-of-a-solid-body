import sympy as sm
import scipy as sp
import matplotlib.pyplot as plt
import numpy as np

#creating Solid_body
class Solid_body:
    def __init__(self, I, omega0, theta):
        self.I = I
        self.omega0 = omega0
        self.theta = theta

    def __get_main_I__(self):
        mainI, main_axes = np.linalg.eig(self.I)
        return np.diag(mainI)

    def __get_main_axes__(self):
        mainI, main_axes = np.linalg.eig(self.I)
        return main_axes

    def __get_S_inv__(self):
        return sp.linalg.inv(self.__get_main_axes__())

#definition of the external force
class External_force:
    def __init__(self, F, Rc):
        self.F = F
        self.Rc = Rc
    def __get_M__(self):
        return np.cross(self.Rc, self.F)

#main part of the project
I = np.array([[1., 1., 1.],
              [4., 2., 5.],
              [2., 8., 7.]], dtype=np.float64)
omega0 = np.array([0., 2., 1.], dtype= np.float64)
theta = np.pi / 6
a = Solid_body(I, omega0, theta)
first_eq = sp.Eq()
print(a.__get_main_I__())
print(a.__get_S_inv__() @ a.I @ a.__get_main_axes__())
print(a.__get_main_axes__())

