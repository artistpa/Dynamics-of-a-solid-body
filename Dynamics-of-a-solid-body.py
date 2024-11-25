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

    def __set_omega__(self, omega):
        self.omega = omega

#definition of the quaternion
class Quaternion:
    def __init__(self, angle, axis):
        self.l0 = np.cos(angle/2)
        self.l1 = axis[0] * np.sin(angle/2)
        self.l2 = axis[1] * np.sin(angle/2)
        self.l3 = axis[2] * np.sin(angle/2)
    def __sum__(self, Quat):
        q = Quaternion(0,[0, 0, 0])
        q.l0 = self.l0 + Quat.l0
        q.l1 = self.l1 + Quat.l1
        q.l2 = self.l2 + Quat.l2
        q.l3 = self.l3 + Quat.l3

#definition of the external forces
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
#print(a.__get_main_I__())
#print(a.__get_S_inv__() @ a.I @ a.__get_main_axes__())
#print(a.__get_main_axes__())
A = a.__get_main_I__()[0][0]
B = a.__get_main_I__()[1][1]
C = a.__get_main_I__()[2][2]
#print(A, B, C)
angle0 = 0
axis0 = np.array([1, 0, 0])
L0 = Quaternion(angle0, axis0)
F = np.array([0, 10, 0])
Rc = np.array([1,1,1])
Fex = External_force(F, Rc)
Mex = Fex.__get_M__()
omega = np.array([0,0,1])
a.__set_omega__(omega)
p = a.omega[0]
q = a.omega[1]
r = a.omega[2]
dpdt =(Mex[0] + (B - C) * q * r) / A
dqdt =(Mex[1] + (C - A) * p * r) / B
drdt =(Mex[2] + (A - B) * q * p) / C