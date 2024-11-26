import sympy as sm
import scipy as sp
import matplotlib.pyplot as plt
import numpy as np

#creating Solid_body
class Solid_body:
    def __init__(self, I, omega, theta):
        self.I = I
        self.omega = omega
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
        self.l0 = 0
        self.l = np.array([0, 0, 0])
        self.l0 = np.cos(angle/2)
        self.l[0] = axis[0] * np.sin(angle/2)
        self.l[1] = axis[1] * np.sin(angle/2)
        self.l[2] = axis[2] * np.sin(angle/2)
    def __sum__(self, Quat):
        q = Quaternion(0,[0, 0, 0])
        q.l0 = self.l0 + Quat.l0
        q.l = self.l + Quat.l
        return q
    def __mult_on_scal__(self, a):
        self.l0 *= a
        self.l *= a

    def __mult_on_quat__(self, Quat):
        q = Quaternion(0, [0, 0, 0])
        q.l0 = self.l0 *  Quat.l0 - np.dot(q.l, self.l)
        q.l = self.l0 * Quat.l + Quat.l0 * self.l + np.cross(self.l, Quat.l)
        return q

#definition of the external forces
class External_force:
    def __init__(self, F, Rc):
        self.F = F
        self.Rc = Rc
    def __get_M__(self):
        return np.cross(self.Rc, self.F)

#derivative dU/dt
def g(t, U):
    dl0dt = -0.5 * (L0.l[0] * p + L0.l[1] * q + L0.l[2] * r)
    dl1dt = (L0.l0 * p + L0.l[1] * r - L0.l[2] * q)
    dl2dt = (L0.l0 * q + L0.l[2] * p - L0.l[0] * r)
    dl3dt = (L0.l0 * r + L0.l[0] * q - L0.l[1] * p)
    dpdt = (Mex[0] + (B - C) * q * r) / A
    dqdt = (Mex[1] + (C - A) * p * r) / B
    drdt = (Mex[2] + (A - B) * q * p) / C
    return np.array([dl0dt, dl1dt, dl2dt, dl3dt, dpdt, dqdt, drdt])

#main part of the project
I = np.array([[1., 1., 1.],
              [4., 2., 5.],
              [2., 8., 7.]], dtype=np.float64)
omega0 = np.array([3., 2., 1.], dtype= np.float64) #in the basis associated with the body
theta = np.pi / 6
body = Solid_body(I, omega0, theta)
#print(a.__get_main_I__())
#print(a.__get_S_inv__() @ a.I @ a.__get_main_axes__())
#print(a.__get_main_axes__())
A = body.__get_main_I__()[0][0]
B = body.__get_main_I__()[1][1]
C = body.__get_main_I__()[2][2]
p = body.omega[0]
q = body.omega[1]
r = body.omega[2]
#print(A, B, C)
angle0 = 0
axis0 = np.array([1, 0, 0])
L0 = Quaternion(angle0, axis0)
F = np.array([0, 10, 0])
Rc = np.array([1,1,1])
Fex = External_force(F, Rc)
Mex = Fex.__get_M__()
dpdt =(Mex[0] + (B - C) * q * r) / A
dqdt =(Mex[1] + (C - A) * p * r) / B
drdt =(Mex[2] + (A - B) * q * p) / C

T = 10
U0 = np.array([L0.l0, L0.l[0], L0.l[1], L0.l[2], body.omega[0], body.omega[1], body.omega[2]])
solution = sp.integrate.solve_ivp(g, (0.0, T), y0=U0, t_eval=np.linspace(0.0, T, 101))