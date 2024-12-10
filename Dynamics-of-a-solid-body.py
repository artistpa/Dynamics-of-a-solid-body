import sympy as sm
import scipy as sp
import matplotlib.pyplot as plt
import numpy as np

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

    def __get_сonjugate_quaternion__(self):
        q = Quaternion(0,[0, 0, 0])
        q.l0 = self.l0
        q.l = -self.l
        return q

    def __normalization__(self):
        l01 = self.l0 / np.sqrt(self.l0 ** 2 + self.l[0] ** 2 + self.l[1] ** 2 + self.l[2] ** 2)
        l1 = self.l / np.sqrt(self.l0 ** 2 + self.l[0] ** 2 + self.l[1] ** 2 + self.l[2] ** 2)
        self.l0 = l01
        self.l = l1

#creating Solid_body
class Solid_body:
    def __init__(self, I, omega, R0):
        self.I = I
        self.omega = omega
        self.R0 = R0

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

    def get_coordinates(self, q):
        q0 = Quaternion(np.pi, R0)
        q1 = q.__get_сonjugate_quaternion__()
        res = (q.__mult_on_quat__(q0)).__mult_on_quat__(q1)
        return np.array([res.l[0], res.l[1], res.l[2]])

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
I = np.array([[10., 0., 0.],
              [0., 10., 0.],
              [0., 0., 10.]], dtype=np.float64)
omega0 = np.array([1, 1, 1], dtype= np.float64) #in the basis associated with the body
R0 = np.array([0, 0, 1]) #the initial coordinates of the center of mass in the basis associated with the body
body = Solid_body(I, omega0, R0)
print(body.__get_main_I__())
print(body.__get_S_inv__() @ body.I @ body.__get_main_axes__())
print(body.__get_main_axes__())
A = body.__get_main_I__()[0][0]
B = body.__get_main_I__()[1][1]
C = body.__get_main_I__()[2][2]
p = body.omega[0]
q = body.omega[1]
r = body.omega[2]
angle0 = np.pi
axis0 = np.array([0, 0, 1])
L0 = Quaternion(angle0, axis0)
F = np.array([0, 0, 0])
Rc = np.array([0, 0, 0])
Fex = External_force(F, Rc)
Mex = Fex.__get_M__()
dpdt =(Mex[0] + (B - C) * q * r) / A
dqdt =(Mex[1] + (C - A) * p * r) / B
drdt =(Mex[2] + (A - B) * q * p) / C

T = 1000
U0 = np.array([L0.l0, L0.l[0], L0.l[1], L0.l[2], body.omega[0], body.omega[1], body.omega[2]])
solution = sp.integrate.solve_ivp(g, (0.0, T), y0=U0, t_eval=np.linspace(0.0, T, 10000))
#print(solution.y)

Quats = []

for i in range(len(solution.y[0])):
    q = Quaternion(0, [0,0,0])
    q.l0 = solution.y[0,i]
    q.l = np.array([solution.y[1,i], solution.y[2,i], solution.y[3,i]])
    q.__normalization__()
    Quats.append(q)

Coordinates_cm = np.array([body.get_coordinates(quat) for quat in Quats])

plt.plot(solution.t, solution.y[6,:], '-ro')
plt.grid()
plt.show()

plt.plot(Coordinates_cm[:,0], Coordinates_cm[:,1], '-ro')
plt.grid()
plt.show()