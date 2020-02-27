import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import newton, minimize
from itertools import product


def func(k, phi, r, n):
    i = np.where(r > (2*np.pi*n - phi)/k + 1.25)
    v = 1/r**15 + (1/r**3)*np.cos(k*(r - 1.25) - phi)
    v[i] = 0
    return v
def OPP(r, rmin, rmax, k, phi):
    cos = math.cos(k * (r - 1.25) - phi)
    sin = math.sin(k * (r - 1.25) - phi)
    V = pow(r, -15) + cos * pow(r, -3)
    F = 15.0 * pow(r, -16) + 3.0 * cos * pow(r, -4) + k * sin * pow(r, -3)
    return (V, F)

# Determine the potential range by searching for extrema
def determineRange(k, phi):
    r = 0.5
    extremaNum = 0
    force1 = OPP(r, 0, 0, k, phi)[1]
    while (extremaNum < 6 and r < 5.0):
        r += 1e-5
        force2 = OPP(r, 0, 0, k, phi)[1]
        if (force1 * force2 < 0.0):
            extremaNum += 1
            force1 = force2
    return r

def f(r, k, phi):
    cos = np.cos(k * (r - 1.25) - phi)
    sin = np.sin(k * (r - 1.25) - phi)
    F = 15.0 * pow(r, -16) + 3.0 * cos * pow(r, -4) + k * sin * pow(r, -3)
    return F

def v(r, k, phi):
    cos = np.cos(k * (r - 1.25) - phi)
    sin = np.sin(k * (r - 1.25) - phi)
    v_ = 1/r**15 + (1/r**3)*np.cos(k*(r - 1.25) - phi)
    return v_


def find_r1_r2(f, k, phi):
    r = np.linspace(0.5, 10, 1000)
    # if k < 1: k=1
    blah  = f(r, k, phi)
    blahblah = blah[1:]*blah[:-1]
    idx = np.argwhere(blahblah <= 0)[[0,2]]
    zero = newton(f, r[idx[0]], args=(k, phi))
    zero_2 = newton(f, r[idx[1]], args=(k, phi))
    return np.array([zero, zero_2 ])

def cost(x, rs):
    r_guess = find_r1_r2(f, *x)
    r = r_guess[0]/r_guess[1] - rs[0]/rs[1]
    return r**2


    
def find_k_phi(x0, rs):
    mini = minimize(cost, x0, args=(rs), method='CG')# options={'maxiter': 1000})
    print(mini)
    return mini.x 


# r_1 = 2.8
# factor = r_1# /1.2
# r_2 = 4.2
# rs = np.array([r_1, r_2])
# k, phi = find_k_phi(np.array([7.5, 0.6]), rs)
# print(k, phi)

# r_guess = find_r1_r2(f, k, phi)
# factor = r_guess[0]/rs[0]
# print(factor)
# rs = rs*factor
# print(rs)
# plt.plot(rs, v(rs, k, phi), 'bo', label='incoming')
r = np.linspace(0.95, 4, 400)
ks = [6, 7.5, 9.5]
phis = [0.4, 0.6, 0.8]
print(list(product(ks, phis)))
for k, phi in list(product(ks, phis)):
    plt.plot(r, v(r, k, phi), label=rf'k:{k}, $\phi$:{phi}')
# plt.plot(r_guess, v(r_guess, k, phi), 'ro', label='estimated')
    plt.legend()
plt.xlabel(r'$r$', size=16)
plt.ylabel(r'$U(r)$', size=16)
plt.tight_layout()
plt.savefig('../../latex/plots/oppselected.png', transparent=True)
plt.show()

#     # if 
#     #     return 0
#     # else:
#     #     return 1/r**15 + (1/r**3)*np.cos(k*(r - 1.25) - phi)



# # v = func(8, 0.4, r, 3)
# # plt.plot(r, v)
# v = func(3.8, 0.8, r, 3)
# plt.plot(r, v)
# # plt.plot(r, v(8, 0.6))
# plt.show()
