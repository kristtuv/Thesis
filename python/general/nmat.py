import math
import hoomd
import hoomd.md
import hoomd.deprecated
import numpy as np

# Define the OPP
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

def run(T_start, T_stop, timeSteps, potential_k, potential_phi):
    hoomd.context.initialize("--gpu 1")
    system = hoomd.deprecated.init.create_random(N = 4096, phi_p = 0.03)
    nl = hoomd.md.nlist.cell()
    all = hoomd.group.all()

    range = determineRange(potential_k, potential_phi)
    print('===================================================================')
    print(range)
    print('===================================================================')
    table = hoomd.md.pair.table(width=1000, nlist=nl)
    table.pair_coeff.set('A', 'A', func = OPP, rmin = 0.5, rmax = range,
                        coeff = dict(k = potential_k, phi = potential_phi))

    filename = "400quasi/quasicrystal_k" + str(potential_k) + "_phi" + str(potential_phi)
    filename += "_T" + str(T_start) + '-' + str(T_stop)
    print(filename)

    hoomd.dump.gsd(filename, period=timeSteps*1e-3, group=all, overwrite=True)
    logger = hoomd.analyze.log(filename = filename + ".log", period = timeSteps * 1e-4,
        quantities = ['time','potential_energy','temperature', 'pressure'], overwrite=True)

    # Integrate at constant temperature
    temperature_ramp = hoomd.variant.linear_interp([(0, T_start), (timeSteps, T_stop)])
    hoomd.md.integrate.nvt(group = all,  kT = temperature_ramp , tau = 1.0,)
    hoomd.md.integrate.mode_standard(dt = 0.01)
    hoomd.run(timeSteps + 1)

if __name__=='__main__':
    T_start = 0.4 
    T_stop = 0.1
    from itertools import product
    k = np.linspace(5.8, 9.5, 20).round(3)
    phi = np.linspace(0.38, 0.8, 20).round(3)
    potentials = product(k, phi)
    # potential_k = np.linspace(7.2, 9.2, 10).round(3)
    timeSteps = 7e7
    # print(list(potentials)[183:])
    # print(list(reversed(list(potentials)))[64:])
    # exit()
    # for p in list(reversed(list(potentials)))[64:]:
    for p in list(potentials)[184:]:
        print('====================================')
        print('Running potential: ', p)
        print('====================================')
        try:
            run(T_start, T_stop, timeSteps, *p)
        except RuntimeError as e:
            print('========================================================')
            print(e)
            print('========================================================')
