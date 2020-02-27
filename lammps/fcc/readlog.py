import lammps_logfile
import matplotlib.pyplot as plt

log = lammps_logfile.File('log.lammps')
T = log.get('Step')
# TotEng = log.get('TotEng')
E_pair = log.get('E_pair')

plt.plot(T, E_pair)
plt.show()
