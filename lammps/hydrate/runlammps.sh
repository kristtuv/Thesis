# for i in 10 100 250 270 290 320 350 370 400
export OMP_NUM_THREADS=20
for i in 250
do
    mpirun lmp < shear_partial_stressmeasure.in -v restartFile data/methane_hydrate_SI_UA_charge.data -v temperature $i;
    # mpirun lmp < hex_ice.in -v restartFile data/ice_ih.data -v temperature $i;
done
