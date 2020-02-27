from ovito.io import import_file, export_file

if __name__=='__main__':
    path400 = '/home/kristtuv/Downloads/polycrystalline_shear_snapshots/polycrystal_L400.0/continuation_step_1000000_T_278.15/continuation_stress_calculation_max_with_deform_temp/restart_step_1560000.data'
    path200 = '/home/kristtuv/Downloads/polycrystalline_shear_snapshots/polycrystal_L200.0/continuation_step_1000000_T_263.15/continuation_stress_calculation_max_with_deform_temp/restart_step_1560000.data'
    path100 = '/home/kristtuv/Downloads/polycrystalline_shear_snapshots/polycrystal_L100.0/continuation_step_1000000_T_263.15/continuation_stress_calculation_max_with_deform_temp/restart_step_1530000.data'
    paths = {'path400': path400, 'path200': path200, 'path100': path100}
    for name, path in paths.items():
        pipe = import_file(path)
        export_file(pipe, f'xyzfiles/{name}.xyz', 'xyz', columns = 
                [ "Particle Type", "Position.X", "Position.Y", "Position.Z"])
    # # Classifier(path)
