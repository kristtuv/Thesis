import numpy as np
from os import listdir
from functools import singledispatch
import os
from ovito.io import import_file, export_file
from ovito.vis import Viewport, ParticlesVis
from ovito.modifiers import ColorCodingModifier
from ovito.pipeline import Pipeline, StaticSource
import re
from collections import Counter


gauss = 'gauss/'
optics = 'optics/'
agglomerative = 'agglomerative/'
dbscan = 'dbscan/'
filetype = 'restart'
files_gauss = [gauss+f for f in listdir(gauss) if f.endswith('evaluation.npy') and f.startswith(filetype)]
files_optics = [optics+f for f in listdir(optics) if f.endswith('evaluation.npy') and f.startswith(filetype)]
files_agglomerative = [agglomerative+f for f in listdir(agglomerative) if f.endswith('evaluation.npy') and f.startswith(filetype)]
files_dbscan = [dbscan+f for f in listdir(dbscan) if f.endswith('evaluation.npy') and f.startswith(filetype)]

def get_files(file_names, evaluation='calinski', reduction='pca'):
    if evaluation == 'calinski':
        idx = 0
        reverse = True
    elif evaluation == 'silhouette':
        idx = 1
        reverse = True
    elif evaluation == 'davies':
        idx = 2
        reverse = False
    eval_criteria = []
    for f in file_names:
        if reduction in f or reduction == 'both':
            try:
                a = np.load(f, allow_pickle=True).item()
            except ValueError:
                a = np.load(f, allow_pickle=True)
                a[0]['name'] = f
                eval_criteria.append((a[0], a[1]))
    return sorted(eval_criteria, key=lambda x: list(x[0].values())[idx], reverse=reverse)

def map_labels(cluster):
    counter = Counter(cluster)
    cc = []
    try:
        counter.pop(-1)
    except KeyError as e:
        e
    c = [i[0] for i in sorted(counter.most_common(), key=lambda kv: (kv[1], kv[0]), reverse=True)]
    for i in range(len(counter)):
        cc.append((c[i], float(len(counter)-i-1)))
    cc.append((-1, -1))
    cc = dict(cc)
    return cc


def map_predictions(predictions):
    counter = Counter(predictions)
    mapping = map_labels(counter)
    new_predictions = []
    for pred in predictions:
        new_predictions.append(mapping[pred])
    return new_predictions


def do_render(pipe, data, start_value, end_value, d_name):
    dump_name = d_name + f'_startvalue{int(start_value)}.png'
    pipe.add_to_scene()
    pipe.modifiers.append(
            ColorCodingModifier(
                property = 'sortedcluster',
                gradient=ColorCodingModifier.Rainbow(),
                start_value=float(start_value),
                end_value=float(end_value)
            ))
    vis_element = pipe.source.data.particles.vis
    vis_element.radius = 0.45
    data.cell.vis.enabled = False
    vp = Viewport()
    # vp.type = Viewport.Type.Back
    vp.type = Viewport.Type.Ortho
    # vp.zoom_all()
    vp.camera_pos = (56.314, 49.9724, 50.0708)
    vp.camera_dir = (-0.00236647, -0.999994, -0.00235447)
    vp.fov = 54
    vp.render_image(size=(800,600), filename=dump_name, background=(0,0,0), frame=0, alpha=True)
    pipe.remove_from_scene()

def render(filename, dump_extention=''):
    dump_path = '/home/kristtuv/Documents/master/latex/plots/clustering_appendix/'
    # dump_path = 'blah/'
    dump_name = filename.split('/')[0]
    # dump_name = dump_path +'best_'+ dump_name + dump_extention# + '.png'
    d_name = dump_path +'best_'+ dump_name + dump_extention# + '.png'
    filename = filename.replace('_evaluation.npy', '')

    pipeline = import_file(filename)
    data = pipeline.compute()
    cluster = data.particles_['cluster'][...]
    new_cluster = map_predictions(cluster)
    start_value = min(new_cluster)
    # dd_name = 'mapped/'+ dump_name + f'_startvalue{int(start_value)}'
    dd_name = 'mapped/best_'+ dump_name + dump_extention + f'_startvalue{int(start_value)}'
    data.particles_.create_property('sortedcluster', data=new_cluster)
    new_pipe = Pipeline(source = StaticSource(data = data))
    export_file(new_pipe, f'{dd_name}.dump', 'lammps/dump', columns=['Particle Identifier', 'Particle Type', 'Position.X', 'Position.Y', 'Position.Z', 'sortedcluster'])

    do_render(new_pipe, data, min(new_cluster), max(new_cluster), d_name)


idx = 1
evaluations = ['calinski', 'davies', 'silhouette']
# evaluations = ['calinski', 'silhouette']
# evaluations = ['silhouette']
reductions = ['encoded', 'pca']
# reductions = ['encoded', ]
all_files = [files_gauss, files_agglomerative, files_optics, files_dbscan]
# all_files = [files_gauss]
# all_files = [files_agglomerative]
# all_files = [files_optics]
# all_files = [files_dbscan]
for files in all_files:
    begintable = r'\begin{table}[!htbp]' + ' \n'
    endtable = r'\end{table}' + ' \n'
    minipage = ''
    for reduction in reductions:
        if reduction == 'encoded':
            caption = r'\caption{Autoencoder}' + ' \n'
        if reduction == 'pca':
            caption = r'\caption{PCA}' + ' \n'
        adjustbox =  r'\begin{adjustbox}{width=0.90\textwidth, height=0.2\textwidth}'  + ' \n'
        endadjustbox = r'\end{adjustbox}' + '\\\\ \n'
        beginminipage = r'\begin{minipage}{0.50\linewidth}' + ' \n'
        endminipage = r'\end{minipage}' + ' \n'
        begintabular = r'\begin{tabular}{cc|cc}' + ' \n'
        toprule = r'\toprule' + ' \n'
        midrule = r'\midrule' + ' \n'
        bottomrule = r'\bottomrule' + ' \n'
        endtabular = r'\end{tabular}' + ' \n'
        tabular = ''

        for evaluation in evaluations:
            dump_extention  = '_' + evaluation + '_' + reduction
            best = get_files(files, evaluation=evaluation, reduction=reduction)[0:idx]
            name = best[0][0]['name']
            method = name.split('/')[0]
            neighbors = re.search('neighbors(\d\d)|nneigh(\d\d)', name)
            neighbors = list(filter(None, neighbors.groups()))[0]
            minpts = re.search('sample(\d+)', name)
            eps = re.search('eps(\d+\.?(\d+)?)', name)
            clusters = str(len(best[0][1]))
            c_score = str(round(best[0][0]['calinski'], 2))
            s_score = str(round(best[0][0]['silhouette'], 2))
            d_score = str(round(best[0][0]['davies'], 2))
            if evaluation == 'calinski':
                line1 = '\multicolumn{4}{c}{Sorted by best Calinski-Harbasz} \\\\ \n'
            if evaluation == 'davies':
                line1 = '\multicolumn{4}{c}{Sorted by best Davies-Boulding} \\\\ \n'
            if evaluation == 'silhouette':
                line1 = '\multicolumn{4}{c}{Sorted by best Silhouette} \\\\ \n'
            line2 = '\multicolumn{2}{c}{Score} & \multicolumn{2}{c}{Parameters} \\\\ \n'
            line3 = f'Calinski & {c_score} & Neighbors & {neighbors} \\\\ \n'
            if minpts:
                m = minpts.group(1)
                e = eps.group(1)
                line4 = f'Silhouette & {s_score} & $MinPts$ & {m} \\\\ \n'
                line5 = f'Davies & {d_score} & $\\varepsilon$ & {e} \\\\ \n'
            if not minpts:
                line4 = f'Silhouette & {s_score} & $MinPts$ & - \\\\ \n'
                line5 = f'Davies & {d_score} & $\\varepsilon$ & - \\\\ \n'
            render(name, dump_extention)

            tabular += (
                    adjustbox
                    +begintabular
                    +toprule
                    +line1
                    +midrule
                    +line2
                    +midrule
                    +line3
                    +line4
                    +line5
                    +bottomrule
                    +endtabular
                    +endadjustbox
                    +' \n'
                    )
        minipage += (
                beginminipage 
                +caption
                +tabular
                +endminipage
                +' \n'
                )
    table = begintable + minipage + endtable
    
    print(table)
            # print(line5)

            # line0 = '\multicolumn{2}{c}{Sorted by ' + f'{2}' +'}'
            # line1 = '\multicolumn{2}{c}{Score}'
            # line2 = 'Calinski'
            # line3 = 'Silhouette'
            # line4 = 'Davies'
            # line5 = '\multicolumn{2}{c}{Parameters}'
            # line6 = 'Neighbors'
            # line7 = 'Clusters'
            # line8 = ''
            # line9 = ''

            # line2+= ' & ' + c_score 
            # line3+= ' & ' + s_score 
            # line4+= ' & ' + d_score 
            # line6+= ' & ' + neighbors 
            # line7+= ' & ' + clusters 

            # if minpts:
            #     m = minpts.group(1)
            #     e = eps.group(1)
            #     line8 += ' & ' + m
            #     line9 += ' & ' + e
        # line1 += '\\\\ \n'
        # line2 += '\\\\ \n'
        # line3 += '\\\\ \n'
        # line4 += '\\\\ \n'
        # line5 += '\\\\ \n'
        # line6 += '\\\\ \n'
        # if line8:
            # line8 = '$MinPts$' + line7 + '\\\\ \n'
            # line9 = '$\\varepsilon$' + line8 + '\\\\ \n'
        # lines = line1 + line2 + line3 + line4 + line5 + line6 + line7 + line8
        # print(lines)
        # exit()


        # print(method)
        # print(evaluation)
        # print(lines)
        # print()
exit()

best_gauss = get_files(files_gauss, evaluation=evaluation, reduction=reduction)[0:idx]
name = best_gauss[0][0]['name']
render(name, dump_extention)
best_optics = get_files(files_optics)[0:idx]
best_agglomerative = get_files(files_agglomerative)[0:idx]
best_dbscan = get_files(files_dbscan)[0:idx]
print(*best_gauss, sep='\n')
print()
print(*best_optics, sep='\n')
print()
print(*best_agglomerative, sep='\n')
print()
print(*best_dbscan, sep='\n')
print()
# os.system('ovito '+str(best_optics[2][0]['name']).replace('_evaluation.npy', ''))



