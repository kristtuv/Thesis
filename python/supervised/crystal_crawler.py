#https://homepage.univie.ac.at/michael.leitner/lattice/spcgrp/index.html
import requests
from bs4 import BeautifulSoup
import re
import numpy as np

# tlds = [
#         'https://homepage.univie.ac.at/michael.leitner/lattice/spcgrp/triclinic.html',
#         'https://homepage.univie.ac.at/michael.leitner/lattice/spcgrp/monoclinic.html',
#         'https://homepage.univie.ac.at/michael.leitner/lattice/spcgrp/orthorhombic.html',
#         'https://homepage.univie.ac.at/michael.leitner/lattice/spcgrp/tetragonal.html',
#         'https://homepage.univie.ac.at/michael.leitner/lattice/spcgrp/trigonal.html',
#         'https://homepage.univie.ac.at/michael.leitner/lattice/spcgrp/cubic.html',
#         ]
tlds = [
        'https://homepage.univie.ac.at/michael.leitner/lattice/pearson/atype.html',
        'https://homepage.univie.ac.at/michael.leitner/lattice/pearson/mtype.html',
        'https://homepage.univie.ac.at/michael.leitner/lattice/pearson/otype.html',
        'https://homepage.univie.ac.at/michael.leitner/lattice/pearson/ttype.html',
        'https://homepage.univie.ac.at/michael.leitner/lattice/pearson/htype.html',
        'https://homepage.univie.ac.at/michael.leitner/lattice/pearson/ctype.html',
        ]
struk_domain = 'https://homepage.univie.ac.at/michael.leitner/lattice'

def get_strukt_links(url, domain):
    r = requests.get(url)
    html = r.text
    soup = BeautifulSoup(html, 'html.parser')
    all_href = list(soup.find_all("a"))
    find_struk = re.compile('href="\.\./struk/(.*)\.html')
    xyz_links = []
    names = []
    pearson_links = []
    for href in all_href:
        href = str(href)
        match = find_struk.search(href)
        if match:
            group = match.group(1)
            if not 'index' in group and not 'additions' in group:
                xyz_link = domain+'/struk.xmol/'+group+'.pos'
                name = group.rsplit('/')[-1]+'.xyz' 
                pearson_link = domain+'/struk/'+group+'.html'
                names.append(name)
                xyz_links.append(xyz_link)
                pearson_links.append(pearson_link)
    return names, xyz_links, pearson_links 

def get_all_struk_links(urls, domain):
    all_names = []
    all_xyz_links = []
    all_pearson_links = []
    for url in urls:
        names, xyz_links, pearson_links = get_strukt_links(url, domain)
        all_names.extend(names)
        all_xyz_links.extend(xyz_links)
        all_pearson_links.extend(pearson_links)
    return all_names, all_xyz_links, all_pearson_links


def write_xyz_files(dump_dir, names, xyz_links):
    strange = []
    for name, link in zip(names, xyz_links):
        print(name, link)
        r = requests.get(link)
        if r.status_code != 200:
            strange.append(link)
            continue
        else:
            text = r.text
            with open(dumpdir+name, 'w') as f:
                f.write(text)
    return strange


def get_pearson_symbols(names, links, dumpdir):
    pearson_dict = {}
    strange = []
    for name, link in zip(names, links):
        print(name, link)
        name = name.split('.')[0]
        r = requests.get(link)
        if r.status_code != 200:
            strange.append(link)
            continue
        else:
            text = r.text
            soup = BeautifulSoup(text, 'html.parser')
            all_href = list(soup.find_all("a"))
            find_struk = re.compile('<a href="../pearson/(?!index).*>(.*)</a>')
            for href in all_href:
                href = str(href)
                match = find_struk.search(href)
                if match:
                    group = match.group(1)
            if not group:
                exit()
            pearson_dict[name] = group

    np.save('pearson_symbols', pearson_dict)
    return strange


if __name__=='__main__':
    dumpdir = 'crystal_files_xyz/'
    names, xyz_links, pearson_links = get_all_struk_links(tlds, struk_domain)
    # names, pearson_links = ['hexdia.xyz'], ['https://homepage.univie.ac.at/michael.leitner/lattice/struk/hexdia.html' ]
    strange = get_pearson_symbols(names, pearson_links, dumpdir)
    print(*strange, sep='\n')
    print()
    strange = write_xyz_files(dumpdir, names, xyz_links)
    print(*strange, sep='\n')

