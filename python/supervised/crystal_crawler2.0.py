import requests
from bs4 import BeautifulSoup
import re
import numpy as np

tlds = [
'triclinic_pearson.html',
'monoclinic_pearson.html',
'orthorhombic_pearson.html',
'tetragonal_pearson.html',
'trig_hex_pearson.html',
'cubic_pearson.html',
        ]
struk_domain = 'http://aflowlib.org/CrystalDatabase/'

def get_strukt_links(url, domain):
    r = requests.get(domain+url)
    html = r.text
    soup = BeautifulSoup(html, 'html.parser')
    all_imgs = list(soup.find_all("img"))

    poscar_links = []
    names = []
    pearson = {} 
    for img in all_imgs:
        img = str(img)
        find_strukture = re.compile('alt="([^_]*_([^_]*)_.*).html')
        match = find_strukture.search(img)
        link_name = match.group(1)
        pearson_symbol = match.group(2)

        names.append(link_name)
        poscar_links.append(struk_domain+'POSCAR/'+link_name+'.poscar')
        pearson[link_name] = pearson_symbol
    print(pearson)
    return names, poscar_links, pearson

def get_all_struk_links(urls, domain):
    all_names = []
    all_poscar_links = []
    all_pearson = {}
    for url in urls:
        names, poscar_links, pearson = get_strukt_links(url, domain)
        all_names.extend(names)
        all_poscar_links.extend(poscar_links)
        all_pearson.update(pearson)
    print('All_pearson', all_pearson)
    return all_names, all_poscar_links, all_pearson



def write_poscar_files(dump_dir, names, links):
    strange = []
    for name, link in zip(names, links):
        print(name, link)
        r = requests.get(link)
        if r.status_code != 200:
            strange.append(link)
            continue
        else:
            text = r.text
            with open(dumpdir+name, 'w') as f:
                f.write(text)
    print(strange)
    return strange


if __name__=='__main__':
    dumpdir = 'crystal_files_poscar/'
    names, poscar_links, pearson = get_all_struk_links(tlds, struk_domain)
    np.save('utility/pearson_symbols', pearson)
    # strange = write_poscar_files(dumpdir, names, poscar_links)
    # print(*strange, sep='\n')

