from matplotlib.colors import ListedColormap
# duse = dict([('gray', '#CAC8E3'), ('green', '#D8FFBF'), ('red', '#FFE5D9'), ('blue', '#9FA2B5')])
# strong = dict([('gray', '#54535E'), ('green', '#A1FF33'), ('red', '#B34310'), ('blue', '#5A67CC')])

dark = dict([('gray', '#788E92'), ('green', '#7CC494'), ('red', '#C47C74'), ('blue', '#729AC4'), ('purple', '#B178C4')])


light = dict([('gray', '#CFDFE6'), ('green', '#91E6AD'), ('red', '#E69187'), ('blue', '#85B4E6'), ('purple', '#CF8CE6')])

cmap_dark = ListedColormap(dark.values())
cmap_light = ListedColormap(light.values())
cmap_complete = ListedColormap(list(light.values()) + list(dark.values()))
