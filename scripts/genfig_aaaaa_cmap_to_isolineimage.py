# the aaaaa is to make this the first in os.listdir() in gen_all_figs

import matplotlib.pyplot as plt
from matplotlib import colormaps
import numpy as np
from PIL import Image

imw = 12096
imh = 592
borderw = 40

cmapnames = ["plasma", "YlGnBu_r", "Greys_r", "PuRd_r", "YlOrRd_r", "Purples_r", "Purples", "magma", "Greys_r"]
regionses = [21, 30, 40, 60, 70, 71, 100]

for regions in regionses:
    for cmapname in cmapnames:
        cmap = colormaps[cmapname]

        imw2 = int(regions*np.ceil(imw/regions))

        ar = cmap(np.reshape(np.tile(np.repeat(np.linspace(0, 1, regions), imw2/regions), imh), (imh, imw2)))

        for i in range(1, regions):
            ar[:, i*(imw2//regions)-(borderw//2):i*(imw2//regions)+(borderw//2), :3] = 0
            ar[:, i*(imw2//regions)-(borderw//2):i*(imw2//regions)+(borderw//2), 3] = 1

        Image.fromarray(np.uint8(ar*255)).save("../results/pngs/" + cmapname + "_isolinemap" + ("" if regions==21 else str(regions)) + ".png")