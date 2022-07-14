from imagen import *
import os
from os import listdir
from os.path import isfile, join
import numpy as np
#The porpuse os this code is to create the database where the matching algorithm will work

def main():
    max_size = 400
    n_pts = 400
    sigma = 1
    dirname = os.path.dirname(__file__)
    images_path = join(dirname, r'images\db_test')
    css_map_path = join(dirname, r'images\css_map_test')
    images_list = [f for f in listdir(images_path) if isfile(join(images_path,f))]
    for image in images_list:
        img = Imagen(join(images_path,image), max_size)
        img.plot_ku(n_pts, sigma)

        np.save(join(css_map_path,image),img.css_map)

if __name__ == '__main__':
    main()