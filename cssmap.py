from boundary_pts import *
from imagen import *
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from scipy.ndimage import gaussian_filter1d
from scipy.signal import convolve
import numpy as np
from utils import *

def compute_ku(img, n_pts, sigma, max_sigma):

    x1,y1,pts = boundary_pts(img, n_pts)

    x = np.linspace(-10, 10, n_pts)
    g, g1, g2 = compute_gaussian(x, sigma)

    xg = gaussian_conv(x1, sigma, max_sigma)
    yg = gaussian_conv(y1, sigma, max_sigma)

    xu = gaussian1d_conv(x1, sigma, max_sigma)
    yu = gaussian1d_conv(y1, sigma, max_sigma) 

    xuu = gaussian2d_conv(x1, sigma, max_sigma)
    yuu = gaussian2d_conv(y1, sigma, max_sigma)

    ku = (xu*yuu - xuu*yu)/(xu**2+yu**2)**(3/2)
    asign = np.sign(ku)
    signchange = ((np.roll(asign, 1) - asign) != 0).astype(int)

    return signchange

def plot_ku(img, n_pts, max_sigma):
    std = np.linspace(0.1, max_sigma, n_pts)
    std = np.flip(std)

    css_map = np.zeros(shape=(1,n_pts))
    for sigma in std:
        newrow = compute_ku(img, n_pts, sigma, max_sigma)
        css_map = np.vstack([css_map, newrow])
    css_map = css_map[0:n_pts][0:]
    return css_map

def maxima_pts(css_map, n_pts):
    line = np.ones((n_pts,), dtype='int8')
    n_inter = []
    index_dict = {}
    y_points = []
    pts = []
    for i in range(n_pts):
        a = np.dot(css_map[i][0:],line)
        n_inter.append(a)

    change_index = []
    for i in range(len(n_inter)-1):
        dif = n_inter[i+1] - n_inter[i]
        if dif > 0:
            change_index.append(i+1)

    for index in change_index:
        p_list = np.where(css_map[index][0:]==1)
        index_dict[index] = p_list[0]

    for index in change_index:
        if len(index_dict[index]) <=2:
            y = int(index_dict[index].sum()/2)
            y_points.append(y)
            pts.append((y,n_pts-index))
        else:
            diff_list = np.diff(index_dict[index])
            first_diff = index_dict[index][0] - (index_dict[index][-1] - n_pts)
            diff_list = np.append(first_diff, diff_list)
            if np.argmin(diff_list) == 0:
                pass
                #Caso particular del borde
            else:
                y = int((index_dict[index][np.argmin(diff_list-1)]+index_dict[index][np.argmin(diff_list)])/2)
                y_points.append(y)
                pts.append((y,n_pts-index))

    return pts

def main():
    dim = 400
    scale = 0.1
    n_pts = 400
    sigma = 1

    img_1 = Imagen(r'C:\Users\diego\Desktop\Programacion\rock-art-vision\images\images_raw_all\Antropomorfos\Ll-43_B52_P1-F4.tif')
    img_1.start(scale, dim)

    css_map = plot_ku(img_1.img, n_pts, sigma)
    plt.imshow(css_map, 'gray')
    pts = maxima_pts(css_map, n_pts)
    print(pts)
    plt.show()
    
if __name__ == '__main__':
    main()