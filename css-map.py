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

    x = np.linspace(-4*max_sigma, 4*max_sigma, n_pts)
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
    std = np.linspace(0.01, max_sigma, n_pts)
    std = np.flip(std)

    css_map = np.zeros(shape=(1,n_pts))
    for sigma in std:
        newrow = compute_ku(img, n_pts, sigma, max_sigma)
        css_map = np.vstack([css_map, newrow])
    css_map = css_map[0:n_pts][0:]
    print(css_map.shape)
    return css_map

def main():
    dim = 400
    scale = 0.1
    n_pts = 400

    img_camelido = Imagen(r'C:\Users\diego\Desktop\Programacion\rock-art-vision\images\images_raw_all\Antropomorfos\Ll-43_B52_P1-F4.tif')
    img_camelido.start(scale, dim)

    css_map = plot_ku(img_camelido.img, n_pts, 6)
    plt.imshow(img_camelido.img, 'gray')
    plt.show()
    plt.imshow(css_map, 'gray')
    plt.show()
    
if __name__ == '__main__':
    main()