from random import gauss
from boundary_pts import *
from imagen import *
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from scipy.ndimage import gaussian_filter1d
import numpy as np


def ku_compute(img, n_pts, sigma):

    x1,y1,pts = boundary_pts(img, n_pts)
    
    N = 1000
    dx = float(len(x1))/N

    x = np.arange(-3*sigma, 3*sigma, dx)
    gaussian = np.exp(-(x/sigma)**2/2)
    gaussian_1d = -(x/sigma**2)*np.exp(-(x/sigma)**2/2)
    gaussian_2d = ((x**2/sigma**4)-(1/sigma**2))*np.exp(-(x/sigma)**2/2)
    
    x1 = x1/np.amax(x1)
    y1 = y1/np.amax(y1)

    xg = np.convolve(x1, gaussian,0)
    yg = np.convolve(y1, gaussian,0)

    xg = xg/np.amax(xg)
    yg = yg/np.amax(yg)

    xu = np.convolve(x1, gaussian_1d)
    yu = np.convolve(y1, gaussian_1d) 

    xuu = np.convolve(x1, gaussian_2d)
    yuu = np.convolve(y1, gaussian_2d) 


    ku = (xu*yuu - xuu*yu)/(xu**2+yu**2)**(3/2)
    ku = np.interp(np.arange(0, n_pts),np.arange(0, len(ku)), ku)
    asign = np.sign(ku)
    signchange = ((np.roll(asign, 1) - asign) != 0).astype(int)

    return signchange

def plot_ku(img, n_pts, max_sigma):
    std = np.linspace(0.01, max_sigma, n_pts)
    std = np.flip(std)

    css_map = np.zeros(shape=(1,n_pts))
    for sigma in std:
        newrow = ku_compute(img, n_pts, sigma)
        css_map = np.vstack([css_map, newrow])

    return css_map

def gaussian_slider():

    dim = 200
    scale = 0.1

    img_camelido = Imagen(r'C:\Users\diego\Desktop\Programacion\rock-art-vision\images\images_raw_all\Ll-43_B5-I_F6.tif')
    img_camelido.start(scale, dim)
    x1,y1,pts = boundary_pts(img_camelido.img, 400)

    fig, ax = plt.subplots()
    line, = plt.plot(y1, -x1)

    plt.subplots_adjust(left=0.25, bottom=0.15)


    axstd = plt.axes([0.1, 0.25, 0.0225, 0.63])
    std_slider = Slider(
        ax=axstd,
        label="Desviacion estandar",
        valmin=0,
        valmax=50,
        valinit=0,
        orientation="vertical"
    )


    # The function to be called anytime a slider's value changes
    def update(val):
        x1,y1,pts = boundary_pts(img_camelido.img, 400)
        x2 = gaussian_filter1d(x1,std_slider.val)
        y2 = gaussian_filter1d(y1,std_slider.val)
        line.set_data(y2, -x2)
        fig.canvas.draw_idle()


    # register the update function with each slider
    std_slider.on_changed(update)

    # Create a `matplotlib.widgets.Button` to reset the sliders to initial values.
    resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
    button = Button(resetax, 'Reset', hovercolor='0.975')


    def reset(event):
        std_slider.reset()
    button.on_clicked(reset)

    plt.show()

def main():
    dim = 200
    scale = 0.1
    n_pts = 400

    img_camelido = Imagen(r'C:\Users\diego\Desktop\Programacion\rock-art-vision\images\images_raw_all\Ll-43_B5-I_F6.tif')
    img_camelido.start(scale, dim)
    
    x, y, pts = boundary_pts(img_camelido.img, n_pts) 
    plt.plot(y,-x)
    plt.show()


if __name__ == '__main__':
    main()