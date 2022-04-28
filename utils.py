from boundary_pts import *
from imagen import *
import numpy as np
import matplotlib.pyplot as plt
from boundary_pts import *
from imagen import *

def compute_elipse(theta, h, k, a, b):
    x = a*np.cos(theta) + h
    y = b*np.sin(theta) + h
    return x, y


def compute_circle(theta, h, k, r):
    x = r*np.cos(theta) + h
    y = r*np.sin(theta) + k
    return x, y

def compute_gaussian(x, sigma):
    gaussian = np.exp(-(x/sigma)**2/2)
    gaussian_1d = -(x/sigma**2)*np.exp(-(x/sigma)**2/2)
    gaussian_2d = ((x**2/sigma**4)-(1/sigma**2))*np.exp(-(x/sigma)**2/2)

    return gaussian, gaussian_1d, gaussian_2d


#We're gonna compute convolve function for close cruves.
def gaussian_conv(arr, sigma, max_sigma):
    n_pts = len(arr)
    t = np.linspace(-3*max_sigma, 3*max_sigma, n_pts)
    g, g1, g2 = compute_gaussian(t, sigma)  
    conv = []
    for i in range(n_pts):
        sumx = np.dot(np.roll(arr,0), np.flip(np.roll(g, i)))
        conv.append(sumx) 
    return np.array(conv)

def gaussian1d_conv(arr, sigma, max_sigma):
    n_pts = len(arr)
    t = np.linspace(-3*max_sigma, 3*max_sigma, n_pts)
    g, g1, g2 = compute_gaussian(t, sigma)  
    conv = []
    for i in range(n_pts):
        sumx = np.dot(np.roll(arr,0), np.flip(np.roll(g1, i)))
        conv.append(sumx) 
    return np.array(conv)

def gaussian2d_conv(arr, sigma, max_sigma):
    n_pts = len(arr)
    t = np.linspace(-3*max_sigma, 3*max_sigma, n_pts)
    g, g1, g2 = compute_gaussian(t, sigma)  
    conv = []
    for i in range(n_pts):
        sumx = np.dot(np.roll(arr,0), np.flip(np.roll(g2, i)))
        conv.append(sumx) 
    return np.array(conv)

def main():
    sigma = 6
    max_sigma = 50
    dim = 200
    scale = 0.1
    n_pts = 400

    img_camelido = Imagen(r'C:\Users\diego\Desktop\Programacion\rock-art-vision\images\images_raw_all\Ll-43_B5-I_F6.tif')
    img_camelido.start(scale, dim)

    x1,y1,pts = boundary_pts(img_camelido.img, n_pts)

    convx = gaussian_conv(x1, sigma, max_sigma)
    convy = gaussian_conv(y1, sigma, max_sigma)
    convx = convx/np.amax(convx)
    convy = convy/np.amax(convy)

    x1 = x1/np.amax(x1)
    y1 = y1/np.amax(y1)
    plt.plot(convy,-convx)
    plt.show()

if __name__ == '__main__':
    main()
