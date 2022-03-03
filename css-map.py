from matplotlib import pyplot as plt
import numpy as np
from scipy import ndimage
from scipy import signal
from imagen import *
from segmentacion import *


def plot(img):
    plt.imshow(img, 'gray')
    plt.show()

def gradient(img):
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    Ix = ndimage.convolve(img, Kx)
    Iy = ndimage.convolve(img, Ky)
    mag = np.hypot(Ix, Iy)
    return Ix, Iy, mag

def gradient_square(img):
    Ix, Iy, mag = gradient(img)
    mag_square = mag**2
    return mag_square

def gaussian(dim, std):
    kernel = np.outer(signal.windows.gaussian(dim, std),
                     signal.windows.gaussian(dim, std))
    return kernel

def convolve(img, kernel):
    conv = signal.convolve(img, kernel, mode='same')
    return conv

def visualizar():

    energy = gaussian(200,10)
    space = Segmentation(energy, 200,50)
    space.start()
    for i in range(100):
        space.contour(1)
        plt.imshow(space.img,'gray')
        plt.draw()
        plt.pause(0.01)
        plt.clf()

    for i in range(100):
        space.contour_2(1)
        plt.imshow(space.img,'gray')
        plt.draw()
        plt.pause(0.01)
        plt.clf()

def viz(space, iter):
    space.start()
    for i in range(50):
        space.contour_1(1)
        plt.imshow(space.img,'gray')
        plt.draw()
        plt.pause(0.01)
        plt.clf()

    for i in range(iter):
        space.contour_2(1)
        plt.imshow(space.img,'gray')
        plt.draw()
        plt.pause(0.01)
        plt.clf()



scale = 0.1
dim = 200
points = 50
kernel = gaussian(10, 6)

img_1 = Imagen(r'C:\Users\diego\Desktop\Programacion\tesis\test\Ll-38_C16-PI-f16.tif')
img_2 = Imagen(r'C:\Users\diego\Desktop\Programacion\tesis\test\Ll-38_C16-PI-f17.tif')
img_3 = Imagen(r'C:\Users\diego\Desktop\Programacion\tesis\test\Ll-43_B1_I_F1.tif')
img_4 = Imagen(r'C:\Users\diego\Desktop\Programacion\tesis\test\Ll-43_B5-I_F6.tif')
img_5 = Imagen(r'C:\Users\diego\Desktop\Programacion\tesis\test\Ll-43_B5-I_F7.tif')
img_6 = Imagen(r'C:\Users\diego\Desktop\Programacion\tesis\test\Ll-43_B5-II_F2.tif')
img_7 = Imagen(r'C:\Users\diego\Desktop\Programacion\tesis\test\Ll-43_B5-II_F7.tif')

images = (img_1, img_2, img_3, img_4, img_5, img_6)

img_2.start(scale, dim)
Ix, Iy, mag = gradient(kernel)
img1 = convolve(img_2.img, Ix)
img2 = convolve(img_2.img, Iy)
blurgrad = np.hypot(img1, img2)
space = Segmentation(blurgrad, dim, points)
plot(blurgrad)

viz(space, 100)
















