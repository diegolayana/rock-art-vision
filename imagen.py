from tkinter import Image
import cv2 as cv
import numpy as np
import math
import matplotlib.pyplot as plt
from skimage import measure
from scipy.interpolate import splprep, splev
from utils import *
import os

class Imagen:

    def __init__(self, path, scale):
        self.path = path
        self.scale = scale
        self.angle = None
        self.load()

    def load(self):
        self.img = cv.imread(self.path, 0)
        self.resize()

    def resize(self):
        self.img = cv.resize(self.img, (0,0), fx=self.scale, fy=self.scale)

    def get_center(self):
        return int(self.img.shape[0]/2), int(self.img.shape[1]/2)

    def thresh(self):
        self.img[self.img < 159] = 0
        self.img[self.img > 161] = 0
        self.img[self.img != 0] = 1
        self.img.astype('int8')

    def square(self, dim):
        if dim > self.img.shape[0] and dim > self.img.shape[1]:
            dx = dim - self.img.shape[0]
            dy = dim - self.img.shape[1]
            if dx % 2 == 0:
                zerox = np.zeros((int(dx/2), self.img.shape[1]))
                self.img = np.append(zerox, self.img, 0)
                self.img = np.append(self.img, zerox, 0)
            else:
                zerox = np.zeros((int(dx/2), self.img.shape[1]))
                zeroxi = np.zeros((int(dx/2) + 1, self.img.shape[1]))
                self.img = np.append(zeroxi, self.img, 0)
                self.img = np.append(self.img, zerox, 0)
            if dy % 2 == 0:
                zeroy = np.zeros((self.img.shape[0], int(dy/2)))
                self.img = np.append(zeroy, self.img, 1)
                self.img = np.append(self.img, zeroy, 1)
            else:
                zeroy = np.zeros((self.img.shape[0], int(dy/2)))
                zeroyi = np.zeros((self.img.shape[0], int(dy/2) + 1))
                self.img = np.append(zeroy, self.img, 1)
                self.img = np.append(self.img, zeroyi, 1)

            self.firstmoment()
            return self.img
        else:
            raise ValueError('No se puede expandir si la dimension es mas pequeña que la imagen.')

    def get_masscenter(self):
        x_c = 0
        y_c = 0
        area = self.img.sum()
        it = np.nditer(self.img, flags=['multi_index'])

        for i in it:
            y_c = i * it.multi_index[1] + y_c
            x_c = i * it.multi_index[0] + x_c

        self.center = int(x_c/area), int(y_c/area)
        return (int(x_c/area), int(y_c/area))

    def firstmoment(self):
        self.load()
        self.thresh()
        x,y = self.get_masscenter()
        image = self.img.copy()
        firstmoment = cv.circle(image, (y,x) , radius = int(self.img.shape[0]*0.02), color=0, thickness=-1)
        self.img = firstmoment

    def secondmoment(self):
        self.load()
        self.thresh()
        x_c , y_c = self.get_masscenter()
        a = 0
        b = 0
        c = 0
        it = np.nditer(self.img , flags=['multi_index'])

        for i in it:
            c = (it.multi_index[0] - x_c)**2 * i + c
            b = (it.multi_index[0] - x_c)*(it.multi_index[1] - y_c) * i + b
            a = (it.multi_index[1] - y_c)**2 * i + a

        b = 2*b

        theta = (1/2) * math.atan2(b,a-c)
        dde_1 = (a - c) * math.cos(2*theta) + b * math.sin(2*theta)
        dde_2 = (a - c) * math.cos(2*theta + math.pi) + b * math.sin(2*theta + math.pi)
        if  dde_1 > 0 and dde_2 <0:
            theta = theta
        else:
            theta = theta + math.pi

        rho = x_c * math.cos(theta) - y_c * math.sin(theta)

        #Plot the orientation line

        x_1 = int(self.img.shape[0]*0.1)
        x_2 = int(self.img.shape[0]*0.9)

        y_1 = (1/math.sin(theta)*(-rho + x_1*math.cos(theta)))
        y_2 = (1/math.sin(theta)*(-rho + x_2*math.cos(theta)))

        p_1 = int(y_1), int(x_1)
        p_2 = int(y_2), int(x_2)

        p_11 = int(x_1), int(y_1)
        p_22 = int(x_2), int(y_2)
        img = self.img.copy()
        secondmoment = cv.line(img, p_1, p_2, 1, thickness=2)

        self.angle = theta
        self.img = secondmoment
        # self.flip()

    def flip(self):
        if self.angle is None:
            self.secondmoment()
            self.load()
            self.thresh()
        if self.angle > 0:
            self.load()
            self.thresh()
            self.img = cv.flip(self.img, 1)

    def boundary_pts(self, n_pts):
        self.flip()
        bound_pts = measure.find_contours(self.img)
        len_contour = []
        if len(bound_pts) > 1:
            for arr in bound_pts:
                len_contour.append(len(arr))
        else:
            return bound_pts

        tck, u = splprep(bound_pts[np.argmax(len_contour)].T, u=None, s=0.0, per = 1)
        u_new = np.linspace(u.min(), u.max(), n_pts)
        x_new,y_new = splev(u_new, tck, der = 0)
        return x_new, y_new

    def compute_ku(self, n_pts, sigma, max_sigma):
        x1,y1 = self.boundary_pts(n_pts)

        xu = gaussian1d_conv(x1, sigma, max_sigma)
        yu = gaussian1d_conv(y1, sigma, max_sigma) 

        xuu = gaussian2d_conv(x1, sigma, max_sigma)
        yuu = gaussian2d_conv(y1, sigma, max_sigma)

        ku = (xu*yuu - xuu*yu)/(xu**2+yu**2)**(3/2)
        asign = np.sign(ku)
        signchange = ((np.roll(asign, 1) - asign) != 0).astype(int)

        return signchange

    def plot_ku(self, n_pts, max_sigma):
        std = np.linspace(0.1, max_sigma, n_pts)
        std = np.flip(std)

        css_map = np.zeros(shape=(1,n_pts))
        for sigma in std:
            newrow = self.compute_ku(n_pts, sigma, max_sigma)
            css_map = np.vstack([css_map, newrow])
        css_map = css_map[0:n_pts][0:]
        
        self.css_map = css_map

def main():

    img = Imagen(r'C:\Users\diego\Desktop\Programacion\rock-art-vision\images\images_raw_all\Ll-43_B5-I_F6.tif', 0.4)
    img.firstmoment()
    plt.imshow(img.img,'gray')
    plt.show()

if __name__ == '__main__':
    main()
