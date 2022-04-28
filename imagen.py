import cv2 as cv
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.widgets import RangeSlider

class Imagen:

    def __init__(self, path):
        self.path = path
        self.img = cv.imread(self.path, 0)
        self.center = int(self.img.shape[0]/2), int(self.img.shape[1]/2)
        self.scale = 1
        self.threshold = 0
        self.angle = 0

    def small(self, scale):
        self.img = cv.resize(self.img, (0,0), fx=scale, fy=scale)
        self.center = int(self.img.shape[0]/2), int(self.img.shape[1]/2)
        self.scale = scale
        return self.img

    def thresh(self):
        self.img[self.img < 159] = 0
        self.img[self.img > 161] = 0
        self.img[self.img != 0] = 1
        self.threshold = 255
        return self.img

    def firstmoment(self):
        x_c = 0
        y_c = 0
        if self.threshold == 0:
            self.thresh()
            area = self.img.sum()
            it = np.nditer(self.img, flags=['multi_index'])

            for i in it:
                y_c = i * it.multi_index[1] + y_c
                x_c = i * it.multi_index[0] + x_c

            center = int(x_c/area), int(y_c/area)

            self.center = center

            image = self.img.copy()
            firstmoment = cv.circle(image, (int(y_c/area), int(x_c/area)) , radius = int(self.img.shape[0]*0.02), color=0, thickness=-1)

            return firstmoment
        else:
            area = self.img.sum()
            it = np.nditer(self.img, flags=['multi_index'])

            for i in it:
                y_c = i * it.multi_index[1] + y_c
                x_c = i * it.multi_index[0] + x_c

            center = int(x_c/area), int(y_c/area)

            self.center = center
            image = self.img.copy()
            firstmoment = cv.circle(image, (int(y_c/area), int(x_c/area)) , radius = int(self.img.shape[0]*0.02), color=0, thickness=-1)

            return firstmoment

    def centered(self):

        if(self.scale <= 0.4):
            center = int(self.img.shape[0]/2), int(self.img.shape[1]/2)
            self.firstmoment()
            dy = int((center[1] - self.center[1]))
            dx = int((center[0] - self.center[0]))
            while abs(dx) > 1 or abs(dy) > 1:
                zerox = np.zeros((abs(dx), self.img.shape[1]))
                if dx > 0:
                    self.img = np.append(zerox, self.img, 0)
                else:
                    self.img = np.append(self.img, zerox, 0)
                zeroy = np.zeros((self.img.shape[0], abs(dy)))
                if dy > 0:
                    self.img = np.append(zeroy, self.img, 1)
                else:
                    self.img = np.append(self.img, zeroy, 1)

                self.firstmoment()
                center = int(self.img.shape[0]/2), int(self.img.shape[1]/2)
                dx = int((center[0] - self.center[0]))
                dy = int((center[1] - self.center[1]))
            return self.img

        else:
            raise ValueError('Ingrese una escala menor al 40%')

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

    def secondmoment(self):

        x_c = self.center[0]
        y_c = self.center[1]
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

        x_1 = int(self.img.shape[0]*0.2)
        x_2 = int(self.img.shape[0]*0.8)

        y_1 = (1/math.sin(theta)*(-rho + x_1*math.cos(theta)))
        y_2 = (1/math.sin(theta)*(-rho + x_2*math.cos(theta)))

        p_1 = int(y_1), int(x_1)
        p_2 = int(y_2), int(x_2)

        p_11 = int(x_1), int(y_1)
        p_22 = int(x_2), int(y_2)
        img = self.img.copy()
        secondmoment = cv.line(img, p_1, p_2, 1, thickness=2)

        self.angle = theta
        return p_11, p_22, secondmoment

    def flip(self):
        if self.angle == 0:
            raise ValueError('Defina la orientación con el segundo momento')
        if self.angle > 0:
            self.img = cv.flip(self.img, 1)
            return self.img
        else:
            return self.img

    def start(self, scale, dim):
        self.small(scale)
        self.centered()
        self.square(dim)
        self.secondmoment()
        self.flip()
        self.img = self.img.astype('int8')


def main():
    dim = 200
    scale = 0.3
    img_camelido = Imagen(r'C:\Users\diego\Desktop\Programacion\rock-art-vision\images\images_raw_all\Ll-43_B5-I_F6.tif')
    img_camelido.small(scale)
    img_camelido.thresh()
    p1,p2, secondmoment = img_camelido.secondmoment()
    plt.imshow(secondmoment, 'gray')
    plt.show()

def thresh_img():

    dim = 200
    scale = 0.3
    # img_camelido = Imagen(r'C:\Users\diego\Desktop\Programacion\rock-art-vision\images\images_raw_all\Ll-43_B5-I_F6.tif')
    img_camelido = Imagen(r'C:\Users\diego\Desktop\Programacion\rock-art-vision\images\images_raw_all\Antropomorfos\Az-Anm-1_B8_P4_F2 (5).jpg')
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    plt.subplots_adjust(bottom=0.25)

    im = axs[0].imshow(img_camelido.img, 'gray')
    axs[1].hist(img_camelido.img.flatten(), bins='auto')
    axs[1].set_title('Histogram of pixel intensities')

    # Create the RangeSlider
    slider_ax = plt.axes([0.20, 0.1, 0.60, 0.03])
    slider = RangeSlider(slider_ax, "Threshold", img_camelido.img.min(), img_camelido.img.max())

    # Create the Vertical lines on the histogram
    lower_limit_line = axs[1].axvline(slider.val[0], color='k')
    upper_limit_line = axs[1].axvline(slider.val[1], color='k')


    def update(val):
        # The val passed to a callback by the RangeSlider will
        # be a tuple of (min, max)

        # Update the image's colormap
        im.norm.vmin = val[0]
        im.norm.vmax = val[1]

        # Update the position of the vertical lines
        lower_limit_line.set_xdata([val[0], val[0]])
        upper_limit_line.set_xdata([val[1], val[1]])

        # Redraw the figure to ensure it updates
        fig.canvas.draw_idle()


    slider.on_changed(update)
    plt.show()
if __name__ == '__main__':
    main()
