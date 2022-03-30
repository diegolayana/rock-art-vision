import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from imagen import *
from scipy import ndimage 
from matplotlib.widgets import RangeSlider
from skimage import morphology

def generate_image_energy(image, std, w_line, w_edge, w_term, w_edge_dist):
    Cx, Cy, Cxx, Cyy, Cxy = get_gradients(image)
    e_line = image
    e_edge = -np.hypot(ndimage.gaussian_filter(Cx, std), ndimage.gaussian_filter(Cy, std))**2
    e_termination = (Cyy*Cx**2 - 2*Cxy*Cx*Cy + Cxx*Cy**2)/((1 + Cx**2 + Cy**2)**(1.5))
    mag = np.hypot(Cx,Cy)
    mag[mag > 0] = 1
    skeleton = morphology.skeletonize(image)
    edge_dist = ndimage.distance_transform_edt(~skeleton)
    e_image = w_line * e_line + w_edge * e_edge + w_term * e_termination + w_edge_dist * edge_dist
    e_image = e_image/np.linalg.norm(e_image)
    
    return e_image  

def get_gradients(image):
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])  
    Cx = ndimage.convolve(image, Kx)
    Cy = ndimage.convolve(image, Ky)  
    Cxx = ndimage.convolve(Cx, Kx)
    Cyy = ndimage.convolve(Cy, Ky)
    Cxy = ndimage.convolve(Cx, Ky)
    
    return Cx, Cy, Cxx, Cyy, Cxy

def main():
    #Parameters
    dim = 200
    n_points = 100

    #Imagen example
    img_camelido = Imagen(r'C:\Users\diego\Desktop\Programacion\rock-art-vision\images\images_raw_all\Ll-43_B5-I_F6.tif')
    img_camelido.start(0.1,200)

    # Define initial parameters
    w_line = 1
    w_edge = 1
    w_term = 1
    w_edge_dist = 1
    std = 1

    # Create the figure and the line that we will manipulate
    fig, ax = plt.subplots()
    im = plt.imshow(generate_image_energy(img_camelido.img, std, w_line, w_edge, w_term, 1), 'gray')

    # adjust the main plot to make room for the sliders
    plt.subplots_adjust(left=0.1, bottom=0.35)

    # Make a horizontal slider to control the frequency.
    ax_line = plt.axes([0.25, 0.1, 0.65, 0.03])
    line_slider = Slider(
        ax=ax_line,
        label='w_line',
        valmin=0,
        valmax=50,
        valinit=w_line,
    )

    # Make a vertically oriented slider to control the amplitude
    ax_edge = plt.axes([0.25, 0.15, 0.65, 0.03])
    edge_slider = Slider(
        ax=ax_edge,
        label="w_edge",
        valmin=0,
        valmax=50,
        valinit=w_edge,
    )

    # Make a vertically oriented slider to control the amplitude
    ax_term = plt.axes([0.25, 0.20, 0.65, 0.03])
    term_slider = Slider(
        ax=ax_term,
        label="w_term",
        valmin=0,
        valmax=50,
        valinit=w_term,
    )

    ax_dist = plt.axes([0.25, 0.25, 0.65, 0.03])
    dist_slider = Slider(
        ax=ax_dist,
        label="w_edge_dist",
        valmin=0,
        valmax=5,
        valinit=w_edge_dist,
    )

    ax_std = plt.axes([0.1, 0.25, 0.0225, 0.63])
    std_slider = Slider(
        ax=ax_std,
        label="Desviación estandar",
        valmin=0,
        valmax=20,
        valinit=1,
        orientation="vertical"
    )


    # The function to be called anytime a slider's value changes
    def update(val):
        im.set_data(generate_image_energy(img_camelido.img, std_slider.val, line_slider.val, edge_slider.val, term_slider.val, dist_slider.val))
        fig.canvas.draw_idle()




    # register the update function with each slider
    line_slider.on_changed(update)
    edge_slider.on_changed(update)
    term_slider.on_changed(update)
    std_slider.on_changed(update)
    dist_slider.on_changed(update)

    # Create a `matplotlib.widgets.Button` to reset the sliders to initial values.
    resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
    button = Button(resetax, 'Reset', hovercolor='0.975')


    def reset(event):
        line_slider.reset()
        edge_slider.reset()
    button.on_clicked(reset)

    plt.show()

if __name__ == '__main__':
    main()