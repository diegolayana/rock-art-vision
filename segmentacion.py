#this is the second try for the segmentation, hope this will work.
from scipy import ndimage, optimize
import numpy as np
from matplotlib import pyplot as plt
from skimage import morphology
from imagen import *
from skimage import measure
from functools import partial
from external_energy import generate_image_energy, get_gradients


def snake_energy(flattened_pts, edge_dist, alpha, beta):

    pts = np.reshape(flattened_pts, (int(len(flattened_pts)/2), 2))
    #external energy
    dist_vals = ndimage.map_coordinates(edge_dist, [pts[:,0],pts[:,1]])
    edge_energy = np.sum(dist_vals)
    external_energy = edge_energy 


    # spacing energy (favors equi-distant points)
    prev_pts = np.roll(pts, 1, axis=0)
    next_pts = np.roll(pts, -1, axis=0)
    displacements = pts - prev_pts
    point_distances = np.sqrt(displacements[:,0]**2 + displacements[:,1]**2)
    mean_dist = np.mean(point_distances)
    spacing_energy = np.sum((point_distances - mean_dist)**2)

    # curvature energy (favors smooth curves)
    curvature_1d = prev_pts - 2*pts + next_pts
    curvature = (curvature_1d[:,0]**2 + curvature_1d[:,1]**2)
    curvature_energy = np.sum(curvature)
    
    return external_energy + alpha*spacing_energy + beta*curvature_energy

def fit_snake(pts, edge_dist, alpha = 0.5, beta = 0.25, nits = 100, point_plot = None):
    if point_plot:
        def callback_function(new_pts):
            callback_function.nits += 1
            y = new_pts[0::2]
            x = new_pts[1::2]
            point_plot.set_data(x,y)
            plt.title('%i iterations' % callback_function.nits)
            point_plot.figure.canvas.draw()
            plt.pause(0.02)
        callback_function.nits = 0
    else:
        callback_function = None
    
    # optimize
    cost_function = partial(snake_energy, alpha=alpha, beta=beta, edge_dist=edge_dist)
    options = {'disp':False}
    options['maxiter'] = nits  # FIXME: check convergence
    method = 'BFGS'  # 'BFGS', 'CG', or 'Powell'. 'Nelder-Mead' has very slow convergence
    res = optimize.minimize(cost_function, pts.ravel(), method=method, options=options, callback=callback_function)
    optimal_pts = np.reshape(res.x, (int(len(res.x)/2), 2))

    return optimal_pts

def init_points(dim, n_points):
    p = dim/2 - int(dim*0.85)
    h = dim/2
    k = dim/2
    theta = np.linspace(0, 2*np.pi, n_points + 1)
    x = np.sin(theta)*p + k
    y = -np.cos(theta)*p + h
    x = x[:-1]
    y = y[:-1]
    pts = np.array([x,y]).T
    return pts

def main():
    #Parameters
    dim = 200
    n_points = 80

    #Image example
    xx, yy = np.mgrid[:dim,:dim]
    circle = (xx-100) **2 + (yy - 100) ** 2
    image = np.logical_and(circle < 2500, circle >= 0).astype(int)

    #Imagen example
    img_camelido = Imagen(r'C:\Users\diego\Desktop\Programacion\rock-art-vision\images\images_raw_all\Ll-43_B5-I_F6.tif')
    img_camelido.start(0.1,200)

    #Plots points
    space = np.zeros((dim,dim), dtype = int)
    for point in init_points(dim, n_points):
        space[int(point[0])][int(point[1])] = 1

    pts = init_points(dim, n_points)

    edge_dist = generate_image_energy(img_camelido.img,1, 40,30, 1, 1)

    fig = plt.figure()
    plt.imshow(img_camelido.img, cmap='gray')
    plt.plot(pts[0], pts[1], 'bo')
    line_obj, = plt.plot(pts[0], pts[1], 'ro')
    plt.axis('off')
        
    plt.ion()
    plt.pause(0.01)
    snake_pts = fit_snake(pts, edge_dist, nits=2000, alpha=0.0001, beta=0.00001, point_plot=line_obj)
    plt.ioff()
    plt.pause(0.01)
    plt.show()

if __name__ == '__main__':
    main()