import numpy as np
import matplotlib.pyplot as plt

def compute_elipse(theta, h, k, a, b):
    x = a*np.cos(theta) + h
    y = b*np.sin(theta) + h
    return x, y

def compute_circle(theta, h, k, r):
    x = r*np.cos(theta) + h
    y = r*np.sin(theta) + k
    return x, y

def compute_gaussian(x, sigma):
    g = gaussian(x, sigma)
    g1 = gaussian1d(x, sigma)
    g2 = gaussian2d(x, sigma)
    return g , g1, g2

def gaussian(x, sigma):
    g = np.exp(-(x**2/(2*(sigma)**2)))/(sigma*np.sqrt(2*np.pi))
    return g

def gaussian1d(x,sigma):
    g = -(np.exp(-(x**2)/(2*sigma**2))*x)/(np.sqrt(2*np.pi)*sigma**3)
    return g

def gaussian2d(x,sigma):
    g = (np.exp(-(x**2)/(2*sigma**2))*(-sigma**2 + x**2))/(np.sqrt(2*np.pi)*sigma**5)
    return g

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
    pass

if __name__ == '__main__':
    main()
