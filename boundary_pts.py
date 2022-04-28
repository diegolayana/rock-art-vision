from matplotlib.pyplot import bone
from imagen import *
from segmentacion import * 
from skimage import measure
from scipy.interpolate import splprep, splev

def boundary_pts(image, pts, pt_spacing = 10):
    bound_pts = measure.find_contours(image)
    len_contour = []
    if len(bound_pts) > 1:
        for arr in bound_pts:
            len_contour.append(len(arr))
    else:
        return bound_pts

    tck, u = splprep(bound_pts[np.argmax(len_contour)].T, u=None, s=0.0, per = 1)
    u_new = np.linspace(u.min(), u.max(), pts)
    x_new,y_new = splev(u_new, tck, der = 0)
    return x_new, y_new, bound_pts[np.argmax(len_contour)]


def main():
    dim = 200
    scale = 0.1
    img_camelido = Imagen(r'C:\Users\diego\Desktop\Programacion\rock-art-vision\images\images_raw_all\Ll-43_B5-I_F6.tif')
    img_camelido.start(scale, dim)
    space = np.zeros((dim,dim)).astype('int8')
    x, y, pts = boundary_pts(img_camelido.img, 200)
    for pt in pts:
        space[int(pt[0])][int(pt[1])] = 1

    fig, ax = plt.subplots()
    print(pts[0])
    ax.imshow(space, 'gray')
    ax.plot(y,x,'ro')
    plt.show()
    

if __name__ == '__main__':
    main()