import numpy as np
import os
from os import listdir
from os.path import isfile, join
from matplotlib import pyplot as plt
import cv2 as cv
import sys

from cssmap import maxima_pts

n_pts = 400
increment = 20
class CSS_matching():
    
    def __init__(self, model, dir_path):
        self.path = dir_path
        self.model = model
        self.load_db()
        self.model_list = maxima_pts(self.model, n_pts)
        self.cost = 0


    def load_db(self):
        css_dict = {}
        css_list = [f for f in listdir(self.path) if isfile(join(self.path,f))]
        for id in css_list:
            css_dict[id[0:7]] = np.load(join(self.path,id))

        self.db = css_dict

    def maxima_pts(self, css_map, n_pts):
        line = np.ones((n_pts,), dtype='int8')
        n_inter = []
        index_dict = {}
        y_points = []
        pts = []
        for i in range(n_pts):
            a = np.dot(css_map[i][0:],line)
            n_inter.append(a)

        change_index = []
        for i in range(len(n_inter)-1):
            dif = n_inter[i+1] - n_inter[i]
            if dif > 0:
                change_index.append(i+1)

        for index in change_index:
            p_list = np.where(css_map[index][0:]==1)
            index_dict[index] = p_list[0]

        for index in change_index:
            if len(index_dict[index]) <=2:
                y = int(index_dict[index].sum()/2)
                y_points.append(y)
                pts.append((y,n_pts-index))
            else:
                diff_list = np.diff(index_dict[index])
                first_diff = index_dict[index][0] - (index_dict[index][-1] - n_pts)
                diff_list = np.append(first_diff, diff_list)
                if np.argmin(diff_list) == 0:
                    pass
                    #Caso particular del borde
                else:
                    y = int((index_dict[index][np.argmin(diff_list-1)]+index_dict[index][np.argmin(diff_list)])/2)
                    y_points.append(y)
                    pts.append((y,n_pts-index))

        return pts

    def select_img(self, filename):
        self.cost = 0
        self.img = np.load(join(self.path,filename +'.npy'))
        self.model_list = maxima_pts(self.model, n_pts)
        self.image_list = maxima_pts(self.img, n_pts)
     
    def append_ceros(self):
        dist = len(self.model_list) - len(self.image_list)
        if dist > 0:
            for i in range(abs(dist)):
                self.image_list.append((0,0))
                self.cost = self.cost + increment

        if dist < 0:
            for i in range(abs(dist)):
                self.model_list.append((0,0))
                self.cost = self.cost + increment
    
    def get_cost(self, filename):
        self.select_img(filename)
        self.append_ceros()
        x_1, y_1 = zip(*self.model_list)
        x_2, y_2 = zip(*self.image_list)
        y_1 = np.array(list(y_1))
        y_2 = np.array(list(y_2))
        self.cost = self.cost + np.sum(abs(y_1-y_2))


def images_list(path):
    list = [f for f in listdir(path) if isfile(join(path,f))]
    return list

def process(filename: str=None) -> None:
    dirname = os.path.dirname(__file__)
    images_path = join(dirname,r'images\images_test')
    path = join(filename, images_path, filename)
    image = cv.imread(path)
    print(path)
    plt.figure()
    plt.imshow(image)

def main():
    if len(sys.argv) >=2:
        model_name = sys.argv[1]
    else:
        model_name = 'H05.tif'

    #Initialization parametres

    dirname = os.path.dirname(__file__)
    db_path = join(dirname,r'images\css-maping')
    images_path = join(dirname,r'images\images_test')
    img_list = images_list(images_path)
    model_name = model_name + '.npy'
    model_path = join(dirname,r'images\css-maping',model_name)
    model = np.load(model_path)

    #Matching Process

    cost_dict = {}
    cost_list = []
    matching = CSS_matching(model, db_path)
    for img in img_list:
        matching.get_cost(img)
        cost_list.append(matching.cost)
        cost_dict[matching.cost] = img

    cost_list = np.array(cost_list)
    cost_list.sort()

    f, axarr = plt.subplots(4,4) 

    axarr[0][0].imshow(cv.imread(join(images_path, cost_dict[cost_list[0]])))
    axarr[0][1].imshow(cv.imread(join(images_path, cost_dict[cost_list[1]])))
    axarr[0][2].imshow(cv.imread(join(images_path, cost_dict[cost_list[2]])))
    axarr[0][3].imshow(cv.imread(join(images_path, cost_dict[cost_list[3]])))

    axarr[1][0].imshow(cv.imread(join(images_path, cost_dict[cost_list[4]])))
    axarr[1][1].imshow(cv.imread(join(images_path, cost_dict[cost_list[5]])))
    axarr[1][2].imshow(cv.imread(join(images_path, cost_dict[cost_list[6]])))
    axarr[1][3].imshow(cv.imread(join(images_path, cost_dict[cost_list[7]])))

    axarr[2][0].imshow(cv.imread(join(images_path, cost_dict[cost_list[8]])))
    axarr[2][1].imshow(cv.imread(join(images_path, cost_dict[cost_list[9]])))
    axarr[2][2].imshow(cv.imread(join(images_path, cost_dict[cost_list[10]])))
    axarr[2][3].imshow(cv.imread(join(images_path, cost_dict[cost_list[11]])))

    axarr[3][0].imshow(cv.imread(join(images_path, cost_dict[cost_list[12]])))
    axarr[3][1].imshow(cv.imread(join(images_path, cost_dict[cost_list[13]])))
    axarr[3][2].imshow(cv.imread(join(images_path, cost_dict[cost_list[14]])))
    axarr[3][3].imshow(cv.imread(join(images_path, cost_dict[cost_list[15]])))
    plt.show()

if __name__ == '__main__':
    main()