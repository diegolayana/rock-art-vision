from turtle import update
import numpy as np
import os
from os import listdir
from os.path import isfile, join
from matplotlib import pyplot as plt
import cv2 as cv
import sys

n_pts = 400
increment = 20
class CSS_matching():

    def __init__(self, model, model_name, css_path, db_path):
        self.path = css_path
        self.model_name = model_name[0:7]
        self.db_path = db_path
        self.model = model
        self.load_css()
        self.load_db()
        self.model_list = self.maxima_pts(self.model, n_pts)
        self.cost = 0

    def load_db(self):
        img_dict = {}
        img_list = [f for f in listdir(self.db_path) if isfile(join(self.db_path,f))]
        for id in img_list:
            img_dict[id] = self.img = cv.imread(join(self.db_path,id), 0)

        self.img_db = img_dict

    def load_css(self):
        css_dict = {}
        css_list = [f for f in listdir(self.path) if isfile(join(self.path,f))]
        for id in css_list:
            css_dict[id[0:7]] = np.load(join(self.path,id))

        self.css_db = css_dict

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
        self.filename = filename
        self.cost = 0
        self.img = np.load(join(self.path,filename +'.npy'))
        self.model_list = self.maxima_pts(self.model, n_pts)
        self.model_list = set(self.model_list)
        self.image_list = self.maxima_pts(self.img, n_pts)
        self.image_list = set(self.image_list)

    def get_cost(self, filename):
        cost = list()
        self.select_img(filename)

        #Case 1
        self.cost = 0
        model_max = self.get_max_point(self.model_list)
        image_max = self.get_max_point(self.image_list)
        self.calculate_cost(model_max, image_max)
        cost.append(self.cost)
        
        #Case 2
        self.cost = 0
        model_max = self.get_second_max_point(self.model_list)
        image_max = self.get_second_max_point(self.image_list)
        self.calculate_cost(model_max, image_max)
        cost.append(self.cost)

        #Case 3
        self.cost = 0
        model_max = self.get_max_point(self.model_list)
        image_max = self.get_second_max_point(self.image_list)
        self.calculate_cost(model_max, image_max)
        cost.append(self.cost)

        #Case 4
        self.cost = 0
        model_max = self.get_second_max_point(self.model_list)
        image_max = self.get_max_point(self.image_list)
        self.calculate_cost(model_max, image_max)
        cost.append(self.cost)

        self.cost = min(cost)
        # print(self.filename, cost)
        
    def calculate_cost(self, model_max, image_max):

        #Componsate the 
        self.compensate(image_max, model_max)
        #Create de model set
        self.model_set = set()
        self.image_set = set()
        model_max = self.current_maxima
        self.model_set.add(model_max)
        self.image_set.add(image_max)
        for i in range(max(len(self.model_list),len(self.image_list))):
            if not (self.get_empty_model() or self.get_empty_image()):
                self.get_model_nearest()
                self.compute_cost()
        
        if len(self.image_list - self.image_set) != 0:
            self.get_residual_image_cost()
        if len(self.model_list - self.model_set) != 0:
            self.get_residual_model_cost()

        # if self.filename == 'A05.tif':
        #     self.plot_maxima()

    def get_empty_model(self):
        cond = self.model_list - self.model_set
        return len(cond) == 0

    def get_empty_image(self):
        cond = self.image_list - self.image_set
        return len(cond) == 0

    def get_residual_image_cost(self):
        my_list =  list(self.image_list - self.image_set)
        x, y = zip(*my_list)
        y = np.array(y)
        cost = np.sum(y)
        self.cost = self.cost + cost
        
    def get_residual_model_cost(self):
        my_list =  list(self.model_list - self.model_set)
        x, y = zip(*my_list)
        y = np.array(y)
        cost = np.sum(y)
        self.cost = self.cost + cost

    def compute_cost(self):
        my_list = list(self.image_list - self.image_set)
        x_model = self.current_maxima[0]
        x,y = zip(*my_list)
        x_to_y = dict()
        for i in range(len(x)):
            x_to_y[x[i]] = y[i]

        x = np.array(x)
        dist = abs(x_model - x)
        perc = dist*100/np.max(x)
        x_nearest = x_model - np.sort(dist)[0]
        perc_nearest = np.sort(perc)[0]
        if perc_nearest < 20:
            if(x_nearest in x):
                y_nearest = x_to_y[x_nearest]
                self.image_set.add((x_nearest,y_nearest))
            else:
                x_nearest = x_model + np.sort(dist)[0]
                y_nearest = x_to_y[x_nearest]
                self.image_set.add((x_nearest,y_nearest))
            cost = abs(y_nearest - self.current_maxima[1])
            self.cost = self.cost + cost
        else:
            self.cost = self.cost + self.current_maxima[1]

    def get_max_point(self, set_point):
        my_list = list(set_point)
        x_list , y_list = zip(*my_list)
        y_to_x = dict()
        for i in range(len(y_list)):
            y_to_x[y_list[i]] = x_list[i]
        y_list = np.array(y_list)
        y_list = np.sort(y_list)
        y = y_list[-1]
        x = y_to_x[y]
        return (x,y)

    def get_second_max_point(self, set_point):
        my_list = list(set_point)
        x_list , y_list = zip(*my_list)
        y_to_x = dict()
        for i in range(len(y_list)):
            y_to_x[y_list[i]] = x_list[i]
        y_list = np.array(y_list)
        y_list = np.sort(y_list)
        y = y_list[-2]
        x = y_to_x[y]
        return (x,y)

    def compensate(self, image_max, model_max):
        alpha = image_max[0] - model_max[0]
        self.model = np.roll(self.model, alpha, 1)
        self.model_list = set(self.maxima_pts(self.model, n_pts))
        self.update_current_maxima(model_max)

    def update_current_maxima(self, model_max):
        x, y = zip(*list(self.model_list))
        y_dict = dict()
        for i in range(len(x)):
            y_dict[y[i]] = x[i]
        self.current_maxima = (y_dict[model_max[1]], model_max[1])

    def get_model_nearest(self):
        my_list = list(self.model_list - self.model_set)
        x_model = self.current_maxima[0]
        x, y = zip(*my_list)
        x_to_y = dict()
        for i in range(len(x)):
            x_to_y[x[i]] = y[i]
        x = np.array(x)
        dist = abs(x_model - x)
        x_nearest = x_model - np.sort(dist)[0]
        if(x_nearest in x):
            y_nearest = x_to_y[x_nearest]
        else:
            x_nearest = x_model + np.sort(dist)[0]
            y_nearest = x_to_y[x_nearest]
        point = (x_nearest,y_nearest)
        mode_set_aux = list(self.model_set)
        mode_set_aux.append(point)
        self.model_set = set(mode_set_aux)
        self.current_maxima = point
        return (x_nearest,y_nearest)

    def plot_maxima(self):
            x1, y1 = zip(*list(self.model_list))
            x2, y2 = zip(*list(self.image_list))
            print(self.filename)
            print(self.get_max_point(self.model_list))
            print(self.get_max_point(self.image_list))
            print(self.cost)
            fig, ax = plt.subplots(2,3)
            ax[0][0].imshow(self.img_db[self.model_name], 'gray')
            ax[0][1].imshow(self.img_db[self.filename], 'gray')
            ax[1][0].imshow(self.model, 'gray')
            ax[1][1].imshow(self.img, 'gray')
            ax[1][2].set_ylim([0,400])
            ax[1][2].set_xlim([0,400])
            ax[1][2].plot(x1,y1, '.')
            ax[1][2].plot(x2,y2, '.')

            plt.show()

def images_list(path):
    list = [f for f in listdir(path) if isfile(join(path,f))]
    return list

def plot_matching(model_name):

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
    matching = CSS_matching(model, model_name,  db_path, images_path)
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

def main():
    if len(sys.argv) >=2:
        model_name = sys.argv[1]
    else:
        model_name = 'A01.tif'

    plot_matching(model_name)
if __name__ == '__main__':
    main()