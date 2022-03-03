import numpy as np
import math

class Segmentation:
    def __init__(self, energy, dim, points):
        self.dim = dim
        self.points = points
        self.img = 0
        self.energy = energy
        self.circle = 0
        self.paststates = set()
        self.win_len = 5

    def start(self):
        d = self.dim
        p = d/2 - int(d*0.9)
        h = d/2
        k = d/2
        circle = []
        points = self.points
        space = np.zeros((d,d), dtype = int)

        theta = np.linspace(0, 2*np.pi, points+1)
        x = np.sin(theta)*p + k
        y = -np.cos(theta)*p  + h

        x = x.astype(int)
        y = y.astype(int)

        for i in range(len(x)):
            circle.append((x[i],y[i]))
            self.paststates.add((x[i],y[i]))

        circle = circle[:-1]
        self.circle = circle

        for i in range(len(x)):
            space[x[i]][y[i]] = 1

        self.img = space

        return self.img

    def window(self, point):
        index = np.array(range(self.win_len))-int(self.win_len/2)
        window = []
        for x in index:
            for y in index:
                window.append(self.energy[x+point[0]][y+point[1]])

        win = np.asarray(window)
        win = np.reshape(win, (self.win_len,self.win_len))
        return win

    def sumenergy(self):
        sum = 0
        for point in self.circle:
            sum = sum + self.energy[point[0]][point[1]]
        return sum

    def newpoint_2(self):

        circle = []
        for point in self.circle:
            win = self.window(point).copy()
            newpoint = np.unravel_index(win.argmax(), win.shape)
            x = point[0] + newpoint[0] - 1
            y = point[1] + newpoint[1] - 1
            newpoint = (x, y)
            while newpoint in circle == True:
                win[1][1] = 0
                newpoint = np.unravel_index(win.argmax(), win.shape)
                x = point[0] + newpoint[0] - 1
                y = point[1] + newpoint[1] - 1
                newpoint = (x, y)
            circle.append(newpoint)
        self.circle = circle

        return point

    def initcircle(self):
        circle = []
        center = (self.dim/2), (self.dim/2)
        paststates = set()
        for point in self.circle:
            win = self.window(point)
            energy_space = {}
            energy = []
            space_point = list(zip(*np.where(win < np.max(self.energy))))
            for i in space_point:
                point_aux = (i[0] + point[0] - int(self.win_len/2), i[1]+point[1] - int(self.win_len/2) )
                energy_aux = math.hypot(point_aux[0] - center[0], point_aux[1] - center[1])
                energy_space[energy_aux] = point_aux
                energy.append(energy_aux)
            energy.sort()
            #Asignacion de la menor energía al punto sin pasar por un estado anterior
            if self.energy[energy_space[energy[0]][0]][energy_space[energy[0]][1]] > 2:
                newpoint = point
                circle.append(newpoint)
                paststates.add(newpoint)
            else:
                i = 0
                if energy_space[energy[0]] == point:
                    newpoint = point
                else:
                    while energy_space[energy[i]] in paststates:
                        i = i+1
                    newpoint = energy_space[energy[i]]
                circle.append(newpoint)
                paststates.add(newpoint)
        self.circle = circle

        return None

    def newpoint(self):
        circle = []
        paststates = set()
        for point in self.circle:
            win = self.window(point)
            space_point = list(zip(*np.where(win < np.max(self.energy))))
            energy_space = {}
            energy = []
            for i in space_point:
                point_aux = (i[0] + point[0] - int(self.win_len/2), i[1]+point[1] - int(self.win_len/2) )
                energy_aux = self.window(point_aux).sum()
                energy_space[energy_aux] = point_aux
                energy.append(energy_aux)
            energy.sort(reverse=True)
            i = 0
            if energy_space[energy[0]] == point:
                newpoint = point
            else:
                while energy_space[energy[i]] in paststates:
                    i = i+1
                newpoint = energy_space[energy[i]]
            circle.append(newpoint)
            paststates.add(newpoint)
        self.circle = circle

        return None

    def contour_1(self, cicles):
        space = np.zeros((self.dim,self.dim), dtype = int)

        for i in range(cicles):
            self.initcircle()
        for i in range(len(self.circle)):
            space[self.circle[i][0]][self.circle[i][1]] = 1

        self.img = space
        return self.img        

    def contour_2(self, cicles):
        space = np.zeros((self.dim,self.dim), dtype = int)
        
        for i in range(cicles):
            self.newpoint()
        for i in range(len(self.circle)):
            space[self.circle[i][0]][self.circle[i][1]] = 1

        self.img = space
        return self.img


def main():
    pass

if __name__ == '__main__':
    main()
    