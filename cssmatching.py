import numpy as np

class Node():
    def __init__(self, state, parent, action):
        self.state = state
        self.parent = parent

class Frontier():
    def __init__(self, frontier, node):
        self.frontier = set(frontier)
        self.node = node

    def add(self, node):
        node_set = set()
        node_set.add(node)
        self.frontier.add(node)

    def remove(self):
        node_x = self.node[0]
        frontier_list = list(self.frontier)
        x, y = zip(*frontier_list)
        dist = abs(np.array(x) - node_x)
        dist = dist.tolist()
        index = dist.index(min(dist))
        node = (x[index],y[index])
        node_set = set()
        node_set.add(node)
        self.frontier = self.frontier - node_set
        self.node = node
            

def maximum(frontier):
        frontier_list = list(frontier)
        x, y = zip(*frontier_list)
        index = y.index(max(y))
        node = (x[index], y[index])
        return node


def main():
    pass

if __name__ == '__main__':
    main()