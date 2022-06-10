from matplotlib import pyplot as plt
import numpy as np


class Grid:
    def __init__(self, width, height, p_random=0.0):
        self.p_random = p_random
        self.width = width
        self.height = height
        self.obstacles = []
        self.goals = []

    def is_in_bounds(self, x, y, size_x, size_y):
        return x >= 0 and x + size_x <= self.width and y >= 0 and y + size_y <= self.height

    def put_obstacle(self, x, y, size_x, size_y):
        assert self.is_in_bounds(x, y, size_x, size_y)
        ob = Square(x, x + size_x, y, y + size_y)
        self.obstacles.append(ob)

    def put_goal(self, x, y, size_x, size_y):
        assert self.is_in_bounds(x, y, size_x, size_y)
        # We split the dirt tile into sx by sy blocks
        sx = 1.0
        sy = 1.0
        for x_i in np.arange(size_x//sx):
            for y_i in np.arange(size_y//sy):
                goal = Square(x+(x_i*sx), x+(x_i*sx)+sx, y+(y_i*sy), y+(y_i*sy)+sy)
                self.goals.append(goal)
        # Then add all remainder goals
        if size_x % sx != 0:
            for y_i in np.arange(size_y // sy):
                goal = Square(x + size_x - (size_x % sx), x + size_x, y + (y_i*sy), y + (y_i*sy) + sy)
                self.goals.append(goal)

        if size_y % sy != 0:
            for x_i in np.arange(size_x // sx):
                goal = Square(x + (x_i*sx), x + (x_i*sx) + sx, y + size_y - (size_y % sy), y + size_y)
                self.goals.append(goal)

        if size_y % sy != 0 and size_x % sx != 0:
            goal = Square(x + size_x - (size_x % sx), x + size_x, y + size_y - (size_y % sy), y + size_y)
            self.goals.append(goal)


    def check_delete_goals(self, robot):
        for i, goal in enumerate(self.goals):
            if goal.intersect(robot.bounding_box):
                self.goals.remove(goal)
                return True
        return False

    def is_blocked(self, box: "Square"):
        return any([ob.intersect(box) for ob in self.obstacles])

    def get_border_coords(self):
        return [0, self.width, self.width, 0, 0], [0, 0, self.height, self.height, 0]


class Square:
    def __init__(self, x1, x2, y1, y2):
        self.x1, self.x2, self.y1, self.y2 = x1, x2, y1, y2
        self.x_size = x2 - x1
        self.y_size = y2 - y1

    def intersect(self, other):
        intersecting = not (self.x2 <= other.x1 or self.x1 >= other.x2 or self.y2 <= other.y1 or self.y1 >= other.y2)
        inside = (other.x1 >= self.x1 and other.x2 <= self.x2 and other.y1 >= self.y1 and other.y2 <= self.y2)
        return intersecting or inside

    def update_pos(self, x, y):
        self.x1, self.x2, self.y1, self.y2 = x, x + self.x_size, y, y + self.y_size