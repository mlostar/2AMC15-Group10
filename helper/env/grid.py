from typing import List

from matplotlib import pyplot as plt
import numpy as np


class Grid:
    def __init__(self, width, height, p_random=0.0):
        self.p_random = p_random
        self.width = width
        self.height = height
        self.obstacles:List[Square] = []
        self.goals: List[Square] = []
        self.dirt_size_x: float = 0.5
        self.dirt_size_y: float = 0.5
        self.is_p_dirt = False
        self.p_dirt = 1.0
        self.small_goal_size_x = 1.0
        self.small_goal_size_y = 1.0

    def is_in_bounds(self, x, y, size_x, size_y):
        return x >= 0 and x + size_x <= self.width and y >= 0 and y + size_y <= self.height

    def put_obstacle(self, x, y, size_x, size_y):
        assert self.is_in_bounds(x, y, size_x, size_y)
        ob = Square(x, x + size_x, y, y + size_y)
        self.obstacles.append(ob)

    def set_dirt_size(self, dirt_size_x, dirt_size_y):
        self.dirt_size_x = dirt_size_x
        self.dirt_size_y = dirt_size_y

    def set_p_dirt(self, is_p_dirt, p_dirt=0.8):
        self.is_p_dirt = is_p_dirt
        self.p_dirt = p_dirt

    def set_small_goal_size(self, small_goal_sx, small_goal_sy):
        self.small_goal_size_x = small_goal_sx
        self.small_goal_size_y = small_goal_sy

    def put_goal(self, x, y, size_x, size_y):
        assert self.is_in_bounds(x, y, size_x, size_y)
        # We split the dirt tile into sx by sy blocks
        sx = self.dirt_size_x
        sy = self.dirt_size_y
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

    def put_random_goal(self, x, y, size_x, size_y):
        assert self.is_in_bounds(x, y, size_x, size_y)
        # We split the dirt tile into sx by sy blocks
        rand_size_x = np.random.rand() * size_x
        rand_size_y = np.random.rand() * size_y
        rand_x = x+np.random.rand()*(size_x - rand_size_x)
        rand_y = y+np.random.rand()*(size_y - rand_size_y)
        if self.p_dirt:
            if np.random.rand() <= self.p_dirt:
                self.put_goal(rand_x, rand_y, rand_size_x, rand_size_y)
        else:
            self.put_goal(rand_x, rand_y, rand_size_x, rand_size_y)

    def put_small_goal(self, x, y, size_x, size_y):
        assert self.is_in_bounds(x, y, size_x, size_y)
        # We split the dirt tile into sx by sy blocks
        sx = self.small_goal_size_x
        sy = self.small_goal_size_y
        for x_i in np.arange(size_x // sx):
            for y_i in np.arange(size_y // sy):
                self.put_random_goal(x + (x_i * sx),  y + (y_i * sy), sx, sy)
        # Then add all remainder goals
        if size_x % sx != 0:
            for y_i in np.arange(size_y // sy):
                self.put_random_goal(x + size_x - (size_x % sx), y + (y_i * sy), size_x % sx, sy)

        if size_y % sy != 0:
            for x_i in np.arange(size_x // sx):
                self.put_random_goal(x + (x_i * sx), y + size_y - (size_y % sy), sx, size_y % sy)

        if size_y % sy != 0 and size_x % sx != 0:
            self.put_random_goal(x + size_x - (size_x % sx), y + size_y - (size_y % sy), size_x % sx, size_y % sy)

    def get_intersected_goals(self, robot):
        intersected_goals = []

        for i, goal in enumerate(self.goals):
            if goal.intersect(robot.bounding_box):
                intersected_goals.append(goal)

        return intersected_goals

    def remove_goals(self, goals):
        for i, goal in enumerate(goals):
            self.goals.remove(goal)

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