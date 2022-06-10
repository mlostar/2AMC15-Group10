from final.env.grid import Square


class Robot:
    def __init__(self, init_position, size=1, battery_drain_p=0, battery_drain_lam=0, move_distance=1.0):
        self.move_distance = move_distance
        self.size = size
        # self.direction_vector = (0, 0)
        self.battery_drain_p = battery_drain_p
        self.battery_drain_lam = battery_drain_lam
        self.battery_lvl = 100
        self.alive = True

        self.pos = init_position
        self.bounding_box = None
        self.set_position(*init_position)

    def set_position(self, start_x=0, start_y=0):
        self.bounding_box = Square(
            start_x,
            start_x + self.size,
            start_y,
            start_y + self.size
        )
