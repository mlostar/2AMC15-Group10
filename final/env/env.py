from copy import deepcopy
from pathlib import Path

import numpy as np
from gym import Env
from gym import spaces
from matplotlib import pyplot as plt

from final.env.robot import Robot
from final.utils.parsing import parse_config


class FloorCleaning(Env):
    # TODO: perhaps it's best to not pass the grid and robot objects on construction
    def __init__(self, config):
        self.rendering_init = False
        grid = config["grid"]
        robot = config["robot"]

        self._original_grid = deepcopy(grid)
        self._original_robot = deepcopy(robot)
        self._robot = deepcopy(robot)
        self._grid = deepcopy(grid)
        self.observation_space = spaces.Dict({
            "distances to borders": spaces.Box(
                low=0,
                high=max(grid.width, grid.height),
                shape=(4,),
                dtype=np.float64
            ),
            "distances to patches": spaces.Box(
                low=0,
                high=max(grid.width, grid.height),
                shape=(4,),
                dtype=np.float64
            )
        })
        self.action_space = spaces.Box(low=0, high=2 * np.pi, shape=(1,))
        self.reward_structure = {
            "wall": -2.,
            "obstacle": -1.,
            "regular": 0.,
            "goal": 1.,
        }

        assert self._grid.is_in_bounds(
            self._robot.bounding_box.x1,
            self._robot.bounding_box.y1,
            self._robot.bounding_box.x_size,
            self._robot.bounding_box.y_size
        )
        assert not self._grid.is_blocked(robot.bounding_box)

    def step(self, action):
        assert self._robot.alive
        done = False

        # Compute the move vector
        if np.random.binomial(n=1, p=self._grid.p_random) == 1:
            action = self.action_space.sample()
        move_vector = np.array([np.cos(action[0]), np.sin(action[0])]) * self._robot.move_distance

        # Set the new position
        new_pos = tuple(np.array(self._robot.pos) + move_vector)
        # Temporarily set the new bounding box to check if it is valid
        new_box = deepcopy(self._robot.bounding_box)
        new_box.update_pos(*new_pos)

        if self._grid.is_blocked(new_box):
            return self._make_observation(), self.reward_structure["obstacle"], False, {}
        elif not self._grid.is_in_bounds(new_pos[0], new_pos[1], self._robot.size, self._robot.size):
            return self._make_observation(), self.reward_structure["wall"], False, {}
        else:
            do_battery_drain = np.random.binomial(1, self._robot.battery_drain_p)

            if do_battery_drain == 1 and self._robot.battery_lvl > 0:
                self._robot.battery_lvl -= (np.random.exponential(self._robot.battery_drain_lam))
                if self._robot.battery_lvl <= 0:
                    self._robot.alive = False
                    self._robot.battery_lvl = 0

                    return self._make_observation(), self.reward_structure["regular"], True, {"reason": "battery drain"}

            del new_box
            self._robot.pos = new_pos
            self._robot.bounding_box.update_pos(*self._robot.pos)

            # What to do if the robot made a valid move, but there were no more goals to begin with
            n_remaining_goals = len(self._grid.goals)
            if n_remaining_goals == 0:
                self._robot.alive = False
                raise AssertionError("Tried to make a step when all of the goals are cleared.")

            # What to do if the robot made a valid move and there were still valid goals left
            intersected_goals = self._grid.get_intersected_goals(self._robot)
            self._grid.remove_goals(intersected_goals)
            if len(intersected_goals) == n_remaining_goals:
                return self._make_observation(), self.reward_structure["goal"], True, {"reason": "goal cleared"}
            elif n_remaining_goals > len(intersected_goals) > 0:
                return self._make_observation(), self.reward_structure["goal"], False, {"reason": "goal cleared"}
            elif len(intersected_goals) == 0:
                return self._make_observation(), self.reward_structure["regular"], False, {"reason": "no intersections"}

            raise AssertionError()

    def _make_observation(self):
        robot_center = (self._robot.bounding_box.x1 + self._robot.bounding_box.x2) / 2, \
                       (self._robot.bounding_box.y1 + self._robot.bounding_box.y2) / 2
        # Compute the distances (n, e, s, w) to either obstacles or walls depending on which one comes first
        distances_to_obstacles = self._get_distances_from_reference(
            reference_point=robot_center,
            squares=self._grid.obstacles,
            fallback_n=robot_center[1],
            fallback_e=self._grid.width - robot_center[0],
            fallback_s=self._grid.height - robot_center[1],
            fallback_w=robot_center[0]
        )
        # Compute the distances to patches. If there are none, the sensor would return the maximum visible distance.
        distances_to_patches = self._get_distances_from_reference(
            reference_point=robot_center,
            squares=self._grid.goals,
            fallback_n=distances_to_obstacles[0],
            fallback_e=distances_to_obstacles[1],
            fallback_s=distances_to_obstacles[2],
            fallback_w=distances_to_obstacles[3]
        )

        return {
            "distances to borders": np.array(distances_to_obstacles),
            "distances to patches": np.array(distances_to_patches)
        }

    @staticmethod
    def _get_distances_from_reference(reference_point, squares, fallback_n, fallback_e, fallback_s, fallback_w):
        distances_e = [ob.x1 - reference_point[0] for ob in squares
                       if ob.y1 <= reference_point[1] <= ob.y2 and ob.x1 >= reference_point[0]]
        distances_w = [reference_point[0] - ob.x2 for ob in squares
                       if ob.y1 <= reference_point[1] <= ob.y2 and ob.x2 <= reference_point[0]]
        distances_n = [reference_point[1] - ob.y2 for ob in squares
                       if ob.x1 <= reference_point[0] <= ob.x2 and ob.y2 <= reference_point[1]]
        distances_s = [ob.y1 - reference_point[1] for ob in squares
                       if ob.x1 <= reference_point[0] <= ob.x2 and ob.y1 >= reference_point[1]]
        nearest_distance_e = fallback_e if not distances_e else min(distances_e)
        nearest_distance_w = fallback_w if not distances_w else min(distances_w)
        nearest_distance_n = fallback_n if not distances_n else min(distances_n)
        nearest_distance_s = fallback_s if not distances_s else min(distances_s)

        return nearest_distance_n, nearest_distance_e, nearest_distance_s, nearest_distance_w

    def render(self, mode="human"):
        if self.rendering_init == False:
            plt.ion()
            plt.gcf()
            plt.show()
            self.rendering_init = True

        plt.gcf()
        plt.plot(*self._grid.get_border_coords(), color='black')

        for goal in self._grid.goals:
            plt.plot(
                [goal.x1, goal.x2, goal.x2, goal.x1, goal.x1],
                [goal.y1, goal.y1, goal.y2, goal.y2, goal.y1],
                color='orange'
            )

        for ob in self._grid.obstacles:
            plt.plot([ob.x1, ob.x2, ob.x2, ob.x1, ob.x1], [ob.y1, ob.y1, ob.y2, ob.y2, ob.y1], color='black')

        robot_box = self._robot.bounding_box
        plt.plot(
            [robot_box.x1, robot_box.x2, robot_box.x2, robot_box.x1, robot_box.x1],
            [robot_box.y1, robot_box.y1, robot_box.y2, robot_box.y2, robot_box.y1],
            color='blue'
        )
        plt.title(f"Battery level: {str(round(self._robot.battery_lvl, 2))}")
        plt.draw()
        plt.pause(0.0001)
        plt.clf()

    def reset(self):
        # TODO: select a random location for the robot
        self.__init__(dict(grid=self._original_grid, robot=self._original_robot))

        return self._make_observation()

    @property
    def grid(self):
        return deepcopy(self._grid)


if __name__ == "__main__":
    from gym.utils.env_checker import check_env

    # Check if the environment conforms to the Gym API
    grid = parse_config(Path("../assets") / "complex.grid")
    robot = Robot(init_position=(0, 0))
    check_env(FloorCleaning(grid, robot))
