from copy import deepcopy
from typing import Tuple, Optional, Union

import numpy as np
from gym import Env, register
from gym import spaces
from gym.core import ActType, ObsType
from matplotlib import pyplot as plt

from final.grid import Grid
from final.robot import Robot
from final.util import parse_config


class FloorCleaning(Env):
    def __init__(self, grid: Grid, robot: Robot):
        self._original_grid = deepcopy(grid)
        self._original_robot = deepcopy(robot)
        self._robot = robot
        self._grid = grid
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
        assert not self._grid.is_blocked(robot)

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        assert self._robot.alive
        done = False

        # Compute the move vector
        if np.random.binomial(n=1, p=self._grid.p_random) == 1:
            action = self.action_space.sample()
        move_vector = np.array([np.cos(action[0]), np.sin(action[0])]) * self._robot.move_distance

        # Set the new position
        new_pos = tuple(np.array(self._robot.pos) + move_vector)
        # Temporarily set the new bounding box to check if it is physically plausible
        new_box = deepcopy(self._robot.bounding_box)
        new_box.update_pos(*new_pos)
        self.bounding_box = new_box

        if self._grid.is_blocked(self._robot):
            return self._make_observation(), self.reward_structure["wall"], False, {}
        elif not self._grid.is_in_bounds(new_pos[0], new_pos[1], self._robot.size, self._robot.size):
            return self._make_observation(), self.reward_structure["obstacle"], False, {}
        else:
            do_battery_drain = np.random.binomial(1, self._robot.battery_drain_p)

            if do_battery_drain == 1 and self._robot.battery_lvl > 0:
                self._robot.battery_lvl -= (np.random.exponential(self._robot.battery_drain_lam))
                if self._robot.battery_lvl <= 0:
                    self._robot.alive = False
                    self._robot.battery_lvl = 0

                    return self._make_observation(), self.reward_structure["regular"], True, {}

            del new_box
            self._robot.pos = new_pos
            self._robot.bounding_box.update_pos(*self._robot.pos)
            self._robot.history.append(self._robot.bounding_box)

            # What to do if the robot made a valid move with enough battery:
            if self._grid.check_delete_goals(self._robot) and len(self._grid.goals) == 0:
                self._robot.alive = False
                return self._make_observation(), self.reward_structure["goal"], False, {}
            elif self._grid.check_delete_goals(self._robot) and len(self._grid.goals) > 0:
                return self._make_observation(), self.reward_structure["goal"], True, {}
            elif not self._grid.check_delete_goals(self._robot):
                return self._make_observation(), self.reward_structure["regular"], True, {}

    def _make_observation(self):
        robot_center = (self._robot.bounding_box.x1 + self._robot.bounding_box.x2) / 2, \
                       (self._robot.bounding_box.y1 + self._robot.bounding_box.y2) / 2

        distances_e = [ob.x1 - robot_center[0] for ob in self._grid.obstacles
                       if ob.y1 <= robot_center[1] <= ob.y2 and ob.x1 >= robot_center[0]]
        distances_w = [robot_center[0] - ob.x2 for ob in self._grid.obstacles
                       if ob.y1 <= robot_center[1] <= ob.y2 and ob.x2 <= robot_center[0]]
        distances_n = [robot_center[1] - ob.y2 for ob in self._grid.obstacles
                       if ob.x1 <= robot_center[0] <= ob.x2 and ob.y2 <= robot_center[1]]
        distances_s = [ob.y1 - robot_center[1] for ob in self._grid.obstacles
                       if ob.x1 <= robot_center[0] <= ob.x2 and ob.y1 >= robot_center[1]]
        nearest_distance_e = self._grid.width - robot_center[0] if not distances_e else min(distances_e)
        nearest_distance_w = robot_center[0] if not distances_e else min(distances_w)
        nearest_distance_n = robot_center[1] if not distances_e else min(distances_n)
        nearest_distance_s = self._grid.height - robot_center[1] if not distances_e else min(distances_s)

        # TODO: Implement the distances to patches
        return {
            "distances to borders": np.array([
                nearest_distance_n,
                nearest_distance_e,
                nearest_distance_s,
                nearest_distance_w
            ]),
            "distances to patches": np.array([
                nearest_distance_n,
                nearest_distance_e,
                nearest_distance_s,
                nearest_distance_w
            ])
        }

    def render(self, mode="human"):
        fig = plt.figure()
        fig.add_subplot(111)
        plt.plot(*self._grid.get_border_coords(), color='black')

        for goal in self._grid.goals:
            plt.plot(
                [goal.x1, goal.x2, goal.x2, goal.x1, goal.x1],
                [goal.y1, goal.y1, goal.y2, goal.y2, goal.y1],
                color='orange'
            )

        for ob in self._grid.obstacles:
            plt.plot([ob.x1, ob.x2, ob.x2, ob.x1, ob.x1], [ob.y1, ob.y1, ob.y2, ob.y2, ob.y1], color='black')

        robot_box = self._robot.history[-1]
        plt.plot(
            [robot_box.x1, robot_box.x2, robot_box.x2, robot_box.x1, robot_box.x1],
            [robot_box.y1, robot_box.y1, robot_box.y2, robot_box.y2, robot_box.y1],
            color='blue'
        )
        plt.title(f"Battery level: {str(round(self._robot.battery_lvl, 2))}")
        plt.draw()
        plt.pause(0.0001)

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            return_info: bool = False,
            options: Optional[dict] = None,
    ) -> Union[ObsType, Tuple[ObsType, dict]]:
        super().reset(seed=seed)

        # TODO: select a random location for the robot
        self.__init__(grid=self._original_grid, robot=self._original_robot)

        return self._make_observation() if not return_info else (self._make_observation(), {})


register(
    id='gym_examples/FloorCleaning-v0',
    entry_point='final.env:FloorCleaning',
    max_episode_steps=300,
)


if __name__ == "__main__":
    from gym.utils.env_checker import check_env

    # Check if the environment conforms to the Gym API
    grid = parse_config('example.grid')
    robot = Robot(init_position=(0, 0))
    check_env(FloorCleaning(grid, robot))
