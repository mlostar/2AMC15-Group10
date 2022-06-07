from pathlib import Path

import numpy as np
from ray.rllib.agents.dqn import DQNTrainer
from ray.rllib.agents.ppo import PPOTrainer
from ray.tune import register_env

from final.env import FloorCleaning
from final.robot import Robot
from final.util import parse_config
import logging

from final.wrappers.descretiser import Discretiser

parent_path = Path(".").resolve().parent
grid = parse_config(parent_path/"assets"/"example.grid")
robot = Robot(init_position=(0, 8))


register_env('FloorCleaning', lambda env_config: Discretiser(FloorCleaning({"robot": robot, "grid": grid}), n_splits=10))
trainer = DQNTrainer(env="FloorCleaning")

checkpoint_path = None
for _ in range(3):
    print(trainer.train())
    checkpoint_path = trainer.save(checkpoint_dir=parent_path/"checkpoints")


trainer.restore(checkpoint_path)
env = Discretiser(FloorCleaning({"robot": robot, "grid": grid}), n_splits=10)
for e in range(10):
    obs = env.reset()
    env.render()

    for s in range(1000):
        move = trainer.compute_action(obs)
        obs, reward, has_ended, info = env.step(move)
        env.render()
        print(f"move: {move*360}, reward: {reward}")

        if has_ended:
            print(f"Game over: {info['reason']}")
            break
