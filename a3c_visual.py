from pathlib import Path

import numpy as np
from ray.rllib.agents.a3c import A3CTrainer

from helper.env.env import FloorCleaning
from helper.env.robot import Robot
from helper.evaluation import get_cleaning_efficiency
from helper.utils.parsing import parse_config

parent_path = Path(".").resolve().parent
grid = parse_config(Path(".").parent/"assets"/"complex_p_dirt.grid")
robot = Robot(init_position=(0, 8))


trainer = A3CTrainer(env=FloorCleaning, config={"env_config": {"robot": robot, "grid": grid},
                                                # Learning rate
                                                "lr": 0.0001,
                                                # Entropy coefficient
                                                "entropy_coeff": 0.01})

env = FloorCleaning(dict(robot=robot, grid=grid))

checkpoint_path = None
for e in range(20):
    print(trainer.train())
    #checkpoint_path = trainer.save(checkpoint_dir=parent_path/"checkpoints")

    efficiency = get_cleaning_efficiency(env, lambda o: trainer.compute_single_action(o), max_steps=100)
    print(f"Epoch: {e} -- Efficiency: {efficiency}")


#trainer.restore(checkpoint_path)
for e in range(10):
    obs = env.reset()
    env.render()

    for s in range(1000):
        move = trainer.compute_action(obs)
        obs, reward, has_ended, info = env.step(move)
        env.render()
        print(f"move: {move/(2*np.pi)*360}, reward: {reward}")

        if has_ended:
            print(f"Game over: {info['reason']}")
            break
