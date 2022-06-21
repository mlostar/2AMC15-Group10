from pathlib import Path

import numpy as np
from ray.rllib.agents.sac import SACTrainer

from helper.env.env import FloorCleaning
from helper.env.robot import Robot
from helper.evaluation import get_cleaning_efficiency
from helper.utils.parsing import parse_config

parent_path = Path(".").resolve().parent
grid = parse_config(Path(".").parent/"assets"/"complex_p_dirt.grid")
robot = Robot(init_position=(0, 8))


# Initialise the trainer and the environment
trainer = SACTrainer(env=FloorCleaning, config={"env_config": {"robot": robot, "grid": grid},
                                                # Learning rate
                                                "lr": 3.13E-05,
                                                "learning_starts": 150,
                                                "optimization": {
                                                    "actor_learning_rate": 0.0002188,
                                                    "critic_learning_rate": 0.0016225,
                                                    "entropy_learning_rate": 0.0009927,
                                                },
                                                "initial_alpha": 0.9995141,
                                                "tau": 0.0290980,
                                                "twin_q": False

                                                })

env = FloorCleaning(dict(robot=robot, grid=grid))

# Train the model
checkpoint_path = None
for e in range(20):
    print(trainer.train())
    #checkpoint_path = trainer.save(checkpoint_dir=parent_path/"checkpoints")

    efficiency = get_cleaning_efficiency(env, lambda o: trainer.compute_single_action(o), max_steps=100)
    print(f"Epoch: {e} -- Efficiency: {efficiency}")

# Play the environment with a visualisation
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
