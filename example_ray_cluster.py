import sys
import time
from collections import Counter
import helper

import ray
from pathlib import Path

import numpy as np
from ray.rllib.agents.a3c import A3CTrainer

from helper.env.env import FloorCleaning
from helper.env.robot import Robot
from helper.evaluation import get_cleaning_efficiency
from helper.utils.parsing import parse_config

""" Run this script locally to execute a Ray program on your Ray cluster on
Kubernetes.

Before running this script, you must port-forward from the local host to
the relevant Kubernetes head service e.g.
kubectl -n ray port-forward service/example-cluster-ray-head 10001:10001.

Set the constant LOCAL_PORT below to the local port being forwarded.
"""
NUMBER_OF_NODES = 1
LOCAL_PORT = 10001
runtime_env = {"working_dir": "./", "py_modules": [helper], "pip": ["tensorflow"]}

parent_path = Path(".").resolve().parent
grid = parse_config(Path(".").parent/"assets"/"simple.grid")
robot = Robot(init_position=(0, 8))
env = FloorCleaning(dict(robot=robot, grid=grid))

@ray.remote
def train(robot, grid):
    trainer = A3CTrainer(env=FloorCleaning, config={"env_config": {"robot": robot, "grid": grid},
                                            # Learning rate
                                            "lr": 0.0001,
                                            # Entropy coefficient
                                            "entropy_coeff": 0.01})
    trainer.train()
    efficiency = get_cleaning_efficiency(env, lambda o: trainer.compute_single_action(o), max_steps=100)
    return efficiency


def wait_for_nodes(expected):
    # Wait for all nodes to join the cluster.
    while True:
        resources = ray.cluster_resources()
        node_keys = [key for key in resources if "node" in key]
        num_nodes = sum(resources[node_key] for node_key in node_keys)
        if num_nodes < expected:
            print(
                "{} nodes have joined so far, waiting for {} more.".format(
                    num_nodes, expected - num_nodes
                )
            )
            sys.stdout.flush()
            time.sleep(1)
        else:
            break


def main():
    wait_for_nodes(NUMBER_OF_NODES)

    result = train.remote(robot, grid)
    print(ray.get(result))
    sys.stdout.flush()

    print("Success!")
    sys.stdout.flush()


if __name__ == "__main__":
    ray.init(f"ray://127.0.0.1:{LOCAL_PORT}", runtime_env=runtime_env, log_to_driver=True)
    main()
