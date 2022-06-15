from pathlib import Path

import gym
import ray
from ray.rllib.agents.dqn import DQNTrainer
from ray.tune import register_env, report, run, uniform
from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest.bayesopt import BayesOptSearch
import plotly.express as px
from final.env.env import FloorCleaning
from final.env.robot import Robot
from final.evaluation import get_cleaning_efficiency
from final.utils.parsing import parse_config

from final.wrappers.descretiser import Discretiser

parent_path = Path(".").resolve().parent
grid = parse_config(parent_path / "assets" / "simple.grid")


N_EPOCHS = 3
MAX_EVAL_STEPS = 100
ray.init(object_store_memory=10**9)


def train(config):
    robot = Robot(init_position=(0, 8))

    env = Discretiser(FloorCleaning({"robot": robot, "grid": grid}), n_splits=10)
    register_env('FloorCleaning',
                 lambda env_config: Discretiser(FloorCleaning({"robot": robot, "grid": grid}), n_splits=10))
    trainer = DQNTrainer(env="FloorCleaning", config=config)

    for _ in range(N_EPOCHS):
        trainer.train()
        # checkpoint_path = trainer.save(checkpoint_dir=parent_path/"checkpoints")

        cleaning_efficiency = get_cleaning_efficiency(
            env=env,
            action_maker=trainer.compute_single_action,
            max_steps=MAX_EVAL_STEPS
        )
        print(cleaning_efficiency)
        report(efficiency=cleaning_efficiency)


parameters = {"gamma": uniform(0, 1.0),
              "lr": uniform(0.0001, 0.1),
              #"num_workers": 0,
              "horizon": 300}
analysis = run(
    train,
    search_alg=BayesOptSearch(metric="efficiency", mode="max"),
    scheduler=ASHAScheduler(metric="efficiency", mode="max"),
    config=parameters,
    time_budget_s=300,
    num_samples=-1,
    resources_per_trial={'cpu': 8, 'gpu': 1},
)
analysis_df = analysis.results_df

print(analysis_df)
parameter_names = list(parameters.keys())
fig = px.scatter(analysis_df, x=f"config.{parameter_names[0]}", y=f"config.{parameter_names[1]}", color="efficiency")
fig.show()
