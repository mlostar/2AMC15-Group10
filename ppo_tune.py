from pathlib import Path

import plotly.express as px
import ray
from ray.rllib.agents.ppo import PPOTrainer
from ray.tune import report, run, uniform
from ray.tune.suggest.bohb import TuneBOHB

from helper.env.env import FloorCleaning
from helper.env.robot import Robot
from helper.evaluation import get_cleaning_efficiency
from helper.utils.parsing import parse_config

parent_path = Path(".").resolve().parent
grid = parse_config(Path(".").parent/"assets"/"complex_p_dirt.grid")

N_EPOCHS = 12
MAX_EVAL_STEPS = 100

ray.init(object_store_memory=10 ** 9)


def train(config):
    try:
        robot = Robot(init_position=(0, 8))

        env = FloorCleaning({"robot": robot, "grid": grid})
        trainer = PPOTrainer(env=FloorCleaning, config={"env_config": {"robot": robot, "grid": grid},
                                                        "num_workers": 0,
                                                        "horizon": 300,
                                                        # "framework": "torch",
                                                        "grad_clip": 4.0,
                                                        "model": {"fcnet_hiddens": [256],
                                                                  "fcnet_activation": "relu"},
                                                        **config})

        for e in range(N_EPOCHS):
            trainer.train()
            # checkpoint_path = trainer.save(checkpoint_dir=parent_path/"checkpoints")

            cleaning_efficiency = get_cleaning_efficiency(
                env=env,
                action_maker=trainer.compute_single_action,
                max_steps=MAX_EVAL_STEPS
            )

        print(cleaning_efficiency)
        report(efficiency=cleaning_efficiency)
    except InterruptedError:
        print("Interrupted")


parameters = {"gamma": uniform(0, 1.0),
              "lr": uniform(0.0001, 0.1)}
analysis = run(
    train,
    search_alg=TuneBOHB(metric="efficiency", mode="max"),
    config=parameters,
    time_budget_s=3600,
    num_samples=-1,
    resources_per_trial={'cpu': 1},
)
analysis_df = analysis.results_df
analysis_df.to_csv("ppo_results.csv", index=False)

print(analysis_df)
parameter_names = list(parameters.keys())
fig = px.scatter(analysis_df, x=f"config.{parameter_names[0]}", y=f"config.{parameter_names[1]}", color="efficiency")
fig.savefig('ppo.png')
