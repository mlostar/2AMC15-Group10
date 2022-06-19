import time
from pathlib import Path

import plotly.express as px
import ray
from ray.rllib.agents.a3c import A3CTrainer
from ray.tune import report, run, uniform, choice
from ray.tune.suggest.bohb import TuneBOHB

from helper.env.env import FloorCleaning
from helper.env.robot import Robot
from helper.evaluation import get_cleaning_efficiency
from helper.utils.parsing import parse_config


N_EPOCHS = 12
MAX_EVAL_STEPS = 100
LOCAL_PORT = 10001
OBJECT_STORE_MEMORY = 10 ** 9
TIME_BUDGET_S= 1800

parent_path = Path(".").resolve().parent
grid = parse_config(Path(".").parent/"assets"/"complex_p_dirt.grid")


def train(config):
    try:
        robot = Robot(init_position=(0, 8))

        env = FloorCleaning({"robot": robot, "grid": grid})
        trainer = A3CTrainer(env=FloorCleaning, config={"env_config": {"robot": robot, "grid": grid},
                                                        "horizon": 300,
                                                        **config})

        for e in range(N_EPOCHS):
            print(f"Epoch: {e}")
            trainer.train()
            # checkpoint_path = trainer.save(checkpoint_dir=parent_path/"checkpoints")

            cleaning_efficiency = get_cleaning_efficiency(
                env=env,
                action_maker=trainer.compute_single_action,
                max_steps=MAX_EVAL_STEPS
            )
        
        print("CLEANING EFFICIENCY: ", cleaning_efficiency)
        report(efficiency=cleaning_efficiency)
    except InterruptedError:
        print("Interrupted")

def tune_search(parameters):
    analysis = run(
        train,
        search_alg=TuneBOHB(metric="efficiency", mode="max"),
        config=parameters,
        time_budget_s=TIME_BUDGET_S,
        num_samples=250,
        resources_per_trial={'cpu': 4},
    )
    
    return analysis.results_df

def main():
    parameters = {"lambda": uniform(0.5, 1),
                "entropy_coeff": uniform(0.01, 0.1),
                "lr": uniform(0.0001, 0.001),
                "model": {"fcnet_hiddens": [choice([32, 64, 128])], "fcnet_activation": "relu"}
                }

    analysis_df = tune_search(parameters)
    
    analysis_df.to_csv("ppo_results.csv", index=False)

    parameter_names = list(parameters.keys())
    # fig = px.scatter(analysis_df, x=f"config.{parameter_names[0]}", y=f"config.{parameter_names[1]}", color="efficiency")
    # fig.show()
    # TODO: Use matplotlib

if __name__ == "__main__":
    ray.init(object_store_memory=OBJECT_STORE_MEMORY, log_to_driver=True)
    
    start_time = time.time()
    main()

    stop_time = time.time()
    print(f"Total elapsed time: {round(stop_time - start_time)}s")
