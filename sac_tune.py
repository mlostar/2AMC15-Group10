import time
from pathlib import Path

import plotly.express as px
import ray
from ray.rllib.agents.sac import SACTrainer
from ray.tune import report, run, uniform, choice
from ray.tune.suggest.bohb import TuneBOHB
from ray.tune.schedulers.hb_bohb import HyperBandForBOHB

from helper.env.env import FloorCleaning
from helper.env.robot import Robot
from helper.evaluation import get_cleaning_efficiency
from helper.utils.parsing import parse_config


N_EPOCHS = 12
MAX_EVAL_STEPS = 100
LOCAL_PORT = 10001
OBJECT_STORE_MEMORY = 10 ** 9
TIME_BUDGET_S= 3600

parent_path = Path(".").resolve().parent
grid = parse_config(Path(".").parent/"assets"/"complex_p_dirt.grid")


def train(config):
    try:
        robot = Robot(init_position=(0, 8))

        env = FloorCleaning({"robot": robot, "grid": grid})
        trainer = SACTrainer(env=FloorCleaning, config={"env_config": {"robot": robot, "grid": grid},
                                                        "horizon": 300,
                                                        "learning_starts": 150,
                                                        **config})

        for e in range(N_EPOCHS):
            trainer.train()
            # checkpoint_path = trainer.save(checkpoint_dir=parent_path/"checkpoints")

            cleaning_efficiency = get_cleaning_efficiency(
                env=env,
                action_maker=trainer.compute_single_action,
                max_steps=MAX_EVAL_STEPS
            )
        
        # print("CLEANING EFFICIENCY: ", cleaning_efficiency)
        report(efficiency=cleaning_efficiency)
    except InterruptedError:
        print("Interrupted")

def tune_search(parameters):
    analysis = run(
        train,
        search_alg=TuneBOHB(metric="efficiency", mode="max"),
        scheduler=HyperBandForBOHB(metric="efficiency", mode="max", max_t=100, stop_last_trials=False),
        config=parameters,
        time_budget_s=TIME_BUDGET_S,
        num_samples=-1,
        resources_per_trial={'cpu': 4},
    )
    
    return analysis.results_df

def main():
    parameters = {"optimization": {
                        "actor_learning_rate": uniform(3e-5, 3e-3),
                        "critic_learning_rate": uniform(3e-5, 3e-3),
                        "entropy_learning_rate": uniform(3e-5, 3e-3),
                    },
                    "initial_alpha": uniform(0.8, 1.0),
                    "tau": uniform(5e-4, 5e-2),
                    "lr": uniform(1e-5, 1e-4),
                    "twin_q": choice([True, False])
                    }

    analysis_df = tune_search(parameters)
    
    analysis_df.to_csv("sac_results.csv", index=False)

    parameter_names = list(parameters.keys())
    # fig = px.scatter(analysis_df, x=f"config.{parameter_names[0]}", y=f"config.{parameter_names[1]}", color="efficiency")
    # fig.show()
    # TODO: Use matplotlib

if __name__ == "__main__":
    ray.init(object_store_memory=OBJECT_STORE_MEMORY, log_to_driver=False)
    
    start_time = time.time()
    main()

    stop_time = time.time()
    print(f"Total elapsed time: {round(stop_time - start_time)}s")
