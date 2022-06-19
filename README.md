# 2AMC15 Assignment 3

## Requirements
This project has a few dependencies. Install it with pip.
```
pip install -r requirements.txt
```

## Running visual agents
Run the visualization of a3c agent.
```
python a3c_visual.py
```

Run the visualization of ppo agent.
```
python ppo_visual.py
```

## Hyperparameter tuning
For the hyper parameter Ray Tune is used.

### Without Ray cluster
Run the a3c tuning.
```
python a3c_tune.py
```

Run the ppo tuning.
```
python ppo_tune.py
```
