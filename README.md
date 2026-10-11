# Bayesian Neural Networks with Conformal Calibration

Code for our paper "Adaptive Prediction Intervals for Concrete Compressive Strength by Integrating Bayesian Neural Networks with Conformal Calibration".

## Data

The dataset isn't included here. Download it from the [UCI repository](https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength) and put it in `Datasets/concrete_strength.xls`.

## Running

```bash
pip install -r requirements.txt
python hyperparameter_study.py
python run_experiments.py
python repeated_experiments.py
```

Run `hyperparameter_study.py` first, since `run_experiments.py` reads the selected architecture from its output. `repeated_experiments.py` runs the 20 random splits.
