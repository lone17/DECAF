# Data Utility Improvement Experiment for DECAF

This repository contains experiments on improving data utility of DECAF using alternating graph during synthesization.

The method is introduced in the paper **DECAF: Generating Fair Synthetic Data Using Causally-Aware Generative networks.**
The original code of DECAF paper is [here]( https://github.com/vanderschaarlab/DECAF).

As the official implementation is imcomplete, the implementation in this repo is adapted from this work **Replication Study of DECAF: Generating Fair Synthetic Data Using Causally-Aware Generative**, whose code can be found [here](https://github.com/ShuaiWang97/UvA_FACT2022).

## Installation

```
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

## Showing the results

The experiments results are precomputed and can be shown using

```
python results.py
```

## Re-run the experiments

The experiments are implemented in the `experiment.py` script
```
python experiment.py
```