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

```
        model    precision       recall     f1-score        auroc          ftu           dp       ftu-f1    ftu-auroc        dp-f1     dp-auroc
0    original  0.877±0.005  0.930±0.007  0.903±0.001  0.764±0.008  0.030±0.009  0.175±0.012  0.935±0.004  0.855±0.006  0.862±0.007  0.793±0.002
1    decaf_nd  0.887±0.021  0.758±0.038  0.816±0.015  0.729±0.023  0.089±0.043  0.347±0.061  0.861±0.023  0.809±0.017  0.724±0.041  0.686±0.024
2   decaf_ftu  0.888±0.016  0.759±0.031  0.818±0.013  0.732±0.018  0.028±0.020  0.297±0.039  0.888±0.011  0.835±0.013  0.756±0.027  0.716±0.016
3    decaf_cf  0.777±0.013  0.879±0.042  0.824±0.015  0.551±0.028  0.036±0.022  0.041±0.029  0.889±0.013  0.701±0.026  0.886±0.018  0.699±0.018
4    decaf_dp  0.762±0.009  0.914±0.034  0.831±0.015  0.518±0.021  0.021±0.021  0.016±0.012  0.899±0.010  0.677±0.014  0.901±0.010  0.679±0.017
5  decaf_cf-y  0.758±0.009  0.970±0.025  0.851±0.006  0.509±0.021  0.012±0.011  0.021±0.021  0.914±0.007  0.672±0.016  0.910±0.011  0.669±0.014
6  decaf_dp-y  0.756±0.005  0.976±0.020  0.852±0.007  0.504±0.012  0.014±0.010  0.020±0.019  0.914±0.008  0.667±0.010  0.911±0.011  0.665±0.013
```

## Re-run the experiments

The experiments are implemented in the `experiment.py` script
```
python experiment.py
```
