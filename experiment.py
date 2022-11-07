from collections import defaultdict

import numpy as np
import pandas as pd

from data import load_adult, preprocess_adult
from metrics import eval_model
from train import gen_decaf, train_decaf

# Define DAG for Adult dataset
DAG = {
    "age": [
        "occupation",
        "hours-per-week",
        "income",
        "workclass",
        "marital-status",
        "education",
        "relationship",
    ],
    "education": [
        "occupation",
        "hours-per-week",
        "income",
        "workclass",
        "relationship",
    ],
    "hours-per-week": ["income"],
    "marital-status": [
        "occupation",
        "hours-per-week",
        "income",
        "workclass",
        "relationship",
        "education",
    ],
    "native-country": [
        "marital-status",
        "hours-per-week",
        "education",
        "workclass",
        "income",
        "relationship",
    ],
    "occupation": ["income"],
    "race": ["occupation", "income", "hours-per-week", "education", "marital-status"],
    "relationship": ["income"],
    "sex": [
        "occupation",
        "marital-status",
        "income",
        "workclass",
        "education",
        "relationship",
    ],
    "workclass": ["income"],
}


def dag_to_idx(df, dag):
    """Convert columns in a DAG to the corresponding indices."""

    dag_idx = []
    for node, children in dag.items():
        for child in children:
            dag_idx.append([df.columns.get_loc(node), df.columns.get_loc(child)])

    return dag_idx


def create_bias_dict(df, edge_map):
    """
    Convert the given edge tuples to a bias dict used for generating
    debiased synthetic data.
    """
    bias_dict = {}
    for key, val in edge_map.items():
        bias_dict[df.columns.get_loc(key)] = [df.columns.get_loc(f) for f in val]

    return bias_dict


def train_models(num_runs=10):
    dataset_train = preprocess_adult(load_adult())
    dataset_test = preprocess_adult(load_adult(load_test=True))

    dag_seed = dag_to_idx(dataset_train, DAG)

    bias_dicts = {
        "nd": {},
        "ftu": create_bias_dict(dataset_train, {"income": ["sex"]}),
        "cf": create_bias_dict(
            dataset_train, {"income": ["marital-status", "sex", "relationship"]}
        ),
        "dp": create_bias_dict(
            dataset_train,
            {
                "income": [
                    "occupation",
                    "hours-per-week",
                    "marital-status",
                    "education",
                    "sex",
                    "workclass",
                    "relationship",
                ]
            },
        ),
        "cf-y": create_bias_dict(
            dataset_train,
            {
                "income": ["sex", "marital-status"],
                "relationship": ["sex"],
            },
        ),
        "dp-y": create_bias_dict(
            dataset_train,
            {
                "income": ["sex"],
                "occupation": ["sex"],
                "marital-status": ["sex"],
                "workclass": ["sex"],
                "education": ["sex"],
                "relationship": ["sex"],
            },
        ),
    }

    results = defaultdict(dict)
    results["original"] = defaultdict(list)
    for ver in bias_dicts.keys():
        results[f"decaf_{ver}"] = defaultdict(list)

    for run in range(num_runs):
        model = train_decaf(
            dataset_train, model_name=f"decaf_run_{run+1}", dag_seed=dag_seed, epochs=50
        )
        for ver, bias_edges in bias_dicts.items():

            synth_data = gen_decaf(model, dataset_train, bias_edges)

            if ver.endswith("-y"):
                synth_X = gen_decaf(model, dataset_train, bias_dicts[ver[:-2]])
                synth_X["income"] = synth_data["income"]
                synth_data = synth_X

            model_results = eval_model(synth_data, dataset_test)

            for key, value in model_results.items():
                results[f"decaf_{ver}"][key].append(value)

        # Original
        model_results = eval_model(dataset_train, dataset_test)
        for key, value in model_results.items():
            results["original"][key].append(value)

    return results


def results_df(results):
    def formatter(results):
        return f"{np.mean(results):.3f}±{np.std(results):.3f}"

    cols = ["model"] + list(results["original"].keys())
    rows = list(results.keys())

    df = pd.DataFrame(np.zeros((len(rows), len(cols))), columns=cols)
    df["model"] = rows

    for model, model_results in results.items():
        for col in cols[1:]:
            df.loc[df["model"] == model, col] = formatter(model_results[col])

    return df


if __name__ == "__main__":
    results = train_models(num_runs=10)
    df = results_df(results)
    print(df)
