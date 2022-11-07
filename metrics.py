"""
Code for FTU and DP metrics was provided by the DECAF authors as part of our
email correspondence with them.
"""

from statistics import harmonic_mean

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

columns_adult = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "income",
]

columns_credit = [
    "male",
    "age",
    "debt",
    "married",
    "bankcustomer",
    "educationlevel",
    "ethnicity",
    "yearsemployed",
    "priordefault",
    "employed",
    "creditscore",
    "driverslicense",
    "citizen",
    "zip",
    "income",
    "approved",
]


def DP(mlp, X_test, dataset="adult"):
    """Calculate fairness metric DP"""

    columns = columns_adult if dataset == "adult" else columns_credit
    X_test_df = pd.DataFrame(X_test, columns=columns[:-1])
    if "ethnicity" in X_test_df:
        X_test_0 = X_test_df[X_test_df["ethnicity"] < 0.5]
        X_test_1 = X_test_df[X_test_df["ethnicity"] > 0.5]
    else:
        X_test_0 = X_test_df[X_test_df["sex"] < 0.5]
        X_test_1 = X_test_df[X_test_df["sex"] > 0.5]
    dp = abs(np.mean(mlp.predict(X_test_0)) - np.mean(mlp.predict(X_test_1)))

    return dp


def FTU(mlp, X_test, dataset="adult"):
    """Calculate fairness metric FTU"""

    columns = columns_adult if dataset == "adult" else columns_credit
    X_test_df = pd.DataFrame(X_test, columns=columns[:-1])
    if "ethnicity" in X_test_df:
        X_test_0 = X_test_df.assign(ethnicity=0)
        X_test_1 = X_test_df.assign(ethnicity=1)
    else:
        X_test_0 = X_test_df.assign(sex=0)
        X_test_1 = X_test_df.assign(sex=1)

    ftu = abs(np.mean(mlp.predict(X_test_0)) - np.mean(mlp.predict(X_test_1)))

    return ftu


def eval_model(dataset_train, dataset_test):
    """Helper function that prints evaluation metrics."""

    dataset = "credit" if "approved" in dataset_train.columns else "adult"
    label_col = "approved" if "approved" in dataset_train.columns else "income"

    X_train, y_train = dataset_train.drop(columns=[label_col]), dataset_train[label_col]
    X_test, y_test = dataset_test.drop(columns=[label_col]), dataset_test[label_col]

    clf = MLPClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1score = harmonic_mean([precision, recall])
    auroc = roc_auc_score(y_test, y_pred)
    dp = DP(clf, X_test, dataset=dataset)
    ftu = FTU(clf, X_test, dataset=dataset)

    dp_f1 = harmonic_mean([1 - dp, f1score])
    dp_auroc = harmonic_mean([1 - dp, auroc])
    ftu_f1 = harmonic_mean([1 - ftu, f1score])
    ftu_auroc = harmonic_mean([1 - ftu, auroc])

    return {
        "precision": precision,
        "recall": recall,
        "f1-score": f1score,
        "auroc": auroc,
        "ftu": ftu,
        "dp": dp,
        "ftu-f1": ftu_f1,
        "ftu-auroc": ftu_auroc,
        "dp-f1": dp_f1,
        "dp-auroc": dp_auroc,
    }
