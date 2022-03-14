from os import listdir
from os.path import join
from itertools import product
from copy import deepcopy
from rich.progress import track

import numpy as np
import pandas as pd
from sklearn.preprocessing._encoders import _BaseEncoder, OneHotEncoder
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTENC, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

from mlresearch.utils import load_datasets, generate_paths, check_pipelines
from mlresearch.data_augmentation import GeometricSMOTE
from rlearn.model_selection import ModelSearchCV


class OHECustom(_BaseEncoder):
    def __init__(
        self,
        categorical_features=None,
        categories="auto",
        drop=None,
        sparse=False,
        dtype=np.float64,
        handle_unknown="error",
    ):
        # NOTE: categorical features should be an array of type bool
        self.categorical_features = categorical_features
        self.categories = categories
        self.drop = drop
        self.sparse = sparse
        self.dtype = dtype
        self.handle_unknown = handle_unknown
        self.ohe = OneHotEncoder(
            categories=categories,
            drop=drop,
            sparse=sparse,
            dtype=dtype,
            handle_unknown=handle_unknown,
        )

    def _check_X_y(self, X, y):
        if type(X) == pd.DataFrame:
            self.is_pandas_ = (
                True if not hasattr(self, "is_pandas_") else self.is_pandas_
            )
            self.columns_ = X.columns
            X_ = X.copy().values
        else:
            self.is_pandas_ = (
                False if not hasattr(self, "is_pandas_") else self.is_pandas_
            )
            X_ = X.copy()
        return X_, y

    def fit(self, X, y=None):
        X_, y = self._check_X_y(X, y)
        return self.ohe.fit(X_[:, self.categorical_features], y)

    def transform(self, X):
        X_, y = self._check_X_y(X, None)
        if self.is_pandas_:
            metric_data = pd.DataFrame(
                X_[:, ~self.categorical_features],
                columns=self.columns_[~self.categorical_features],
            )
            encoded_data = pd.DataFrame(
                self.ohe.transform(X_[:, self.categorical_features]),
                columns=self.ohe.get_feature_names_out(
                    self.columns_[self.categorical_features]
                ),
            )
            data = pd.concat([metric_data, encoded_data], axis=1)
        else:
            metric_data = X_[:, ~self.categorical_features]
            encoded_data = self.ohe.transform(X_[:, self.categorical_features])
            data = np.concatenate([metric_data, encoded_data], axis=1).astype(
                np.float64
            )

        return data

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)


CONFIG = {
    "oversamplers": [
        ("NONE", None, {}),
        ("SMOTENC", SMOTENC(categorical_features=None), {}),
        ("RAND-OVER", RandomOverSampler(), {"k_neighbors": [3, 5]}),
        ("RAND-UNDER", RandomUnderSampler(), {}),
        (
            "G-SMOTE",
            GeometricSMOTE(),
            {
                "k_neighbors": [3, 5],
                "selection_strategy": ["combined", "minority", "majority"],
                "truncation_factor": [-1.0, -0.5, 0.0, 0.25, 0.5, 0.75, 1.0],
                "deformation_factor": [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0],
            },
        ),
    ],
    "classifiers": [
        ("CONSTANT", DummyClassifier(strategy="prior"), {}),
        (
            "LR",
            LogisticRegression(),
            [
                {"penalty": ["none", "l2"], "solver": "lbfgs"},
                {"penalty": ["l1", "l2"], "solver": "liblinear"},
            ],
        ),
        ("KNN", KNeighborsClassifier(), {"n_neighbors": [3, 5]}),
        ("DT", DecisionTreeClassifier(), {"max_depth": [3, 6]}),
        (
            "RF",
            RandomForestClassifier(),
            {"max_depth": [None, 3, 6], "n_estimators": [10, 50, 100]},
        ),
    ],
    "scoring": ["accuracy", "f1_macro", "geometric_mean_score_macro"],
    "n_splits": 5,
    "n_runs": 3,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": 1,
}


if __name__ == "__main__":

    DATA_PATH, RESULTS_PATH, ANALYSIS_PATH = generate_paths(__file__)

    # Get objects
    datasets = load_datasets(DATA_PATH)

    for dataset, oversampler in track(
        list(product(datasets, CONFIG["oversamplers"])), description="Experiment"
    ):

        file_name = (
            f'{dataset[0].replace(" ", "_").lower()}__'
            + f'{oversampler[0].replace("-", "").lower()}.pkl'
        )

        # Skip experiment if it is already saved
        if file_name in listdir(RESULTS_PATH):
            continue

        categorical_features = dataset[1][0].columns.str.startswith("cat_")
        unique_values = [
            dataset[1][0][cat].unique()
            for cat in dataset[1][0].columns[categorical_features]
        ]

        # Set up dataset-specific params for oversampler
        oversampler = deepcopy(oversampler)
        if hasattr(oversampler[1], "categorical_features"):
            oversampler[1].set_params(categorical_features=categorical_features)

        # Set up one hot encoder
        ohe = (
            "OHE",
            OHECustom(
                categorical_features=categorical_features,
                categories=unique_values,
                sparse=False,
            ),
            {},
        )

        # Set up pipelines
        estimators, param_grids = check_pipelines(
            [[oversampler], [ohe], CONFIG["classifiers"]],
            random_state=CONFIG["random_state"],
            n_runs=CONFIG["n_runs"],
        )

        # Define and fit experiment
        experiment = ModelSearchCV(
            estimators=estimators,
            param_grids=param_grids,
            scoring=CONFIG["scoring"],
            n_jobs=CONFIG["n_jobs"],
            cv=StratifiedKFold(
                n_splits=CONFIG["n_splits"],
                shuffle=True,
                random_state=CONFIG["random_state"],
            ),
            verbose=CONFIG["verbose"],
            return_train_score=True,
            refit=False,
        ).fit(*dataset[1])

        # Save results
        pd.DataFrame(experiment.cv_results_).to_pickle(join(RESULTS_PATH, file_name))
