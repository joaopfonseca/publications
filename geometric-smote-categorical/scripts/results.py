from os.path import join
from itertools import product
from copy import deepcopy
from rich.progress import track

import numpy as np
from sklearn.preprocessing._encoders import _BaseEncoder, OneHotEncoder
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTENC

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

    def fit(self, X, y=None):
        # NOTE: X must be a numpy array
        return self.ohe.fit(X[:, self.categorical_features], y)

    def transform(self, X):
        # NOTE: X must be a numpy array
        metric_data = X[:, ~self.categorical_features]
        encoded_data = self.ohe.transform(X[:, self.categorical_features])
        return np.concatenate([metric_data, encoded_data], axis=1).astype(np.float64)

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)


CONFIG = {
    "oversamplers": [
        ("NONE", None, {}),
        ("SMOTENC", SMOTENC(categorical_features=None), {}),
        ("G-SMOTE", GeometricSMOTE(), {}),
    ],
    "classifiers": [
        ("CONSTANT", DummyClassifier(strategy="prior"), {}),
        ("LR", LogisticRegression(), {}),
        ("KNN", KNeighborsClassifier(), {}),
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

    for dataset, oversampler in track(list(product(datasets, CONFIG["oversamplers"]))):

        categorical_features = dataset[1][0].columns.str.startswith("cat_")

        # Set up dataset-specific params for oversampler
        oversampler = deepcopy(oversampler)
        if hasattr(oversampler[1], "categorical_features"):
            oversampler[1].set_params(categorical_features=categorical_features)

        # Set up one hot encoder
        ohe = (
            "OHE",
            OHECustom(categorical_features=categorical_features, sparse=False),
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
        file_name = (
            f'{dataset[0].replace(" ", "_").lower()}__'
            + f'{oversampler[0].replace("-", "").lower()}.pkl'
        )
        experiment.results_.to_pickle(join(RESULTS_PATH, file_name))
