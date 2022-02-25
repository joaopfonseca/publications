from os.path import join
from itertools import product
from copy import deepcopy
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTENC
from mlresearch.data_augmentation import GeometricSMOTE
from mlresearch.utils import load_datasets, generate_paths, check_pipelines
from rlearn.tools import ImbalancedExperiment


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
}


if __name__ == "__main__":

    DATA_PATH, RESULTS_PATH, ANALYSIS_PATH = generate_paths(__file__)

    # Get assets
    datasets = load_datasets(DATA_PATH)
    oversamplers = CONFIG["oversamplers"]
    classifiers = CONFIG["classifiers"]

    for dataset, oversampler in product(datasets, oversamplers):

        categorical_features = dataset[1][0].columns.str.startswith("cat_")

        oversampler = deepcopy(oversampler)
        if hasattr(oversampler[1], "categorical_features"):
            oversampler[1].set_params(categorical_features=categorical_features)

        pipelines = check_pipelines(
            [[oversampler], classifiers],
            random_state=CONFIG["random_state"],
            n_runs=CONFIG["n_runs"],
        )

        # Define and fit experiment
        experiment = ImbalancedExperiment().fit(dataset)

        file_name = (
            f'{dataset[0].replace(" ", "_").lower()}__'
            + f'{oversampler[0].replace("-", "").lower()}.pkl'
        )
        experiment.results_.to_pickle(join(RESULTS_PATH, file_name))
