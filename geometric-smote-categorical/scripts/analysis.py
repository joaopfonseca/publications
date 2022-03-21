"""
Analyze the experimental results.
"""

# Author: João Fonseca <jpfonseca@novaims.unl.pt>
# License: MIT

from os.path import join
from collections import Counter
from itertools import product
import numpy as np
import pandas as pd
from rlearn.tools import select_results
from mlresearch.utils import generate_paths, load_datasets


def summarize_datasets(datasets):
    """Returns a dataframe with a basic description of the datasets."""

    summarized = pd.DataFrame(
        {
            "Dataset": [dataset[0] for dataset in datasets],
            "Metric": [
                dataset[1][0].columns.str.startswith("cat_").sum()
                for dataset in datasets
            ],
            "Non-Metric": [
                (~dataset[1][0].columns.str.startswith("cat_")).sum()
                for dataset in datasets
            ],
            "Obs.": [dataset[1][0].shape[0] for dataset in datasets],
            "Min. Obs.": [
                Counter(dataset[1][1]).most_common()[-1][-1] for dataset in datasets
            ],
            "Maj. Obs.": [
                Counter(dataset[1][1]).most_common()[0][-1] for dataset in datasets
            ],
        }
    )

    formatter = "{0:.%sf}" % 2
    summarized["IR"] = (summarized["Maj. Obs."] / summarized["Min. Obs."]).apply(
        formatter.format
    )
    summarized["Classes"] = [dataset[1][1].unique().shape[0] for dataset in datasets]

    return summarized


if __name__ == "__main__":
    # define paths and basic variables
    DATA_PATH, RESULTS_PATH, ANALYSIS_PATH = generate_paths(__file__)
    datasets = load_datasets(data_dir=DATA_PATH)

    DATASETS = [dataset[0].lower().replace(" ", "_") for dataset in datasets]
    OVERSAMPLERS = [
        "NONE", "RAND-OVER", "RAND-UNDER", "SMOTENC", "G-SMOTE"
    ]
    CLASSIFIERS = [
        "LR",
        "KNN",
        "DT",
        "RF",
    ]
    RESULTS_NAMES = [
        f"{dataset}__{oversampler.lower().replace('-', '')}.pkl"
        for dataset, oversampler in product(DATASETS, OVERSAMPLERS)
    ]

    # datasets description
    summarize_datasets(datasets).to_csv(
        join(ANALYSIS_PATH, "datasets_description.csv"), index=False
    )

    # load results
    results = []
    for name in RESULTS_NAMES:
        file_path = join(RESULTS_PATH, name)
        df = pd.read_pickle(file_path)
        df["Dataset"] = name.split("__")[0].replace("_", " ").title()
        df["Oversampler"], _, df["Classifier"] = np.array(
            df["models"].str.split("|").tolist()
        ).T
        df.set_index(["Dataset", "Oversampler", "Classifier"], inplace=True)
        results.append(df)

    # combine and select results
    results = pd.concat(results, axis=0, sort=True)
    results = select_results(
        results, oversamplers_names=OVERSAMPLERS, classifiers_names=CLASSIFIERS
    )
