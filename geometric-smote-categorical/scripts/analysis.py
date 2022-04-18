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
from mlresearch.utils import (
    generate_paths,
    load_datasets,
    make_bold,
)  # , load_plt_sns_configs


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

    return summarized.sort_values("Dataset")


def calculate_wide_optimal(results):

    mean_scoring_cols = results.columns[results.columns.str.contains("mean_test")]

    optimal = results[mean_scoring_cols]

    # Calculate maximum score per group key
    keys = ["Dataset", "Oversampler", "Classifier"]
    agg_measures = {score: max for score in optimal.columns}
    optimal = optimal.groupby(keys).agg(agg_measures).reset_index()

    # Format as long table
    optimal = optimal.melt(
        id_vars=keys,
        value_vars=mean_scoring_cols,
        var_name="Metric",
        value_name="Score",
    )

    # Cast to categorical columns
    optimal_cols = keys + ["Metric"]
    for col in optimal_cols:
        optimal[col] = pd.Categorical(optimal[col], optimal[col].unique())

    # Sort values
    optimal = optimal.sort_values(optimal_cols)

    # Move oversamplers to columns
    optimal = optimal.pivot_table(
        index=["Dataset", "Classifier", "Metric"],
        columns=["Oversampler"],
        values="Score",
    )

    return optimal


def format_table(df):
    df = df.copy()
    index_cols = list(df.index.names)
    df.reset_index(inplace=True)

    if "Metric" in df.columns:
        df["Metric"] = df["Metric"].map(METRICS)

    df = df.set_index(index_cols) if index_cols[0] is not None else df

    df = df.apply(lambda row: make_bold(row, num_decimals=3), axis=1)

    return df


def save_longtable(df, path=None, caption=None, label=None):

    wo_tex = (
        df.to_latex(
            longtable=True,
            caption=caption,
            label=label,
            index=False,
            column_format="c" * df.shape[1],
        )
        .replace(r"\textbackslash ", "\\")
        .replace(r"\{", "{")
        .replace(r"\}", "}")
    )

    if path is not None:
        open(path, "w").write(wo_tex)
    else:
        return wo_tex


if __name__ == "__main__":
    # define paths and basic variables
    DATA_PATH, RESULTS_PATH, ANALYSIS_PATH = generate_paths(__file__)
    datasets = load_datasets(data_dir=DATA_PATH)

    DATASETS = [dataset[0].lower().replace(" ", "_") for dataset in datasets]
    OVERSAMPLERS = ["NONE", "RAND-OVER", "RAND-UNDER", "SMOTENC", "G-SMOTE"]
    CLASSIFIERS = [
        "LR",
        "KNN",
        "DT",
        "RF",
    ]
    METRICS = {
        "mean_test_accuracy": "OA",
        "mean_test_f1_macro": "F-Score",
        "mean_test_geometric_mean_score_macro": "G-Mean",
    }
    RESULTS_NAMES = [
        f"{dataset}__{oversampler.lower().replace('-', '')}.pkl"
        for dataset, oversampler in product(DATASETS, OVERSAMPLERS)
        if not dataset.endswith(")")
    ]

    # datasets description
    datasets_description = summarize_datasets(datasets)

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
    wide_optimal = calculate_wide_optimal(results)

    # Save all tables to latex
    TBL_OUTPUTS = (
        (
            "datasets_description",
            datasets_description,
            (
                "Description of the datasets collected after data preprocessing. The"
                " sampling strategy is similar across datasets. Legend: (IR) Imbalance Ratio"
            ),
        ),
        (
            "wide_optimal",
            format_table(wide_optimal).reset_index(),
            "Wide optimal results",
        ),
    )

    for name, df, caption in TBL_OUTPUTS:
        save_longtable(df, join(ANALYSIS_PATH, f"{name}.tex"), caption, f"tbl:{name}")
