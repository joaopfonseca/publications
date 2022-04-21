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
    generate_mean_std_tbl_bold,
)  # , load_plt_sns_configs


def summarize_datasets(datasets):
    """Returns a dataframe with a basic description of the datasets."""

    summarized = pd.DataFrame(
        {
            "Dataset": [dataset[0].title() for dataset in datasets],
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

    summarized["base_names"] = summarized["Dataset"].apply(lambda x: x.split(" (")[0])
    summarized["ir_sort"] = summarized["IR"].astype(float)

    summarized = summarized\
        .groupby("base_names")\
        .apply(lambda df: df.sort_values("ir_sort").reset_index(drop=True))\
        .reset_index(drop=True)\
        .drop(columns=["ir_sort", "base_names"])

    return summarized


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


def calculate_mean_sem_rankings(wide_optimal):
    ranks = wide_optimal\
        .rank(axis=1, ascending=False)\
        .reset_index()\
        .groupby(["Classifier", "Metric"])
    return ranks.mean(), ranks.sem(ddof=0)


def calculate_mean_sem_scores(wide_optimal):
    scores = wide_optimal\
        .reset_index()\
        .groupby(["Classifier", "Metric"])
    return scores.mean(), scores.sem(ddof=0)


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
        .replace(r"\$", "$")
    )

    if path is not None:
        open(path, "w").write(wo_tex)
    else:
        return wo_tex


def format_table(df, with_sem=True, maximum=True, decimals=3):

    if with_sem:
        df = generate_mean_std_tbl_bold(*df, maximum=maximum, decimals=decimals)
    else:
        df = df.copy()
        df = df.apply(
            lambda row: make_bold(row, maximum=maximum, num_decimals=decimals),
            axis=1
        )

    index_cols = list(df.index.names)
    df.reset_index(inplace=True)

    if "Metric" in df.columns:
        df["Metric"] = df["Metric"].map(METRICS)

    df.rename(columns=OVERSAMPLERS, inplace=True)
    df = df.set_index(index_cols) if index_cols[0] is not None else df

    # reorder oversamplers
    if df.columns.isin(OVERSAMPLERS.values()).sum() == len(OVERSAMPLERS):
        ovs_cols = df.columns.isin(OVERSAMPLERS.values())
        df_index = df.loc[:, ~ovs_cols]
        df_ovs = df.loc[:, OVERSAMPLERS.values()]
        df = pd.concat([df_index, df_ovs], axis=1)

    return df


if __name__ == "__main__":

    # Define paths and basic variables
    DATA_PATH, RESULTS_PATH, ANALYSIS_PATH = generate_paths(__file__)
    datasets = load_datasets(data_dir=DATA_PATH)

    DATASETS = [
        dataset[0].lower().replace(" ", "_") for dataset in datasets
        if dataset[0].lower().split(" ")[0] != "thyroid"
    ]
    OVERSAMPLERS = {
        "G-SMOTE": "G-SMOTE",
        "NONE": "NONE",
        "SMOTENC": "SMOTENC",
        "RAND-OVER": "ROS",
        "RAND-UNDER": "RUS",
    }
    CLASSIFIERS = ["LR", "KNN", "DT", "RF"]
    METRICS = {
        "mean_test_accuracy": "OA",
        "mean_test_f1_macro": "F-Score",
        "mean_test_geometric_mean_score_macro": "G-Mean",
    }
    RESULTS_NAMES = [
        f"{dataset}__{oversampler.lower().replace('-', '')}.pkl"
        for dataset, oversampler in product(DATASETS, OVERSAMPLERS.keys())
    ]

    # Filter datasets that are too easy to solve
    datasets = [
        dataset for dataset in datasets
        if dataset[0].lower().replace(" ", "_") in DATASETS
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
        results, oversamplers_names=OVERSAMPLERS.keys(), classifiers_names=CLASSIFIERS
    )
    wide_optimal = calculate_wide_optimal(results)

    # Get mean rankings
    ranks = calculate_mean_sem_rankings(wide_optimal)

    # Get mean scores
    scores = calculate_mean_sem_scores(wide_optimal)

    # Save all tables to latex
    TBL_OUTPUTS = (
        (
            "datasets_description",
            datasets_description,
            (
                "Description of the datasets collected after data preprocessing. The"
                " sampling strategy is similar across datasets. Legend: (IR) Imbalance "
                "Ratio"
            ),
        ),
        (
            "wide_optimal",
            format_table(wide_optimal, with_sem=False).reset_index(),
            "Wide optimal results",
        ),
        (
            "mean_sem_ranks",
            format_table(ranks, maximum=False, decimals=2).reset_index(),
            (
                "Mean rankings over the different datasets, folds and runs used in the "
                "experiment."
            )
        ),
        (
            "mean_sem_scores",
            format_table(scores, maximum=True, decimals=2).reset_index(),
            (
                "Mean scores over the different datasets, folds and runs used"
                " in the experiment"
            )
        )
    )

    for name, df, caption in TBL_OUTPUTS:
        save_longtable(df, join(ANALYSIS_PATH, f"{name}.tex"), caption, f"tbl:{name}")
