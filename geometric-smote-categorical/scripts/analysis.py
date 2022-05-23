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
from scipy.stats import wilcoxon, ttest_rel
from statsmodels.stats.multitest import multipletests
from rlearn.tools import select_results
from rlearn.tools.reporting import _extract_pvalue
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


def apply_wilcoxon_test(wide_optimal, dep_var, OVRS_NAMES, alpha):
    """Performs a Wilcoxon signed-rank test"""
    pvalues = []
    for ovr in OVRS_NAMES:
        mask = np.repeat(True, len(wide_optimal))

        pvalues.append(
            wilcoxon(
                wide_optimal.loc[mask, ovr], wide_optimal.loc[mask, dep_var]
            ).pvalue
        )
    wilcoxon_results = pd.DataFrame(
        {
            "Oversampler": OVRS_NAMES,
            "p-value": pvalues,
            "Significance": np.array(pvalues) < alpha,
        }
    )
    return wilcoxon_results


def generate_statistical_results(wide_optimal, alpha=0.05, control_method="NONE"):
    """Generate the statistical results of the experiment."""

    # Get results
    results = wide_optimal.copy()

    # Calculate rankings
    ranks = results.rank(axis=1, ascending=0).reset_index()

    # Friedman test
    friedman_test = (
        ranks.groupby(["Classifier", "Metric"])
        .apply(_extract_pvalue)
        .reset_index()
        .rename(columns={0: "p-value"})
    )

    friedman_test["Significance"] = friedman_test["p-value"] < alpha
    friedman_test["p-value"] = friedman_test["p-value"].apply(
        lambda x: "{:.1e}".format(x)
    )

    # Wilcoxon signed rank test
    # Optimal proposed method vs oversampling framework
    wilcoxon_test = []
    for dataset in results.reset_index().Dataset.unique():
        wilcoxon_results = apply_wilcoxon_test(
            results[(results.reset_index()["Dataset"] == dataset).values],
            "G-SMOTE",
            ["NONE", "RAND-OVER", "RAND-UNDER", "SMOTENC"],
            alpha,
        ).drop(columns="Significance")
        wilcoxon_results["Dataset"] = dataset.replace("_", " ").title()
        wilcoxon_test.append(
            wilcoxon_results.pivot("Dataset", "Oversampler", "p-value")
        )

    wilcoxon_test = pd.concat(wilcoxon_test, axis=0)
    wilcoxon_test = wilcoxon_test.reset_index()

    # Holms test
    # Optimal proposed framework vs baseline framework
    ovrs_names = results.columns.to_list()
    ovrs_names.remove(control_method)

    # Define empty p-values table
    pvalues = pd.DataFrame()

    # Populate p-values table
    for name in ovrs_names:
        pvalues_pair = results.groupby(["Classifier", "Metric"])[
            [name, control_method]
        ].apply(lambda df: ttest_rel(df[name], df[control_method])[1])
        pvalues_pair = pd.DataFrame(pvalues_pair, columns=[name])
        pvalues = pd.concat([pvalues, pvalues_pair], axis=1)

    # Corrected p-values
    holms_test = pd.DataFrame(
        pvalues.apply(
            lambda col: multipletests(col, method="holm")[1], axis=1
        ).values.tolist(),
        columns=ovrs_names,
    )
    holms_test = holms_test.set_index(pvalues.index).reset_index()

    # Return statistical analyses
    statistical_results_names = ("friedman_test", "wilcoxon_test", "holms_test")
    statistical_results = zip(
        statistical_results_names, (friedman_test, wilcoxon_test, holms_test)
    )
    return statistical_results


def save_longtable(df, path=None, caption=None, label=None):
    """
    Exports a pandas dataframe to longtable format.

    This function replaces ``df.to_latex`` when there are latex commands in
    the table.
    """

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
        "G-SMOTE": "G-SMOTENC",
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

    # Get statistical analyses
    friedman_, wilcoxon_, holms_ = [
        stat[1] for stat in generate_statistical_results(
            wide_optimal, alpha=0.05, control_method="NONE"
        )
    ]

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
        ),
        (
            "friedman_test",
            friedman_,
            (
                "Results for Friedman test. Statistical significance is tested at a "
                "level of $\alpha = 0.05$. The null hypothesis is that there is no "
                "difference in the classification outcome across oversamplers."
            )
        ),
        (
            "wilcoxon_test",
            wilcoxon_,
            (
                "Results for Wilcoxon signed-rank method. Statistical significance is tested at a "
                "level of $\alpha = 0.05$. The null hypothesis is that the performance of the "
                "proposed oversampler is similar to the remaining oversamplers."
            )
        ),
        (
            "holms_test",
            holms_,
            (
                "Adjusted p-values the Holm-Bonferroni test. Statistical significance is"
                " tested at a level of $\alpha = 0.05$. The null hypothesis is that "
                "the benchmark methods do not perform better than the control method "
                "(G-SMOTENC)."
            )
        ),
    )

    for name, df, caption in TBL_OUTPUTS:
        save_longtable(df, join(ANALYSIS_PATH, f"{name}.tex"), caption, f"tbl:{name}")
