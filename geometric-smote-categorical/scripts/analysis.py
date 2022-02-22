from os.path import join
from collections import Counter
import pandas as pd
from mlresearch.utils import generate_paths, load_datasets


def summarize_datasets(datasets):
    """Returns a dataframe with a basic description of the datasets."""

    summarized = pd.DataFrame(
        {
            "Dataset": [dataset[0] for dataset in datasets],
            "Features": [dataset[1][0].shape[1] for dataset in datasets],
            "Instances": [dataset[1][0].shape[0] for dataset in datasets],
            "Min. Instances": [
                Counter(dataset[1][1]).most_common()[-1][-1] for dataset in datasets
            ],
            "Maj. Instances": [
                Counter(dataset[1][1]).most_common()[0][-1] for dataset in datasets
            ],
        }
    )

    formatter = "{0:.%sf}" % 2
    summarized["IR"] = (
        summarized["Maj. Instances"] / summarized["Min. Instances"]
    ).apply(formatter.format)
    summarized["Classes"] = [dataset[1][1].unique().shape[0] for dataset in datasets]

    return summarized


if __name__ == "__main__":
    DATA_PATH, RESULTS_PATH, ANALYSIS_PATH = generate_paths(__file__)
    datasets = load_datasets(data_dir=DATA_PATH)

    # datasets description
    summarize_datasets(datasets).to_csv(
        join(ANALYSIS_PATH, "datasets_description.csv"), index=False
    )
