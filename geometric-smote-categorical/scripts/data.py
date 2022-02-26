from collections import Counter
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from mlresearch.datasets import ContinuousCategoricalDatasets
from mlresearch.utils import generate_paths

DATA_PATH, RESULTS_PATH, ANALYSIS_PATH = generate_paths(__file__)

if __name__ == "__main__":

    # Get data
    datasets = ContinuousCategoricalDatasets()
    datasets.download()

    # Remove datasets that are too small
    min_obs = 500
    filtered_content = [
        (name, data) for name, data in datasets.content_ if data.shape[0] >= min_obs
    ]

    # Sample datasets
    min_n_samples, n_obs, rnd_seed = 15, 5000, 5
    content = []
    for name, data in filtered_content:

        cat_feats = data.columns[data.columns.str.startswith("cat_")].tolist()

        if data.shape[0] > n_obs:
            data = data.sample(n=n_obs, random_state=rnd_seed)

        classes = [
            cl for cl, count in Counter(data.target).items() if count >= min_n_samples
        ]
        data = data[data.target.isin(classes)].reset_index(drop=True)
        data = pd.concat(
            [
                pd.DataFrame(
                    MinMaxScaler().fit_transform(
                        data.drop(columns=cat_feats + ["target"])
                    )
                ),
                data[cat_feats],
                data.target,
            ],
            axis=1,
        )
        content.append((name, data))

    # Save database
    datasets.content_ = content
    datasets.save(DATA_PATH, "gsmote-categorical")
