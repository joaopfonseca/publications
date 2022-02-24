from itertools import product
from mlresearch.utils import load_datasets, generate_paths
from rlearn.tools import ImbalancedExperiment

CONFIG = {}

if __name__ == "__main__":

    DATA_PATH, RESULTS_PATH, ANALYSIS_PATH = generate_paths(__file__)

    # Get assets
    datasets = load_datasets(DATA_PATH)
    oversamplers = CONFIG["oversamplers"]

    for dataset, oversampler in product(datasets, oversamplers):

        # Define and fit experiment
        experiment = ImbalancedExperiment().fit(dataset)

        file_name = (
            f'{dataset[0].replace(" ", "_").lower()}__' +
            f'{oversampler[0].replace("-", "").lower()}.pkl'
        )
        experiment.results_.to_pickle(join(results_dir, file_name))
