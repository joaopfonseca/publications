from mlresearch.datasets import ContinuousCategoricalDatasets
from mlresearch.utils import generate_paths

DATA_PATH, RESULTS_PATH, ANALYSIS_PATH = generate_paths(__file__)

data = ContinuousCategoricalDatasets()
data.download()
data.save(DATA_PATH, 'gsmote-categorical')
