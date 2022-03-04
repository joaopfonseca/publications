# Publications

This repository contains the source code of most of the experiments developed
in my research. The LaTeX and Python code for generating the paper,
experiments' results and visualizations reported in each paper is available
(whenever possible) in the paper's directory.

Additionally, contributions at the algorithm level are available in the
package ``ml-research``. The ``ml-research`` package can be found
[here](https://github.com/joaopfonseca/ml-research).

## Publication List

- Fonseca, J., Bacao, F. (-). [S<sup>4</sup>AD-Learning: Self-supervised Semi-supervised Active Deep Learning](deep-active-learning). Working paper.
- Fonseca, J., Douzas, G., Bacao, F. (-). [Geometric SMOTENC: A geometrically enhanced drop-in replacement for SMOTENC](geometric-smote-categorical). Working paper.
- Fonseca, J., Bacao, F. (2022). [Improving Active Learning Performance Through the Use of Data Augmentation](active-learning-augmentation). Submitted to Knowledge-Based Systems.
- Fonseca, J., Bacao, F. (2022). [Research Trends and Applications of Data Augmentation Algorithms](data-augmentation-trends). Submitted to Artificial Intelligence Review.
- Fonseca, J., Douzas, G., Bacao, F. (2021). [Increasing the Effectiveness of Active Learning: Introducing Artificial Data Generation in Active Learning for Land Use/Land Cover Classification](2021-al-generator-lulc). Remote Sensing, 13(13), 2619. https://doi.org/10.3390/rs13132619
- Fonseca, J., Douzas, G., Bacao, F. (2021). [Improving Imbalanced Land Cover Classification with K-Means SMOTE: Detecting and Oversampling Distinctive Minority Spectral Signatures](2021-kmeans-smote-lulc). Information, 12(7), 266.
- Crayton A, Fonseca J, Mehra K, Ng M, Ross J, Sandoval-Castañeda M, von Gnecht R. (2021). [Narratives and Needs: Analyzing Experiences of Cyclone Amphan Using Twitter Discourse](2020-amphan-preprint), in IJCAI 2021 Workshop on AI for Social Good. https://crcs.seas.harvard.edu/publications/narratives-and-needs-analyzing-experiences-cyclone-amphan-using-twitter-discourse
- Douzas, G., Bacao, F., Fonseca, J., & Khudinyan, M. (2019). [Imbalanced Learning in Land Cover Classification: Improving Minority Classes’ Prediction Accuracy Using the Geometric SMOTE Algorithm](2019-lucas). Remote Sensing, 11(24), 3040. https://doi.org/10.3390/rs11243040

## Installation

To install the basic project dependencies, first activate a Python 3 virtual
environment and from the root of the project run the command:

    pip install -r requirements.txt


## Project structure

Every research project contains the scripts, data, results, analysis and content
directories.

## scripts

It is the entry point every project. To install the required dependencies from
the scripts directory run the command:

    pip install -r requirements.txt

In order to generate the content of the publication in a reproducible format,
various scripts are provided.

**data.py**

Download and save the datasets used for the experiments:

    python data.py

**results.py**

Run the experiments and get the results:

    python results.py

**analysis.py**

Analyze the results of experiments:

    python analysis.py

### data

It contains the experimental data. They are downloaded and
saved, using the ``data.py`` script.

### results

It contains the results of experiments as pickled pandas dataframes. They are
generated, using the ``results.py`` script.

### analysis

It contains the analysis of experiments' results in various formats. They are
generated, using the ``analysis.py`` script.

### content

It contains the LaTex source files of the project.

