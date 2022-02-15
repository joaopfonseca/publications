# Publications

This repository contains the source code of most of the experiments developed
in my research. The LaTeX and Python code for generating the paper,
experiments' results and visualizations reported in each paper is available
(whenever possible) in the paper's directory.

Additionally, contributions at the algorithm level are available in the
package ``ml-research``. The ``ml-research`` package can be found
[here](https://github.com/joaopfonseca/ml-research).

## Publication List

TODO


## Installation

To install the basic project dependencies, first activate a Python 3 virtual
environment and from the root of the project run the command:

    pip install .


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

