====
Data
====

To download and preprocess the data, run the command::

    python data.py

The data are saved in the *data* directory as a database (.db) file. 

=======
Results
=======

To run the experiments and get the experimental results, run the command::

    python results.py

The results for each oversampler are saved in the *results* directory as pickled
pandas dataframes.

========
Analysis
========

To analyze the experimental results, run the command::

    python analysis.py

The outcome is saved in the *analysis* directory in various formats.

**Note:** the script ``al-example.py`` contains the code to produce the
figure with a comparison of active learning methods
(``analysis/al-example.pdf``).
