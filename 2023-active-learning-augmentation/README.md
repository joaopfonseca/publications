# Improving Active Learning Performance Through the Use of Data Augmentation

[Link to publication](https://www.hindawi.com/journals/ijis/2023/7941878/)

## Abstract

Active learning (AL) is a well-known technique to optimize data usage in
training, through the interactive selection of unlabeled observations, out of
a large pool of unlabeled data, to be labeled by a supervisor. Its focus is to
find the unlabeled observations that, once labeled, will maximize the
informativeness of the training dataset, therefore reducing data-related
costs. The literature describes several methods to improve the effectiveness
of this process. Nonetheless, there is a paucity of research developed around
the application of artificial data sources in AL, especially outside image
classification or NLP. This paper proposes a new AL framework, which relies on
the effective use of artificial data. It may be used with any classifier,
generation mechanism, and data type and can be integrated with multiple other
state-of-the-art AL contributions. This combination is expected to increase
the ML classifier’s performance and reduce both the supervisor’s involvement
and the amount of required labeled data at the expense of a marginal increase
in computational time. The proposed method introduces a hyperparameter
optimization component to improve the generation of artificial instances
during the AL process as well as an uncertainty-based data generation
mechanism. We compare the proposed method to the standard framework and an
oversampling-based active learning method for more informed data generation in
an AL context. The models’ performance was tested using four different
classifiers, two AL-specific performance metrics, and three classification
performance metrics over 15 different datasets. We demonstrated that the
proposed framework, using data augmentation, significantly improved the
performance of AL, both in terms of classification performance and data
selection efficiency (all the codes and preprocessed data developed for this
study are available at https://github.com/joaopfonseca/publications/).
