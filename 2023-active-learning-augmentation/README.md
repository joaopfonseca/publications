# Improving Active Learning Performance Through the Use of Data Augmentation

## Abstract

Active Learning (AL) is a technique that is used to iteratively select
unlabeled observations out of a large pool of unlabeled data to be labeled by
a supervisor. Its focus is to find the unlabeled observations that, once
labeled, will maximize the informativeness of the training dataset. However,
the manual labeling of observations involves human resources with domain
expertise, making it an expensive and time-consuming task. The literature
describes various methods to improve the effectiveness of this process, but
there is little research developed around the usage of artificial data sources
in AL. In this paper we propose a new framework for AL, which allows for an
effective use of artificial data. The proposed method implements a data
augmentation policy that optimizes the generation of artificial instances to
improve the AL process. We compare the proposed method to the standard
framework by using 4 different classifiers, 2 AL-specific performance metrics
and 3 classification performance metrics over 10 different datasets. We show
that the proposed framework, using data augmentation, significantly improves
the performance of AL, both in terms of classification performance and data
selection efficiency. 
