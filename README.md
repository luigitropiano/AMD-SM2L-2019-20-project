# AMD-SM2L-2019-20-project

«Algorithms for massive datasets»
«Statistical methods for ML»

Joint project for 2019-20

The project is based on the analysis of the «[2013 American Community Survey](https://www.kaggle.com/census/2013-american-community-survey)» dataset published on Kaggle and released under the public domain license (CC0).

The task is to implement from scratch a learning algorithm for regression with square loss (e.g., ridge regression). The label to be predicted must be selected among the following 5 attributes, removing the remaining 4 from the dataset:

    PERNP (Person's earnings)
    PINCP (Person's income)
    WAGP (Wages or salary income past 12 months)
    HINCP (Household income)
    FINCP (Family income)

Important: the techniques used in order to infer the predictor should be time and space efficient, and scale up to larger datasets.

The project can be carried out individually, or in groups of two students. Code should be written in Python 3 (different choices must be preliminarily agreed with both instructors).

The project report, preferably written in LaTeX, will be evaluated according to the following criteria:

    Correctness of the general methodological approach
    Reproducibility of the experiments
    Correctness of the approach used for choosing the hyperparameters
    Scalability of the proposed solution
    Clarity of exposition

The report should contain the following information:

    Which parts of the dataset have been considered
    How data have been organized
    The applied pre-processing techniques
    The considered regression algorithms and their implementations
    How the proposed solution scales up with data size
    Description of the experiments
    Comment on the experimental results

The report must also contain the following declaration: “I/We declare that this material, which I/We now submit for assessment, is entirely my/our own work and has not been taken from the work of others, save and to the extent that such work has been cited and acknowledged within the text of my/our work. I/We understand that plagiarism, collusion, and copying are grave and serious offences in the university and accept the penalties that would be imposed should I engage in plagiarism, collusion or copying. This assignment, or any part of it, has not been previously submitted by me/us or any other person for assessment on this or any other course of study.“

If the proposed solution is based on the ones published in Kaggle, this must be clearly stated, and the report should explain the differences and compare the experimental results.

The project should be made available through a public github repository, containing code and report. The dataset should not be added to the repository, but downloaded during code execution, for instance via the kaggle API (https://github.com/Kaggle/kaggle-api).

Once the project has been finalized, students should send an email to Prof. Cesa-Bianchi (nicolo DOT cesa-bianchi AT unimi DOT it), Prof. Malchiodi (malchiodi AT di DOT unimi DOT it) and Dr. Paudice (andrea DOT paudice AT unimi DOT it), specifying

    Their names and student IDs
    The program they are enrolled in (Master in Computer Science, Master in Data Science for Economics, etc.)
    The github link to the project

After the project is evaluated, students will be able to schedule an appointment for the oral discussion.

This project is valid for the academic year 2019/20.
