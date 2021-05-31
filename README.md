# AMD-SM2L-2019-20-project

«Algorithms for massive datasets»
«Statistical methods for ML»

## Joint project for 2019-20

The project is based on the analysis of the «[2013 American Community Survey](https://www.kaggle.com/census/2013-american-community-survey)» dataset published on Kaggle and released under the public domain license (CC0).

The task is to implement from scratch a learning algorithm for regression with square loss (e.g., ridge regression). The label to be predicted must be selected among the following 5 attributes, removing the remaining 4 from the dataset:

    PERNP (Person's earnings)
    PINCP (Person's income)
    WAGP (Wages or salary income past 12 months)
    HINCP (Household income)
    FINCP (Family income)

Important: the techniques used in order to infer the predictor should be time and space efficient, and scale up to larger datasets.

The project can be carried out individually, or in groups of two students. Code should be written in Python 3 (different choices must be preliminarily agreed with both instructors).

This project is valid for the academic year 2019/20.

## Running the project

The project have been tested with python 3.7+ . In order to run our code it is
necessary to install the requirements contained in `requirements.txt`. 
We have created a different file for each of the experiment explained in the
project report. We performed a total of 5 different experiments, that are:

- Experiment 1: Only numericals and ordinals features (no categoricals)
- Experiment 2: All numericals + all ordinals + 25 of 150 categorical features
- Experiment 3: All numericals + all ordinals + 100 of 150 categorical features
- Experiment 4: All features + PCA
- Experiment 5: All features

## Quick start guide

To start the code using a python virtualenv it is possible to use the following
commands:

`~$ git clone https://github.com/tr0p1x/AMD-SM2L-2019-20-project && cd AMD-SM2L-2019-20-project`

`~$ python3 -m venv ./venv && source ./venv/bin/activate`

`~$ pip install -r requirements.txt`

`~$ python3 ./start_experiment[1-5].py`
