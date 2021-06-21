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
project report. We performed a total of 4 different experiments, that are:

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

It is possible to start our project by choosing one of the 3 following ways:

- **Local mode**

  To start the project using the Apache Spark local mode, just run

  `~$ python3 ./start_experiment[1-5].py`

  Apache Spark local mode is a non-distributed single-JVM deployment mode, where Spark spawns all the execution components - driver, executor, backend, and master - in the same single JVM. We do not recommend to execute the project using the local mode, as it seems to have a non optimal memory management.

- **Standalone mode**

  To start the prject using the Apache Spark standalone mode, instead, run:

  `~$ ./spark_standalone_mode.sh && ./spark/spark-3.1.1-bin-hadoop3.2/sbin/start-all.sh`

  `~$ python3 ./start_experiment[1-5].py -H 127.0.1.1`

  This will start two different processes: one master process on port 7077, and an isolated worker process with 12 GB of ram and with 7 allocated cores. It will also create one executor running within the worker process with 9 GB of ram and 6 allocated cores.
  This has shown to be an effective setup on our 8 core / 16 threads machine with 16 GB of ram. 
  From our tests, the standalone mode was much more stable than the local mode.
  
  When finished, in order to stop the standalone cluster, just run: 

  `./spark/spark-3.1.1-bin-hadoop3.2/sbin/stop-all.sh`

  It is possible to allocate a different amount of resources by editing the spark-env.sh file and re-creating the standalone cluster.

- **Existing cluster**

  If you already have a working Apache Spark setup and would like to submit our code to your cluster, you can pass the hostname and the port of your Master process as shown in the following command:
  
  `~$ python3 ./start_experiment[1-5].py -H <hostname> -P <port>`
  
  If not specified, `<port>` will default to `7077`
