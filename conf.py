# -*- coding: utf-8 -*-
import os
import findspark
import pyspark
from pyspark.sql import SparkSession

os.environ["SPARK_HOME"] = "./spark/spark-3.1.1-bin-hadoop3.2"
os.environ["HADOOP_HOME"] = "./spark/spark-3.1.1-bin-hadoop3.2"

def load_conf_default():

    findspark.init()

    configArray = [
        ('spark.local.dir', './spark/spark_tmp'),
#        ('spark.cores.max', 2),
#        ('spark.executor.cores', 2),
        ('spark.executor.memory', '9g'),
        ('spark.driver.memory','10g'),
        ("spark.driver.maxResultSize", "90g"),
#        ('spark.driver.cores', 2),
#        ("spark.sql.shuffle.partitions", "50"),
#        ("spark.default.parallelism", "1"),
#        ('spark.network.timeout', '480s'),
#        ('spark.executor.heartbeatInterval', '40s'),
#        ('spark.ui.enabled', 'false'),
#        ("spark.python.worker.reuse", False)
    ]

    config = pyspark.SparkConf().setMaster("local[*]").setAll(configArray)
    #config = pyspark.SparkConf().setMaster("spark://thinkzen.localdomain:7077")
    spark = SparkSession.builder.config(conf=config).getOrCreate()

    return spark

def load_conf(spark_master_url, spark_master_port):

    findspark.init()

    config = pyspark.SparkConf().setMaster(f'spark://{spark_master_url}:{spark_master_port}')
    spark = SparkSession.builder.config(conf=config).getOrCreate()

    return spark
