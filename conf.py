# -*- coding: utf-8 -*-
import os
import findspark
import pyspark
from pyspark.sql import SparkSession

def load_conf():
    os.environ["SPARK_HOME"] = "./spark/spark-3.1.1-bin-hadoop3.2"
    os.environ["HADOOP_HOME"] = "./spark/spark-3.1.1-bin-hadoop3.2"
    #os.environ["PYTHONPATH"] = "%SPARK_HOME%/python;%SPARK_HOME%/python/lib/py4j-0.10.9-src.zip:%PYTHONPATH%"
    #os.environ["PYTHONPATH"] = "%SPARK_HOME%/python;%SPARK_HOME%/python/lib/py4j-0.10.9-src.zip:%$"
    ##os.environ["SPARK_CUDF_JAR"] = "C:/tmp/spark/cuda/cudf-0.18.1-cuda11.jar"
    ##os.environ["SPARK_RAPIDS_PLUGIN_JAR"] = "C:/tmp/spark/cuda/rapids-4-spark_2.12-0.4.1.jar"
    ##os.environ["SPARK_HOME"] = "C:/tmp/spark/spark-3.0.2-bin-hadoop3.2"
    ##os.environ["HADOOP_HOME"] = "C:/tmp/spark/spark-3.0.2-bin-hadoop3.2"
    
    findspark.init()
    
    #configArray = [
    #    ('spark.executor.memory', '10g'),
    #    ('spark.executor.cores', '3'),
    #    ('spark.cores.max', '3'),
    #    ('spark.driver.memory','10g'),
    #    #('spark.executor.resource.gpu.amount','1'),
    #    #('spark.task.resource.gpu.amount','0.0833333333333333'),
    #    #('spark.executor.resource.gpu.discoveryScript','./getGpusResources.bat'),
    #
    #    #('spark.worker.resourcesFile','./resources.txt')
    #]
    configArray = [
        ('spark.driver.memory','9g'),
        ('spark.executor.memory', '9g'), 
        ('spark.executor.cores', 4), 
        ('spark.driver.cores', 1), 
        ('spark.cores.max', 4),
        ("spark.driver.maxResultSize", "12g"),
        ("spark.sql.shuffle.partitions", "50")
    ]
    
    config = pyspark.SparkConf().setMaster("local[4]").setAll(configArray)
    #config = pyspark.SparkConf().setAll(configArray)
    
    spark = SparkSession.builder.config(conf=config).getOrCreate()
    debug=False
    if debug:
        logger = spark._jvm.org.apache.log4j
        logger.LogManager.getLogger("org"). setLevel( logger.Level.DEBUG )
        logger.LogManager.getLogger("akka").setLevel( logger.Level.DEBUG )

    return spark
