# -*- coding: utf-8 -*-
import os
import findspark
import pyspark
from pyspark.sql import SparkSession

def load_conf():
    os.environ["SPARK_HOME"] = "./spark/spark-3.1.1-bin-hadoop3.2"
    os.environ["HADOOP_HOME"] = "./spark/spark-3.1.1-bin-hadoop3.2"
    
    findspark.init()
    
    configArray = [
#        ('spark.cores.max', 2),
#        ('spark.executor.cores', 2),
        ('spark.executor.memory', '9g'),
        ('spark.driver.memory','10g'),
        ("spark.driver.maxResultSize", "90g"),
#        ('spark.driver.cores', 2),
#        ("spark.sql.shuffle.partitions", "50"),
#        ("spark.default.parallelism", "1"),
        ('spark.network.timeout', '480s'),
        ('spark.executor.heartbeatInterval', '40s'),
        ('spark.local.dir', './spark_tmp')
#        ('spark.ui.enabled', 'false'),
#        ("spark.python.worker.reuse", False)
    ]
    
    config = pyspark.SparkConf().setMaster("local[5]").setAll(configArray)
    
    spark = SparkSession.builder.config(conf=config).getOrCreate()
#    debug=False
#    if debug:
#        logger = spark._jvm.org.apache.log4j
#        logger.LogManager.getLogger("org"). setLevel( logger.Level.DEBUG )
#        logger.LogManager.getLogger("akka").setLevel( logger.Level.DEBUG )

    return spark
