import os, tarfile
import urllib.request
from src import utils

def download(PATH):
    SPARK = PATH + '/spark'
    if not os.path.exists(SPARK):
        os.makedirs(SPARK)
    if not os.path.exists(SPARK + '/spark_tmp'):
        os.makedirs(SPARK + '/spark_tmp')
    if not os.path.exists(SPARK + '/spark-3.1.1-bin-hadoop3.2'):
        thetarfile = "https://downloads.apache.org/spark/spark-3.1.1/spark-3.1.1-bin-hadoop3.2.tgz"
        ftpstream = urllib.request.urlopen(thetarfile)
        thetarfile = tarfile.open(fileobj=ftpstream, mode="r|gz")
        thetarfile.extractall(path = SPARK)
