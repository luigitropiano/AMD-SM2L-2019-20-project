import os, tarfile
import urllib.request
from src import utils

def download(PATH):
    if not os.path.exists(PATH + '/spark'):
        os.makedirs(PATH + '/spark')
    if not os.path.exists(PATH + '/spark/spark_tmp'):
        os.makedirs(PATH + '/spark/spark_tmp')
    if not os.path.exists(PATH + '/spark/spark-3.1.1-bin-hadoop3.2'):
        thetarfile = "https://downloads.apache.org/spark/spark-3.1.1/spark-3.1.1-bin-hadoop3.2.tgz"
        ftpstream = urllib.request.urlopen(thetarfile)
        thetarfile = tarfile.open(fileobj=ftpstream, mode="r|gz")
        thetarfile.extractall()
