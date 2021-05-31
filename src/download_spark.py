import os, tarfile
import urllib.request
from src import utils

def download(DATA_PATH):
    if not os.path.exists(DATA_PATH + 'spark'):
        thetarfile = "https://downloads.apache.org/spark/spark-3.1.1/spark-3.1.1-bin-hadoop3.2.tgz"
        ftpstream = urllib.request.urlopen(thetarfile)
        thetarfile = tarfile.open(fileobj=ftpstream, mode="r|gz")
        thetarfile.extractall()
