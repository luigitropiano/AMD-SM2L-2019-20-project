#!/usr/bin/bash

mkdir -p spark
mkdir -p spark/spark_tmp
rm -rf spark/spark-3.1.1-bin-hadoop3.2
rm -rf spark/spark-3.1.1-bin-hadoop3.2

wget -O ./spark/spark-3.1.1-bin-hadoop3.2.tgz 'https://downloads.apache.org/spark/spark-3.1.1/spark-3.1.1-bin-hadoop3.2.tgz'
cd ./spark
tar -zxf spark-3.1.1-bin-hadoop3.2.tgz 
cd ../

cp -a ./spark-env.sh ./spark/spark-3.1.1-bin-hadoop3.2/conf/spark-env.sh
cp -a ./spark-defaults.conf ./spark/spark-3.1.1-bin-hadoop3.2/conf/spark-defaults.conf

echo 'SPARK_LOCAL_DIRS='$PWD'/spark_tmp/' >> ./spark/spark-3.1.1-bin-hadoop3.2/conf/spark-env.sh 

