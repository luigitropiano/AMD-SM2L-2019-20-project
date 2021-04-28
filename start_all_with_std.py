# -*- coding: utf-8 -*-
import sklearn.metrics
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.storagelevel import StorageLevel

## CUSTOM IMPORT
from src import ridge_regression as rr
from src import american_community_survey as amc
from src import preprocessing
from src import utils
import conf

## START
    
###############################################################
## PREPROCESSING: CLEANING
spark = conf.load_conf()
# path to dataset
utils.printNowToFile("starting:")
DATA_PATH = './dataset/'
df = amc.load_dataset(DATA_PATH, spark)

###############################################################
## PREPROCESSING: FEATURES ENGINEERING

# name of the target column
target = 'PINCP'
df = df.dropna(subset = target)


#
skip_list = ['PERNP', 'WAGP', 'HINCP', 'FINCP']

numericals = ['SERIALNO', 'NP', 'BDSP', 'CONP', 'ELEP', 'FULP', 'INSP', 'MHP', 'MRGP', 'RMSP', 'RNTP', 'SMP', 'VALP', 'WATP', 'GRNTP', 'GRPIP', 'GASP', 'NOC', 'NPF', 'NRC', 'OCPIP', 'SMOCP', 'AGEP', 'INTP', 'JWMNP', 'OIP', 'PAP', 'RETP', 'SEMP', 'SSIP', 'SSP', 'WKHP', 'POVPIP']

################################################################3

#norm = ['features_norm', 'scaledFeatures']
#skip_norm = numericals + target + norm
skip_norm = numericals + [target]
categorical_ft = [var for var in df.columns if var not in skip_norm]

df = df.fillna(0, numericals)

'''
utils.printNowToFile("starting STDSCALER")
df = preprocessing.apply_stdscaler_to_df(df, numericals, 'scaledFeatures')
utils.printNowToFile("end STDSCALER")
'''

utils.printNowToFile("starting OHE:")
df = preprocessing.apply_ohe_to_df(df, categorical_ft)
utils.printNowToFile("end OHE:")

'''
# CREATE VECTORASSEMBLER FOR PCA
norm = df.select('scaledFeatures').schema.names
cit_encode =df.select(df.colRegex("`.+(_encode)`")).schema.names
features = cit_encode
df = VectorAssembler(inputCols=features, outputCol='vec_ohe', handleInvalid='keep').transform(df)

encode = df.select(df.colRegex("vec_ohe")).schema.names
features_end = norm + encode
df = VectorAssembler(inputCols=features_end, outputCol='features_', handleInvalid='keep').transform(df)
'''

# STD SU TUTTE
cit_encode =df.select(df.colRegex("`.+(_encode)`")).schema.names
num_ohe = numericals + cit_encode

utils.printNowToFile("starting STDSCALER on all")
df = preprocessing.apply_stdscaler_to_df(df, inputCols=num_ohe, outputCol='scaledFeatures')
utils.printNowToFile("end STDSCALER on all")

'''
# PCA
utils.printNowToFile("starting PCA:")
df = preprocessing.apply_pca_to_df(df, inputCols='', outputCol='')
utils.printNowToFile("after PCA:")
'''

#printToFile('len: {0}'.format(len(df_pca.select("features_final").take(1)[0][0])))

#df_pers = df_pca.persist(StorageLevel.MEMORY_AND_DISK)
#df_pers = df.persist(StorageLevel.MEMORY_AND_DISK)
df_pers = df.persist(StorageLevel.DISK_ONLY)

#( train_set, test_set, val_set ) = df_pers.randomSplit([0.7, 0.2, 0.1])
( train_set, test_set ) = df_pers.randomSplit([0.7, 0.3])

#train_set = train_set.repartition(10000)

utils.printNowToFile("starting SparkRidgeRegression:")

srr = rr.SparkRidgeRegression(reg_factor=0.1)
train_set = train_set.withColumn('features_final', train_set.scaledFeatures)
test_set = test_set.withColumn('features_final', train_set.scaledFeatures)
utils.printNowToFile("pre srr fit:")
srr.fit(train_set)
utils.printNowToFile("post srr fit:")

result = srr.predict_many(test_set)
#utils.printToFile('result: {0}'.format(srr.r2(result.select('PINCP', 'new_column'))))
utils.printToFile('result: {0}'.format(rr.r2(result.select('PINCP', 'new_column'))))


lin_reg = LinearRegression(standardization = False, featuresCol = 'features_final', labelCol='PINCP', maxIter=10, regParam=1.0, elasticNetParam=0.0, fitIntercept=True)
utils.printNowToFile("starting linear transform:")
linear_mod = lin_reg.fit(train_set)
utils.printNowToFile("after linear transform:")


predictions = linear_mod.transform(test_set)
y_true = predictions.select("PINCP").toPandas()
y_pred = predictions.select("prediction").toPandas()
r2_score = sklearn.metrics.r2_score(y_true, y_pred)
utils.printToFile('r2_score before: {0}'.format(r2_score))

'''
predict_target = ['PINCP', 'new_column']
predictions=rg.predict_many(test_set).select(*predict_target)
y_true = predictions.select("PINCP").toPandas()
y_pred = predictions.select("new_column").toPandas()
import sklearn.metrics
r2_score = sklearn.metrics.r2_score(y_true, y_pred)
utils.printToFile('r2_score after: {0}'.format(r2_score))
'''

utils.printNowToFile("done:")
