# -*- coding: utf-8 -*-
import sklearn.metrics
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.storagelevel import StorageLevel

## CUSTOM IMPORT
from src import ridge_regression as rr
from src import american_community_survey as amc
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

# COLUMNS SETTING
skipping = ['PERNP', 'WAGP', 'HINCP', 'FINCP']
numericals = ['SERIALNO', 'NP', 'BDSP', 'CONP', 'ELEP', 'FULP', 'INSP', 'MHP', 'MRGP', 'RMSP', 'RNTP', 'SMP', 'VALP', 'WATP', 'GRNTP', 'GRPIP', 'GASP', 'NOC', 'NPF', 'NRC', 'OCPIP', 'SMOCP', 'AGEP', 'INTP', 'JWMNP', 'OIP', 'PAP', 'RETP', 'SEMP', 'SSIP', 'SSP', 'WKHP', 'POVPIP']
ordinals = ['AGS', 'YBL', 'MV', 'TAXP', 'CITWP', 'DRAT', 'JWRIP', 'MARHT', 'MARHYP', 'SCHG', 'SCHL', 'WKW', 'YOEP', 'DECADE', 'JWAP', 'JWDP', 'SFN']
categoricals = [col for col in df.columns if col not in skipping + numericals + ordinals + [target]]

################################################################

df = df.fillna(0, numericals)

###############################################################

from pyspark.ml import Pipeline
from pyspark.ml.feature import PCA
from pyspark.ml.feature import OneHotEncoder
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.feature import StandardScaler, RobustScaler, MinMaxScaler

utils.printNowToFile("starting StringIndexer + OneHotEncoder pipeline:")

stages = [
    StringIndexer(inputCol=col, outputCol=col+"_index", handleInvalid='keep') for col in ordinals + categoricals
    OneHotEncoder(inputCol=col+"_index", outputCol=col+"_encode", dropLast = True) for col in categoricals
]

df = Pipeline(stages = stages).fit(df).transform(df)

# SPLIT DATASET
#df = df.persist(StorageLevel.MEMORY_AND_DISK)
( train_set, val_set, test_set ) = df.randomSplit([0.6, 0.2, 0.2])

###############################################################

utils.printNowToFile("starting VectorAssembler + StandardScaler pipeline:")

ordinals_input = [col+"_index" for col in ordinals]
categoricals_input = [col+"_encode" for col in categoricals]
scaledFeatures = ['numericals_std', 'ordinals_std', 'categoricals_std']
minmaxFeatures = ['numericals_minmax', 'ordinals_minmax', 'categoricals_minmax']
robustFeatures = ['numericals_robust', 'ordinals_robust', 'categoricals_robust']

vector = [
    VectorAssembler(inputCols = numericals, outputCol = 'numericals_vector', handleInvalid='keep'),
    VectorAssembler(inputCols = ordinals_input, outputCol = 'ordinals_vector'),
    VectorAssembler(inputCols = categoricals_input, outputCol = 'categoricals_vector')
]

std_pipeline = [
    StandardScaler(inputCol = 'numericals_vector', outputCol = 'numericals_std', withStd=True, withMean=True),
    StandardScaler(inputCol = 'ordinals_vector', outputCol = 'ordinals_std', withStd=True, withMean=True),
    StandardScaler(inputCol = 'categoricals_vector', outputCol = 'categoricals_std', withStd=True, withMean=True),
    VectorAssembler(inputCols = scaledFeatures, outputCol = 'features_std')
]

minmax_pipeline = [
    MinMaxScaler(inputCol = 'numericals_vector', outputCol = 'numericals_minmax'),
    MinMaxScaler(inputCol = 'ordinals_vector', outputCol = 'ordinals_minmax'),
    MinMaxScaler(inputCol = 'categoricals_vector', outputCol = 'categoricals_minmax'),
    VectorAssembler(inputCols = minmaxFeatures, outputCol = 'features_minmax')

]

robust_pipeline = [
    RobustScaler(inputCol='numericals_vector', outputCol='numericals_robust', withScaling=True, withCentering=False, lower=0.25, upper=0.75),
    RobustScaler(inputCol='ordinals_vector', outputCol='ordinals_robust', withScaling=True, withCentering=False, lower=0.25, upper=0.75),
    RobustScaler(inputCol='categoricals_vector', outputCol='categoricals_robust', withScaling=True, withCentering=False, lower=0.25, upper=0.75),
    VectorAssembler(inputCols = robustFeatures, outputCol = 'features_robust')
]

#pipeline = Pipeline(stages = vector + std_pipeline + minmax_pipeline + robust_pipeline).fit(train_set)

pipeline1 = Pipeline(stages = vector + std_pipeline).fit(train_set)
train_set = pipeline1.transform(train_set)
test_set = pipeline1.transform(test_set)
val_set = pipeline1.transform(val_set)
utils.printNowToFile("end pipeline std")

pipeline2 = Pipeline(stages = vector + minmax_pipeline).fit(train_set)
train_set = pipeline2.transform(train_set)
test_set = pipeline2.transform(test_set)
val_set = pipeline2.transform(val_set)
utils.printNowToFile("end pipeline min max")

pipeline3 = Pipeline(stages = vector + robust_pipeline).fit(train_set)
train_set = pipeline3.transform(train_set)
test_set = pipeline3.transform(test_set)
val_set = pipeline3.transform(val_set)
utils.printNowToFile("end pipeline robust")

#Drop useless features
utils.printNowToFile("dropping useless columns:")
useless_col = ordinals_input + categoricals_input + numericals

train_set = train_set.drop(*useless_col)
test_set = test_set.drop(*useless_col)
val_set = val_set.drop(*useless_col)

################################################################
utils.printNowToFile("starting CrossValidation:")

from functools import reduce  # For Python 3.x
from pyspark.sql import DataFrame
import numpy as np

def unionAll(*dfs):
    return reduce(DataFrame.unionByName, dfs)

fold_1, fold_2, fold_3, fold_4, fold_5 = train_set.randomSplit([0.2, 0.2, 0.2, 0.2, 0.2])

scores_dict = {}
folds = [fold_1, fold_2, fold_3, fold_4, fold_4]

for alpha in [0.01, 0.1, 1]:
    utils.printNowToFile('trying alpha = ' + str(alpha))
    partial_scores = np.array([])
    srrcv = rr.SparkRidgeRegression(reg_factor=alpha)
    for fold in folds:
        folds_to_merge = [f for f in folds if f != fold]
        cv = unionAll(*folds_to_merge)
        cv = cv.coalesce(100)
        srrcv.fit(cv)
        result = srrcv.predict_many(test_set)
        partial_scores = np.append(partial_scores, srrcv.r2(result.select('PINCP', 'new_column')))
    final_score = np.mean(partial_scores)
    scores_dict[alpha] = final_score

for k in scores_dict:
    utils.printNowToFile('alpha ' + str(k) + ' - r2 score ' + str(scores_dict[k]))

best_alpha = max(scores_dict, key=scores_dict.get)
utils.printNowToFile('selected alpha: ' + str(best_alpha))

###############################################################
utils.printNowToFile("starting SparkRidgeRegression:")

train_set = train_set.persist(StorageLevel.DISK_ONLY)

srr = rr.SparkRidgeRegression(reg_factor=best_alpha)
#train_set = train_set.withColumn('features_final', train_set.scaledFeatures)
#test_set = test_set.withColumn('features_final', train_set.scaledFeatures)
utils.printNowToFile("pre srr fit:")
srr.fit(train_set)
utils.printNowToFile("post srr fit:")

result = srr.predict_many(test_set)
utils.printToFile('result: {0}'.format(srr.r2(result.select('PINCP', 'new_column'))))


lin_reg = LinearRegression(standardization = False, featuresCol = 'features_final', labelCol='PINCP', maxIter=10, regParam=best_alpha, elasticNetParam=0.0, fitIntercept=True)
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
