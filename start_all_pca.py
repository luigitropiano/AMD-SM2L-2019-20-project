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

# name of the target column and remove all the rows where 'PINCP' is null
target = 'PINCP'
df = df.dropna(subset = target)

# COLUMNS SETTING
skipping = ['PERNP', 'WAGP', 'HINCP', 'FINCP']
numericals = ['NP', 'BDSP', 'CONP', 'ELEP', 'FULP', 'INSP', 'MHP', 'MRGP', 'RMSP', 'RNTP', 'SMP', 'VALP', 'WATP', 'GRNTP', 'GRPIP', 'GASP', 'NOC', 'NPF', 'NRC', 'OCPIP', 'SMOCP', 'AGEP', 'INTP', 'JWMNP', 'OIP', 'PAP', 'RETP', 'SEMP', 'SSIP', 'SSP', 'WKHP', 'POVPIP']
ordinals = ['AGS', 'YBL', 'MV', 'TAXP', 'CITWP', 'DRAT', 'JWRIP', 'MARHT', 'MARHYP', 'SCHG', 'SCHL', 'WKW', 'YOEP', 'DECADE', 'JWAP', 'JWDP', 'SFN']
categoricals = [col for col in df.columns if col not in skipping + numericals + ordinals + [target]]

################################################################
#fill all null numericals value with 0
df = df.fillna(0, numericals)

###############################################################
#INDEXING AND ENCODING
from pyspark.ml import Pipeline
from pyspark.ml.feature import PCA
from pyspark.ml.feature import OneHotEncoder
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.feature import StandardScaler
from pyspark.sql.functions import rand

utils.printNowToFile("starting StringIndexer + OneHotEncoder pipeline:")

ordinals_input = [col+"_index" for col in ordinals]
categoricals_input = [col+"_encode" for col in categoricals]

indexers = [StringIndexer(inputCol=col, outputCol=col+"_index", handleInvalid='keep') for col in ordinals + categoricals]
encoders = [OneHotEncoder(inputCol=col+"_index", outputCol=col+"_encode", dropLast = True) for col in categoricals]
assemblers = [
    VectorAssembler(inputCols = numericals, outputCol = 'numericals_vector', handleInvalid='keep'),
    VectorAssembler(inputCols = ordinals_input, outputCol = 'ordinals_vector'),
    VectorAssembler(inputCols = categoricals_input, outputCol = 'categoricals_vector')
]

df = Pipeline(stages = indexers + encoders + assemblers).fit(df).transform(df)

#Drop useless features
utils.printNowToFile("dropping useless columns:")
useless_col = numericals + ordinals_input + categoricals_input

df = df.drop(*useless_col)

# SPLIT DATASET
#df = df.persist(StorageLevel.MEMORY_AND_DISK)
( train_set, test_set ) = df.orderBy(rand()).randomSplit([0.7, 0.3])

###############################################################
#SCALING

utils.printNowToFile("starting scalers pipelines:")

scaledFeatures = ['numericals_scaled', 'ordinals_scaled', 'categoricals_scaled']

stages = [
    StandardScaler(inputCol = 'numericals_vector', outputCol = 'numericals_scaled', withStd=True, withMean=True),
    StandardScaler(inputCol = 'ordinals_vector', outputCol = 'ordinals_scaled', withStd=True, withMean=True),
    StandardScaler(inputCol = 'categoricals_vector', outputCol = 'categoricals_scaled', withStd=True, withMean=True),
    VectorAssembler(inputCols = scaledFeatures, outputCol = 'scaledFeatures'),
    PCA(k=75, inputCol='scaledFeatures', outputCol='features_final')
]

pipeline = Pipeline(stages = stages).fit(train_set)
train_set = pipeline.transform(train_set)
test_set = pipeline.transform(test_set)
utils.printNowToFile("end pipeline std")

#Drop useless features
utils.printNowToFile("dropping useless columns:")
useless_col = ['numericals_vector', 'ordinals_vector', 'categoricals_vector']
train_set = train_set.drop(*useless_col)
test_set = test_set.drop(*useless_col)

################################################################
#TUNING WITH K-FOLD CROSS VALIDATION
utils.printNowToFile("starting CrossValidation:")

from functools import reduce  # For Python 3.x
from pyspark.sql import DataFrame
import numpy as np

def unionAll(*dfs):
    return reduce(DataFrame.unionByName, dfs)

features_columns = ['features_final']
for features_column in features_columns:

    utils.printNowToFile("starting CrossValidation for " + features_column + ":")

    fold_1, fold_2, fold_3, fold_4, fold_5 = train_set.randomSplit([0.2, 0.2, 0.2, 0.2, 0.2])

    scores_dict = {}
    folds = [fold_1, fold_2, fold_3, fold_4, fold_5]

    for alpha in [0.01, 0.1, 1]:
        utils.printNowToFile('trying alpha = ' + str(alpha))
        partial_scores = np.array([])
        srrcv = rr.SparkRidgeRegression(reg_factor=alpha)
        for fold in folds:
            folds_to_merge = [f for f in folds if f != fold]
            cv = unionAll(*folds_to_merge)
            cv = cv.coalesce(100)
            srrcv.fit(cv, features_column)
            result = srrcv.predict_many(test_set, features_column, 'new_column')
            partial_scores = np.append(partial_scores, srrcv.r2(result.select('PINCP', 'new_column')))
        final_score = np.mean(partial_scores)
        scores_dict[alpha] = final_score

    for k in scores_dict:
        utils.printNowToFile('alpha ' + str(k) + ' - r2 score ' + str(scores_dict[k]))

    best_alpha = max(scores_dict, key=scores_dict.get)
    utils.printNowToFile('selected alpha: ' + str(best_alpha))

################################################################

    utils.printNowToFile("starting SparkRidgeRegression:")

    train_set = train_set.persist(StorageLevel.DISK_ONLY)

    utils.printNowToFile("pre srr fit:")
    srr = rr.SparkRidgeRegression(reg_factor=best_alpha)
    srr.fit(train_set, features_column)
    utils.printNowToFile("post srr fit:")

    result = srr.predict_many(test_set, features_column, 'new_column')
    utils.printToFile('result: {0}'.format(srr.r2(result.select('PINCP', 'new_column'))))

    utils.printNowToFile("starting linear transform:")
    lin_reg = LinearRegression(standardization = False, featuresCol = features_column, labelCol='PINCP', maxIter=10, regParam=best_alpha, elasticNetParam=0.0, fitIntercept=True)
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
