# -*- coding: utf-8 -*-
import sklearn.metrics
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.storagelevel import StorageLevel

## CUSTOM IMPORT
import conf
from src import american_community_survey as amc
from src import utils

## START
    
###############################################################
## PREPROCESSING: CLEANING
spark = conf.load_conf()

spark.sparkContext.addPyFile('ridge_regression.py')
import ridge_regression as rr

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

# SPLIT DATASET
from pyspark.sql.functions import rand
#df = df.persist(StorageLevel.MEMORY_AND_DISK)
( train_set, test_set ) = df.orderBy(rand()).randomSplit([0.7, 0.3])

###############################################################
#INDEXING AND ENCODING

from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.feature import StandardScaler

utils.printNowToFile("starting pipeline")

ordinals_input = [col+"_index" for col in ordinals]
categoricals_input = [col+"_encode" for col in categoricals]
stdFeatures = ['numericals_std', 'ordinals_std', 'categoricals_std']

# stages for index and encoding pipeline
stages = [
    # numericals
    VectorAssembler(inputCols = numericals, outputCol = 'numericals_vector', handleInvalid='keep'),
    StandardScaler(inputCol = 'numericals_vector', outputCol = 'numericals_std', withStd=True, withMean=True),

    # ordinals
    *[StringIndexer(inputCol=col, outputCol=col+"_index", handleInvalid='keep') for col in ordinals],
    VectorAssembler(inputCols = ordinals_input, outputCol = 'ordinals_vector'),
    StandardScaler(inputCol = 'ordinals_vector', outputCol = 'ordinals_std', withStd=True, withMean=True),

    # categoricals
    *[StringIndexer(inputCol=col, outputCol=col+"_index", handleInvalid='keep') for col in categoricals],
    *[OneHotEncoder(inputCol=col+"_index", outputCol=col+"_encode", dropLast = True) for col in categoricals],
    VectorAssembler(inputCols = categoricals_input, outputCol = 'categoricals_vector'),
    StandardScaler(inputCol = 'categoricals_vector', outputCol = 'categoricals_std', withStd=True, withMean=True),

    # final assembler
    VectorAssembler(inputCols = stdFeatures, outputCol = 'features_std')
]

pipeline = Pipeline(stages=stages).fit(train_set)
train_set = pipeline.transform(train_set)
test_set = pipeline.transform(test_set)

###############################################################

final_columns = [target, 'features_std']

#Drop useless features
utils.printNowToFile("dropping useless columns:")
train_set = train_set.select(final_columns)
test_set = test_set.select(final_columns)

################################################################
#TUNING WITH K-FOLD CROSS VALIDATION
utils.printNowToFile("starting CrossValidation:")

from functools import reduce  # For Python 3.x
from pyspark.sql import DataFrame
import numpy as np

def unionAll(*dfs):
    return reduce(DataFrame.unionByName, dfs)

for features_column in [col for col in final_columns if col != target]:

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

    result = srr.predict_many(test_set, features_column, 'target_predictions')
    utils.printToFile('result: {0}'.format(srr.r2(result.select('PINCP', 'target_predictions'))))

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
