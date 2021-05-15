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
numericals = ['NP', 'BDSP', 'CONP', 'ELEP', 'FULP', 'INSP', 'MHP', 'MRGP', 'RMSP', 'RNTP', 'SMP', 'VALP', 'WATP', 'GRNTP', 'GRPIP', 'GASP', 'NOC', 'NPF', 'NRC', 'OCPIP', 'SMOCP', 'AGEP', 'INTP', 'JWMNP', 'OIP', 'PAP', 'RETP', 'SEMP', 'SSIP', 'SSP', 'WKHP', 'POVPIP']
ordinals = ['AGS', 'YBL', 'MV', 'TAXP', 'CITWP', 'DRAT', 'JWRIP', 'MARHT', 'MARHYP', 'SCHG', 'SCHL', 'WKW', 'YOEP', 'DECADE', 'JWAP', 'JWDP', 'SFN']
categoricals = [col for col in df.columns if col not in skipping + numericals + ordinals]

################################################################
#fill all null numericals value with 0
df = df.fillna(0, numericals)

###############################################################

#ENCODING and INDEXING categorical features

from pyspark.ml import Pipeline
from pyspark.ml.feature import PCA
from pyspark.ml.feature import OneHotEncoder
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.feature import StandardScaler

utils.printNowToFile("starting StringIndexer + OneHotEncoder pipeline:")

indexers = [StringIndexer(inputCol=col, outputCol=col+"_index", handleInvalid='keep') for col in ordinals + categoricals]
encoders = [OneHotEncoder(inputCol=col+"_index", outputCol=col+"_encode", dropLast = True) for col in categoricals]

#list of features indexed and encoded
ordinals_input = [col+"_index" for col in ordinals]
categoricals_input = [col+"_encode" for col in categoricals]


#vector assembler pipeline to create numerical, ordinals and categoricals vector input for scaler
vector = [
    VectorAssembler(inputCols = numericals, outputCol = 'numericals_vector', handleInvalid='keep'),
    VectorAssembler(inputCols = ordinals_input, outputCol = 'ordinals_vector'),
    VectorAssembler(inputCols = categoricals_input, outputCol = 'categoricals_vector')
]

df = Pipeline(stages = indexers + encoders + vector).fit(df).transform(df)

#Drop useless features
utils.printNowToFile("dropping useless columns:")
useless_col = ordinals_input + categoricals_input + numericals

df = df.drop(*useless_col)

# SPLIT DATASET
df = df.persist(StorageLevel.MEMORY_AND_DISK)
( train_set, val_set, test_set ) = df.randomSplit([0.6, 0.2, 0.2])

###############################################################
#STD SCALER 

utils.printNowToFile("starting VectorAssembler + StandardScaler pipeline:")

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
val_set = pipeline.transform(val_set)

#####################################################################
#TUNING WITH K-FOLD CROSS VALIDATION
utils.printNowToFile("starting CrossValidation:")

from functools import reduce  # For Python 3.x
from pyspark.sql import DataFrame
import numpy as np

def unionAll(*dfs):
    return reduce(DataFrame.unionByName, dfs)

#split train set in 5 fold
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
        partial_scores = np.append(partial_scores, srrcv.r2(result.select('PINCP', 'target_predictions')))
    final_score = np.mean(partial_scores)
    scores_dict[alpha] = final_score

for k in scores_dict:
    utils.printNowToFile('alpha ' + str(k) + ' - r2 score ' + str(scores_dict[k]))

#find best value for alpha
best_alpha = max(scores_dict, key=scores_dict.get)
utils.printNowToFile('selected alpha: ' + str(best_alpha))


################################################################
#PREDICTION
utils.printNowToFile("starting SparkRidgeRegression:")

#persist train_set on disk memory
train_set = train_set.persist(StorageLevel.DISK_ONLY)

#Fit ridge regression model on train set
srr = rr.SparkRidgeRegression(reg_factor=best_alpha)
#train_set = train_set.withColumn('features_final', train_set.scaledFeatures)
#test_set = test_set.withColumn('features_final', train_set.scaledFeatures)
utils.printNowToFile("pre srr fit:")
srr.fit(train_set)
utils.printNowToFile("post srr fit:")

#r2 score on test set
result = srr.predict_many(test_set)
utils.printToFile('result: {0}'.format(srr.r2(result.select('PINCP', 'target_predictions'))))

#fit model with pyspark ridge regression
lin_reg = LinearRegression(standardization = False, featuresCol = 'features_final', labelCol='PINCP', maxIter=10, regParam=best_alpha, elasticNetParam=0.0, fitIntercept=True)
utils.printNowToFile("starting linear transform:")
linear_mod = lin_reg.fit(train_set)
utils.printNowToFile("after linear transform:")

#r2 score on test set
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
