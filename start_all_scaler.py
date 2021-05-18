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

# name of the target column and emrove all the rows where 'PINCP' is null
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
from pyspark.ml.feature import OneHotEncoder
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.feature import StandardScaler, RobustScaler, MinMaxScaler
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

stdFeatures = ['numericals_std', 'ordinals_std', 'categoricals_std']
minmaxFeatures = ['numericals_minmax', 'ordinals_minmax', 'categoricals_minmax']
robustFeatures = ['numericals_robust', 'ordinals_robust', 'categoricals_robust']

#std scaler 
std_pipeline = [
    StandardScaler(inputCol = 'numericals_vector', outputCol = 'numericals_std', withStd=True, withMean=True),
    StandardScaler(inputCol = 'ordinals_vector', outputCol = 'ordinals_std', withStd=True, withMean=True),
    StandardScaler(inputCol = 'categoricals_vector', outputCol = 'categoricals_std', withStd=True, withMean=True),
    VectorAssembler(inputCols = stdFeatures, outputCol = 'features_std')
]

#min max scaler
minmax_pipeline = [
    MinMaxScaler(inputCol = 'numericals_vector', outputCol = 'numericals_minmax'),
    MinMaxScaler(inputCol = 'ordinals_vector', outputCol = 'ordinals_minmax'),
    MinMaxScaler(inputCol = 'categoricals_vector', outputCol = 'categoricals_minmax'),
    VectorAssembler(inputCols = minmaxFeatures, outputCol = 'features_minmax')
]

#robust scaler
robust_pipeline = [
    RobustScaler(inputCol='numericals_vector', outputCol='numericals_robust', withScaling=True, withCentering=True, lower=0.2, upper=0.8),
    RobustScaler(inputCol='ordinals_vector', outputCol='ordinals_robust', withScaling=True, withCentering=True, lower=0.2, upper=0.8),
    RobustScaler(inputCol='categoricals_vector', outputCol='categoricals_robust', withScaling=True, withCentering=True, lower=0.2, upper=0.8),
    VectorAssembler(inputCols = robustFeatures, outputCol = 'features_robust')
]

#pipeline = Pipeline(stages = vector + std_pipeline + minmax_pipeline + robust_pipeline).fit(train_set)

pipeline = Pipeline(stages = std_pipeline).fit(train_set)
train_set = pipeline.transform(train_set)
test_set = pipeline.transform(test_set)
utils.printNowToFile("end pipeline std")

pipeline = Pipeline(stages = minmax_pipeline).fit(train_set)
train_set = pipeline.transform(train_set)
test_set = pipeline.transform(test_set)
utils.printNowToFile("end pipeline min max")

pipeline = Pipeline(stages = robust_pipeline).fit(train_set)
train_set = pipeline.transform(train_set)
test_set = pipeline.transform(test_set)
utils.printNowToFile("end pipeline robust")

#Drop useless features
utils.printNowToFile("dropping useless columns:")
useless_col = ['numericals_vector', 'ordinals_vector', 'categoricals_vector']
train_set = train_set.drop(*useless_col)
test_set = test_set.drop(*useless_col)

################################################################
alpha = 0.1

features_columns = ['features_std', 'features_minmax, features_robust']
for features_column in features_columns:

    utils.printNowToFile("starting SparkRidgeRegression:")

    train_set = train_set.persist(StorageLevel.DISK_ONLY)

    utils.printNowToFile("pre srr fit:")
    srr = rr.SparkRidgeRegression(reg_factor=alpha)
    srr.fit(train_set, features_column)
    utils.printNowToFile("post srr fit:")

    result = srr.predict_many(test_set, features_column, 'target_predictions')
    utils.printToFile('result: {0}'.format(srr.r2(result.select('PINCP', 'target_predictions'))))

    utils.printNowToFile("starting linear transform:")
    lin_reg = LinearRegression(standardization = False, featuresCol = features_column, labelCol='PINCP', maxIter=10, regParam=alpha, elasticNetParam=0.0, fitIntercept=True)
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
