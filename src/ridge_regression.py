# -*- coding: utf-8 -*-
import numpy as np
from pyspark.ml.linalg import DenseVector, SparseVector, Vectors, VectorUDT
from pyspark.sql import functions as F
from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType, DoubleType


@udf(VectorUDT())
def add_interceptor(v):
    if v is None:
        return v
    return DenseVector(np.append(np.ones(1), v))

def squared_error(target, prediction):
    return (target - prediction) ** 2 # power of 2

def root_mean_squared_error(predictions):
    return np.sqrt(predictions.map(lambda p: squared_error(*p)).mean())

def mean_squared_error(predictions):
    return predictions.map(lambda p: squared_error(p[0], p[1])).mean()

def mean_absolute_error(predictions):
    return np.abs(predictions.map(lambda prediction: prediction[1] - prediction[0]).reduce(lambda a, b: a + b))/predictions.count()

def r2(predictions):
    mean_ = predictions.rdd.map(lambda t: t[0]).mean()
    sum_squares = predictions.rdd.map(lambda t: (t[0] - mean_)**2).sum()
    residual_sum_squares = predictions.rdd.map(lambda t: squared_error(*t)).sum()
    return 1 - (residual_sum_squares / sum_squares)

def inv_mul(example):
    row = example.features_final.toArray()[np.newaxis]
    return row.T @ row

def map_row(example):
    row = example.features_final.toArray()
    return [((x, y), row[x]*row[y]) for x in range(len(row)) for y in range(len(row))]

def prod_inv_row(inverse, example):
    value, key = example
    row = value.features_final.toArray()[np.newaxis]
    product = inverse @ row.T
    return (key, product)

def prod_join(example):
    key, value = example
    v1, v2 = value
    return (v1 * v2)

def spark_to_numpy(reduce_by_key_result):
    result = np.array([])
    for ((k1, k2), v) in sorted(reduce_by_key_result):
        result = np.append(result, v)

    size = int(np.sqrt(len(result)))
    return result.reshape(size,size)

class SparkRidgeRegression(object):
    def __init__(self, reg_factor):
        self.reg_factor = reg_factor

    def predict(self, example):
        thetas = self.thetas
        X_predictor = np.c_[np.ones((example.shape[0], 1)), example]
        self.predictions = X_predictor.dot(thetas)
        #X_predictor = X.withColumn('features_final', add_interceptor('features_final'))
        #self.predictions = X_predictor.features_final.toArray().dot(thetas)
        return self.predictions
    
    def predict_many(self, examples):
        thetas = self.thetas
        dot_prod_udf = F.udf(lambda example: float(thetas.dot(example)), DoubleType())
        examples = examples.withColumn('features_final', add_interceptor('features_final'))
        return examples.withColumn('new_column', dot_prod_udf('features_final'))

    def fit(self, X):
        X_with_intercept = X.withColumn('features_final', add_interceptor('features_final'))
        features_number = len(X_with_intercept.take(1)[0].features_final)
    
        # Identity matrix of dimension compatible with our X_intercept Matrix
        A = np.identity(features_number) 
    
        # set first 1 on the diagonal to zero so as not to include a bias term for
        # the intercept
        A[0, 0] = 0
    
        # We create a bias term corresponding to alpha for each column of X not
        # including the intercept
        A_biased = self.reg_factor * A

        #prod1 = X_with_intercept.rdd.map(lambda example: inv_mul(example)).reduce(lambda x, y: x + y) + A_biased
        prod1 = X_with_intercept.rdd.flatMap(lambda example: map_row(example)).reduceByKey(lambda x, y: x + y).collect()
        prod1 = spark_to_numpy(prod1) + A_biased
        print("end prod1 with shape: " + str(prod1.shape))

        inverse = np.linalg.inv(prod1)
        print("end inverse computation")

        prod2 = X_with_intercept.rdd.zipWithIndex().map(lambda example: prod_inv_row(inverse, example))
        y_rdd = X_with_intercept.select('PINCP').rdd.zipWithIndex().map(lambda x: (x[1], x[0][0]))
        thetas = prod2.join(y_rdd).map(lambda example: prod_join(example)).reduce(lambda x, y: x + y)

        self.thetas = np.array(thetas).squeeze()
    
        return self

