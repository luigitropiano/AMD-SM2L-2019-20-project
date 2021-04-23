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

def inv_mul(example):
    row = DenseVector(example.features_final).toArray()[np.newaxis]
    return row.T @ row

def mat_mul(inverse, example):
    row = DenseVector(example.features_final).toArray()[np.newaxis]
    return inverse @ row.T

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
        self.y = np.array(X_with_intercept.select('PINCP').collect())
        features_number = len(X_with_intercept.take(1)[0].features_final)
    
        # Identity matrix of dimension compatible with our X_intercept Matrix
        A = np.identity(features_number) 
    
        # set first 1 on the diagonal to zero so as not to include a bias term for
        # the intercept
        A[0, 0] = 0
    
        # We create a bias term corresponding to alpha for each column of X not
        # including the intercept
        A_biased = self.reg_factor * A
        prod1 = X_with_intercept.rdd.map(lambda example: inv_mul(example)).reduce(lambda x, y: x + y) + A_biased
        inverse = np.linalg.inv(prod1)
        prod2 = X_with_intercept.rdd.map(lambda example: mat_mul(inverse, example)).reduce(lambda x, y: np.hstack((x, y)))
        #prod1 = X_with_intercept.rdd.map(lambda example: self.inv_mul(example)).reduce(lambda x, y: x + y) + A_biased
        #self.inverse = np.linalg.inv(prod1)
        #prod2 = X_with_intercept.rdd.map(lambda example: self.mat_mul(example)).reduce(lambda x, y: np.hstack((x, y)))
    
        thetas = prod2 @ self.y
    
        self.thetas = thetas.squeeze()
        return self

