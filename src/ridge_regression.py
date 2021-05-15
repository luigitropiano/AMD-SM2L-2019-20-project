# -*- coding: utf-8 -*-
import numpy as np
from pyspark.sql import functions as F
from pyspark.sql.types import FloatType, DoubleType
from pyspark.ml.linalg import DenseVector, SparseVector, Vector, VectorUDT
from pyspark.sql.functions import udf

@udf(VectorUDT())
def add_intercept_udf(v):
	return DenseVector(np.append(np.ones(1), v))

class SparkRidgeRegression(object):
    def __init__(self, reg_factor):
        self.reg_factor = reg_factor

    def squared_error(self, target, prediction):
        return (target - prediction) ** 2 # power of 2

    def root_mean_squared_error(self, predictions):
        return np.sqrt(predictions.map(lambda p: self.squared_error(*p)).mean())

    def mean_squared_error(self, predictions):
        return predictions.map(lambda p: self.squared_error(p[0], p[1])).mean()

    def mean_absolute_error(self, predictions):
        return np.abs(predictions.map(lambda prediction: prediction[1] - prediction[0]).reduce(lambda a, b: a + b))/predictions.count()

    def r2(self, predictions):
        mean_ = predictions.rdd.map(lambda t: t[0]).mean()
        sum_squares = predictions.rdd.map(lambda t: (t[0] - mean_)**2).sum()
        residual_sum_squares = predictions.rdd.map(lambda t: self.squared_error(*t)).sum()
        return 1 - (residual_sum_squares / sum_squares)

    def add_intercept(self, sample):
        row = sample.features_final.toArray()
        row = np.insert(row, 0, 1)
        return row

    def inv_mul(self, sample):
        row = sample[np.newaxis]
        return row.T @ row

    def map_row(self, sample):
        row = sample
        # return one key per row
        return [(x, row[x]*row) for x in range(len(row))]
        # return one key per value
        #return [((x, y), row[x]*row[y]) for x in range(len(row)) for y in range(len(row))]

    def prod_inv_row(self, inverse, sample):
        value, key = sample
        row = value[np.newaxis]
        product = inverse @ row.T
        return (key, product)

    def prod_join(self, sample):
        key, value = sample
        v1, v2 = value
        return (v1 * v2)
    
    def spark_to_numpy(self, reduce_by_key_result):
        result = np.array([])
        # unpack one key per row
        for (k, v) in sorted(reduce_by_key_result):
        # unpack one key per value
        #for ((k1, k2), v) in sorted(reduce_by_key_result):
            result = np.append(result, v)
    
        size = int(np.sqrt(len(result)))
        return result.reshape(size,size)


    def predict(self, example):
        thetas = self.thetas
        X_predictor = np.c_[np.ones((example.shape[0], 1)), example]
        self.predictions = X_predictor.dot(thetas)
        #X_predictor = X.withColumn('features_final', add_intercept('features_final'))
        #self.predictions = X_predictor.features_final.toArray().dot(thetas)
        return self.predictions

    #def predict_many(self, examples):
    #    thetas = self.thetas
    #    # apply our function to RDD
    #    return examples.rdd.map(lambda row: add_intercept(row)).map(lambda row: thetas.dot(row)).collect()

    def predict_many(self, examples):
        thetas = self.thetas
        dot_prod_udf = F.udf(lambda example: float(thetas.dot(example)), DoubleType())
        examples = examples.withColumn('features_final', add_intercept_udf('features_final'))
        return examples.withColumn('target_predictions', dot_prod_udf('features_final'))


    def fit(self, X):

        # Count number of columns and add 1 for the intercept term
        features_number = len(X.take(1)[0].features_final) + 1

        # Identity matrix of dimension compatible with our X_intercept Matrix
        A = np.identity(features_number)

        # set first 1 on the diagonal to zero so as not to include a bias term for
        # the intercept
        A[0, 0] = 0

        # We create a bias term corresponding to alpha for each column of X not
        # including the intercept
        A_biased = self.reg_factor * A


        #prod1 = X.rdd.map(lambda row: add_intercept(row)).rdd.map(lambda row: inv_mul(row)).reduce(lambda x, y: x + y) + A_biased
        prod1 = X.rdd.map(lambda row: self.add_intercept(row)).flatMap(lambda row: self.map_row(row)).reduceByKey(lambda x, y: x + y, int(np.sqrt(features_number))).collect()
        prod1 = self.spark_to_numpy(prod1) + A_biased
        print("end prod1 with shape: " + str(prod1.shape))

        inverse = np.linalg.inv(prod1)
        print("end inverse computation")

        prod2 = X.rdd.map(lambda row: self.add_intercept(row)).zipWithIndex().map(lambda row: self.prod_inv_row(inverse, row))
        y_rdd = X.select('PINCP').rdd.zipWithIndex().map(lambda x: (x[1], x[0][0]))
        thetas = prod2.join(y_rdd).map(lambda row: self.prod_join(row)).reduce(lambda x, y: x + y)

        self.thetas = np.array(thetas).squeeze()
    
        return self

