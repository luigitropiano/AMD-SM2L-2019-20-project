from pyspark.ml import Pipeline
from pyspark.ml.feature import PCA
from pyspark.ml.feature import OneHotEncoder
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.feature import StandardScaler


def apply_stdscaler_to_df(df, inputCols, outputCol):

    assembler_column = 'assembled_features'
    vecAssembler = VectorAssembler(inputCols=inputCols, outputCol=assembler_column, handleInvalid='keep')
    transformed = vecAssembler.transform(df)
    
    scaler = StandardScaler(inputCol=assembler_column, outputCol=outputCol, withStd=True, withMean=True)
    
    scalerModel =  scaler.fit(transformed.select(assembler_column))
    df = scalerModel.transform(transformed)
    return df


def apply_ohe_to_df(df, categorical_ft):
    indexers = [StringIndexer(inputCol=col, outputCol=col+"_index", handleInvalid='keep') for col in categorical_ft]
    encoder = [OneHotEncoder(inputCol=col+"_index", outputCol=col+"_encode", dropLast = True) for col in categorical_ft]

    pipeline = Pipeline(stages = indexers + encoder)
    df = pipeline.fit(df).transform(df)
    
    indexed_ft = df.select(df.colRegex("`.+(_index)`"))
    df = df.drop(*indexed_ft.schema.names)
    return df


# PCA
def apply_pca_to_df(df, k, inputCols, outputCol):
    pca = PCA(k=k, inputCol=inputCols, outputCol=outputCol)
    model = pca.fit(df)
    
    final_featuress = ['PINCP', outputCol]
    df_pca= model.transform(df).select(*final_featuress)
    return df_pca
    
