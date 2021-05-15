import os, zipfile
from src import utils


def load_dataset(DATA_PATH, spark):

    csv_files = [x for x in os.listdir(DATA_PATH) if 'csv' in x]
    
    # if dataset not already unzipped, unzip it
    if not csv_files:
        with zipfile.ZipFile(DATA_PATH + '2013-american-community-survey.zip','r') as zip_ref:
            zip_ref.extractall(DATA_PATH)
    #del csv_files
    

    #dataframe people dataset
    pfiles = ["ss13pusa.csv", "ss13pusb.csv"]
    hfiles = ["ss13husa.csv", "ss13husb.csv"]
    df_p = spark.read.csv([DATA_PATH + f for f in pfiles], header = True, inferSchema = True)
    df_h = spark.read.csv([DATA_PATH + f for f in hfiles], header = True, inferSchema = True)

    # drop columns in housing and person
    dropping_list = ['PERNP', 'WAGP', 'HINCP', 'FINCP', 'RT', 'DIVISION', 'REGION', 'ADJINC', 'ADJHSG', 'WGTP', 'PWGTP', 'SPORDER', 'VACS' ]
    #
    join_list = ['SERIALNO', 'PUMA', 'ST']

    df_p = df_p.drop(*dropping_list)
    df_h = df_h.drop(*dropping_list)
    
    col_p = df_p.columns
    col_h = df_h.columns
    
    #join dei due dataframe
    utils.printNowToFile("join df started:")
    df = df_p.join(df_h, on=join_list, how='inner')
    utils.printNowToFile("join df end:")
    
    del df_h
    del df_p
    
    df = df.drop('PUMA')
    df = df.drop('SERIALNO')
    
    weight_list_p = df.select(df.colRegex("`(pwgtp)+?.+`"))
    weight_list_h = df.select(df.colRegex("`(wgtp)+?.+`"))
    flag_list = df.select(df.colRegex("`(?!FOD1P|FOD2P|FIBEROP|FULP|FPARC|FINCP)(F)+?.+(P)`"))
    
    df = df.drop(*weight_list_p.schema.names)
    df = df.drop(*weight_list_h.schema.names)
    df = df.drop(*flag_list.schema.names)
    
    return df
