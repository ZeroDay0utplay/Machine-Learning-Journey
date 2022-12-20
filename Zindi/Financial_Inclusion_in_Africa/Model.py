import tensorflow as tf
import pandas as pd


CSV_FILE = "Train.csv"



def data_preproc(csv_file):
    df = pd.read_csv("Train.csv")
    del df["uniqueid"]
    print(df.job_type.unique())




data_preproc(CSV_FILE)



