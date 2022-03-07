import pandas as pd
import numpy as np

def join_enrich(df):
    df_right = pd.read_csv("enrichment\/below_poverty_level_by_community.csv")
    return  df.merge(df_right, on=["Community Area"], how="left")
