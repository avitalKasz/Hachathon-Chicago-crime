import pandas as pd
import numpy as np
from ast import literal_eval as make_tuple
import importlib
import mpu

possible_primary_types = {"BATTERY", "ASSAULT", "THEFT", "CRIMINAL "
                                                         "DAMAGE",
                          "DECEPTIVE PRACTICE"}
GEO_FEATURE_NAME = "Community Area"
K_NEIGHBORS = 30
classes_num = {"BATTERY": 0, "THEFT": 1, "CRIMINAL DAMAGE": 2,
               "DECEPTIVE PRACTICE": 3, "ASSAULT": 4}
num_classes = {0: "BATTERY", 1: "THEFT", 2: "CRIMINAL DAMAGE",
               3: "DECEPTIVE PRACTICE", 4: "ASSAULT"}
nan_pc = str("nan")


def load():
    # load
    return pd.read_csv("dataset_crimes.csv")

    # drop unnedded
    # df = df.drop(columns=["ID", "Case Number", "Updated On"])


# def clean_geo_data(X_train):
#     # df_acc['Latitude'] = df_acc['Latitude'].astype(float)
#     # df_acc['Longitude'] = df_acc['Longitude'].astype(float)
#     # heat_df = df_acc[['ID','Latitude', 'Longitude', "Primary Type"]]
#     X_train = X_train.dropna(subset=["Location", GEO_FEATURE_NAME])
#     # df = df[df["Location"].notna()]
#     X_train = X_train.reset_index()
#     return X_train


def get_neghibors_crimes(X_train, y_train):
    X_train = X_train.reset_index()
    y_train = y_train.reset_index()
    y_train = y_train.drop(columns=["index"])
    f = X_train[["Location", GEO_FEATURE_NAME]]
    f["Primary Type"] = y_train
    X_train["from_K-neighbors_amount_BATTERY"] = 0
    X_train["from_K-neighbors_amount_THEFT"] = 0
    X_train["from_K-neighbors_amount_CRIMINAL DAMAGE"] = 0
    X_train["from_K-neighbors_amount_DECEPTIVE PRACTICE"] = 0
    X_train["from_K-neighbors_amount_ASSAULT"] = 0
    for i, samp in enumerate(f.values):
        if str(samp[0]) == nan_pc or str(samp[1]) == nan_pc or str(samp[2]) == nan_pc:
            continue
        # if (i % 100 == 0):
        #     print("progress {} from {}".format(i, len(f.values)))
        location = make_tuple(f.at[i, "Location"])
        district = f.at[i, GEO_FEATURE_NAME]
        lst = f[f[GEO_FEATURE_NAME] == district].dropna().values.tolist()
        # for sample in lst:
        #     sample[0] = make_tuple(sample[0])
        #     # coordinate
        lst.sort(key=lambda coord: mpu.haversine_distance(location, make_tuple(
            coord[0])))
        lst = lst[:K_NEIGHBORS]
        for type in possible_primary_types:
            counter = 0
            for near_sample in lst:
                if num_classes[near_sample[2]] == type:
                    counter += 1
            current_col = "from_K-neighbors_amount_{}".format(type)
            X_train.at[i, current_col] = counter
    return X_train
