import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from lazypredict.Supervised import LazyClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import neighbors_crimes

from sklearn.preprocessing import PolynomialFeatures
import joblib
import pickle

num_classes = {0: "BATTERY", 1: "THEFT", 2: "CRIMINAL DAMAGE", 3: "DECEPTIVE PRACTICE", 4: "ASSAULT"}
classes_num = {"BATTERY": 0, "THEFT": 1, "CRIMINAL DAMAGE": 2, "DECEPTIVE PRACTICE": 3, "ASSAULT": 4}


# print settings
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 10000)


def load(path):
    return pd.read_csv(path)


def prepare_X(X, train_flag):
    """
    Clean and coordinate
    :param X:
    :return:
    """
    # drop unnedded
    X = X.drop(columns=["Unnamed: 0", "ID", "Case Number", "Updated On", "Year"])

    # remove duplicates
    dup = X[X.duplicated()]

    # drop locations
    #TODO: change urgently!
    X = X.drop(columns=["Latitude", "Longitude", "Y Coordinate", "X Coordinate", "Block", "IUCR", "Description", "FBI Code"])

    # change data types
    X["Location Description"] = X["Location Description"].apply(str)
    X["Ward"] = X["Ward"].apply(str)
    X["Beat"] = X["Beat"].apply(str)
    X["District"] = X["District"].apply(str)
    X["Community Area"] = X["Community Area"].apply(str)
    X["Date"] = pd.to_datetime(X["Date"])

    # remove nans

    # TODO:
    # Beat, District, Ward, Community Area,
    #const_imputer = SimpleImputer(strategy='constant', fill_value=0)
    #df[["Date"]] = const_imputer.transform(df[["Date"]])
    #const_imputer.fit(df[["Date"]])
    freq_imputer = SimpleImputer(strategy='most_frequent')
    const_imputer = SimpleImputer(strategy='constant', fill_value=0)

    freq_imputer.fit(X[['Arrest', 'Domestic', 'Location Description']])
    const_imputer.fit(X[["Beat", "District", "Ward", "Community Area"]])

    X[['Arrest', 'Domestic', 'Location Description']] = freq_imputer.transform(X[['Arrest', 'Domestic', 'Location Description']])
    X[["Beat", "District", "Ward", "Community Area"]] = const_imputer.transform(X[["Beat", "District", "Ward", "Community Area"]])
    X[X["Date"].isnull()] = pd.to_datetime('20200101', format='%Y%m%d', errors='ignore')

    # convert data-time
    X['crime_day'] = X["Date"].dt.dayofweek
    X['crime_month'] = X["Date"].dt.month
    X['crime_year'] = X["Date"].dt.year
    X['time'] = X['Date'].dt.hour
    X['sin_time'] = np.sin(2 * np.pi * (X['Date'].dt.hour * 60 * 60) / (24 * 60 * 60))
    X['cos_time'] = np.cos(2 * np.pi * (X['Date'].dt.hour * 60 * 60) / (24 * 60 * 60))
    X = X.drop(columns=["Date"])


    # encoding
    X = X.reset_index()
    # Load from file
    enc = None
    if not train_flag:
        enc = joblib.load("one_hot_encoder.pkl")
    else:
        enc = OneHotEncoder(handle_unknown='ignore')
        enc.fit(X[['Beat','Location Description', 'Arrest', 'Domestic', 'District', 'Ward', 'Community Area']])
    enc_X = enc.transform(X[['Beat','Location Description', 'Arrest', 'Domestic', 'District', 'Ward', 'Community Area']])
    X = X.join(pd.DataFrame(enc_X.toarray()))
    X = X.drop(columns=['Beat', 'Location Description', 'Arrest', 'Domestic', 'District', 'Ward'])
    if train_flag:
        joblib.dump(enc, "one_hot_encoder.pkl")

    # add polynomial
    poly = PolynomialFeatures(2)
    #X = poly.fit_transform(X)
    return X


def enrich(X, y):
    X = neighbors_crimes.get_neghibors_crimes(X, y)
    X = X.drop(columns=["Community Area", "Location"])
    return X


def lazy_predict(X_train, X_test, y_train, y_test):
    clf = LazyClassifier(predictions=True)
    models, predictions = clf.fit(X_train, X_test, y_train, y_test)
    print(models)


def split(df):
    y = df["Primary Type"]
    X = df.drop(columns=["Primary Type"])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test


def prepare_y(y):
    y = y.to_frame()
    y['Primary Type'] = y['Primary Type'].apply(lambda x: classes_num[x])
    return y


df = load("Dataset_crimes.csv")[0:10000]
X_train, X_val, X_test, y_train, y_val, y_test = split(df)
X_train = prepare_X(X_train, True)
y_train = prepare_y(y_train)
X_train = enrich(X_train, y_train)
X_train.to_csv("train_csv")
#X_train = pd.read_csv("around_csv")

# fit model no training data
model = XGBClassifier()
model.fit(X_train, y_train)
print(model)


# #predict
X_test = prepare_X(X_test, False)
y_test = prepare_y(y_test)
X_test = enrich(X_test, y_test)
X_test.to_csv("test_csv")
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

lazy_predict(X_train, X_test, y_train, y_test)

# convert to y classes





# class Preprocess:
#     def __init__(self):
#
#
#     def clean(self):
#
#
#     def encode(self):

#
# p = Preprocess()
# p.clean()
# p.encode()