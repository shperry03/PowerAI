import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MaxAbsScaler
from tpot.builtins import OneHotEncoder


# NOTE: Make sure that the outcome column is labeled 'target' in the data file
df = pd.read_csv(r'openpowerlifting.csv')
df = df[df.Tested == "Yes"]
df = df[df.Event == "SBD"]
df = df[["Age", "AgeClass", "Sex", "BodyweightKg", "MeetCountry", "Equipment", "Wilks", "Glossbrenner","McCulloch", "Place"]]
df.Sex = df.Sex.map({'M': 0, 'F': 1})
countries = df.MeetCountry.unique()

add,i = {}, 0
for country in  countries:
    add[country] = i
    i+=1

df.MeetCountry = df.MeetCountry.map(add)
equip = df.Equipment.unique()

values,i  = {},0
for equi in equip:
    values[equi] = i
    i+=1

df.Equipment = df.Equipment.map(values)
df = df.fillna(round(df.Age.mean(),2))

age_range = {
    '16-17':16, 
    '20-23':20, 
    '18-19':18, 
    '24-34':24, 
    '35-39':35, 
    '40-44':40, 
    '70-74':70,
    '55-59':55, 
    '45-49':45, 
    '50-54':50, 
    '60-64':60, 
    '13-15':13, 
    '65-69':65, 
    '75-79':75,
    '80-999':80, 
    '5-12':5,
    28.72: 24
}

df.AgeClass = df.AgeClass.map(age_range)
df.Wilks.replace(28.72, round(df.Wilks.mean(),2))
df.Glossbrenner.replace(28.72, round(df.Glossbrenner.mean(),2))
df.McCulloch.replace(28.72,round(df.Glossbrenner.mean(),2))
df.Place.replace('DQ', '120',inplace=True)
df.Place.replace('G', '120',inplace=True)
df.Place.replace('DD', '120',inplace=True)
df.Place.replace('NS', '120',inplace=True)

df.Place = pd.to_numeric(df.Place)

bins = [0,4,15,50,150]
labels = [0,1,2,3]# 0 = Excellent, 1 = good, 2 = Alright, 3 = poorly

df['Place']= pd.cut(x=df['Place'], bins=bins, labels=labels)
df.Age = pd.to_numeric(df.Age)
df.Wilks = pd.to_numeric(df.Wilks)
df.Glossbrenner = pd.to_numeric(df.Glossbrenner)
df.McCulloch = pd.to_numeric(df.McCulloch)

features = df.drop('Place', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, df['Place'], random_state=None)

# Average CV score on the training set was: 0.7176863300509296
exported_pipeline = make_pipeline(
    MaxAbsScaler(),
    OneHotEncoder(minimum_fraction=0.15, sparse=False, threshold=10),
    ExtraTreesClassifier(bootstrap=True, criterion="entropy", max_features=0.15000000000000002, min_samples_leaf=7, min_samples_split=15, n_estimators=100)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
joblib.dump(exported_pipeline,'tree_model.joblib')
