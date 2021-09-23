# -*- coding: utf-8 -*-

#1.kutuphanler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#2.veri onisleme
#2.1.veri yukleme

veriler = pd.read_csv('Churn_Modelling.csv')
print(veriler)

# veri on isleme

X = veriler.iloc[:,3:13].values
Y = veriler.iloc[:,13].values


# encoder: Kategorik -> Numeric (Veri Birleştirme)

from sklearn import preprocessing

le = preprocessing.LabelEncoder()
X[:,1] = le.fit_transform(X[:,1])

le2 = preprocessing.LabelEncoder()
X[:,2] = le2.fit_transform(X[:,2])




# OneHotEncoder: kolon başlıklarını etiketlere taşımak ve her etiketin altına 1 veya 0 diyerek oraya ait veya değil demektir.
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

ohe = ColumnTransformer([("ohe", OneHotEncoder(dtype=float),[1])],
                        remainder="passthrough")

X = ohe.fit_transform(X)
X = X[:,1:]



# verilerin train ve test olarak bölünmesi
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2,, random_state=0)

from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_pred,y_test)
print(cm)











