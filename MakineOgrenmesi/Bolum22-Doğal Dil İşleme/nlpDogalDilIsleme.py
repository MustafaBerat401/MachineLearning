# -*- coding: utf-8 -*-


# HATA VAR !??!!! => ValueError: Input contains NaN, infinity or a value too large for dtype('float64').
import numpy as np
import pandas as pd


yorumlar = pd.read_csv('Restaurant_Reviews.csv',error_bad_lines=False)

import re 

import nltk
 
nltk.download('stopwords')
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

nltk.download('stopwords')
from nltk.corpus import stopwords

# Preprocessing (Önişleme)
comments = []

for i in range(716):
    
    yorum = re.sub('[^a-zA-Z]',' ',yorumlar['Review'][i])
      
    yorum = yorum.lower()
    yorum = yorum.split()
    yorum = [ps.stem(kelime) for kelime in yorum if not kelime in set(stopwords.words('english'))]
    yorum = ' '.join(yorum)
    comments.append(yorum)

# Feature Extraction (Öznitelik Çıkarımı)
# Bag of Words (BOW)

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features = 500)
X = cv.fit_transform(comments).toarray() # bağımsız değişken
y = yorumlar.iloc[:,1].values # bağımlı değişken

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train,y_train)

y_pred = gnb.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)



















