# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# veri yükleme

veriler = pd.read_csv('eksikveriler.csv')

print(veriler)


# veri ön işleme

boy = veriler[['boy']]

print(boy)

boyKilo = veriler[['boy','kilo']]
print(boyKilo)

# eksik veriler

# sci - kit learn kütüphanesi

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values = np.nan, strategy='mean')

yas = veriler.iloc[:,1:4].values
print(yas)

imputer = imputer.fit(yas[:,1:4])
yas[:,1:4] = imputer.transform(yas[:,1:4])
print(yas)