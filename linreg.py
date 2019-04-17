import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
df = pd.read_csv('Admission_Predict.csv')
from sklearn.model_selection import train_test_split
X = df[['GRE Score', 'TOEFL Score', 'University Rating', 'CGPA']]
y = df[['Chance of Admit ']]
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
from sklearn.linear_model import LogisticRegression
lm = LogisticRegression()
lm.fit(x_train, y_train)
Yhat = lm.predict(x_test)

