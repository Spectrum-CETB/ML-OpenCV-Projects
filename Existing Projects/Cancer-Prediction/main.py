import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np

data=pd.read_csv("cancer.csv")
x=data.iloc[:,[0:33]]
y=data.iloc[:,33]

x_train,y_train,x_test,y_test=train_test_split(x,y,test_size=0.30,random_state=0)
lr=LogisticRegression()
lr.fit_transform(x_train,y_train)

y_pred=lr.predict(x_test)
