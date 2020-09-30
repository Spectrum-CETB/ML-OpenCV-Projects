  
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,GridSearchCV
import keras
from keras.models import Sequential
from keras.layers import Dropout,Dense,Activation
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn import metrics
import matplotlib.pyplot as plt

data=pd.read_csv("datasets_180_408_data.csv")
data.drop(labels='Unnamed: 32',axis=1,inplace=True)
data['diagnosis'].replace('M',0,inplace=True)
data['diagnosis'].replace('B',1,inplace=True)
data.head(6)

x=data.iloc[:,2:33].values
y=data.iloc[:,1].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

s=SimpleImputer(missing_values=0,strategy='mean')
x_train=s.fit_transform(x_train)
x_test=s.fit_transform(x_test)

scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.fit_transform(x_test)

def hyper_parameter_training(layer,activation):
  model=Sequential()
  for i,node in enumerate(layer):
    if i==0:
      model.add(Dense(node,input_dim=x_train.shape[1]))
      model.add(Activation(activation))
      model.add(Dropout(0.3))
    else:
      model.add(Dense(node))
      model.add(Activation(activation))
      model.add(Dropout(0.3))
  model.add(Dense(units=1,kernel_initializer='glorot_uniform',activation='sigmoid'))
  model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
  return model

model=KerasClassifier(hyper_parameter_training)

layers=[(10,),(30,20,10),(40,20)]
activations=['sigmoid','relu']

param_grid=dict(layer=layers,activation=activations,batch_size=[128,256],epochs=[30])

grid=GridSearchCV(model,param_grid,cv=5)

grid.fit(x_train,y_train)

y_pred=grid.predict(x_test)
y_pred=(y_pred>0.5)

val=metrics.accuracy_score(y_test,y_pred)
print("accuracy is =",str(val*100)+" %")
