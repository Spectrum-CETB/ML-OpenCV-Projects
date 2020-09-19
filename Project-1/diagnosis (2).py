import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split

data=pd.read_csv("datasets_180_408_data.csv")
data.head()

x=data.iloc[:,2:33].values
y=data.iloc[:,2].values

x_train,x_test,y_train_y_test=train_test_split(x,y,test_size=0.2,random_state=0)

classifier=LogisticRegression(random_state=0)
classifier.fit(x_train,y_train)

y_pred=classifier.predict(x_test)



