import pandas as pd
import numpy as np

import warnings
import pickle
warnings.filterwarnings("ignore")

rds=pd.read_csv("C:/Users/saika/Downloads/heart_failure_clinical_records_dataset.csv")
#print(rds.columns)

dropcol=[1,2,4,6,7,8,11]
rds2=rds.drop(rds.columns[dropcol],axis=1)
x=rds2.iloc[:,0:5]
y=rds2.iloc[:,5]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x, y, test_size=0.2, random_state=42)
from sklearn.ensemble import RandomForestRegressor
reg=RandomForestRegressor()
reg.fit(x_train,y_train)
rsquare=reg.score(x_train,y_train)
print(rsquare)

y_pred=reg.predict([[50,0,1,0,0]])
print(float(y_pred))
with open('model.pkl', 'wb') as f:
    pickle.dump(reg, f)

# Load the model from the pickle file
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)