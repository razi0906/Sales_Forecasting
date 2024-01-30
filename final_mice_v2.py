# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 13:57:13 2020

@author: Yutish-pc
Implemented MICE with correctness in dataset
"""

# importing libraries
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

#####Data Preprocessing-------------------------------------------------

# getting the dataset
train_data = pd.read_csv('train.csv') 
features_data = pd.read_csv('features.csv')
stores_data = pd.read_csv('stores.csv')

#filling the missing values
from impyute.imputation.cs import mice
imputed_training=mice(features_data.iloc[:,2:11])

imputed_training[imputed_training < 0] = 0 

for i in range (0,7):
    features_data.iloc[:,4+i] = imputed_training[2+i]
    
#merging data
result = pd.merge(train_data,features_data,how = 'inner',
                  left_on = ['Store','Date','IsHoliday'],
                  right_on = ['Store','Date','IsHoliday'])

dataset = pd.merge(result,stores_data,on = 'Store')


# creating dummy variables
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelEncoder_data = LabelEncoder()
dataset.iloc[:, 4] = labelEncoder_data.fit_transform(dataset.iloc[:, 4])
dataset.iloc[:, 14] = labelEncoder_data.fit_transform(dataset.iloc[:, 14])

dataset['Date'] = dataset['Date'].str.replace('\D', '').astype(int)

onehotencoder  = OneHotEncoder(categorical_features= [0,1,14])
dataset = onehotencoder.fit_transform(dataset).toarray()

# setting X and y
X = dataset
y = dataset[:,130]
X = np.delete(X,[130],1)

# splitting of data
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.2, 
                                                 random_state = 0)

# fearute scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

'''.............................................
sc_y = StandardScaler()
#y_train = sc_y.fit_transform(y_train)
#y_test = sc_y.transform(y_test)
#to check later..................................
'''

# making the model (Random Forest regression)................................
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=100 , random_state = 0)
regressor.fit(X_train,y_train)

# predictiong the result
pred_test = regressor.predict(X_test)


#extra part..............
pred_train = regressor.predict(X_train)

#xtra part...............

# plotting the results.............................................
from sklearn.metrics import accuracy_score, confusion_matrix 

acc_test = accuracy_score(y_test.astype(np.int64), pred_test.astype(np.int64))
acc_train = accuracy_score(y_train.astype(np.int64), pred_train.astype(np.int64))

cm = confusion_matrix(y_test.astype(np.int64), pred_test.astype(np.int64))

plt.plot(y_test)
plt.show()
plt.plot(pred_test)
plt.show()

#%%

fig, ax= plt.subplots()
ax.scatter(y_test,pred_test,edgecolors=(0, 0, 0))
ax.plot([y_test.min(),y_test.max()],[y_test.min(),y_test.max()],'k--',lw=1)
ax.set_xlabel('Actual')
ax.set_ylabel('Predicted')
ax.set_title('Ground truth vs Predicted')
plt.show()



from sklearn.metrics import accuracy_score, confusion_matrix 

acc_test = accuracy_score(Y_test.astype(np.int64), pred_test.astype(np.int64))
acc_train = accuracy_score(Y_train.astype(np.int64), pred_train.astype(np.int64))

cm = confusion_matrix(Y_test.astype(np.int64), pred_test.astype(np.int64))

plt.plot(Y_test)
plt.show()
plt.plot(pred_test)
plt.show()




fig, ax= plt.subplots()
ax.scatter(Y_test,pred_test,edgecolors=(0, 0, 0))
ax.plot([Y_test.min(),Y_test.max()],[Y_test.min(),Y_test.max()],'k--',lw=1)
ax.set_xlabel('Actual')
ax.set_ylabel('Predicted')
ax.set_title('Ground truth vs Predicted')
plt.show()




print(regression_model.score(Y_train,pred_train))













