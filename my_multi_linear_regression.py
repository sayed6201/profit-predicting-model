# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset with the help of panda
dataset = pd.read_csv('50_Startups.csv')


#X(represents independent variables)=dataset.iloc[:, :-1(excluding the last column)].values
X = dataset.iloc[:, :-1].values
#y(is the dependent var) = dataset.iloc[:,4(all the data of 4th column)]
y = dataset.iloc[:, 4].values


# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
#X[:,3(3rd index is catagorical)]
X[:, 3] = labelencoder.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()


#avoiding dummy variable trap
#(removing 1st column) X = X[:(all lines of x),1:(all the column,starting from index 1)]
#the frameware takes it automatically, no need to it manually
X = X[:,1:]

  
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Feature Scaling
#libarary takes care of it, no need to do that
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""


#-------------------------------Coedeing begins-----------------------------------------


#fitting Multiple Linear regression to the training set
from sklearn.linear_model import LinearRegression
#creating object pf LinearRegression CLass
regressor = LinearRegression()
regressor.fit(X_train, y_train)


#predicting the Teat set results
y_pred = regressor.predict(X_test)


#---------------------------------------Step 2-----------------------------------------


#building optimal model using Backward elimination
#for backward elimination we need to prepare some stuff
#all indeoendent variables doesn'r have same segnificance Level,vars having less SL will be eliminated
# y = b0x0+b1x1+b2x2...bnxn, x0=1, we need to add x0=1 in the out matrix(indeoendent variables)
import statsmodels.formula.api as sm


#we are out independet variables to columns of 1's, having 59 lines....
#np.ones() function of numpy that adds ones, 
#axis =1/0, 1 if we are adding column/1 if raws or lines
X = np.append(arr = np.ones((50,1)).astype(int), values = X,axis=1)


#-------------------------------------Step 3-------------------------------------
#BACKWARD ELIMINATION STARTS FROM HERE................

#step 1: our SL = 0.05

#step 2: Fit the full model with all possible predictors
#X_opt have the required vars that has most impact on profiet, lower P-values means higher Significant varibale with respect to dependent variable
X_opt = X[:,[0,1,2,3,4,5]]
#sm.OLS() is from statsmodel library, 
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()

#step 3: looking for predcitor with higest  p values
#it will show all independent variables with their p values
regressor_OLS.summary()

#Step 4: removing Indepnedt var with higher p-value, it's X2[P-value = 0.990]
X_opt = X[:,[0,1,3,4,5]]
#step 5: fitting into the regressor without the variable
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
#step 3 Again: looking for predcitor with higest  p values
#it will show all independent variables with their p values
regressor_OLS.summary()

#Step 4: removing Indepnedt var with higher p-value, it's X1[P-value = 0.940]
X_opt = X[:,[0,3,4,5]]
#step 5: fitting into the regressor without the variable
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
#step 3 Again: looking for predcitor with higest  p values
#it will show all independent variables with their p values
regressor_OLS.summary()

#Step 4: removing Indepnedt var with higher p-value, it's X4[P-value = 0.602]
X_opt = X[:,[0,3,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

#Step 4: removing Indepnedt var with higher p-value, it's X5[P-value = 0.060]
X_opt = X[:,[0,3]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()