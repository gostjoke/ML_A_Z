# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values
y = y.astype(float)
# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
### fix the categorical_features
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

# Avoiding the Dummy Variable Trap
X = X[:, 1:]
X = X.astype(float)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()  
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Building the optimal model using Backward Elimination
# Building the optimal model using Backward Elimination
import statsmodels.api as sm

X_train = np.append(arr=np.ones((40, 1)).astype(int), values=X_train, axis=1)
X_opt = X_train[:, [0, 1, 2, 3, 4, 5]]
regressor_OLS = sm.OLS(endog= y_train, exog= X_opt).fit()
regressor_OLS.summary()

# Eliminate feature with the highest p-value (if greater than 0.05)
X_opt = X_train[:, [0, 1, 3, 4, 5]]
regressor_OLS = sm.OLS(y_train, X_opt).fit()
regressor_OLS.summary()

# Continue eliminating features with high p-values
X_opt = X_train[:, [0, 3, 4, 5]]
regressor_OLS = sm.OLS(y_train, X_opt).fit()
regressor_OLS.summary()

X_opt = X_train[:, [0, 3, 5]]
regressor_OLS = sm.OLS(y_train, X_opt).fit()
regressor_OLS.summary()

X_opt = X_train[:, [0, 3]]
regressor_OLS = sm.OLS(y_train, X_opt).fit()
regressor_OLS.summary()
print(regressor_OLS.summary())


