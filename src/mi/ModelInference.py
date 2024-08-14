import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import os

# Read the CSV files into dataframes
X_train_1 = pd.read_csv('/mnt/data/X_train_1.csv')
y_train = np.load('/mnt/data/y_train.npy')
X_test_1 = pd.read_csv('/mnt/data/X_test_1.csv')
X_test = pd.read_csv('/mnt/data/X_test.csv')

#Test the best Model on the Test Set
y_test = pd.read_csv('/mnt/data/sample_submission.csv', index_col = 'id')

kn_reg = KNeighborsRegressor(n_neighbors = 9)

kn_reg.fit(X_train_1, y_train)
y_pred = kn_reg.predict(X_test_1)

print('Training score: %.4f' %kn_reg.score(X_train_1, y_train))
print('Mean Absolute Error: %.4f' %mean_absolute_error(y_test, y_pred))
print('Root Mean Squared Error: %.4f' %np.sqrt(mean_squared_error(y_test, y_pred)))

pd.DataFrame(y_pred).describe()

y_pred = pd.Series(y_pred, index = X_test.index, name = 'predicted_proba')
results = pd.concat([y_test, y_pred], axis = 1)
results
