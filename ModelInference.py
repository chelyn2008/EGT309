import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# variables to trf here: X_train_scaled, X_val_scaled, X_train_1, X_val_1, y_train, y_val

# Instantiate baseline regression algorithms
lin_reg = LinearRegression()
kn_reg = KNeighborsRegressor()
tree = DecisionTreeRegressor(random_state = 42)
forest = RandomForestRegressor(random_state = 42)
grad_boost = GradientBoostingRegressor(random_state = 42)

# Define a function that fits and trains training data then test it on the test data 
def building_model(model, model_name, X_train, X_val):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
        
    train_score.append(round(model.score(X_train, y_train), 3))
    test_score.append(round(model.score(X_val, y_val), 3))
    r2.append(round(r2_score(y_val, y_pred), 3))
    mae.append(round(mean_absolute_error(y_val, y_pred), 3))
    rmse.append(round(np.sqrt(mean_squared_error(y_val, y_pred)), 3))

# Make a list of the algorithms that require scaling with their names
models_need_scaling = [lin_reg, kn_reg]
name_models_need_scaling = ['Linear Regression', 'K-Neighbors Regressor'] 

train_score, test_score, r2, mae, rmse = [], [], [], [], []

# Apply the function defined in the previous code snippet
for model, model_name in zip(models_need_scaling, name_models_need_scaling):
    building_model(model, model_name, X_train_scaled, X_val_scaled)

# Create a data frame that displays the metrics corresponding to each model 
scores_df1 = pd.DataFrame({'model': name_models_need_scaling,
                          'train_score': train_score,
                          'test_score': test_score,
                          'r2_score': r2,
                          'mean_absolute_error': mae,
                          'root_mean_squared_error': rmse}).sort_values(by = 'r2_score',
                                                                        ascending = False).set_index('model')
print(scores_df1)

# Make a list of the algorithms that don't require scaling with their names
models_without_scaling = [tree, forest, grad_boost] 
name_models_without_scaling = ['Decision Tree Regression', 
                               'Random Forest Regression', 
                               'Gradient Boosting Regressor'] 

train_score, test_score, r2, mae, rmse = [], [], [], [], []

features = ['human_average', 'environmental_average', 'infrastructure_average', 'weather_average']

# Apply the function defined in the previous code snippet
for model, model_name in zip(models_without_scaling, name_models_without_scaling):
    building_model(model, model_name, X_train_1[features], X_val_1[features])

# Create a data frame that displays the metrics corresponding to each model 
scores_df2 = pd.DataFrame({'model': name_models_without_scaling,
                          'train_score': train_score,
                          'test_score': test_score,
                          'r2_score': r2,
                          'mean_absolute_error': mae,
                          'root_mean_squared_error': rmse}).sort_values(by = 'r2_score',
                                                                        ascending = False).set_index('model')
print("", scores_df2)

# define a function that fits and the train the model on the training data for all parameters while using 
# cross-validation
def hyperparameter_tuning(model, params, X_train):
    clf = GridSearchCV(estimator = model, 
                       param_grid = params, 
                       scoring = 'neg_mean_squared_error', 
                       n_jobs = -1,
                       return_train_score = True,
                       refit = True,
                       cv = 5)
    clf.fit(X_train, y_train)
    
    print('Best estimator: ', clf.best_estimator_)
    print('Best score: ', clf.best_score_)

# KNN
params_kn = {'n_neighbors' : [5, 7, 9],
             'weights' : ['uniform', 'distance'],
             'metric' : ['minkowski', 'euclidean', 'manhattan']}
kn_reg = KNeighborsRegressor()
print("KNN model")
hyperparameter_tuning(kn_reg, params_kn, X_train_scaled)

# DT
params_tree = {'max_depth': [10, None],
               'min_samples_split': [2, 5],
               'min_samples_leaf': [1, 2]}
tree = DecisionTreeRegressor()
print("Decision Tree Regressor model")
hyperparameter_tuning(tree, params_tree, X_train_1[features])

# linear models
lin_reg = LinearRegression()
kn_reg = KNeighborsRegressor(n_neighbors = 9)
tree = DecisionTreeRegressor(min_samples_leaf = 2, min_samples_split = 5)

models_need_scaling = [lin_reg, kn_reg]
models_names_need_scaling = ['Linear Regression', 'k-Neighbors Regression']

train_score, test_score, r2, mae, rmse = [], [], [], [], []

for model, model_name in zip(models_need_scaling, models_names_need_scaling):
    building_model(model, model_name, X_train_scaled, X_val_scaled)

# Linear regression & KNN
scores_df3 = pd.DataFrame({'model': models_names_need_scaling,
                          'train_score': train_score,
                          'test_score': test_score,
                          'r2_score': r2,
                          'mean_absolute_error': mae,
                          'root_mean_squared_error': rmse}).sort_values(by = 'r2_score',
                                                                        ascending = False).set_index('model')
print("Models that need scaling: \n", scores_df3)

train_score, test_score, r2, mae, rmse = [], [], [], [], []

building_model(tree, 'Decision Tree Regressor', X_train_1[features], X_val_1[features])

# Decision Tree metrics
scores_df4 = pd.DataFrame({'model': 'Decision Tree Regressor',
                          'train_score': train_score,
                          'test_score': test_score,
                          'r2_score': r2,
                          'mean_absolute_error': mae,
                          'root_mean_squared_error': rmse}).sort_values(by = 'r2_score',
                                                                        ascending = False).set_index('model')

# comparison between all models
scores_df = pd.concat([scores_df3, scores_df4], axis = 0).sort_values(by = 'r2_score',
                                                                      ascending = False)
print("Models that no need scaling: \n", scores_df4)