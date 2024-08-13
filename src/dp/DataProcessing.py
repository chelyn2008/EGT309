# Import the basic libraries
import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

# Load the training and test sets
train = pd.read_csv('/mnt/data/train.csv', index_col = 'id')
test = pd.read_csv('/mnt/data/test.csv', index_col = 'id')

# Display the shape of the data frames
print('Shape of the:')
print('Training set: ', train.shape)
print('Test set: ', test.shape)

# Create a function to compute the duplicates 
def duplicates(df, name_df):
    print(f'The {name_df} set has %d duplicates.' %df.duplicated().sum())

# Determine the duplicates in the training set
duplicates(train, 'training')

# Determine the duplicates in the test set
duplicates(test, 'test')

#compute empty cell
print(train.isna().sum().any())
print(test.isna().sum().any())

#printing info summary
train.info()

#descriptive analysis
train.describe()

#define feature and target variable
X = train.drop('FloodProbability', axis = 1)
y = train['FloodProbability']

#spitting training data into training and validation set
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2, random_state = 42)
X_test = test.copy()

print('For the training set:')
print('Shape of the features: ', X_train.shape)
print('Shape of the response variable: ', y_train.shape)
print('*'*60)
print('*'*60)
print('For the validation set:')
print('Shape of the features: ', X_val.shape)
print('Shape of the response variable: ', y_val.shape)
print('*'*60)
print('*'*60)
print('For the test set:')
print('Shape of the features: ', X_test.shape)

#checking for null values in trained df
train_df = pd.concat([X_train, y_train], axis = 1)
train_df.isna().sum().any()

# Display the min and max values of the numerical variables
min_val = [X_train[col].min() for col in X_train.columns]
max_val = [X_train[col].max() for col in X_train.columns]

minmax_df = pd.DataFrame({'feature': X_train.columns,
                         'min_value': min_val,
                         'max_value': max_val}).set_index('feature')
minmax_df


#feature selection
from sklearn.feature_selection import mutual_info_regression

def compute_mi_scores(X, y):
    mi_scores = mutual_info_regression(X, y)
    mi_scores_df = pd.DataFrame({'feature': X.columns,
                                 'mi_scores': mi_scores}).set_index('feature').sort_values(by = 'mi_scores', 
                                                                                           ascending = False)
    return mi_scores_df

compute_mi_scores(X_train, y_train)

#printing the X train columns
X_train.columns

#feature engineering
weather_features = ['MonsoonIntensity', 'ClimateChange']
infrastructure_features = ['TopographyDrainage', 'RiverManagement', 'DamsQuality', 
                  'DrainageSystems', 'DeterioratingInfrastructure'] 

environmental_features = ['Deforestation',  'Siltation', 'CoastalVulnerability', 'WetlandLoss', 'Landslides', 'Watersheds']
human_features = ['Urbanization', 'PopulationScore', 'InadequatePlanning', 'PoliticalFactors', 
        'IneffectiveDisasterPreparedness', 'AgriculturalPractices', 'Encroachments']

X_train_1 = X_train.copy()
X_val_1 = X_val.copy()
X_test_1 = X_test.copy()

X_train_1['weather_average'] = round(X_train_1[weather_features].mean(axis = 1), 2)
X_val_1['weather_average'] = round(X_val_1[weather_features].mean(axis = 1), 2)
X_test_1['weather_average'] = round(X_test_1[weather_features].mean(axis = 1), 2)

X_train_1['infrastructure_average'] = round(X_train_1[infrastructure_features].mean(axis = 1), 2)
X_val_1['infrastructure_average'] = round(X_val_1[infrastructure_features].mean(axis = 1), 2)
X_test_1['infrastructure_average'] = round(X_test_1[infrastructure_features].mean(axis = 1), 2)

X_train_1['environmental_average'] = round(X_train_1[environmental_features].mean(axis = 1), 2)
X_val_1['environmental_average'] = round(X_val_1[environmental_features].mean(axis = 1), 2)
X_test_1['environmental_average'] = round(X_test_1[environmental_features].mean(axis = 1), 2)

X_train_1['human_average'] = round(X_train_1[human_features].mean(axis = 1), 2)
X_val_1['human_average'] = round(X_val_1[human_features].mean(axis = 1), 2)
X_test_1['human_average'] = round(X_test_1[human_features].mean(axis = 1), 2)

X_train_1.head()

features = ['human_average', 'environmental_average', 'infrastructure_average', 'weather_average']


#scaling the features into 3 different sets
X_train_scaled = X_train_1[features].copy()
X_val_scaled = X_val_1[features].copy()
X_test_scaled = X_test_1[features].copy()

# Implement min-max scaling to make the range of all variables between 0 and 1
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train_scaled)
X_val_scaled = scaler.transform(X_val_scaled)
X_test_scaled = scaler.transform(X_test_scaled)

print("Shape of X_train_scaled: ", X_train_scaled.shape)
print("Shape of X_val_scaled: ", X_val_scaled.shape)
print("Shape of X_train_1: ", X_train_1.shape)
print("Shape of X_val_1: ", X_val_1.shape)
print("Shape of y_train: ", y_train.shape)
print("Shape of X_test: ", X_test.shape)
print("Shape of X_test_1: ", X_test_1.shape)
print("Shape of X_test_scaled: ", X_test_scaled.shape)
print("Shape of X_train_1: ", X_train_1.shape)
print("Shape of X_val: ", X_val.shape)
print("Shape of y_val: ", y_val.shape)

# exporting datasets
X_test.to_csv('/mnt/data/X_test.csv')
X_test_1.to_csv('/mnt/data/X_test_1.csv')
np.save('/mnt/data/X_test_scaled.npy', X_test_scaled)
np.save('/mnt/data/y_train.npy', y_train)
X_train_1.to_csv('/mnt/data/X_train_1.csv')
np.save('/mnt/data/X_train_scaled.npy', X_train_scaled)
X_val.to_csv('/mnt/data/X_val.csv')
X_val_1.to_csv('/mnt/data/X_val_1.csv')
np.save('/mnt/data/X_val_scaled.npy', X_val_scaled)
np.save('/mnt/data/y_val.npy', y_val)
