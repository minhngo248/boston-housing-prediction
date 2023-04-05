"""
Created 5th April 2023

@author : minh.ngo
"""

from sklearn.model_selection import cross_validate
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
import numpy as np
import pandas as pd
import argparse

def cross_validation(model, _x, _y, _cv) -> float:
    """
    Parameters
    ----------
    model       : model to be used for cross validation
    _x          : features to train
    _y          : target to train
    _cv         : int
                number of folds

    Returns
    -------
    results     : dictionary of average cross validation results 
                
    """
    _scoring = ['r2', 'neg_mean_squared_error']
    results = cross_validate(model, _x, _y, cv=_cv, scoring=_scoring)
    results['test_mean_squared_error'] = -results['test_neg_mean_squared_error']
    results['mean_r2'] = np.mean(results['test_r2'])
    return results

parser = argparse.ArgumentParser()
parser.add_argument('--nb-folds', type=int, default=5)
parser.add_argument('--test-size', type=float, default=0.2)
args = parser.parse_args()

# Read data
df = pd.read_csv('kc_house_data.csv', index_col='id')
# Drop columns
new_df = df.drop(['date', 'zipcode', 'yr_renovated'], axis=1)

# Normalization
normalizer = Normalizer()
new_df['sqft_living'] = normalizer.fit_transform(new_df['sqft_living'].values.reshape(-1,1))
new_df['sqft_lot'] = normalizer.fit_transform(new_df['sqft_lot'].values.reshape(-1,1))
new_df['sqft_above'] = normalizer.fit_transform(new_df['sqft_above'].values.reshape(-1,1))
new_df['sqft_basement'] = normalizer.fit_transform(new_df['sqft_basement'].values.reshape(-1,1))
new_df['sqft_living15'] = normalizer.fit_transform(new_df['sqft_living15'].values.reshape(-1,1))
new_df['sqft_lot15'] = normalizer.fit_transform(new_df['sqft_lot15'].values.reshape(-1,1))
new_df['yr_built'] = normalizer.fit_transform(new_df['yr_built'].values.reshape(-1,1))
new_df['lat'] = normalizer.fit_transform(new_df['lat'].values.reshape(-1,1))
new_df['long'] = normalizer.fit_transform(new_df['long'].values.reshape(-1,1))

# Split data
X = new_df.drop('price', axis=1)
y = new_df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=0)

# Cross validation
dict_models = {'linear_regression': LinearRegression()}
for key, model in dict_models.items():
    print(key)
    print(cross_validation(model, X_train, y_train, args.nb_folds)['mean_r2'])
