import pandas as pd
import numpy as np
import matplotlib as plt
import sklearn as sk
from sklearn.model_selection import train_test_split
import seaborn as sb

# from sklearn.datasets import fetch_california_housing
# housing = fetch_california_housing()
# housing_df = pd.DataFrame(housing["data"],columns=housing['feature_names'])
# housing_df['target'] = housing['target']
# housing_df = housing_df.drop('MedHouseVal', axis = 1)

# from sklearn.linear_model import Ridge
# np.random.seed(42)

# X = housing_df.drop('target', axis=1)
# y = housing_df['target']

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# model = Ridge()
# model.fit(X_train, y_train)

# print('The Ridge model score is: {:.4f}'.format(model.score(X_test, y_test)))

# from sklearn.ensemble import RandomForestRegressor

# np.random.seed(42)

# X = housing_df.drop('target', axis=1)
# y = housing_df['target']

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# model = RandomForestRegressor()
# model.fit(X_train, y_train)

# print('The RandomForestRegressor model score is: {:.4f}'.format(model.score(X_test, y_test)))

heart_disease = pd.read_csv('../sklearn/heart-disease.csv')
# from sklearn.svm import LinearSVC
# np.random.seed(42)
# X = heart_disease.drop('target',axis=1)
# y = heart_disease['target']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# clf = LinearSVC(max_iter=1000)
# clf.fit(X_train, y_train)
# print('The LinearSVC model score is: {:.4f}'.format(clf.score(X_test, y_test)))

from sklearn.ensemble import RandomForestClassifier
np.random.seed(42)
X = heart_disease.drop('target',axis=1)
y = heart_disease['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)
print('The LinearSVC model score is: {:.4f}'.format(clf.score(X_test, y_test)))
