import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score
#1-loading the dataset
housing=pd.read_csv("housing.csv")

#2-creating a stratified test set
housing['income_cat']=pd.cut(housing['median_income'],bins=[0.0,1.5,3.0,4.5,6.0,np.inf],labels=[1,2,3,4,5])

split=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)

for train_index,test_index in split.split(housing,housing['income_cat']):
    strat_train_set=housing.iloc[train_index].drop('income_cat',axis=1)
    strat_test_set=housing.iloc[test_index].drop('income_cat',axis=1)

#3-now working with copy of training data
housing=strat_train_set.copy()

#4-separating features and labels
housing_labels=housing['median_house_value'].copy()
housing=housing.drop('median_house_value',axis=1)

#5-separating numerical and categorical columns
num_attributes=housing.drop('ocean_proximity',axis=1).columns.tolist()
cat_attributes=['ocean_proximity']

#6- making a pipeline

#for numerical columns
num_pipeline=Pipeline([
    ("imputer",SimpleImputer(strategy="median")),
    ("scalar",StandardScaler())
])

#for categorical columns
cat_pipeline=Pipeline([
    ("onehot",OneHotEncoder(handle_unknown="ignore"))
])

#7-constructing a full pipeline
full_pipeline=ColumnTransformer([
    ("num",num_pipeline,num_attributes),
    ("cat",cat_pipeline,cat_attributes)
])

#8-transforming the data
housing_prepared=full_pipeline.fit_transform(housing)
print(housing_prepared.shape)

#7-Training the model
#linear Regression Model
lin_reg=LinearRegression()
lin_reg.fit(housing_prepared,housing_labels)
lin_preds=lin_reg.predict(housing_prepared)
# lin_rmse=root_mean_squared_error(housing_labels,lin_preds)
# print(f"The root mean squared error for Linear Regression is {lin_rmse}")
lin_rmses=-cross_val_score(lin_reg,housing_prepared,housing_labels,scoring="neg_root_mean_squared_error",cv=10)
print(pd.Series(lin_rmses).describe())

#Decision Tree Model
dec_reg=DecisionTreeRegressor()
dec_reg.fit(housing_prepared,housing_labels)
dec_preds=lin_reg.predict(housing_prepared)
# dec_rmse=root_mean_squared_error(housing_labels,dec_preds)
# print(f"The root mean squared error for Decision Tree is {dec_rmse}")
dec_rmses=-cross_val_score(dec_reg,housing_prepared,housing_labels,scoring="neg_root_mean_squared_error",cv=10)
print(pd.Series(dec_rmses).describe())

#Random Forest Model
random_reg=RandomForestRegressor()
random_reg.fit(housing_prepared,housing_labels)
random_preds=lin_reg.predict(housing_prepared)
# random_rmse=root_mean_squared_error(housing_labels,random_preds)
# print(f"The root mean squared error for Random Forrest Regression is {random_rmse}")
random_rmses=-cross_val_score(random_reg,housing_prepared,housing_labels,scoring="neg_root_mean_squared_error",cv=10)
print(pd.Series(random_rmses).describe())