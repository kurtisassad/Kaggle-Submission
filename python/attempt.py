import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from numpy import linspace
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer,mean_squared_error
from sklearn.cross_validation import KFold


predict_these = pd.read_csv("test_predictors.csv")
data = pd.read_csv("trainingdata.csv")

def rmse(y, y_pred):
        return np.sqrt(np.mean((y_pred - y)**2))

X_data = data.drop(['y'], axis=1)
y_data = data['y']
predict_these = predict_these.drop(['1:500'],axis=1)
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2)
params={
    'max_depth': [3,5,7],
    'subsample': [0.5,1.0],
    'colsample_bytree': [0.5,1.0],
    'n_estimators': [2000,2500,3000],
    'reg_alpha': [0,0.01,0.04],
}
model = XGBRegressor(objective='reg:linear')
scorer = make_scorer(rmse)
gs = GridSearchCV(model,
                  params,
                  cv=5,
                  scoring=scorer)
gs = gs.fit(X_train, y_train)
model = gs.best_estimator_
model.fit(X_train, y_train)

predictions = pd.Series(model.predict(predict_these))
predictions = pd.Series(predictions)
predictions.to_csv("Submission.csv")
