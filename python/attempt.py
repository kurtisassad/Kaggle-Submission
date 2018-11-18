import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from numpy import linspace
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
import statsmodels.api as sm
import seaborn as sns

predict_these = pd.read_csv("test_predictors.csv")
data = pd.read_csv("trainingdata.csv")
rmse = lambda x: np.sqrt(mean_squared_error(x))

# f,ax = plt.subplots(figsize=(18, 18))
# sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
# plt.show()

y = data.y
predictors = ["X1","X2","X3","X4","X5","X6","X7","X8","X12","X23","X25"]
# predictors = ["X1","X2","X3","X4","X5","X6","X7","X8","X9","X10","X11","X13","X14","X15","X16","X17","X18","X19","X20","X21","X22","X24","X12","X23","X25"]
X = data[predictors]
model = GradientBoostingRegressor()
model.fit(X,y)

#model building
def build_model(n_estimators,early_stopping_rounds,X_train,X_test,y_train,y_test):
  model = XGBRegressor(n_estimators=n_estimators,eval_metric="rmse")
  model.fit(X_train,y_train,eval_set=[(X_test, y_test)], verbose=False)
  predictions = model.predict(X_test)
  rmse = np.sqrt(mean_squared_error(predictions,y_test))
  print(rmse)
  return rmse


def cv(times,vals):
  for i in range(times):
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

    x_axis = linspace(1900,2000,vals)
    for j in range(vals):
      rmse_list[j] += build_model(int(x_axis[j]),5,X_train,X_test,y_train,y_test)


model = XGBRegressor(n_estimators=n_est)
model.fit(X_train,y_train,eval_set=[(X_test, y_test)], verbose=False)
predictions = pd.Series(model.predict(predict_these[predictors]))
predictions = pd.Series(predictions)
predictions.to_csv("Submission.csv")

times = 3
vals = 10
rmse_list = np.zeros(vals)
cv(times,vals)

n_est = int(x_axis[np.argmin(rmse_list)])
print(n_est)
plt.plot(x_axis,rmse_list)
plt.xlabel('n_estimators')
plt.ylabel('mean absolute error')
plt.title('hyperparameter tuning')
plt.show()
