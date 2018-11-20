import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error

predict_these = pd.read_csv("test_predictors.csv")
data = pd.read_csv("trainingdata.csv")
all_predictors = data.columns.values
all_predictors = np.delete(all_predictors,0)

predictors = ["X1","X3","X4","X5","X6","X7","X8","X12","X23","X25"]
use_data = pd.concat([data[predictors],data["X3"]*data["X5"],data["X5"]*data["X7"]],axis=1)
predict_these = pd.concat([predict_these[predictors],predict_these["X3"]*predict_these["X5"],predict_these["X5"]*predict_these["X7"]],axis=1)
y = data.y
X = use_data

#3 way interactions X4:X5:X12 X4:X5:X25
# use_data = pd.concat([data[predictors],data["X3"]*data["X5"],data["X5"]*data["X7"],data["X4"]*data["X5"]*data["X12"],data["X4"]*data["X5"]*data["X25"]],axis=1)
# y = data.y
# X = use_data

# hyperparameter selection
'''
def build_model(hparam,X_train,X_test,y_train,y_test):
  model = XGBRegressor(n_estimators=2300,gamma=0.06868686868686869,colsample_bytree=0.8383838384,subsample=0.36969696969696975,reg_lambda=0.4444444444444445,max_depth=1,eval_metric="rmse",reg_alpha=0.4050707070707071)
  model.fit(X_train,y_train,eval_set=[(X_test, y_test)], verbose=False)
  predictions = model.predict(X_test)
  rmse = np.sqrt(mean_squared_error(predictions,y_test))
  print(rmse)
  return rmse

times = 10
vals = np.linspace(0,0.4,100)
rmse_list = np.zeros(len(vals))

def cv(times,vals):
  for i in range(times):
    print('iteration:',i+1,'/',len(range(times)))
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
    for j in range(len(vals)):
      rmse_list[j] += build_model(vals[j],X_train,X_test,y_train,y_test)

cv(times,vals)
rmse_list/=times
print(rmse_list)
hparam = vals[np.argmin(rmse_list)]
print('the best hyperparameter',hparam)
plt.plot(vals,rmse_list) #1963 optimal n_estimators
plt.xlabel('hyperparameter')
plt.ylabel('mean absolute error')
plt.title('hyperparameter tuning')
plt.show()
'''
#model fit for final predictions
model = XGBRegressor(n_estimators=2300,gamma=0.06868686868686869,colsample_bytree=0.8383838384,subsample=0.36969696969696975,reg_lambda=0.4444444444444445,max_depth=1,eval_metric="rmse",reg_alpha=0.4050707070707071)
model.fit(X,y, verbose=False)
predictions = pd.Series(model.predict(predict_these))
predictions.to_csv("Submission.csv")
f = open("Submission.csv","r")
f_out = open("Submission_out.csv","w")
for line in f.readlines():
  line = line.split(',')
  line[0] = str(int(line[0]) + 1)
  line = ",".join(line)
  f_out.write(line)
f.close()
f_out.close()
