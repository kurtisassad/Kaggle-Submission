import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

predict_these = pd.read_csv("test_predictors.csv")
data = pd.read_csv("trainingdata.csv")
all_predictors = data.columns.values
all_predictors = np.delete(all_predictors,0)

f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()

2 way interaction finder
for i in range(1,len(all_predictors)):
  for j in range(i+1,len(all_predictors) - 1):
    temp_data = pd.concat([data['y'],data['X{}'.format(i)]*data['X{}'.format(j)]],axis=1)
    temp_data.columns = ['y', 'x']
    correlation = abs(temp_data['y'].corr(temp_data['x']))
    if(correlation >=0.15):
      print('X{}:X{}'.format(i,j),'correlation',correlation)
    f,ax = plt.subplots(figsize=(1, 1))
    sns.heatmap(temp_data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
    plt.show()

3 way interaction finder
for i in range(1,len(all_predictors)):
  for j in range(i+1,len(all_predictors)):
    for k in range(j+1,len(all_predictors)):
      if((i != j and j!=k) or (i!=k and j!=k)):
        temp_data = pd.concat([data['y'],data['X{}'.format(i)]*data['X{}'.format(j)]*data['X{}'.format(k)]],axis=1)
        temp_data.columns = ['y', 'x']
        correlation = abs(temp_data['y'].corr(temp_data['x']))
        if(correlation >=0.2):
          print('X{}:X{}:X{}'.format(i,j,k),'correlation',correlation)

newdf = pd.concat([data,data['y'], np.sqrt(data['X6']*data['X1']),np.sqrt(data['X7']*data['X1']),np.sqrt(data['X8']*data['X1'])], axis=1)
f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(newdf.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()
