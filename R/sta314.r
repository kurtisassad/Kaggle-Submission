require(caret)
require(gsubfn)
require(glmnet)

if(getwd() != "/home/kurt/Desktop/sta314/data"){
  setwd("/home/kurt/Desktop/sta314/data")
  data = read.csv("trainingdata.csv")
  predict_these = read.csv("test_predictors.csv")
}
if(getwd() != "/home/kurt/Desktop/sta314/R"){
  setwd("/home/kurt/Desktop/sta314/R")
}

rmse <- function(actual,predicted){
  sqrt(sum(actual-predicted)^2/length(actual))
}

partition_data <- function(data,split_percent){
  train_index = createDataPartition(data$y,p=split_percent, list=FALSE)
  train_data = data[train_index,]
  test_data = data[-train_index,]
  return(list(train=train_data,test=test_data))
}

important_preds <- function(data){
  #build datasets
  y_data = data$y
  x_data = data[,-1]
  #take these preds
  important = (summary(lm(y~.,data=data))$coefficients[,4] < 0.05)
  return(important)
}

build_lin_model <- function(data){
  use_data = data[,which(important_preds(data))]
  use_data$y = data[,"y"]
  return(lin_reg = lm(y~.,data=use_data))
}

compute_rmse <- function(predictor,data){
  predictions <- predict(predictor,newdata=data)
  print(rmse(predictions,data[,"y"]))
}

#finish this later
p_val_forward_selection <- function(data,max_exp){
  important_predictors <- important_preds(data)
  i <- 1
  j <- 1
  new_preds <- data[,-which(important_predictors)]
  new_preds <- preds[,names(preds)!="y"]
  print(ncol(preds))
  while( i < ncol(preds)){
    while(j < max_exp+1){
      new_data <- data[,important_predictors]
      new_data$y <- data$y
      new_data[(names(new_preds[i]))] <- data[names(new_preds[i])]
      lm(y~.,data=new_data)
      j <- j+ 1
    }
    i <- i + 1
    j <- 1
  }
}

lasso_prediciton <- function(data,lambda){
  x <- model.matrix(y ~ . ,data )[,-1]
  lasso.mod = glmnet(x,data$y,alpha = 1, lambda = lambda)
}

turn_into_csv <- function(data){
  da.sample = data.frame(cbind(1:500,data))
  names(da.sample) = c('id','y') 
  write.csv(da.sample,file='Submission.csv',row.names=FALSE)
}


list[train_data,test_data] = partition_data(data,0.8)
lin_model <- build_lin_model(train_data)
y <- predict(lin_model,predict_these)
turn_into_csv(y)
p_val_forward_selection(train_data,2)
compute_rmse(lin_model,test_data)

