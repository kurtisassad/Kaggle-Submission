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
  x_train <- train_data
  x_train$y <- NULL
  y_train <- train_data$y
  x_test <- test_data
  x_test$y <- NULL
  y_test <- test_data$y
  return(list(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test))
}

k_fold_partition <- function(data,k){
  n = nrow(data)
  fold_size = floor(n/k)
  indicies <- sample(1:n,n)
  partitions = list()
  i <- 1
  take <- 0
  while(take < n){
    partitions[[i]] <- data[indicies[(take+1):(take+fold_size)],]
    take <- take + fold_size
    i<- i + 1
  }
  return(partitions)
}

important_preds <- function(x_train,y_train){
  #take these preds
  x_train$y <- y_train
  important = (summary(lm(y~.,data=x_train))$coefficients[,4] < 0.05)
  return(important)
}

build_lin_model <- function(x_train,y_train,x_test,y_test){
  use_data = data[,which(important_preds(x_train,y_train))]
  use_data$y = x_train$y
  lin_reg = lm(y~.,data=use_data)
  err<-rmse(predict(lin_reg,x_test),y_test)
  return(list(model = lin_reg,err=err))
}

#finish this later
p_val_forward_selection <- function(data,max_exp){
  important_predictors <- important_preds(data)
  i <- 1
  j <- 1
  new_preds <- data[,-which(important_predictors)]
  new_preds$y <- NULL
  while( i < ncol(new_preds)){
    while(j < max_exp+1){
      #new_data <- data[,important_predictors]
      #new_data$y <- data$y
      #new_data[(names(new_preds[i]))] <- data[names(new_preds[i])]
      #lm(y~.,data=new_data)
      j <- j+ 1
    }
    i <- i + 1
    j <- 1
  }
}

lasso_prediciton <- function(x_train,y_train,x_test,y_test,lambda){
  x_train <- as.matrix(x_train)
  x_test <- as.matrix(x_test)
  lasso_model <- glmnet(x_train, y_train, alpha = 1, lambda = lambda)
  err <- rmse(predict(lasso_model,newx=x_test),y_test)
  return(list(model=lasso_model,err=err))
}

ridge_prediciton <- function(x_train,y_train,x_test,y_test,lambda){
  x_train <- as.matrix(x_train)
  x_test <- as.matrix(x_test)
  ridge_model <- glmnet(x_train, y_train, alpha = 0, lambda = lambda)
  err <- rmse(predict(ridge_model,newx=x_test),y_test)
  return(list(model=ridge_model,err=err))
}

turn_into_csv <- function(data){
  da.sample = data.frame(cbind(1:500,data))
  names(da.sample) = c('id','y') 
  write.csv(da.sample,file='Submission.csv',row.names=FALSE)
}

k_fold_model <- function(k,hparam_vals,model_function,iters){
  rmse_list <- array(0,length(hparam_vals))
  for(l in 1:iters){
    partitions <- k_fold_partition(data,k)
    for(m in 1:length(hparam_vals)){
      hparam <- hparam_vals[m]
      for(i in 1:length(partitions)){
        test_data <- partitions[[i]]
        train_data <- data.frame()
        for(j in 1:length(partitions)){
          if(j != i){
            train_data <- rbind(train_data,partitions[[j]])
          }
        }
        x_train <- train_data
        x_train$y <- NULL
        y_train <- train_data$y
        x_test <- test_data
        x_test$y <- NULL
        y_test <- test_data$y
        list[model,err] <- model_function(x_train,y_train,x_test,y_test,hparam)
        rmse_list[m] <- rmse_list[m] + err
      }
    }
    for(i in 1:length(hparam_vals)){
      rmse_list[i] <- rmse_list[i]/(length(partitions)*length(iters))
    }
  }
  plot(hparam_vals,rmse_list)
  print(min(rmse_list))
  best_hparam = hparam_vals[which.min(rmse_list)]
  print(best_hparam)
  return(best_hparam)
}
hparam_vals = seq(0, 0.2, length.out = 100)
best_hparam = k_fold_model(5,hparam_vals,lasso_prediciton,10)

#if(F){
#fit final model to all data
x_train <- data
x_train$y <- NULL
y_train <- data$y
x_test <- data
x_test$y <- NULL
y_test <- data$y
#build model
list[model,err] = lasso_prediciton(x_train,y_train,x_test,y_test,best_hparam)
print(coef(model))
predict_these$X1.500 <- NULL
#turn into predictions
y <- predict(model,newx=as.matrix(predict_these))
turn_into_csv(y)
#}

#partition data into training and validaion set
#list[x_train,y_train,x_test,y_test] = partition_data(data,0.8)
#list[lin_model,lin_rmse] <- build_lin_model(x_train,y_train,x_test,y_test)
#p_val_forward_selection(train_data,2)
#list[lasso_model,lasso_rmse] <- lasso_prediciton(x_train,y_train,x_test,y_test,0.1)
#print(lin_rmse)
#print(lasso_rmse)
#turn_into_csv(y)