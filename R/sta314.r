require(caret)
require(gsubfn)
require(glmnet)
require(nnet)
require(rpart)

if(getwd() != "/home/kurt/Desktop/sta314/data"){
  setwd("/home/kurt/Desktop/sta314/data")
  data = read.csv("trainingdata.csv")
  predict_these = read.csv("test_predictors.csv")
}
if(getwd() != "/home/kurt/Desktop/sta314/R"){
  setwd("/home/kurt/Desktop/sta314/R")
}

if(F){
temp = data$y
data = predict(preProcess(data[, -1], method=c("center", "scale")),data[,-1])
data$y = temp
predict_these= predict(preProcess(predict_these[, -1], method=c("center", "scale")),predict_these[,-1])
}
rmse <- function(actual,predicted){
  sqrt(sum(actual-predicted)^2/length(actual))
}

partition_data <- function(data,split_percent){
  train_index = createDataPartition(data$y,p=split_percent, list=FALSE)
  train_data = data[train_index,]
  test_data = data[-train_index,]
  return(list(train_data=train_data,test_data=test_data))
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

lasso_prediction <- function(x_train,y_train,x_test,y_test,lambda){
  x_train <- as.matrix(train_data[,-1])
  x_test <- as.matrix(test_data[,-1])
  lasso_model <- glmnet(x_train, train_data[,1], alpha = 1, lambda = lambda)
  err <- rmse(predict(lasso_model,newx=x_test),test_data[,1])
}

ridge_prediction <- function(train_data,test_data,lambda){
  x_train <- as.matrix(train_data[,-1])
  x_test <- as.matrix(test_data[,-1])
  ridge_model <- glmnet(x_train, train_data[,1], alpha = 0, lambda = lambda)
  return(rmse(predict(ridge_model,newx=x_test),test_data[,1]))
  
}

nn_prediction <- function(train_data,test_data,hparam){
  nn_model <- nnet(y~.,data=train_data,linout=TRUE,size=floor(hparam),maxit=100,trace=F)
  return(rmse(predict(nn_model,newdata=test_data[,-1]),test_data[,1]))
}

dtree_prediction <- function(train_data,test_data,hparam){
  tree_model <- rpart(y~.,data=train_data)
  tree_model <- prune(tree_model,cp=hparam)
  return(rmse(predict(tree_model,newdata=test_data[,-1]),test_data[,1]))
}

turn_into_csv <- function(data){
  da.sample = data.frame(cbind(1:500,data))
  names(da.sample) = c('id','y') 
  write.csv(da.sample,file='Submission.csv',row.names=FALSE)
}

k_fold_model <- function(data,k,hparam_vals,model_function,iters){
  rmse_list <- array(0,length(hparam_vals))
  for(l in 1:iters){
    print(paste("iteration",l))
    partitions <- k_fold_partition(data,k)
    for(i in 1:length(partitions)){
      test_data <- partitions[[i]]
      train_data <- data.frame()
      for(j in 1:length(partitions)){
        if(j != i){
          train_data <- rbind(train_data,partitions[[j]])
        }
      }
      for(m in 1:length(hparam_vals)){
        hparam <- hparam_vals[m]
        err <- model_function(train_data,test_data,hparam)
        rmse_list[m] <- rmse_list[m] + err
      }
    }
  }
  for(i in 1:length(hparam_vals)){
    rmse_list[i] <- rmse_list[i]/(k*iters)
  }
  plot(hparam_vals,rmse_list)
  print("RMSE list corresponding to hyperparameters",rmse_list)
  print(paste("minimum RMSE",min(rmse_list)))
  best_hparam = hparam_vals[which.min(rmse_list)]
  print(paste("best hyperparameter",best_hparam))
  return(best_hparam)
}


if(F){
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
}

#partition data into training and validaion set
#list[x_train,y_train,x_test,y_test] = partition_data(data,0.8)
#list[lin_model,lin_rmse] <- build_lin_model(x_train,y_train,x_test,y_test)
#p_val_forward_selection(train_data,2)
#list[lasso_model,lasso_rmse] <- lasso_prediciton(x_train,y_train,x_test,y_test,0.1)
#print(lin_rmse)
#print(lasso_rmse)
#turn_into_csv(y)

#fresh attempt
factors = c("X11","X15","X17","X12")
maybe = c("X13","X12")
data[,factors] <- lapply(round(data[,factors]),FUN=factor)

#hparam_vals = 10^seq(0, -10, length.out = 20)
hparam_vals = seq(1, 10, length.out = 10)
best_hparam = k_fold_model(data,5,hparam_vals,nn_prediction,5)

#list[train_data,test_data] = partition_data(data,0.8)

#plot(tree_model,uniform=T)
#text(tree_model)
#barplot(tree_model$variable.importance,las=2)
#print(summary(mod))








