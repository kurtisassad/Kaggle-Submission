require(gplots)
require(RColorBrewer)

if(getwd() != "/home/kurt/Desktop/sta314/data"){
  setwd("/home/kurt/Desktop/sta314/data")
  data = read.csv("trainingdata.csv")
  predict_these = read.csv("test_predictors.csv")
}
if(getwd() != "/home/kurt/Desktop/sta314/R"){
  setwd("/home/kurt/Desktop/sta314/R")
}

hists <- function(data){
  for (i in 2:ncol(data)) {
    hist(data[,i],main=names(data)[i]) 
  }
}
ks_matrix <- function(data){
  len = ncol(data)
  pvals <- matrix( rep( 0, len=(len-1)^2), nrow = (len-1))
  for (i in 2:len) {
    for (j in 2:len){
      pvals[i-1,j-1] = ks.test(data[,i],data[,j])$p.value
    }
  }
  return(pvals)
}

plot_heat_map <- function(mat){
  rownames(mat) <- names(data)[-1]
  my_palette <- colorRampPalette(c("red", "yellow", "green"))(n = 299)
  heatmap.2(mat,dendrogram='none', Rowv=TRUE, Colv=TRUE,main="Pvalues of KS test",col=my_palette)
}

#hists(data)
#plot_heat_map(ks_matrix(data))
