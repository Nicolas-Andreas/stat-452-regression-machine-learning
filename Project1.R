#Project 1

get.folds = function(n, k) {
  n.fold = ceiling(n / k)
  fold.ids.raw = rep(1:k, times = n.fold)
  fold.ids = fold.ids.raw[1:n]
  folds.rand = fold.ids[sample.int(n)]
  return(folds.rand)
}

getMSPE = function(y, y.hat) {
  resid = y - y.hat
  resid.sq = resid^2
  SSPE = sum(resid.sq)
  MSPE = SSPE / length(y)
  return(MSPE)
}

rescale = function(x1, x2) {
  for(col in 1:ncol(x1)) {
    a = min(x2[,col])
    b = max(x2[,col])
    x1[,col] = (x1[,col] - a) / (b - a)
  }
  x1
}

data = read.csv("Data2020.csv")
pairs(data)

#Set up training and test set
n = nrow(data)
group1 = rep(1, times = round(n * 0.75))
group2 = rep(2, times = n - round(n * 0.75))
group.raw = c(group1, group2)
group = group.raw[sample.int(n)]

data.train = data[group == 1,]
data.valid = data[group == 2,]
y.valid = data.valid$Y

#Testing models

#Linear Regression
fit.lm = lm(Y ~ ., data = data.train)
pred.lm = predict(fit.lm, data.valid)
MSPE.lm = getMSPE(pred.lm, y.valid)
MSPE.lm

fit.lm = lm(Y ~ X12 + X4 + X2, data = data.train)
pred.lm = predict(fit.lm, data.valid)
MSPE.lm = getMSPE(pred.lm, y.valid)
MSPE.lm

fit.lm2 = lm(Y ~ .^2, data = data.train)
pred.lm2 = predict(fit.lm2, data.valid)
MSPE.lm2 = getMSPE(pred.lm2, y.valid)
MSPE.lm2

fit.start = lm(Y ~ 1, data = data)
fit.end = lm(Y ~ ., data = data)

#Stepwise Regression
step.BIC = step(fit.start, list(upper = fit.end), k = log(nrow(data.train)))
pred.BIC = predict(step.BIC, data.valid)
MSPE.BIC = getMSPE(y.valid, pred.BIC)

#GAM
library(mgcv)
fit.gam = gam(Y ~ s(X1) + s(X2) + s(X3) + X4 + s(X5) + s(X6) + s(X7) + s(X8) + s(X9) + X10
              + s(X11) + X12 + s(X13) + s(X14) + s(X15), data = data)
summary(fit.gam)

#all sub regression
library(leaps)
matrix = model.matrix(Y ~ ., data = data)
y = data$Y

all.subsets = regsubsets(x = matrix, y = y, nvmax = 20, intercept = FALSE)
info.subsets = summary(all.subsets)$which
n.models = nrow(info.subsets)
all.BIC = rep(0, times = n.models)

for(i in 1:n.models) {
  this.data.matrix = matrix[,info.subsets[i,]]
  fit = lm(y ~ this.data.matrix - 1)
  
  this.BIC = extractAIC(fit, k = log(nrow(data)))[2]
  all.BIC[i] = this.BIC
}

bestBIC = info.subsets[which.min(all.BIC),]


#Models with tuning

#Neural Nets
library(nnet)
nnetRep = 10
all.n.hidden = c(1, 3, 5, 7)
all.shrink = c(0.1, 0.5, 1, 2)
all.pars = expand.grid(n.hidden = all.n.hidden, shrink = all.shrink)
n.pars = nrow(all.pars)

K = 10
folds = get.folds(nrow(data), K)

CV.MSPEs = array(0, dim = c(K, n.pars))
for(i in 1:K) {
  print(paste0(i, " of ", K))
  
  data.train = data[folds != i,]
  x.train.raw = data.train[, -1]
  x.train = rescale(x.train.raw, x.train.raw)
  y.train = data.train[, 1]
  
  data.valid = data[folds == i,]
  x.valid.raw = data.valid[, -1]
  x.valid = rescale(x.valid.raw, x.train.raw)
  y.valid = data.valid[, 1]
  
  for(j in 1:n.pars) {
    this.n.hidden = all.pars[j,1]
    this.shrink = all.pars[i,2]
    
    all.nnets = list(1:nnetRep)
    all.SSEs = rep(0, times = nnetRep)
    for(l in 1:nnetRep) {
      fit.nnet = nnet(x.train, y.train, linout = TRUE, size = this.n.hidden, decay = this.shrink, maxit = 500, trace = FALSE)
      SSE.nnet = fit.nnet$value
      
      all.nnets[[l]] = fit.nnet
      all.SSEs[l] = SSE.nnet
    }
    
    ind.best = which.min(all.SSEs)
    fit.nnet.best = all.nnets[[ind.best]]
    
    pred.nnet = predict(fit.nnet.best, x.valid)
    MSPE.nnet = getMSPE(y.valid, pred.nnet)
    
    CV.MSPEs[i, j] = MSPE.nnet
  }
}

#Random forest
library(randomForest)

fit.rf = randomForest(Y ~ ., data = data.train, importance = T)

importance(fit.rf)
varImpPlot(fit.rf)

oob.pred = predict(fit.rf)
oob.MSPE = getMSPE(data$Y, oob.pred)
sample.pred = predict(fit.rf, data.valid)
sample.MSPE = getMSPE(y.valid, sample.pred)


all.mtry = 3:9
all.nodesize = c(2, 3, 5)
all.pars = expand.grid(mtry = all.mtry, nodesize = all.nodesize)
n.pars = nrow(all.pars)

M = 5

OOB.MSPEs = array(0, dim = c(M, n.pars))

for(i in 1:n.pars) {
  print(paste0(i, " of ", n.pars))
  
  this.mtry = all.pars[i, "mtry"]
  this.nodesize = all.pars[i, "nodesize"]
  
  for(j in 1:M) {
    fit.rf = randomForest(Y ~ ., data = data, importance = FALSE, mtry = this.mtry, nodesize = this.nodesize)
    
    OOB.pred = predict(fit.rf)
    OOB.MSPE = getMSPE(data$Y, OOB.pred)
    OOB.MSPEs[j, i] = OOB.MSPE
  }
}

names.pars = paste0(all.pars$mtry, "-", all.pars$nodesize)
colnames(OOB.MSPEs) = names.pars
boxplot(OOB.MSPEs, las = 2)

OOB.RMSPEs = apply(OOB.MSPEs, 1, function(w) w/min(w))
OOB.RMSPEs = t(OOB.RMSPEs)
boxplot(OOB.RMSPEs, las = 2)

fit.rf.2 = randomForest(Y ~ ., data = data.train, importance = TRUE, mtry = 3, nodesize = 2)
plot(fit.rf.2)
varImpPlot(fit.rf.2)
sample.pred.2 = predict(fit.rf.2, data.valid)
sample.MSPE.2 = getMSPE(y.valid, sample.pred.2)

#CV Comparison
library(mgcv)
library(randomForest)
library(nnet)
data = read.csv("Data2020.csv")

set.seed(6232493)

n = nrow(data)
k = 20
folds = get.folds(n, k)

all.models = c("LS", "LSpart", "Step", "GAM", "RF", "NNET")
all.MSPEs = array(0, dim = c(k, length(all.models)))
colnames(all.MSPEs) = all.models

max.terms = 15
for(i in 1:k) {
  print(paste0(i, " of ", k))
  
  data.train = data[folds != i,]
  x.train.raw = data.train[, -1]
  x.train = rescale(x.train.raw, x.train.raw)
  y.train = data.train[, 1]
  
  data.valid = data[folds == i,]
  x.valid.raw = data.valid[, -1]
  x.valid = rescale(x.valid.raw, x.train.raw)
  
  n.train = nrow(data.train)
  
  y.train = data.train$Y
  y.valid = data.valid$Y
  
  fit.ls = lm(Y ~ ., data = data.train)
  pred.ls = predict(fit.ls, newdata = data.valid)
  MSPE.ls = getMSPE(y.valid, pred.ls)
  all.MSPEs[i, "LS"] = MSPE.ls
  
  fit.ls.part = lm(Y ~ X12 + X4 + X2, data = data.train)
  pred.ls.part = predict(fit.ls.part, newdata = data.valid)
  MSPE.ls.part = getMSPE(y.valid, pred.ls.part)
  all.MSPEs[i, "LSpart"] = MSPE.ls.part
  
  fit.gam = gam(Y ~ s(X1) + s(X2) + s(X3) + X4 + s(X5) + s(X6) + s(X7) + s(X8) + s(X9) + X10
                + s(X11) + X12 + s(X13) + s(X14) + s(X15), data = data.train)
  pred.gam = predict(fit.gam, data.valid)
  MSPE.gam = getMSPE(y.valid, pred.gam)
  all.MSPEs[i, "GAM"] = MSPE.gam
  
  fit.start = lm(Y ~ 1, data = data)
  fit.end = lm(Y ~ ., data = data)
  
  step.BIC = step(fit.start, list(upper = fit.end), k = log(nrow(data.train)), trace = FALSE)
  pred.BIC = predict(step.BIC, data.valid)
  MSPE.BIC = getMSPE(y.valid, pred.BIC)
  all.MSPEs[i, "Step"] = MSPE.BIC

  fit.rf.7.3 = randomForest(Y ~ ., data = data.train, importance = TRUE, mtry = 7, nodesize = 3)
  sample.pred.7.3 = predict(fit.rf.7.3, data.valid)
  sample.MSPE.7.3 = getMSPE(y.valid, sample.pred.7.3)
  all.MSPEs[i, "RF"] = sample.MSPE.7.3

  all.nnets = list(1:nnetRep)
  all.SSEs = rep(0, times = nnetRep)
  for(l in 1:nnetRep) {
    fit.nnet = nnet(x.train, y.train, linout = TRUE, size = 1, decay = 0.1, maxit = 500, trace = FALSE)
    SSE.nnet = fit.nnet$value

    all.nnets[[l]] = fit.nnet
    all.SSEs[l] = SSE.nnet
  }

  ind.best = which.min(all.SSEs)
  fit.nnet.best = all.nnets[[ind.best]]

  pred.nnet = predict(fit.nnet.best, x.valid)
  MSPE.nnet = getMSPE(y.valid, pred.nnet)

  all.MSPEs[i, "NNET"] = MSPE.nnet
  
}

boxplot(all.MSPEs)

all.RMSPE = apply(all.MSPEs, 1, function(w) {
  best = min(w)
  return(w / best)
})
all.RMSPE = t(all.RMSPE)

boxplot(all.RMSPE)


#Prediction
library(mgcv)
testData2020 = read.csv("Data2020testX.csv")
set.seed(4828347)
fit.gam = gam(Y ~ s(X1) + s(X2) + s(X3) + X4 + s(X5) + s(X6) + s(X7) + s(X8) + s(X9) + X10
              + s(X11) + X12 + s(X13) + s(X14) + s(X15), data = data)
pred.gam = predict(fit.gam, testData2020)
write.table(pred.gam, "Project1Prediction.txt", sep = ",", row.names = F, col.names =
              F)

