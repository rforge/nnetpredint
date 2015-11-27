# Test Case for {neuralnet} package
set.seed(500)
library(MASS)
data <- Boston

# normalization
maxs <- apply(data, 2, max)
mins <- apply(data, 2, min)
scaled <- as.data.frame(scale(data, center = mins, scale = maxs - mins))
index <- sample(1:nrow(data),round(0.75*nrow(data)))
train_ <- scaled[index,]
test_ <- scaled[-index,]

# Training
library(neuralnet)
n <- names(train_)
f <- as.formula(paste("medv ~", paste(n[!n %in% "medv"], collapse = " + ")))
nn <- neuralnet(f,data=train_,hidden=c(5,3),linear.output=FALSE)
plot(nn)

# Getting Confidence Interval
library(nnetpredint)
x = train_[,-14]
y = train_[,14]
yFit = c(nn$net.result[[1]])
nodeNum = c(13,5,3,1)
m = 3
wtsList = nn$weights[[1]]
Wts = transWeightListToVect(wtsList,m)
newData = test_[,-14]
yPredInt = nnetPredInt(x, y, yFit, nodeNum, Wts, newData)
print(yPredInt[1:20,])

# Compare to the predict values from the neuralnet Compute method
predValue = compute(nn,newData)
print(matrix(predValue$net.result[1:20]))