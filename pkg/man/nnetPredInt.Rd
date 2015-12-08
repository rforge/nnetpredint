\name{nnetPredInt}
\alias{nnetPredInt}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
%%  ~~function to do ... ~~
Prediction intervals of neural networks
}
\description{
%%  ~~ A concise (1-5 lines) description of what the function does. ~~
Get the prediction intervals of new dataset at certain confidence level based on the training datasets and the gradient at weight parameters of the neural network model.
}
\usage{
## S3 method for object of class 'nnet', 'nn', 'rsnns', etc:
nnetPredInt(object, xTrain, yTrain, newData, ...)

## Default S3 method:
nnetPredInt(xTrain, yTrain, yFit, node, wts, newData,
    alpha = 0.05 , lambda = 0.5, funName = 'sigmoid')
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{object}{
  object of class: nnet as returned by nnet package, nn as returned by neuralnet package, rsnns as returned by RSNNS package.
  }
  \item{xTrain}{
  matrix or data frame of input values for the training dataset.
  }
  \item{yTrain}{
  vector of target values for the training dataset.
  }
  \item{newData}{
  matrix or data frame of the prediction dataset.
  }
  \item{yFit}{
  vector of the fitted values, as the output produced by the training model, e.g. nnet$fitted.values (nnet) , nn$net.result[[1]] (neuralnet) and rsnns$fitted.values (RSNNS)
}
  \item{node}{
a vector of integers specifying the number of hidden nodes in each layer. Multi-layer network has the structure (s0, s1, ..., sm), in which s0 denotes the dimension for input layer and sm denotes the dimension of the output layer. sm is usually set as 1.
  }
   \item{wts}{
  a numeric vector of optimal weight parameters as the output of the neural network training model. The order of wts parameter is as follows: For any node i in layer k: c(bias ik, wi1k,wi2k,...wijk).
  }
  \item{}{
  nnet object, returned by nnet package. We can directly set the wts as: wts = nnet$wts
  }
  \item{}{
  nn object, returned by neuralnet package. We need to use \link{transWeightListToVect} function to transform the list of weights to a single vector first: wts = transWeightListToVect(wtsList, m).
  }
  \item{}{
  rsnns object, returned by RSNNS package. We need to transform and combine the weight and bias parameters to a single vector: weightMatrix(object) and extractNetInfo(object)$unitDefinitions$unitBias.
  }
  \item{alpha}{
  confidence level. The confidence level is set to (1-alpha). In default, alpha = 0.05.
  }
  \item{lambda}{
  decay parameter of weights when the Jacobian matrix of training dataset is singular. In default, lamda is set to 0.5 .
  }
  \item{funName}{
  activation function name of neuron, e.g. 'sigmoid', 'tanh', etc. In default, it is set to 'sigmoid'.
  }
}
\details{
%%  ~~ If necessary, more details than the description above ~~
}
\value{
  data frame of the prediction intervals, including prediction value, lower and upper bounds of the interval.
  \item{yPredValue}{
%%     ~~Describe \code{x} here~~
the column of prediction value in the data frame.
  }
  \item{lowerBound}{
%%     ~~Describe \code{x} here~~
the column of prediction lower bounds in the data frame.
  }
  \item{upperBound}{
%%     ~~Describe \code{x} here~~
the column of prediction upper bounds in the data frame.
  }
}
\references{
%% ~put references to the literature/web site here ~
De Veaux R. D., Schumi J., Schweinsberg J., Ungar L. H., 1998, "Prediction intervals for neural networks via nonlinear regression", Technometrics 40(4): 273-282.

Chryssolouris G., Lee M., Ramsey A., "Confidence interval prediction for neural networks models", IEEE Trans. Neural Networks, 7 (1), 1996, pp. 229-232.

neuralnet package by Stefan Fritsch, Frauke Guenther.

nnet package by Brian Ripley, William Venables.

RSNNS package by Christoph Bergmeir, Jose M. Benitez.
}
\author{
%%  ~~who you are~~
Xichen Ding <rockingdingo@gmail.com>
}
\note{
%%  ~~further notes~~
}

%% ~Make other sections like Warning with \section{Warning }{....} ~

\seealso{
%% ~~objects to See Also as \code{\link{help}}, ~~~
\code{\link{transWeightListToVect}}
}
\examples{
# Example 1: Using the nn object trained by neuralnet package
set.seed(500)
library(MASS)
data <- Boston
maxs <- apply(data, 2, max)
mins <- apply(data, 2, min)
scaled <- as.data.frame(scale(data, center = mins, scale = maxs - mins)) # normalization
index <- sample(1:nrow(data),round(0.75*nrow(data)))
train_ <- scaled[index,]
test_ <- scaled[-index,]

library(neuralnet) # Training
n <- names(train_)
f <- as.formula(paste("medv ~", paste(n[!n \%in\% "medv"], collapse = " + ")))
nn <- neuralnet(f,data = train_,hidden = c(5,3),linear.output = FALSE)
plot(nn)

library(nnetpredint) # Getting prediction confidence interval
x <- train_[,-14]
y <- train_[,14]
newData <- test_[,-14]

# S3 generic method: Object of nn
yPredInt <- nnetPredInt(nn, x, y, newData)
print(yPredInt[1:20,])

# S3 default method: xTrain,yTrain,yFit,...
yFit <- c(nn$net.result[[1]])
nodeNum <- c(13,5,3,1)
m <- 3
wtsList <- nn$weights[[1]]
wts <- transWeightListToVect(wtsList,m)
yPredInt2 <- nnetPredInt(x, y, yFit, nodeNum, wts, newData, alpha = 0.05)
print(yPredInt2[1:20,])

# Compare to the predict values from the neuralnet Compute method
predValue <- compute(nn,newData)
print(matrix(predValue$net.result[1:20]))



# Example 2: Using the nnet object trained by nnet package
library(nnet)
xTrain <- rbind(cbind(runif(150,min = 0, max = 0.5),runif(150,min = 0, max = 0.5)) ,
		cbind(runif(150,min = 0.5, max = 1),runif(150,min = 0.5, max = 1))
		)
nObs <- dim(xTrain)[1]
yTrain <- 0.5 + 0.4 * sin(2* pi * xTrain \%*\% c(0.4,0.6)) +rnorm(nObs,mean = 0, sd = 0.05)
plot(xTrain \%*\% c(0.4,0.6),yTrain)

# Training nnet models
net <- nnet(yTrain ~ xTrain,size = 3, rang = 0.1,decay = 5e-4, maxit = 500)
yFit <- c(net$fitted.values)
nodeNum <- c(2,3,1)
wts <- net$wts

# New data for prediction intervals
library(nnetpredint)
newData <- cbind(seq(0,1,0.05),seq(0,1,0.05))
yTest <- 0.5 + 0.4 * sin(2* pi * newData \%*\% c(0.4,0.6))+rnorm(dim(newData)[1],mean = 0, sd = 0.05)

# S3 generic method: Object of nnet
yPredInt <- nnetPredInt(net, xTrain, yTrain, newData)
print(yPredInt[1:20,])

# S3 default method: xTrain,yTrain,yFit,...
yPredInt2 <- nnetPredInt(xTrain, yTrain, yFit, node = nodeNum, wts = wts, newData,
	alpha = 0.05, funName = 'sigmoid')

plot(newData \%*\% c(0.4,0.6),yTest,type = 'b')
lines(newData \%*\% c(0.4,0.6),yPredInt$yPredValue,type = 'b',col='blue')
lines(newData \%*\% c(0.4,0.6),yPredInt$lowerBound,type = 'b',col='red')   # lower bound
lines(newData \%*\% c(0.4,0.6),yPredInt$upperBound,type = 'b',col='red')   # upper bound



# Example 3: Using the rsnns object trained by RSNNS package
library(RSNNS)
data(iris)
#shuffle the vector
iris <- iris[sample(1:nrow(iris),length(1:nrow(iris))),1:ncol(iris)]
irisValues <- iris[,1:4]
irisTargets <- decodeClassLabels(iris[,5])[,'setosa']

iris <- splitForTrainingAndTest(irisValues, irisTargets, ratio=0.15)
iris <- normTrainingAndTestSet(iris)
model <- mlp(iris$inputsTrain, iris$targetsTrain, size=5, learnFuncParams=c(0.1),
	maxit=50, inputsTest=iris$inputsTest, targetsTest=iris$targetsTest)
predictions <- predict(model,iris$inputsTest)

# Generating prediction intervals
library(nnetpredint)
# S3 Method for rsnns class prediction intervals
xTrain <- iris$inputsTrain
yTrain <- iris$targetsTrain
newData <- iris$inputsTest
yPredInt <- nnetPredInt(model, xTrain, yTrain, newData)
print(yPredInt[1:20,])
}

% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
