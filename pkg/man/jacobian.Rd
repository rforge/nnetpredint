\name{jacobian}
\alias{jacobian}
\alias{jacobian.default}
\alias{jacobian.nnet}
\alias{jacobian.nn}
\alias{jacobian.rsnns}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
%%  ~~function to do ... ~~
Jacobian Matrix of Gradient Function for Training Datasets
}
\description{
%%  ~~ A concise (1-5 lines) description of what the function does. ~~
Calculate the Jacobian matrix of gradient function for the training dataset. It takes input from neural network models and the gradient at each weight parameters. The matrix has dimension of R [nObs * nPara], nObs denotes the number of training observations and nPara denotes the number of weights parameters.
}
\usage{
jacobian(object, ...)

\method{jacobian}{nnet}(object, xTrain, funName = 'sigmoid',\dots)
\method{jacobian}{nn}(object, xTrain, funName = 'sigmoid',\dots)
\method{jacobian}{rsnns}(object, xTrain, funName = 'sigmoid',\dots)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{object}{
  object of class: nnet as returned by 'nnet' package, nn as returned by 'neuralnet' package, rsnns as returned by 'RSNNS' package.
  }
  \item{xTrain}{
  matrix or data frame of input values for the training dataset.
  }
  \item{funName}{
  activation function name of neuron, e.g. 'sigmoid', 'tanh', etc. In default, it is set to 'sigmoid'.
  }
  \item{...}{
  additional arguments passed to the method.
  }
}
\details{
%%  ~~ If necessary, more details than the description above ~~
  Jacobian matrix with gradient function, in which J[ij] element denotes the gradient function at the jth weight parameters for the ith training observation. The dimension is equal to nObs * nPara.
}
\value{
matrix which denotes the Jacobian matrix for training datasets.
}
\author{
%%  ~~who you are~~
Xichen Ding <rockingdingo@gmail.com>
}
\seealso{
%% ~~objects to See Also as \code{\link{help}}, ~~~
\code{\link{nnetPredInt}}
}
\examples{
library(nnet)
xTrain <- rbind(cbind(runif(150,min = 0, max = 0.5),runif(150,min = 0, max = 0.5)) ,
		cbind(runif(150,min = 0.5, max = 1),runif(150,min = 0.5, max = 1))
		)
nObs <- dim(xTrain)[1]
yTrain <- 0.5 + 0.4 * sin(2* pi * xTrain \%*\% c(0.4,0.6)) +rnorm(nObs,mean = 0, sd = 0.05)
# Training nnet models
net <- nnet(yTrain ~ xTrain,size = 3, rang = 0.1,decay = 5e-4, maxit = 500)

# Calculating Jacobian Matrix of the training samples
library(nnetpredint)
jacobMat = jacobian(net,xTrain)
dim(jacobMat)
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
