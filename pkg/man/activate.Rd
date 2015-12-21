\name{activate}
\alias{activate}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
%%  ~~function to do ... ~~
Neuron Activation Function
}
\description{
%%  ~~ A concise (1-5 lines) description of what the function does. ~~
activation function for each neuron node, we provide the popular activation functions including 'sigmoid', 'tanh', etc.
}
\usage{
activate(x, funName)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{x}{
%%     ~~Describe \code{x} here~~
input value to the activation function
}
  \item{funName}{
%%     ~~Describe \code{funName} here~~
This package provides the most popular activation functions which can be used by setting the funName parameter to the following strings: 'sigmoid', 'tanh'.
}
  \item{}{
%%     ~~Describe \code{funName} here~~
'sigmoid' sigmoid function 1/(1 + exp(-x)).
}
  \item{}{
%%     ~~Describe \code{funName} here~~
'tanh' tanh is the hyperbolic tangent function equal to (exp(x) - exp(-x)) / (exp(x) + exp(-x)).
}
}
\examples{
x <- c(-5:5)
y1 <- activate(x, funName = 'sigmoid')
y2 <- activate(x, funName = 'tanh')
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
