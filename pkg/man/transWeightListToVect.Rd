\name{transWeightListToVect}
\alias{transWeightListToVect}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
%%  ~~function to do ... ~~
Transform the list of optimal weights to a numeric vector
}
\description{
%%  ~~ A concise (1-5 lines) description of what the function does. ~~
This function transforms the list of weight parameters,typically the output of neuralnet package (nn$weights) to a single numeric vector in specific order, which will be used as the input to the prediciton interval function. The order of weight parameters is the same as the output of nnet package (nnet$wts).
}
\usage{
transWeightListToVect(wtsList,m = 2)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{wtsList}{
%%     ~~Describe \code{x} here~~
the list of weights found by neuralnet package (nn$weights). For nnet weights output(nnet$wts), there is no need for transformation and it can be used directly by the prediction interval method.
}
  \item{m}{
%%     ~~Describe \code{x} here~~
the number of layers of the neural networks, which is the number of hidden layer plus + 1. m is default to 2 for single hidden layer networks.
}
}
\details{
%%  ~~ If necessary, more details than the description above ~~
}
\value{
  \item{wtsVect}{
%%     ~~Describe \code{x} here~~
the numeric vector of optimal weights found by neural networks.
}
}
\references{
%% ~put references to the literature/web site here ~
{neuralnet} package by Stefan Fritsch, Frauke Guenther
}
\author{
Xichen Ding <rockingdingo@gmail.com>
}
\note{
%%  ~~further notes~~
}

%% ~Make other sections like Warning with \section{Warning }{....} ~

\seealso{
%% ~~objects to See Also as \code{\link{help}}, ~~~
}
\examples{


% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
}
