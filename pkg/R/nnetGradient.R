# Purpose : calculate prediction interval of the neural network models
# Author: Xichen Ding
# Date: 2015-11-27

# Sigmoid function
sigmoid<-function(x) {
	y = 1/(1+exp(-x))
	return (y)
}
# y = sigmoid(2)

# Sigmoid Differentiation function
sigmoidDeri<-function(x){
	y = sigmoid(x)*(1-sigmoid(x))
	return (y)
}

tanhFunc<-function(x) {
	y = tanh(x)
	return (y)
}

tanhDeri<-function(x) {
	y = 1 - (tanh(x))^2
	return (y)
}

activate<-function(x,funName) {
	if (funName =='sigmoid') {
		res = sigmoid(x)
	} else if (funName == 'tanh') {
		res = tanhFunc(x)
	} else {
		res = "function name not found"
	}
	return (res)
}

activateDeri<-function(x,funName) {
	if (funName =='sigmoid') {
		res = sigmoidDeri(x)
	} else if (funName == 'tanh') {
		res = tanhDeri(x)
	} else {
		res = "function name not found"
	}
	return (res)
}

getOutVectList<-function(xVect,W,B,m,funName) {
	outVecList = list()
	aVectList = list()
	curAVect = xVect
	for (k in 1:m) {
		wk = W[[k]]
		bVect = B[[k]]
		curOutVect = wk %*% matrix(curAVect) + matrix(bVect)
		curAVect = activate(curOutVect,funName)
		outVecList = c(outVecList,list(curOutVect))
		aVectList = c(aVectList,list(curAVect))
	}
	res = list(out = outVecList, aVect = aVectList)
	return (res)
}
# outVect = getNetOutVectList(xVect,W,B,m)

getPredVal<-function(xVect,W,B,m,funName) {
	yPred = 0
	curAVect = xVect
	for (k in 1:m) {
		wk = W[[k]]
		bVect = B[[k]]
		curOutVect = wk %*% matrix(curAVect) + matrix(bVect)
		curAVect = activate(curOutVect,funName)
	}
	yPred = c(curAVect)
	return (yPred)
}

transWeightPara<-function(Wts,m,nodeNum) {
	# nnet package output
	idxPara = 0
	W = list()
	B = list()
	for (k in 1:m) {
		Sk= nodeNum[k+1] # kth layer
		Sk_1= nodeNum[k]   # (k-1)th layer
		curWts = Wts[(idxPara + 1) : (idxPara + Sk * (Sk_1 + 1))]
		curWtsMat = t(matrix(curWts,nrow=(Sk_1 + 1)))
		W = c(W, list(matrix(curWtsMat[,2:(Sk_1 + 1)] , nrow = Sk)))
		B = c(B, list(matrix(curWtsMat[,1:1], nrow = Sk)))
		idxPara = idxPara + Sk * (Sk_1 + 1)
	}
	res = list(W = W, B = B)
	return (res)
}

transWeightPara2<-function(wtsList,m,nodeNum) {
	# neuralnet package output
	W = list()
	B = list()
	for (k in 1:m) {
		curWtsMat = wtsList[[k]]
		B = c(B, list(matrix(curWtsMat[1:1,], ncol = 1)))
		W = c(W, list(t(curWtsMat[-1,])))
	}
	res = list(W = W, B = B)
	return (res)
}

transModelWts<-function(model, m, nodeNum) {
	if (class(model) == 'nnet') {
		res = transWeightPara(model$wts,m,nodeNum)
	} else if (class(model) == 'nn') {
		res = transWeightPara2(model$weights[[1]],m,nodeNum)
	}
	return (res)
}

transWeightListToVect<-function(wtsList,m = 2){
	wtsVect = c()
	for (k in 1:m) {
		curWtsMat = wtsList[[k]]
		curWtsVect = c(curWtsMat)
		wtsVect = c(wtsVect,curWtsVect)
	}
	return (wtsVect)
}
# wtsVect = transWeightListToVect(wtsList,m)

calcSigmaEst<-function(yTarVal,yPredVal,nObs,nPara){
	nDegreeFree = nObs - nPara
	varEst = sum((yTarVal - yPredVal)^2)/nDegreeFree
	sigma = sqrt(varEst)
	return (sigma)
}

# Wijk , k: layer, m Total Layer, idx between (k,m), outMat is a list outVect[[i]] = W[[i]] * A + B, A = active(outVect[[i]])
# Parallel
parOutVectGradMat<-function (idx, k, m, outVect, W, funName) {
	if (idx == k) {
		nDim = dim(outVect[[idx]])[1]
		diagMat = matrix(rep(0,nDim*nDim),nrow = nDim)
		diag(diagMat) = c(activateDeri(outVect[[idx]],funName))
		res = diagMat
	} else if (idx > k) {
		nDim = dim(outVect[[idx]])[1]
		diagMat = matrix(rep(0,nDim*nDim),nrow = nDim)
		diag(diagMat) = c(activateDeri(outVect[[idx]], funName))
		res = diagMat %*% W[[idx]] %*% parOutVectGradMat(idx -1, k, m, outVect, W, funName)
	}
	return (res)
}

# R 1*p, nodeNum is the vector of node number with length (1+m), c(s0,s1,...,sm)
# outVect ~ xVect
# nPara number of parameter

getParaGrad<-function (xVect,W,B,m,nPara,funName) {
	outVect = getOutVectList(xVect,W,B,m,funName)$out
	aVect = getOutVectList(xVect,W,B,m,funName)$aVect
	gradParaVect = c()
	for (k in 1:m) {
		wk = W[[k]]
		wkDeriVect = c()
		
		# Wijk and bik
		curGradActiMat = parOutVectGradMat(idx = m, k, m, outVect, W, funName)
		if (k==1) {
			multiplier = matrix(xVect)
		} else {
			multiplier = matrix(activateDeri(outVect[[k-1]],funName))
		}
		numSk = dim(wk)[1]
		numSk_1 = dim(wk)[2]
		
		for (i in 1:numSk) {
			wkiVect = rep(0,(numSk_1 + 1))
			# bik
			bVectDeri = 0 * B[[k]]
			bVectDeri[i] = 1
			bikDeriVal = curGradActiMat %*% matrix(bVectDeri)
			wkiVect[1] = bikDeriVal
			for (j in 1:numSk_1) {
				# Wijk
				wkDeri = 0 * wk
				wkDeri[i,j] = 1 # derivative of Wijk at position (i,j)
				wijkDeriVal = curGradActiMat %*% wkDeri %*% multiplier
				wkiVect[j+1] = wijkDeriVal
			}
			wkDeriVect=c(wkDeriVect,wkiVect)
		}
		gradParaVect = c(gradParaVect,wkDeriVect)
	}
	return (gradParaVect)
}

parGetParaGrad<-function(idx,xMat,W,B,m,nPara,funName){
	curWtsDeri = getParaGrad(matrix(as.numeric(xMat[idx,])),W,B,m,nPara,funName)
	return (curWtsDeri)
}

getJacobianMatrix<-function(xTrain,W,B,m,nPara,funName) {
	# xTrain is the input training matrix:  R nObs * s0
	nObs = dim(xTrain)[1]
	# nPara is number of parameters
	jacobianMat = matrix(rep(0.0,nObs * nPara),nrow = nObs)
	
	idx = c(1:nObs)
	res = sapply(idx,FUN = parGetParaGrad,xMat = xTrain,W = W,B = B,m = m,nPara = nPara,funName = funName)
	jacobianMat = t(res)
	return (jacobianMat)
}

nnetPredInt<-function(xTrain, yTrain, yFit, node, wts, newData, alpha = 0.05 ,lambda = 0.5, funName = 'sigmoid') {
	nObs = dim(xTrain)[1]
	nPara = length(wts)
	nPred = dim(newData)[1]
	m = length(node) - 1 # Node Number vector S0-Sm
	transWts = transWeightPara(wts, m, node)
	W = transWts$W 
	B = transWts$B 
	predIntMat = matrix(rep(0, 2 * nPred), ncol = 2)
	yPredVect = c()
	
	# sigma estimation
	sigmaGaussion = calcSigmaEst(yTrain, yFit, nObs,nPara)
	tQuant = qt(alpha/2, df = (nObs - nPara))
	
	# Calc Jacobian Matrix: default decay parameter for singular matrix lambda = 0.5
	jacobianMat = getJacobianMatrix(xTrain,W,B,m,nPara,funName)
	detValue = det(t(jacobianMat) %*% jacobianMat)
	
	if (abs(detValue) < 1e-4) { # matrix t(J) %*% J is singular, Need lambda decay parameters
		jacobianInvError = solve(t(jacobianMat) %*% jacobianMat + lambda * diag(nPara)) %*% t(jacobianMat) %*% jacobianMat %*% solve(t(jacobianMat) %*% jacobianMat + lambda * diag(nPara))
	} else if (abs(detValue) >= 1e-4) {
		jacobianInvError = solve(t(jacobianMat) %*% jacobianMat)
	}
	
	for (i in 1:nPred) {
		xVect = matrix(as.numeric(newData[i:i,]))    # Input Matrix
		yPred = getPredVal(xVect,W,B,m,funName)
		wtsGradVect = getParaGrad(xVect,W,B,m,nPara,funName)
		f = matrix(wtsGradVect)
		sigmaModel = sqrt(1 + t(f) %*% jacobianInvError %*% f)
		confWidth = abs(tQuant * sigmaGaussion * sigmaModel)
		predIntVect = c(yPred - confWidth,yPred + confWidth)
		predIntMat[i:i,] = predIntVect
		yPredVect = c(yPredVect,yPred)
	}
	resDf = data.frame(yPredValue = yPredVect,lowerBound = predIntMat[,1:1] , upperBound = predIntMat[,2:2])
	return (resDf)
}

