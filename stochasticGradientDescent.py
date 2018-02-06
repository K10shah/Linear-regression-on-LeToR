import numpy as np
import pandas as pd
import scipy.stats as stats
import sklearn.cluster as cl
import regressionEssentials as re
from sklearn.metrics import mean_squared_error
from math import sqrt

# Perform the stochastic update batch wise whilst keeping the track of minimum validation set error.
def stochasticUpdate(inputData, inputValidationData, outputData, outputValidationData, designMatrix, currentW, learningRate, batchSize, numberOfClusters, lambdaRegularization):
	idealW = currentW 
	# Initial minimum error set to a large value.
	rmsErrorMin = 9999
	# Calculate design matrix for the validation input set.
	clusteredDataInfo = re.calculateM(inputValidationData, numberOfClusters)
	clusteredData = re.createDataClusters(inputValidationData, clusteredDataInfo, numberOfClusters)
	covarianceMatrixClusters = re.covariancePerCluster(clusteredData)
	designMatrixValidation = re.calculateDesignMatrix(inputValidationData, clusteredDataInfo, covarianceMatrixClusters, numberOfClusters)
	# Run the update step once for each batch.
	index = len(inputData)//batchSize
	for i in range(0, index):
		# Calculate the batch change in E for the batch.
		deltaE = calculateBatchDeltaE(designMatrix, outputData, i*batchSize, (i+1)*batchSize, currentW, lambdaRegularization)
		# Update the parameters usind the batch gradient and the learning rate.
		currentW = np.subtract(currentW, learningRate*deltaE)
		# Calculate the new target values with the updated parameters.
		predictedTarget = computeTargetValuesSGD(inputValidationData, designMatrixValidation, numberOfClusters, currentW)
		# Calculate the rms error for the updated parameters.
		rmsError = re.calculateRootMeanSquaredError(outputValidationData, predictedTarget)
		if (rmsError < rmsErrorMin):
			# Got a new minimum rms error.
			# Save the parameters.
			idealW = currentW
			# Save the minimum rms error till now.
			rmsErrorMin = rmsError
	return idealW

# Learn the paramters given the input training and validation sets using stochastic gradient descent.
def learnParameterSGD (inputData, inputValidationData, outputData, outputValidationData, batchSize, numberOfClusters, lambdaRegularization):
	# Learning rate fixed.
	learningRate = 1
	# Cluster the input data to decide the number of basis functions.
	clusteredDataInfo = re.calculateM(inputData, numberOfClusters)	
	clusteredData = re.createDataClusters(inputData, clusteredDataInfo, numberOfClusters)
	covarianceMatrixClusters = re.covariancePerCluster(clusteredData)
	designMatrix = re.calculateDesignMatrix(inputData, clusteredDataInfo, covarianceMatrixClusters, numberOfClusters)
	# Choose a random w.
	randomW = np.random.rand(1, numberOfClusters+1)
	# Run the update on parameters using the stochastic gradient update.
	W = stochasticUpdate(inputData, inputValidationData, outputData, outputValidationData, designMatrix, randomW, learningRate, batchSize, numberOfClusters, lambdaRegularization)
	return W

# Compute the target values using the learned parameters, input data and the desing matrix for the input data.
def computeTargetValuesSGD(inputData, designMatrix, numberOfClusters, W):
	# Target matrix to be returned.
	target = np.empty((len(inputData),1,))
	target[:] = np.NAN
	target = np.asmatrix(target)
	for i in range(0, len(designMatrix)):
		target[i][0] = np.matmul(W, designMatrix[i].T)
	return target

# Compute the target values using the input data and the learned parameters.
def computeTargetSGD(inputData, numberOfClusters, W):
	# Compute the design matrix for the input.
	clusteredDataInfo = re.calculateM(inputData, numberOfClusters)	
	clusteredData = re.createDataClusters(inputData, clusteredDataInfo, numberOfClusters)
	covarianceMatrixClusters = re.covariancePerCluster(clusteredData)
	designMatrix = re.calculateDesignMatrix(inputData, clusteredDataInfo, covarianceMatrixClusters, numberOfClusters)	
	# Target matrix to be returned.
	target = np.empty((len(inputData),1,))
	target[:] = np.NAN
	target = np.asmatrix(target)
	if(W.shape[1] == 1):
		# Ensure that the dot product is performed without any errors.
		W = W.T
	for i in range(0, len(designMatrix)):
		target[i][0] = np.matmul(W, designMatrix[i].T)
	return target
	
# Calculate the gradient for a batch of the input row vectors.
def calculateBatchDeltaE(designMatrix, outputData, start, end, currentW, lambdaRegularization):
	# Slice of the design matrix.
	phi = designMatrix[start:end,:]
	# Slice of the output matrix.
	t = outputData[start:end]
	t = np.asmatrix(t)
	if(t.shape[0] == 1):
		t = t.T
	E_D = np.matmul((np.matmul(phi, currentW.T) - t).T, phi)
	# Gradient accumulated for the batch.
	E = (E_D + lambdaRegularization * currentW)/(end-start)
	return E