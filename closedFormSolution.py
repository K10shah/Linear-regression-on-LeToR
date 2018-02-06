import numpy as np
import pandas as pd
import scipy.stats as stats
import sklearn.cluster as cl
import regressionEssentials as re
from sklearn.metrics import mean_squared_error
from math import sqrt

# Function to calculate the parameters given the design matrix, target and the regularization hyperparameter.
def computeClosedFormSolution(lambdaRegularization, designMatrix, outputData):
	regularizationMatrix = lambdaRegularization * np.identity(designMatrix.shape[1])
	Wml = np.matmul(np.matmul((np.add(regularizationMatrix, np.matmul(designMatrix.T, designMatrix))).I, designMatrix.T), outputData)
	return Wml

# Compute the output vector using the input data and the learned parameters.
def computeTargetValuesClosedFormSolution(inputData, numberOfClusters, Wml):
	# Calculate the design matrix.
	clusteredDataInfo = re.calculateM(inputData, numberOfClusters)
	clusteredData = re.createDataClusters(inputData, clusteredDataInfo, numberOfClusters)
	covarianceMatrixClusters = re.covariancePerCluster(clusteredData)
	designMatrix = re.calculateDesignMatrix(inputData, clusteredDataInfo, covarianceMatrixClusters, numberOfClusters)
	# Initialise the output matrix for the number of row vectors in the input.
	target = np.empty((len(inputData),1,))
	target[:] = np.NAN
	target = np.asmatrix(target)
	if (Wml.shape[1] == 1):
		Wml = Wml.T
	# Calculate the output matrix using the learned parameters.
	for i in range(0, len(designMatrix)):
		target[i][0] = np.matmul(Wml, designMatrix[i].T)
	# Return the output matrix.
	return target

# Learn the parameters using the training input data and training output data.
def learnParametersClosdFormSolution(inputData, outputData, numberOfClusters, lambdaRegularization):
	# Calculate the design matrix.
	clusteredDataInfo = re.calculateM(inputData, numberOfClusters)	
	clusteredData = re.createDataClusters(inputData, clusteredDataInfo, numberOfClusters)
	covarianceMatrixClusters = re.covariancePerCluster(clusteredData)
	designMatrix = re.calculateDesignMatrix(inputData, clusteredDataInfo, covarianceMatrixClusters, numberOfClusters)
	Wml = computeClosedFormSolution(lambdaRegularization, designMatrix, outputData)
	# Return the parameters.
	return Wml
