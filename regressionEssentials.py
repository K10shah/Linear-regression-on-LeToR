import numpy as np
import pandas as pd
import scipy.stats as stats
import sklearn.cluster as cl
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt

# Cluster the data into K clusters using KMeans clustering.
def calculateM(data, k):
	# Calculates the number of basis funcitons required using k-means clustering.
	k_means = cl.KMeans(init='k-means++', n_clusters=k, n_init=10, n_jobs=-1)
	k_means.fit(data)
	k_means_labels = k_means.labels_
	centroids = k_means.cluster_centers_
	clusteredDataInfo = [None]*2
	clusteredDataInfo[0] = centroids
	clusteredDataInfo[1] = k_means_labels
	return clusteredDataInfo

# Create a datastructure that holds data for same clusters together.
def createDataClusters(inputData, clusteredDataInfo, k):
	#print("In createDataClusters.")
	# Define array that stores cluster data.
	clusteredData = [None]*k
	for i in range(0,k):
		clusteredData[i] = []
	for i in range(0, len(inputData)):
		label = clusteredDataInfo[1][i]
		clusteredData[label].append(inputData[i])
	return clusteredData

# Calculate the covariance matrix for each cluster of the data. 
def covariancePerCluster(clusteredData):
	#print("In covariancePerCluster.")
	# Return covariance matrix per cluster.
	k = len(clusteredData)
	covarianceMatrixClusters = []
	for i in range(0,k):
		clusterDataFrame = pd.DataFrame(clusteredData[i])
		covariance = clusterDataFrame.cov()
		covarianceMatrixClusters.append(covariance)
	return covarianceMatrixClusters

# Calculate the design matrix for the input data using M basis functions where M is the number of clusters the data is partitioned into.
def calculateDesignMatrix(inputData, clusteredDataInfo, covarianceMatrixClusters, k):
	#print("In calculateDesignMatrix.")
	# Calculates design matrix using M Gaussian Model basis functions.
	designMatrix = np.empty((len(inputData),(k+1),))
	designMatrix[:] = np.NAN
	for p in range(0, len(inputData)):
		designMatrix[p][0] = 1
	for i in range(0, len(inputData)):
		for j in range(1, k+1):
			# Calculating all the values Gaussian Basis functions for each input vector.
			inputMatrix = np.matrix(inputData[i])
			meanMatrix = np.matrix(clusteredDataInfo[0][j-1])
			covarianceMatrix = np.asmatrix(np.array(covarianceMatrixClusters[j-1]))
			inverse = np.linalg.pinv(covarianceMatrix)
			subtractionResult = np.subtract(inputMatrix, meanMatrix)
			subtractionTranspose = subtractionResult.T
			basisFunctionValue = np.exp(np.multiply((-0.5), np.matmul(np.matmul(subtractionResult, inverse), subtractionTranspose)))
			designMatrix[i][j] = basisFunctionValue[0][0]
	designMatrix = np.asmatrix(designMatrix)
	return designMatrix

# Function to calculate the root mean square error given the known target and the target predicted by the model we have learned.
def calculateRootMeanSquaredError(predictedTarget, target):
	rms = sqrt(mean_squared_error(predictedTarget, target))
	return rms

# Draw an over lapping line graph of the given output and predicted output.
def plotOverlappingLineGraph(predicted, given, filename):
	plotArray1 = [[], []]
	plotArray2 = [[], []]
	# Convert numpy matrices to python lists to plot for sample count against output value for given and predicted data.
	predicted = np.array(predicted).reshape(-1,).tolist()
	given = np.array(given).reshape(-1,).tolist()
	# Build the plotting datastructure.
	for i in range(0, len(predicted)):
		plotArray2[0].append(i)
		plotArray2[1].append(given[i])
		plotArray1[0].append(i)
		plotArray1[1].append(predicted[i])
	fig = plt.figure(figsize=(20,10))
	ax1 = fig.add_subplot(111)
	ax1.plot(plotArray2[0], plotArray2[1], label="given_output")
	ax1.plot(plotArray1[0], plotArray1[1], label="predicted_output")
	plt.xlabel('Sample Count')
	plt.ylabel('Output Values')
	handles, labels = ax1.get_legend_handles_labels()
	lgd = ax1.legend(handles, labels, loc='upper center', bbox_to_anchor=(1.15,1))
	ax1.grid('on')
	# Save the plot to a png file in the location defined by filename.
	plt.savefig(filename)
	# Create a smaller plot.
	fig = plt.figure(figsize=(20,10))
	ax1 = fig.add_subplot(111)
	predicted = predicted[0:100]
	given = given[0:100]
	smallerPlotArray1 = [[], []]
	smallerPlotArray2 = [[], []]
	for i in range(0, len(predicted)):
		smallerPlotArray2[0].append(i)
		smallerPlotArray2[1].append(given[i])
		smallerPlotArray1[0].append(i)
		smallerPlotArray1[1].append(predicted[i])
	ax1.plot(smallerPlotArray2[0], smallerPlotArray2[1], label="given_output")
	ax1.plot(smallerPlotArray1[0], smallerPlotArray1[1], label="predicted_output")
	plt.xlabel('Sample Count')
	plt.ylabel('Output Values')
	handles, labels = ax1.get_legend_handles_labels()
	lgd = ax1.legend(handles, labels, loc='upper center', bbox_to_anchor=(1.15,1))
	ax1.grid('on')
	# Save the plot to a png file in the location defined by filename.
	plt.savefig("smaller-plot-" + filename)