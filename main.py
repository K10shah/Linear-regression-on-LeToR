import stochasticGradientDescent as sgd
import closedFormSolution as cfs
import regressionEssentials as re
import numpy as np

# Load all the data into the globally declared variables.
# 80 percent data is used as training data, 10 percent is used as validation data and the remaining 10 percent is used as test data.

# Load synthetic input dataset.
syn_input_data = np.genfromtxt('./synthetic_data_input.csv', delimiter=',').reshape([-1,1])
# Synthetic input training dataset.
syn_input_train_data = syn_input_data[0:16000]
# Synthetic input validation dataset.
syn_input_validation_data = syn_input_data[16000:18000]
# Synthetic input test dataset.
syn_input_test_data = syn_input_data[18000:20000]

# Load synthetic output dataset.
syn_output_data = np.genfromtxt('./synthetic_data_output.csv', delimiter=',').reshape([-1,1])
# Synthetic output train dataset.
syn_output_train_data = syn_output_data[0:16000]
# Synthetic output validation dataset.
syn_output_validation_data = syn_output_data[16000:18000]
# Synthetic output test datasetset.
syn_output_test_data = syn_output_data[18000:20000]

# Load letor input data.
letor_input_data = np.genfromtxt('./letor_data_input.csv', delimiter=',')
# Letor input training dataset.
letor_input_train_data = letor_input_data[0:55698]
# Letor input validation dataset.
letor_input_validation_data = letor_input_data[55698:62660]
# Letor input test dataset.
letor_input_test_data = letor_input_data[62660:69623]

# Load letor output data.
letor_output_data = np.genfromtxt('./letor_data_output.csv', delimiter=',')
# Letor output training data.
letor_output_train_data = letor_output_data[0:55698]
# Letor output validation dataset.
letor_output_validation_data = letor_output_data[55698:62660]
# Letor output test data.
letor_output_test_data = letor_output_data[62660:69623]

# Fix the lambda used in regularization.
lambda_Regularization = 0.1
# Fix the number of clusters to 15.
number_Of_Clusters = 15
# Fix the batch size for the batch stochastic gradient descent.
batchSize = 100

print("--------------------------------------")
# Letor closed form solution.
print("Training the model for the letor input data set with closed form solution...")
Wml = cfs.learnParametersClosdFormSolution(letor_input_train_data, letor_output_train_data, number_Of_Clusters, lambda_Regularization)
print("The parameters are -")
print(Wml)
predicted_Target_For_Validation = cfs.computeTargetValuesClosedFormSolution(letor_input_validation_data, number_Of_Clusters, Wml)
rms_Error_Validation = re.calculateRootMeanSquaredError(letor_output_validation_data, predicted_Target_For_Validation)
print("RMSE for letor validation dataset is - " + str(rms_Error_Validation))
predicted_Target_For_Test = cfs.computeTargetValuesClosedFormSolution(letor_input_test_data, number_Of_Clusters, Wml)
rms_Error_Test = re.calculateRootMeanSquaredError(letor_output_test_data, predicted_Target_For_Test)
print("RMSE for letor test dataset is - " + str(rms_Error_Test))
re.plotOverlappingLineGraph(predicted_Target_For_Test, letor_output_test_data, "letor-cfs-test-output-plot")
print("Graphical representation of the output can be found in the file titled letor-cfs-test-output-plot.png!")
print("--------------------------------------")

# Fix the number of clusters to 35.
number_Of_Clusters = 35
# synthetic closed form solution.
print("Training the model for the synthetic input data set with closed form solution...")
Wml = cfs.learnParametersClosdFormSolution(syn_input_train_data, syn_output_train_data, number_Of_Clusters, lambda_Regularization)
print("The parameters are -")
print(Wml)
predicted_Target_For_Validation = cfs.computeTargetValuesClosedFormSolution(syn_input_validation_data, number_Of_Clusters, Wml)
rms_Error_Validation = re.calculateRootMeanSquaredError(syn_output_validation_data, predicted_Target_For_Validation)
print("RMSE for synthetic validation dataset is - " + str(rms_Error_Validation))
predicted_Target_For_Test = cfs.computeTargetValuesClosedFormSolution(syn_input_test_data, number_Of_Clusters, Wml)
rms_Error_Test = re.calculateRootMeanSquaredError(syn_output_test_data, predicted_Target_For_Test)
print("RMSE for synthetic test dataset is - " + str(rms_Error_Test))
re.plotOverlappingLineGraph(predicted_Target_For_Test, syn_output_test_data, "synthetic-cfs-test-output-plot")
print("Graphical representation of the output can be found in the file titled synthetic-cfs-test-output-plot.png!")
print("--------------------------------------")

# Fix the number of clusters to 35.
number_Of_Clusters = 35
# Letor sgd solution
print("Training the model for the letor input data set with stochastic gradient descent...")
W = sgd.learnParameterSGD(letor_input_train_data, letor_input_validation_data, letor_output_train_data, letor_output_validation_data, batchSize, number_Of_Clusters, lambda_Regularization)
print("The parameters are -")
print(Wml)
predicted_Target_For_Validation = sgd.computeTargetSGD(letor_input_validation_data, number_Of_Clusters, W)
rms_Error_Validation = re.calculateRootMeanSquaredError(letor_output_validation_data, predicted_Target_For_Validation)
print("RMSE for letor validation dataset is - " + str(rms_Error_Validation))
predicted_Target_For_Test = sgd.computeTargetSGD(letor_input_test_data, number_Of_Clusters, W)
rms_Error_Test = re.calculateRootMeanSquaredError(letor_output_test_data, predicted_Target_For_Test)
print("RMSE for letor test dataset is - " + str(rms_Error_Test))
re.plotOverlappingLineGraph(predicted_Target_For_Test, letor_output_test_data, "letor-sgd-test-output-plot")
print("Graphical representation of the output can be found in the file titled letor-sgd-test-output-plot.png!")
print("--------------------------------------")

# Fix the number of clusters to 25.
number_Of_Clusters = 25
# Synthetic sgd solution
print("Training the model for the synthetic input data set with stochastic gradient descent...")
W = sgd.learnParameterSGD(syn_input_train_data, syn_input_validation_data, syn_output_train_data, syn_output_validation_data, batchSize, number_Of_Clusters, lambda_Regularization)
print("The parameters are -")
print(Wml)
predicted_Target_For_Validation = sgd.computeTargetSGD(syn_input_validation_data, number_Of_Clusters, W)
rms_Error_Validation = re.calculateRootMeanSquaredError(syn_output_validation_data, predicted_Target_For_Validation)
print("RMSE for synthetic validation dataset is - " + str(rms_Error_Validation))
predicted_Target_For_Test = sgd.computeTargetSGD(syn_input_test_data, number_Of_Clusters, W)
rms_Error_Test = re.calculateRootMeanSquaredError(syn_output_test_data, predicted_Target_For_Test)
print("RMSE for synthetic test dataset is - " + str(rms_Error_Test))
re.plotOverlappingLineGraph(predicted_Target_For_Test, syn_output_test_data, "synthetic-sgd-test-output-plot")
print("Graphical representation of the output can be found in the file titled synthetic-sgd-test-output-plot.png!")
print("--------------------------------------")