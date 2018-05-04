"""
Here we will check our Neural Network model performance on iris dataset of scikit learn

"""
import time
import numpy as np
from Neural_Network import NeuralNetwork as NN
from sklearn import preprocessing

# load iris dataset from sklearn
from sklearn.datasets import load_iris

iris = load_iris()

# Store input and output in separate array
X = iris.data
Y = iris.target
#normalise the data
X = preprocessing.scale(X)

# extract size of X 
N, m=X.shape # 'N' is the number of inputs and 'm' is number of features

#Randomly shuffle the data and separate out training and testing set
index_all = np.arange(0, N)
index_shuffle = np.random.permutation(N)#contains the randomly shuffled indices
index_train = []
index_train.append(index_shuffle[0:int(0.75*N)])
index_test = np.delete(index_all,index_train)
X_train, Y_train = X[index_train], Y[index_train]
X_test, Y_test = X[index_test], Y[index_test]

# determine number of classes
n_class = len(np.unique(Y))

#define neural network model-1 parameters
hidden_layers = [5,5] # hidden layer size is passed as a list ith position value determines number of neuron in ith layer
l_rate = 0.6  # learning rate
epochs = 1000 # number of training epochs

#check the timing of model-1
t0=time.time()#start time

#define model-1
model1 = NN(input_layer_size=m, output_layer_size=n_class, hidden_layer_size=hidden_layers)
model1.train(X_train, Y_train, l_rate=l_rate, n_epochs=epochs)
#end time
t1=time.time()
total_time = t1 - t0

#prediction of our model on test and training data
Y_train_predict = model1.predict(X_train)
Y_test_predict = model1.predict(X_test)
accuracy_train = 100*np.sum(Y_train==Y_train_predict)/len(Y_train)
accuracy_test = 100*np.sum(Y_test==Y_test_predict)/len(Y_test)
print("total time for training model-1:",total_time)
print("accuracy of model-1 on training data", accuracy_train)
print("accuracy of model-1 on test data", accuracy_test)

#define neural network model-2 parameters
hidden_layers = [100] # hidden layer size is passed as a list ith position value determines number of neuron in ith layer
l_rate = 0.3  # learning rate
epochs = 800 # number of training epochs

#check the timing of model-2
t0=time.time()#start time

#define model-2
model2 = NN(input_layer_size=m, output_layer_size=n_class, hidden_layer_size=hidden_layers)
model2.train(X_train, Y_train, l_rate=l_rate, n_epochs=epochs)
#end time
t1=time.time()
total_time = t1 - t0

#prediction of our model on test and training data
Y_train_predict = model2.predict(X_train)
Y_test_predict = model2.predict(X_test)
accuracy_train = 100*np.sum(Y_train==Y_train_predict)/len(Y_train)
accuracy_test = 100*np.sum(Y_test==Y_test_predict)/len(Y_test)
print("total time for training model-2:",total_time)
print("accuracy of model-2 on training data", accuracy_train)
print("accuracy of model-2 on test data", accuracy_test)
