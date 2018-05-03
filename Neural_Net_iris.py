"""
Here we will check our Neural Network model performance on iris dataset of scikit learn

"""
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

#define neural network model parameters
hidden_layers = [5,5] # hidden layer size is passed as a list ith position value determines number of neuron in ith layer
l_rate = 0.6  # learning rate
epochs = 1000 # number of training epochs

model = NN(input_layer_size=m, output_layer_size=n_class, hidden_layer_size=hidden_layers)
model.train(X_train, Y_train, l_rate=l_rate, n_epochs=epochs)
Y_train_predict = model.predict(X_train)
Y_test_predict = model.predict(X_test)
accuracy=100*np.sum(Y_test==Y_test_predict)/len(Y_test)
print(accuracy)
