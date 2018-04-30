from numpy import exp,dot,random,array
class neural_network():
  def __init__(self):
    #seed is to generate same random number every times the program runs
    random.seed(10)
    
    #initialise random weights for single perceptron each between -1 to 1
    self.weights = random.random((3,1))*2 -1
    
    # sigmoid function is taken as the activation function
    # input to this function is weighted sum of input
  def sigmoid(self,x):
    return 1/(1 + exp(-x))
  
  #gradient of sigmoid is x*(1-x)
  def sigmoid_gradient(self,x):
    return x*(1-x)
  
  # method "predict" predicts the neural net outputs using current weights
  # this is done by forward propagation
  def predict(self,inputs):
    return self.sigmoid(dot(inputs,self.weights))
  
  # 'train' is to update the weights by doing back propagation
  # this is done by calculating errors
  def train(self,train_inputs,train_outputs,epochs):
    for i in range(epochs):
      #calculate the current prediction
      output=self.predict(train_inputs)
      #calculate the current error
      error=train_outputs-output
      # adjudt the weights accordingly
      adjustment = dot(train_inputs.T,error*self.sigmoid_gradient(output))
      self.weights += adjustment
      
neural_network=neural_network()
print ("Initial synaptic weights after training: ")
print (neural_network.weights)
print (neural_network.predict(array([1, 0, 0])))
training_set_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
training_set_outputs = array([[0, 1, 1, 0]]).T
# Train the neural network using a training set.
# Do it 10,000 times and make small adjustments each time.
neural_network.train(training_set_inputs, training_set_outputs, 10000)
print ("New synaptic weights after training: ")
print (neural_network.weights)

# Test the neural network with a new situation.
print ("Considering new situation [1, 0, 0] -> ?: ")
print (neural_network.predict(array([1, 0, 0])))
  
    
