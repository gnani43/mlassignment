#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
df=pd.read_csv(r"C:\Users\gnani\Downloads\Custom_CNN_Features.csv")
df


# In[5]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron


# In[6]:


df=df.drop(columns=['Filename', 'Label'])
df


# In[7]:


# logic for AND gate
X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_and = np.array([0, 0, 0, 1])
perceptron_and = Perceptron(max_iter=1000)
perceptron_and.fit(X_and, y_and)
# Logic for XOR gate
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = np.array([0, 1, 1, 0])
perceptron_xor = Perceptron(max_iter=1000)
perceptron_xor.fit(X_xor, y_xor)


# In[8]:


import numpy as np
import matplotlib.pyplot as plt

# Defining the perceptron function
def perceptron(inputs, weights):
    summation = np.dot(inputs, weights)
    if summation > 0:
        return 1
    else:
        return 0

# Defining the AND gate inputs and weight
inputs_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
weights_and = np.array([10, 0.2, -0.75])

# Defining the XOR gate inputs and weights
inputs_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
weights_xor = np.array([-10, 20, 20])

# Evaluate the perceptron for the AND gate
output_and = []
for input in inputs_and:
    output_and.append(perceptron(input, weights_and))
output_and = np.array(output_and)

# Evaluate the perceptron for the XOR gate
output_xor = []
for input in inputs_xor:
    output_xor.append(perceptron(input, weights_xor))
output_xor = np.array(output_xor)

# Plot the results
plt.scatter(inputs_and[:, 0], inputs_and[:, 1], c=output_and)
plt.title('Perceptron - AND Gate')
plt.xlabel('Input 1')
plt.ylabel('Input 2')
plt.show()

plt.scatter(inputs_xor[:, 0], inputs_xor[:, 1], c=output_xor)
plt.title('Perceptron - XOR Gate')
plt.xlabel('Input 1')
plt.ylabel('Input 2')
plt.show()


# In[1]:


import numpy as np
import matplotlib.pyplot as plt
#And gate truthtable 
# AND gate input 
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# AND gate output 
y = np.array([0, 0, 0, 1])

# as given in the assignment weights and bias 
W0, W1, W2 = 10, 0.2, -0.75
learning_rate = 0.05

#  the function for step activation
def step_activation(z):
    return 1 if z >= 0 else 0

epochs = 0
errors = []

while True:
    error_sum = 0
    for i in range(len(X)):
        xi = X[i]
        target = y[i]
        
        # weighted sum
        z = W0 + W1 * xi[0] + W2 * xi[1]
        
        #  predicted output
        predicted = step_activation(z)
        
        #  the error
        error = target - predicted
        error_sum += error ** 2
        
        # Update weights and bias
        W0 += learning_rate * error
        W1 += learning_rate * error * xi[0]
        W2 += learning_rate * error * xi[1]
    
    epochs += 1
    errors.append(error_sum)
    
    # Check for convergence condition or maximum epochs
    if error_sum <= 0.002 or epochs >= 1000:
        break

# the Plot epochs vs. the error values
plt.plot(range(epochs), errors)
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.title('Epochs vs. Error for AND Gate Perceptron')
plt.grid(True)
plt.show()

# Printing the learned weights and bias
print(f"Learned Weights: W0 = {W0}, W1 = {W1}, W2 = {W2}")


# In[2]:


# The function for Bi-Polar Step activation 
def bipolar_step_activation(z):
    return 1 if z > 0 else -1

#  tracking epochs and errors
epochs = 0
errors = []

while True:
    error_sum = 0
    for i in range(len(X)):
        xi = X[i]
        target = y[i]
        
        #  weighted sum
        z = W0 + W1 * xi[0] + W2 * xi[1]
        
        #  predicted output
        predicted = bipolar_step_activation(z)
        
        #  the error
        error = target - predicted
        error_sum += error ** 2
        
        # Update weights and bias
        W0 += learning_rate * error
        W1 += learning_rate * error * xi[0]
        W2 += learning_rate * error * xi[1]
    
    epochs += 1
    errors.append(error_sum)
    
    # Check for convergence condition or maximum epochs
    if error_sum <= 0.002 or epochs >= 1000:
        break

# Plot epochs vs. error values
plt.plot(range(epochs), errors)
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.title('Epochs vs. Error for Bi-Polar Step Perceptron')
plt.grid(True)
plt.show()

# Print the learned weights and bias
print(f"Learned Weights: W0 = {W0}, W1 = {W1}, W2 = {W2}")



# In[9]:


#A1
import numpy as np
import pandas as pd
# Define initial weights and learning rate
W0 = 10
W1 = 0.2
W2 = -0.75
learning_rate = 0.05
# Training data for AND gate
# AND gate truth table: inputs and corresponding outputs
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
targets = np.array([0, 0, 0, 1])

def activate(sum):
    return 1 if sum >= 0 else 0

# Perceptron training function
def train_perceptron(weights, learning_rate, max_epochs, data):
    errors = []  # To store error values for each epoch
    for epoch in range(max_epochs):
        total_error = 0
        for i in range(len(data)):
            x1, x2 = data[i]
            target = targets[i]
            # Calculate the weighted sum
            weighted_sum = weights[0] + weights[1] * x1 + weights[2] * x2
            # Calculate the error
            error = target - activate(weighted_sum)
            total_error += error
            # Update weights
            weights[0] += learning_rate * error
            weights[1] += learning_rate * error * x1
            weights[2] += learning_rate * error * x2
        errors.append(total_error)
        if total_error == 0:
            break
    return weights, errors

# Train the perceptron and collect errors
trained_weights, error_values = train_perceptron([W0, W1, W2], learning_rate, 100, inputs)

# Print the trained weights
print("Trained Weights:")
print(f"W0: {trained_weights[0]}, W1: {trained_weights[1]}, W2: {trained_weights[2]}")

# Test the perceptron
def test_perceptron(weights, data):
    correct = 0
    for i in range(len(data)):
        x1, x2 = data[i]
        target = targets[i]
        weighted_sum = weights[0] + weights[1] * x1 + weights[2] * x2
        prediction = activate(weighted_sum)
        if prediction == target:
            correct += 1
        print(f"Input: ({x1}, {x2}), Target: {target}, Prediction: {prediction}")
    accuracy = (correct / len(data)) * 100
    print(f"Accuracy: {accuracy}%")

# Test the trained perceptron
print("\nTesting the Trained Perceptron:")
test_perceptron(trained_weights, inputs)


# In[10]:


# Plot epochs against error values
plt.plot(range(len(error_values)), error_values)
plt.xlabel('Epochs')
plt.ylabel('Sum-Square-Error')
plt.title('Epochs vs. Sum-Square-Error in Perceptron Training')
plt.show()


# In[11]:


#A2
import numpy as np

# Define initial weights and learning rate
W0 = 10
W1 = 0.2
W2 = -0.75
learning_rate = 0.05

# Training data for AND gate
# AND gate truth table: inputs and corresponding outputs
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
targets = np.array([0, 0, 0, 1])

# Activation functions
def bi_polar_step_activation(sum):
    return -1 if sum < 0 else 1

def sigmoid_activation(sum):
    return 1 / (1 + np.exp(-sum))

def relu_activation(sum):
    return max(0, sum)

# Perceptron training function
def train_perceptron(weights, learning_rate, max_epochs, data, activation_fn):
    errors = []  # To store error values for each epoch
    for epoch in range(max_epochs):
        total_error = 0
        for i in range(len(data)):
            x1, x2 = data[i]
            target = targets[i]
            # Calculate the weighted sum
            weighted_sum = weights[0] + weights[1] * x1 + weights[2] * x2
            # Apply the selected activation function
            activated_sum = activation_fn(weighted_sum)
            # Calculate the error
            error = target - activated_sum
            total_error += error
            # Update weights
            weights[0] += learning_rate * error
            weights[1] += learning_rate * error * x1
            weights[2] += learning_rate * error * x2
        errors.append(total_error)
        if total_error == 0:
            break
    return weights, errors

# Test different activation functions and compare iterations
activation_functions = [("Bi-Polar Step", bi_polar_step_activation),
                        ("Sigmoid", sigmoid_activation),
                        ("ReLU", relu_activation)]

for activation_name, activation_fn in activation_functions:
    print(f"\nTraining with {activation_name} Activation Function:")
    trained_weights, error_values = train_perceptron([W0, W1, W2], learning_rate, 100, inputs, activation_fn)
    print("Trained Weights:")
    print(f"W0: {trained_weights[0]}, W1: {trained_weights[1]}, W2: {trained_weights[2]}")

    print("\nTesting the Trained Perceptron:")
    test_perceptron(trained_weights, inputs)
    print(f"Number of Iterations to Converge: {len(error_values)}")

    import matplotlib.pyplot as plt
    plt.plot(range(1, len(error_values) + 1), error_values)
    plt.xlabel('Epochs')
    plt.ylabel('Total Error')
    plt.title(f'Error Convergence ({activation_name} Activation Function)')
    plt.grid(True)
    plt.show()


# In[12]:


#A3
def activate(sum):
    return 1 if sum >= 0 else 0

# Perceptron training function
def train_perceptron(weights, learning_rate, max_epochs, data):
    errors = []  # To store error values for each epoch
    for epoch in range(max_epochs):
        total_error = 0
        for i in range(len(data)):
            x1, x2 = data[i]
            target = targets[i]
            # Calculate the weighted sum
            weighted_sum = weights[0] + weights[1] * x1 + weights[2] * x2
            # Calculate the error
            error = target - activate(weighted_sum)
            total_error += error
            # Update weights
            weights[0] += learning_rate * error
            weights[1] += learning_rate * error * x1
            weights[2] += learning_rate * error * x2
        errors.append(total_error)
        if total_error == 0:
            break
    return errors

# Varying learning rates
learning_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
iteration_counts = []

for lr in learning_rates:
    # Clone the initial weights to keep them the same for each learning rate
    weights = [W0, W1, W2]
    # Train the perceptron and collect errors
    error_values = train_perceptron(weights, lr, 100, inputs)
    iteration_counts.append(len(error_values))

# Plot the number of iterations vs. learning rates
plt.plot(learning_rates, iteration_counts, marker='o', linestyle='-', color='b')
plt.xlabel('Learning Rate')
plt.ylabel('Number of Iterations to Converge')
plt.title('Convergence Analysis with Varying Learning Rates')
plt.grid(True)
plt.show()


# In[13]:


#A4
import numpy as np
def activate(sum):
    return 1 if sum >= 0 else 0

# Perceptron training function
def train_perceptron(weights, learning_rate, max_epochs, data):
    errors = []  # To store error values for each epoch
    for epoch in range(max_epochs):
        total_error = 0
        for i in range(len(data)):
            x1, x2 = data[i]
            target = targets[i]
            # Calculate the weighted sum
            weighted_sum = weights[0] + weights[1] * x1 + weights[2] * x2
            # Calculate the error
            error = target - activate(weighted_sum)
            total_error += error
            # Update weights
            weights[0] += learning_rate * error
            weights[1] += learning_rate * error * x1
            weights[2] += learning_rate * error * x2
        errors.append(total_error)
        if total_error == 0:
            break
    return weights, errors

# Train the perceptron and collect errors
trained_weights, error_values = train_perceptron([W0, W1, W2], learning_rate, 100, inputs)

# Print the trained weights
print("Trained Weights:")
print(f"W0: {trained_weights[0]}, W1: {trained_weights[1]}, W2: {trained_weights[2]}")

# Test the perceptron
def test_perceptron(weights, data):
    correct = 0
    for i in range(len(data)):
        x1, x2 = data[i]
        target = targets[i]
        weighted_sum = weights[0] + weights[1] * x1 + weights[2] * x2
        prediction = activate(weighted_sum)
        if prediction == target:
            correct += 1
        print(f"Input: ({x1}, {x2}), Target: {target}, Prediction: {prediction}")
    accuracy = (correct / len(data)) * 100
    print(f"Accuracy: {accuracy}%")

# Test the trained perceptron
print("\nTesting the Trained Perceptron:")
test_perceptron(trained_weights, inputs)


# In[14]:


#A4
import numpy as np

# Define initial weights and learning rate
W0 = 10
W1 = 0.2
W2 = -0.75
learning_rate = 0.05

# Training data for AND gate
# XOR gate truth table: inputs and corresponding outputs
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
targets = np.array([0, 1, 1, 0])

# Activation functions
def bi_polar_step_activation(sum):
    return -1 if sum < 0 else 1

def sigmoid_activation(sum):
    return 1 / (1 + np.exp(-sum))

def relu_activation(sum):
    return max(0, sum)

# Perceptron training function
def train_perceptron(weights, learning_rate, max_epochs, data, activation_fn):
    errors = []  # To store error values for each epoch
    for epoch in range(max_epochs):
        total_error = 0
        for i in range(len(data)):
            x1, x2 = data[i]
            target = targets[i]
            # Calculate the weighted sum
            weighted_sum = weights[0] + weights[1] * x1 + weights[2] * x2
            # Apply the selected activation function
            activated_sum = activation_fn(weighted_sum)
            # Calculate the error
            error = target - activated_sum
            total_error += error
            # Update weights
            weights[0] += learning_rate * error
            weights[1] += learning_rate * error * x1
            weights[2] += learning_rate * error * x2
        errors.append(total_error)
        if total_error == 0:
            break
    return weights, errors

# Test different activation functions and compare iterations
activation_functions = [("Bi-Polar Step", bi_polar_step_activation),
                        ("Sigmoid", sigmoid_activation),
                        ("ReLU", relu_activation)]

for activation_name, activation_fn in activation_functions:
    print(f"\nTraining with {activation_name} Activation Function:")
    trained_weights, error_values = train_perceptron([W0, W1, W2], learning_rate, 100, inputs, activation_fn)
    print("Trained Weights:")
    print(f"W0: {trained_weights[0]}, W1: {trained_weights[1]}, W2: {trained_weights[2]}")

    print("\nTesting the Trained Perceptron:")
    test_perceptron(trained_weights, inputs)
    print(f"Number of Iterations to Converge: {len(error_values)}")

    import matplotlib.pyplot as plt
    plt.plot(range(1, len(error_values) + 1), error_values)
    plt.xlabel('Epochs')
    plt.ylabel('Total Error')
    plt.title(f'Error Convergence ({activation_name} Activation Function)')
    plt.grid(True)
    plt.show()


# In[15]:


#A5
import numpy as np

# Define initial weights and learning rate
W0 = 0.1
W1 = 0.1
W2 = 0.1
W3 = 0.1
learning_rate = 0.1

# Training data
data = np.array([
    [20, 6, 2, 386],
    [16, 3, 6, 289],
    [27, 6, 2, 393],
    [19, 1, 2, 110],
    [24, 4, 2, 280],
    [22, 1, 5, 167],
    [15, 4, 2, 271],
    [18, 4, 2, 274],
    [21, 1, 4, 148],
    [16, 2, 4, 198]
])

# Target values (High Value or Low Value)
targets = np.array([1, 1, 1, 0, 1, 0, 1, 1, 0, 0])

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def predict(x1, x2, x3, x4):
    weighted_sum = W0 + W1 * x1 + W2 * x2 + W3 * x3
    return sigmoid(weighted_sum)

def train_perceptron(max_epochs, data, targets):
    global W0, W1, W2, W3  # Declare global variables
    for epoch in range(max_epochs):
        total_error = 0
        for i in range(len(data)):
            x1, x2, x3, x4 = data[i]
            target = targets[i]
            prediction = predict(x1, x2, x3, x4)
            error = target - prediction
            total_error += error
            W0 += learning_rate * error
            W1 += learning_rate * error * x1
            W2 += learning_rate * error * x2
            W3 += learning_rate * error * x3
        if total_error == 0:
            break

# Train the perceptron
train_perceptron(1000, data, targets)

# Test the perceptron and print the results
for i in range(len(data)):
    x1, x2, x3, x4 = data[i]
    prediction = predict(x1, x2, x3, x4)
    classification = "Yes" if prediction >= 0.5 else "No"
    print(f"Customer C_{i + 1}: {classification}")


# In[16]:


#A6
# Calculate the pseudo-inverse of the data
pseudo_inverse = np.linalg.pinv(data)
print("Pseudo inverse is",pseudo_inverse)


# In[17]:


#A7
import numpy as np

class ANDGateNeuralNetwork:
    def __init__(self, learning_rate=0.05):
        self.weights_ih = np.random.randn(2, 2)
        self.weights_ho = np.random.randn(1, 2)
        self.learning_rate = learning_rate

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward_propagate(self, inputs):
        h = self.sigmoid(np.dot(self.weights_ih, inputs))
        o = self.sigmoid(np.dot(self.weights_ho, h))
        return o

    def backpropagate(self, inputs, target_output, actual_output, h):
        error = target_output - actual_output

        gradient_ho = error * actual_output * (1 - actual_output)
        gradient_ih = (gradient_ho @ self.weights_ho) * h * (1 - h)

        self.weights_ho += self.learning_rate * np.outer(gradient_ho, h)
        self.weights_ih += self.learning_rate * np.outer(gradient_ih, inputs)

    def train(self, training_examples):
        for inputs, target_output in training_examples:
            h = self.sigmoid(np.dot(self.weights_ih, inputs))
            actual_output = self.forward_propagate(inputs)
            self.backpropagate(inputs, target_output, actual_output, h)

    def predict(self, inputs):
        return self.forward_propagate(inputs)

# Create a new AND gate neural network
network = ANDGateNeuralNetwork()

# Train the network on the AND gate truth table
training_examples = [(np.array([0, 0]), 0), (np.array([0, 1]), 0), (np.array([1, 0]), 0), (np.array([1, 1]), 1)]
network.train(training_examples)

# Test the network for multiple inputs
inputs_list = [np.array([0, 0]), np.array([0, 1]), np.array([1, 0]), np.array([1, 1])]

for inputs in inputs_list:
    output = network.predict(inputs)
    print(f"Input: {inputs}, Output: {output}")


# In[18]:


#A8
import numpy as np

class XORGateNeuralNetwork:
    def __init__(self, learning_rate=0.05):
        self.weights_ih = np.random.randn(2, 2)
        self.weights_ho = np.random.randn(1, 2)
        self.learning_rate = learning_rate

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward_propagate(self, inputs):
        h = self.sigmoid(np.dot(self.weights_ih, inputs))
        o = self.sigmoid(np.dot(self.weights_ho, h))
        return o

    def backpropagate(self, inputs, target_output, actual_output, h):
        error = target_output - actual_output

        gradient_ho = error * actual_output * (1 - actual_output)
        gradient_ih = (gradient_ho @ self.weights_ho) * h * (1 - h)

        self.weights_ho += self.learning_rate * np.outer(gradient_ho, h)
        self.weights_ih += self.learning_rate * np.outer(gradient_ih, inputs)

    def train(self, training_examples):
        for inputs, target_output in training_examples:
            h = self.sigmoid(np.dot(self.weights_ih, inputs))
            actual_output = self.forward_propagate(inputs)
            self.backpropagate(inputs, target_output, actual_output, h)

    def predict(self, inputs):
        return self.forward_propagate(inputs)

# Create a new XOR gate neural network
network = XORGateNeuralNetwork()

# Train the network on the XOR gate truth table
training_examples = [(np.array([0, 0]), 0), (np.array([0, 1]), 1), (np.array([1, 0]), 1), (np.array([1, 1]), 0)]
network.train(training_examples)

# Test the network for multiple inputs
inputs_list = [np.array([0, 0]), np.array([0, 1]), np.array([1, 0]), np.array([1, 1])]

for inputs in inputs_list:
    output = network.predict(inputs)
    print(f"Input: {inputs}, Output: {output}")


# In[19]:


#A9
class ANDGatePerceptron:
    def __init__(self, learning_rate=0.05):
        self.weights = np.random.randn(2)
        self.bias = np.random.randn(1)
        self.learning_rate = learning_rate

    def forward_propagate(self, inputs):
        weighted_sum = np.dot(inputs, self.weights) + self.bias
        output = np.where(weighted_sum >= 0, 1, 0)
        return output

    def backpropagate(self, inputs, target_output, actual_output):
        error = target_output - actual_output
        delta = error * self.learning_rate
        self.weights += delta * inputs
        self.bias += delta

    def train(self, training_examples):
        for inputs, target_output in training_examples:
            actual_output = self.forward_propagate(inputs)
            self.backpropagate(inputs, target_output, actual_output)

    def predict(self, inputs):
        return self.forward_propagate(inputs)

# Create a new perceptron
perceptron = ANDGatePerceptron()

# Create a training dataset
training_examples = [(np.array([0, 0]), np.array([0])), (np.array([0, 1]), np.array([0])), (np.array([1, 0]), np.array([0])), (np.array([1, 1]), np.array([1]))]

# Train the perceptron
perceptron.train(training_examples)

# Test the perceptron
inputs = np.array([1, 1])
output = perceptron.predict(inputs)

# Print the output
print(output)



import numpy as np
from sklearn.neural_network import MLPClassifier

# Create an XOR gate dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])

# Create an MLPClassifier with one hidden layer
mlp = MLPClassifier(hidden_layer_sizes=(2,), activation='logistic', max_iter=10000, random_state=42)

# Train the model
mlp.fit(X, y)

# Test the model
inputs = np.array([[1, 1]])
output = mlp.predict(inputs)

# Print the output
print(output)



# In[20]:


#A9
import numpy as np

class Perceptron:
    def __init__(self, num_inputs, learning_rate=0.05):
        self.weights = np.random.randn(num_inputs)
        self.learning_rate = learning_rate

    def forward_propagate(self, inputs):
        weighted_sum = np.dot(inputs, self.weights)
        output = 1 / (1 + np.exp(-weighted_sum))  # Use sigmoid activation
        return output

    def backpropagate(self, inputs, target_output, actual_output):
        error = target_output - actual_output
        delta = error * actual_output * (1 - actual_output)
        self.weights += self.learning_rate * delta * inputs

    def train(self, training_examples, num_epochs=100):
        for epoch in range(num_epochs):
            for inputs, target_output in training_examples:
                actual_output = self.forward_propagate(inputs)
                self.backpropagate(inputs, target_output, actual_output)

    def predict(self, inputs):
        return self.forward_propagate(inputs)

# Create a new perceptron with 2 input features (for AND and XOR)
num_inputs = 2
perceptron = Perceptron(num_inputs)

# Create a training dataset for AND
training_and = [
    (np.array([0, 0]), 0),
    (np.array([0, 1]), 0),
    (np.array([1, 0]), 0),
    (np.array([1, 1]), 1)
]

# Create a training dataset for XOR
training_xor = [
    (np.array([0, 0]), 0),
    (np.array([0, 1]), 1),
    (np.array([1, 0]), 1),
    (np.array([1, 1]), 0)
]

# Train the perceptron for AND
perceptron.train(training_and)

# Test the perceptron for AND
inputs_and = np.array([1, 1])
output_and = perceptron.predict(inputs_and)
print("AND Gate:", output_and)

# Train the perceptron for XOR
perceptron.train(training_xor)

# Test the perceptron for XOR
inputs_xor = np.array([1, 1])
output_xor = perceptron.predict(inputs_xor)
print("XOR Gate:", output_xor)

for inputs in inputs_list:
    output = network.predict(inputs)
    print(f"Input: {inputs}, Output: {output}")


# In[21]:


#A10
import numpy as np
from sklearn.neural_network import MLPClassifier

# Training data for AND gate
# AND gate truth table: inputs and corresponding outputs
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 0, 0, 1])

# Create an MLPClassifier with one hidden layer
mlp = MLPClassifier(hidden_layer_sizes=(2,), activation='logistic', solver='sgd', learning_rate_init=0.05, max_iter=100)

# Train the classifier
mlp.fit(X, y)

# Print the trained weights and biases
print("Trained Weights (Coefs):")
print(mlp.coefs_)
print("Trained Biases (Intercepts):")
print(mlp.intercepts_)

# Test the trained classifier
def test_classifier(classifier, data, targets):
    predictions = classifier.predict(data)
    accuracy = (sum(predictions == targets) / len(targets)) * 100
    print("Predictions:", predictions)
    print("Accuracy:", accuracy, "%")

# Test the trained classifier
print("\nTesting the Trained Classifier:")
test_classifier(mlp, X, y)


# In[23]:


#A10
from sklearn.neural_network import MLPClassifier

# Training data for XOR gate
# XOR gate truth table: inputs and corresponding outputs
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])

# Create an MLPClassifier with one hidden layer
mlp = MLPClassifier(hidden_layer_sizes=(2,), activation='logistic', solver='sgd', learning_rate_init=0.05, max_iter=100)

# Train the classifier
mlp.fit(X, y)

# Print the trained weights and biases
print("Trained Weights (Coefs):")
print(mlp.coefs_)
print("Trained Biases (Intercepts):")
print(mlp.intercepts_)

# Test the trained classifier
def test_classifier(classifier, data, targets):
    predictions = classifier.predict(data)
    accuracy = (sum(predictions == targets) / len(targets)) * 100
    print("Predictions:", predictions)
    print("Accuracy:", accuracy, "%")

# Test the trained classifier
print("\nTesting the Trained Classifier:")
test_classifier(mlp, X, y)


# In[26]:


#A11
import numpy as np

# Define the initial weights and learning rate
W0 = 10
W1 = 0.2
W2 = -0.75
learning_rate = 0.05
# Reshape your data into a 2D format (samples x features)
num_samples = X.shape[0]   # Total number of samples
num_features = y.shape[1]  # Number of features for each sample

inputs = X.reshape(num_samples, num_features)
targets = y.reshape(num_samples)

# Initialize variables for tracking epochs and errors
epochs = 0
errors = []

# Define the Step activation function
def step_activation(x):
    return 1 if x >= 0 else 0

# Training the perceptron
max_epochs = 1000  # Set a maximum number of epochs to avoid infinite looping

while epochs < max_epochs:
    total_error = 0
    
    for i in range(len(inputs)):
        # Calculate the weighted sum of inputs
        weighted_sum = W0 + W1 * inputs[i][0] + W2 * inputs[i][1]
        
        # Apply the Step activation function
        prediction = step_activation(weighted_sum)
        
        # Calculate the error
        error = targets[i] - prediction
        
        # Update weights
        W0 = W0 + learning_rate * error
        W1 = W1 + learning_rate * error * inputs[i][0]
        W2 = W2 + learning_rate * error * inputs[i][1]
        
        total_error += error ** 2
    
    # Append the total error for this epoch to the list
    errors.append(total_error)
    
    # Check for convergence
    if total_error == 0:
        break
    
    epochs += 1

# Test the trained perceptron
for i in range(len(inputs)):
    weighted_sum = W0 + W1 * inputs[i][0] + W2 * inputs[i][1]
    prediction = step_activation(weighted_sum)
    print(f"Input: {inputs[i]}, Target: {targets[i]}, Predicted: {prediction}")

print(f"Final Weights: W0 = {W0}, W1 = {W1}, W2 = {W2}")
print(f"Number of Epochs: {epochs}")


# In[29]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load data from the uploaded Excel file
data = pd.read_csv(r"C:\Users\gnani\Downloads\Custom_CNN_Features.csv")

# Assuming 'f0' contains the text data, replace it with the actual column name
text_column = 'f0'

# Drop rows with missing values in the text column
data = data.dropna(subset=[text_column])

# Convert the text column to strings
data[text_column] = data[text_column].astype(str)

# Convert 'Classification' to categorical if needed
data['Label'] = data['Label'].astype('category')

# Split the data into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Create a TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Fit and transform the training text data
X_train_tfidf = tfidf_vectorizer.fit_transform(train_data[text_column])

# Transform the testing text data
X_test_tfidf = tfidf_vectorizer.transform(test_data[text_column])

# Define the labels
y_train = train_data['Label']
y_test = test_data['Label']

# Create a decision tree classifier
classifier = DecisionTreeClassifier()

# Train the classifier on the training data
classifier.fit(X_train_tfidf, y_train)

# Make predictions on the testing data
y_pred = classifier.predict(X_test_tfidf)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Plot the accuracy
plt.figure(figsize=(8, 6))
plt.bar(['Accuracy'], [accuracy], color='grey')
plt.ylim(0, 1)  # Set the y-axis limits between 0 and 1
plt.title('Classifier Accuracy')
plt.ylabel('Accuracy')
plt.show()

