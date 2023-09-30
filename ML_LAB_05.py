#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np

# Initialize weights and bias
W0 = 10
W1 = 0.2
W2 = -0.75
learning_rate = 0.05

# Define the activation function (step function in this case)
def activate(z):
    if z >= 0:
        return 1
    else:
        return 0

# Define the perceptron function
def perceptron(input_data, weights):
    # Calculate the weighted sum of inputs
    z = W0 + np.dot(input_data, weights)
    # Apply the activation function
    output = activate(z)
    return output

# Training data (you can modify this as needed)
input_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Two input features
target_output = np.array([0, 0, 0, 1])  # Adjust this for your specific problem

# Training loop
epochs = 100  # Number of training iterations
for epoch in range(epochs):
    for i in range(len(input_data)):
        # Compute the predicted output
        prediction = perceptron(input_data[i], [W1, W2])
        # Compute the error
        error = target_output[i] - prediction
        # Update the weights and bias
        W0 = W0 + learning_rate * error
        W1 = W1 + learning_rate * error * input_data[i][0]
        W2 = W2 + learning_rate * error * input_data[i][1]

# Print the final weights
print("Final Weights:")
print("W0 =", W0)
print("W1 =", W1)
print("W2 =", W2)


# In[10]:


import numpy as np
import matplotlib.pyplot as plt

# Initialize weights and bias
W0 = 10
W1 = 0.2
W2 = -0.75
learning_rate = 0.05

# Define the Sigmoid activation function
def activate_sig(z):
    return 1 / (1 + np.exp(-z))

def perceptron_sig(input_data, weights):
    # Calculate the weighted sum of inputs
    z = np.dot(input_data, weights)
    # Apply the activation function
    output = activate_bi(z)
    return output

# Training data (you can modify this as needed)
input_data = np.array([[1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])  # Bias term included as 1
target_output = np.array([0, 0, 0, 1])  # Adjust this for your specific problem

# Training loop
epochs = 100  # Number of training iterations
error_history = []  # To store error values for plotting

for epoch in range(epochs):
    total_error = 0  # Initialize the total error for this epoch
    
    for i in range(len(input_data)):
        # Compute the predicted output
        prediction = perceptron_sig(input_data[i], [W0, W1, W2])
        
        # Compute the error
        error = target_output[i] - prediction
        
        # Update the weights and bias
        W0 = W0 + learning_rate * error * input_data[i][0]
        W1 = W1 + learning_rate * error * input_data[i][1]
        W2 = W2 + learning_rate * error * input_data[i][2]
        
        # Calculate the sum-square-error and add it to the total error
        total_error += error ** 2
    
    # Append the total error for this epoch to the error history
    error_history.append(total_error)

# Plot epochs against error values
plt.plot(range(1, epochs + 1), error_history)
plt.xlabel('Epochs')
plt.ylabel('Sum-Square-Error')
plt.title('Error vs. Epochs')
plt.grid(True)
plt.show()

# Print the final weights
print("Final Weights:")
print("W0 =", W0)
print("W1 =", W1)
print("W2 =", W2)



# In[14]:


import numpy as np
import matplotlib.pyplot as plt

# Initialize weights and bias
W0 = 10
W1 = 0.2
W2 = -0.75
learning_rate = 0.05

def activate_re(z):
    return max(0, z)

# Define the perceptron function
def perceptron_re(input_data, weights):
    # Calculate the weighted sum of inputs
    z = np.dot(input_data, weights)
    # Apply the activation function
    output = activate_bi(z)
    return output

# Training data (you can modify this as needed)
input_data = np.array([[1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])  # Bias term included as 1
target_output = np.array([0, 0, 0, 1])  # Adjust this for your specific problem

# Training loop
epochs = 100  # Number of training iterations
error_history = []  # To store error values for plotting

for epoch in range(epochs):
    total_error = 0  # Initialize the total error for this epoch
    
    for i in range(len(input_data)):
        # Compute the predicted output
        prediction = perceptron_re(input_data[i], [W0, W1, W2])
        
        # Compute the error
        error = target_output[i] - prediction
        
        # Update the weights and bias
        W0 = W0 + learning_rate * error * input_data[i][0]
        W1 = W1 + learning_rate * error * input_data[i][1]
        W2 = W2 + learning_rate * error * input_data[i][2]
        
        # Calculate the sum-square-error and add it to the total error
        total_error += error ** 2
    
    # Append the total error for this epoch to the error history
    error_history.append(total_error)

# Plot epochs against error values
plt.plot(range(1, epochs + 1), error_history)
plt.xlabel('Epochs')
plt.ylabel('Sum-Square-Error')
plt.title('Error vs. Epochs')
plt.grid(True)
plt.show()

# Print the final weights
print("Final Weights:")
print("W0 =", W0)
print("W1 =", W1)
print("W2 =", W2)



# In[15]:


def bipolar_step_activation(x):
    return 1 if x >= 0 else -1

W = np.array([10, 0.2, -0.75])
learning_rate = 0.05
convergence_error = 0.002
max_epochs = 1000

input_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
target_output = np.array([0, 0, 0, 1])
error_values = []
epochs = 0
while True:
    total_error = 0  
    for i in range(len(input_data)):
        weighted_sum = W[0] + W[1] * input_data[i, 0] + W[2] * input_data[i, 1]
        predicted_output = bipolar_step_activation(weighted_sum)
        error_i = target_output[i] - predicted_output
        total_error += error_i ** 2  
        W[0] += learning_rate * error_i
        W[1] += learning_rate * error_i * input_data[i, 0]
        W[2] += learning_rate * error_i * input_data[i, 1]
 
    error_values.append(total_error) 
    epochs += 1
    if total_error <= convergence_error or epochs >= max_epochs:
        break

print(" bipolar step activation : ")
print(f"Converged in {epochs} epochs.")
print("Final weights:", W)


# In[20]:


W0 = 10
W1 = 0.2
W2 = -0.75
learning_rate = 0.05

input_data = np.array([[1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])  # Bias term included as 1
target_output = np.array([0, 0, 0, 1])  # Adjust this for your specific problem

def bipolar_step(x):
    return np.where(x >= 0, 1, -1)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def train_perceptron(data, activation_function, learning_rate, max_iterations=1000):
    num_inputs = data.shape[1] - 1
    weights = np.random.rand(num_inputs)
    bias = np.random.rand()
    iterations = 0
    
    for _ in range(max_iterations):
        converged = True
        for example in data:
            x = example[:num_inputs]
            y = example[num_inputs]
            y_pred = activation_function(np.dot(weights, x) + bias)
            
            if y != y_pred:
                error = y - y_pred
                weights += learning_rate * error * x
                bias += learning_rate * error
                converged = False
        
        iterations += 1
        if converged:
            break
    
    return weights, bias, iterations

learning_rate = 0.05
weights_bipolar, bias_bipolar, iterations_bipolar = train_perceptron(input_data, bipolar_step, learning_rate)
print("Bipolar Step Function:")
print("Weights:", weights_bipolar)
print("Bias:", bias_bipolar)
print("Iterations to Converge:", iterations_bipolar)


learning_rate = 0.05
weights_sigmoid, bias_sigmoid, iterations_sigmoid = train_perceptron(input_data, sigmoid, learning_rate)
print("\nSigmoid Function:")
print("Weights:", weights_sigmoid)
print("Bias:", bias_sigmoid)
print("Iterations to Converge:", iterations_sigmoid)


learning_rate = 0.05
weights_relu, bias_relu, iterations_relu = train_perceptron(input_data, relu, learning_rate)
print("\nReLU Function:")
print("Weights:", weights_relu)
print("Bias:", bias_relu)
print("Iterations to Converge:", iterations_relu)


# In[21]:


import numpy as np
import matplotlib.pyplot as plt

# Initialize weights and bias
W0 = 10
W1 = 0.2
W2 = -0.75
learning_rate = 0.05

# Define the Sigmoid activation function
def activate_sig(z):
    return 1 / (1 + np.exp(-z))

def perceptron_sig(input_data, weights):
    # Calculate the weighted sum of inputs
    z = np.dot(input_data, weights)
    # Apply the activation function
    output = activate_bi(z)
    return output

# Training data (you can modify this as needed)
input_data = np.array([[1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])  # Bias term included as 1
target_output = np.array([0, 1, 1, 0])  # Adjust this for your specific problem

# Training loop
epochs = 100  # Number of training iterations
error_history = []  # To store error values for plotting

for epoch in range(epochs):
    total_error = 0  # Initialize the total error for this epoch
    
    for i in range(len(input_data)):
        # Compute the predicted output
        prediction = perceptron_sig(input_data[i], [W0, W1, W2])
        
        # Compute the error
        error = target_output[i] - prediction
        
        # Update the weights and bias
        W0 = W0 + learning_rate * error * input_data[i][0]
        W1 = W1 + learning_rate * error * input_data[i][1]
        W2 = W2 + learning_rate * error * input_data[i][2]
        
        # Calculate the sum-square-error and add it to the total error
        total_error += error ** 2
    
    # Append the total error for this epoch to the error history
    error_history.append(total_error)

# Plot epochs against error values
plt.plot(range(1, epochs + 1), error_history)
plt.xlabel('Epochs')
plt.ylabel('Sum-Square-Error')
plt.title('Error vs. Epochs')
plt.grid(True)
plt.show()

# Print the final weights
print("Final Weights:")
print("W0 =", W0)
print("W1 =", W1)
print("W2 =", W2)



# In[22]:


W0 = 10
W1 = 0.2
W2 = -0.75
learning_rate = 0.05

input_data = np.array([[1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])  # Bias term included as 1
target_output = np.array([0, 1, 1, 0])  # Adjust this for your specific problem

def bipolar_step(x):
    return np.where(x >= 0, 1, -1)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def train_perceptron(data, activation_function, learning_rate, max_iterations=1000):
    num_inputs = data.shape[1] - 1
    weights = np.random.rand(num_inputs)
    bias = np.random.rand()
    iterations = 0
    
    for _ in range(max_iterations):
        converged = True
        for example in data:
            x = example[:num_inputs]
            y = example[num_inputs]
            y_pred = activation_function(np.dot(weights, x) + bias)
            
            if y != y_pred:
                error = y - y_pred
                weights += learning_rate * error * x
                bias += learning_rate * error
                converged = False
        
        iterations += 1
        if converged:
            break
    
    return weights, bias, iterations

learning_rate = 0.05
weights_bipolar, bias_bipolar, iterations_bipolar = train_perceptron(input_data, bipolar_step, learning_rate)
print("Bipolar Step Function:")
print("Weights:", weights_bipolar)
print("Bias:", bias_bipolar)
print("Iterations to Converge:", iterations_bipolar)


learning_rate = 0.05
weights_sigmoid, bias_sigmoid, iterations_sigmoid = train_perceptron(input_data, sigmoid, learning_rate)
print("\nSigmoid Function:")
print("Weights:", weights_sigmoid)
print("Bias:", bias_sigmoid)
print("Iterations to Converge:", iterations_sigmoid)


learning_rate = 0.05
weights_relu, bias_relu, iterations_relu = train_perceptron(input_data, relu, learning_rate)
print("\nReLU Function:")
print("Weights:", weights_relu)
print("Bias:", bias_relu)
print("Iterations to Converge:", iterations_relu)


# In[ ]:




