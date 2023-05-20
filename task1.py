import numpy as np
import torch
import matplotlib.pyplot as plt

# Read the CSV file with measurements and store in the data variable
data = np.loadtxt('measurements.csv', delimiter=',', dtype=np.float32)

# Split the data into inputs and targets and convert to PyTorch tensors
inputs = torch.tensor(data[:, 0], dtype=torch.float32).unsqueeze(1)
targets = torch.tensor(data[:, 1], dtype=torch.float32).unsqueeze(1)

# Define the parameterized function as a PyTorch module
class ParametricFunction(torch.nn.Module):
    def __init__(self):
        super(ParametricFunction, self).__init__()
        # Define the parameters
        self.a = torch.nn.Parameter(torch.tensor(1.0)) # Controls the amplitude
        self.b = torch.nn.Parameter(torch.tensor(1.0)) # Controls the frequency
        self.c = torch.nn.Parameter(torch.tensor(1.0)) # Controls the damping factor

    # Define the forward pass of the module
    def forward(self, x):
        # Apply the parametric function to the input x
        # (multiply by 2 pi to convert angular frequency to standard frequency)
        return self.a * torch.cos(2 * torch.pi * x / self.b) * torch.exp(-self.c * x)

# Train the model
model = ParametricFunction() # Create an instance of the ParametricFunction module
criterion = torch.nn.MSELoss() # Define the mean squared error loss function
optimizer = torch.optim.SGD(model.parameters(), lr=0.1) # Define the optimizer with stochastic gradient descent and a learning rate of 0.1

for epoch in range(10000): # Loop over the epochs for training
    optimizer.zero_grad() # Reset the gradients
    y_pred = model(inputs) # Predict the output for the current input
    loss = criterion(y_pred, targets) # Calculate the loss between the predicted and actual output
    loss.backward() # Compute the gradients of the loss with respect to the parameters
    optimizer.step() # Update the parameters using the gradients

# Evaluate the model
with torch.no_grad(): # Turn off gradient tracking for evaluation
    outputs = model(inputs) # Predict the output for the input data
    loss = criterion(outputs, targets) # Calculate the loss between the predicted and actual output
    print(f'Loss: {loss.item()}') # Print the loss value as a scalar

# Evaluate the model on the input data
with torch.no_grad(): # Turn off gradient tracking for plotting
    outputs = model(inputs) # Predict the output for the input data

# Plot the actual data and the predicted data
plt.plot(inputs, targets, '.', label='Actual Data')
plt.plot(inputs, outputs, '.', label='Predicted Data')
plt.title('Actual vs. Predicted Data')
plt.xlabel('Time')
plt.ylabel('Voltage')
plt.legend()

# Show the plot
plt.show()