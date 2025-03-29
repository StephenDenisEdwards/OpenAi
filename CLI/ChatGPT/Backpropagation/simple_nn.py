import matplotlib.pyplot as plt

# Define the simple neural network parameters
X = 1.5       # Input value
Y = 0.5       # Target output value
W = 0.8       # Initial weight (randomly chosen)
alpha = 0.1   # Learning rate
steps = 20    # Number of training iterations

# Store the history of training for plotting
weights = []
predictions = []
errors = []

# Training the neural network
for step in range(steps):
    # Forward propagation: Compute predicted output
    Y_hat = X * W

    # Calculate the error (Mean Squared Error)
    error = 0.5 * (Y - Y_hat)**2

    # Record history for plotting
    weights.append(W)
    predictions.append(Y_hat)
    errors.append(error)

    # Compute gradient (derivative of error w.r.t. weight)
    gradient = (Y_hat - Y) * X

    # Update weight using gradient descent
    W = W - alpha * gradient

# Plot the predictions and errors to visualize training progress
fig, axs = plt.subplots(2, 1, figsize=(8, 10))

# Plot predictions vs target output
axs[0].plot(range(steps), predictions, marker='o', label='Predictions')
axs[0].axhline(y=Y, color='r', linestyle='--', label='Target Output')
axs[0].set_xlabel('Training Steps')
axs[0].set_ylabel('Output Value')
axs[0].set_title('Neural Network Output Over Training Steps')
axs[0].legend()
axs[0].grid()

# Plot error reduction over training steps
axs[1].plot(range(steps), errors, marker='o', color='orange')
axs[1].set_xlabel('Training Steps')
axs[1].set_ylabel('Error (MSE)')
axs[1].set_title('Error Reduction Over Training Steps')
axs[1].grid()

# Display the plots
plt.tight_layout()
plt.show(block=True)
