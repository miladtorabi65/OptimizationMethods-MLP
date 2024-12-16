import numpy as np
import pandas as pd
from scipy.optimize import minimize
import time


class NeuralNetwork:
    def __init__(self,
                 input_size,
                 hidden_layer_sizes,
                 lambda_reg=0.01,
                 initialization_mode='he',
                 activation='relu'):
        """
        Initialize the neural network with the specified architecture and hyperparameters.

        Parameters:
        - input_size (int): Number of input features.
        - hidden_layer_sizes (list of int): Number of neurons in each hidden layer.
        - lambda_reg (float): Regularization strength (L2 regularization).
        - initialization_mode (str): Weight initialization strategy ('he', 'xavier', etc.).
        - activation (str): Activation function ('relu', 'tanh', etc.).
        """
        self.input_size = input_size
        self.hidden_layer_sizes = hidden_layer_sizes
        self.lambda_reg = lambda_reg
        self.initialization_mode = initialization_mode
        self.activation = activation.lower()
        # Initialize weights and biases
        self.weights, self.biases = self.initialize_weights_and_biases(mode=self.initialization_mode)
        # Initialize an empty list to store loss values during training
        self.loss_history = []

    def model_info_show(self):
        """
        Print detailed information about the model's architecture and parameters.
        """
        print(f'Input size: {self.input_size}')
        print(f'Hidden layer sizes: {self.hidden_layer_sizes}')
        print(f'Lambda (regularization strength): {self.lambda_reg}')
        print(f'Initialization mode: {self.initialization_mode}')
        print(f'Activation function: {self.activation}')
        print(f'Weights:')
        for idx, W in enumerate(self.weights):
            print(f' Layer {idx+1} weights shape: {W.shape}')
        print(f'Biases:')
        for idx, b in enumerate(self.biases):
            print(f' Layer {idx+1} biases shape: {b.shape}')

    def initialize_weights_and_biases(self, mode='xavier'):
        """
        Initialize weights and biases for the neural network.

        Parameters:
        - mode (str): Weight initialization strategy ('xavier', 'standard', 'he').

        Returns:
        - weights (list of np.ndarray): List of weight matrices for each layer.
        - biases (list of np.ndarray): List of bias vectors for each layer.
        """
        weights = []  # To store weight matrices
        biases = []  # To store bias vectors

        # Define sizes of all layers: input, hidden, and output
        layer_sizes = [self.input_size] + self.hidden_layer_sizes + [1]  # 1 for the output layer

        # Loop through each layer to initialize weights and biases
        for i in range(len(layer_sizes) - 1):
            size_in, size_out = layer_sizes[i], layer_sizes[i + 1]

            # Initialize weights based on the chosen method
            if mode == 'xavier':
                # Xavier/Glorot initialization
                limit = np.sqrt(6 / (size_in + size_out))
                W = np.random.uniform(-limit, limit, (size_in, size_out))
            elif mode == 'standard':
                # Standard initialization (small random values)
                W = np.random.randn(size_in, size_out) * 0.01
            elif mode == 'he':
                # He initialization
                stddev = np.sqrt(2 / size_in)
                W = np.random.randn(size_in, size_out) * stddev
            else:
                # Raise error for unknown initialization mode
                raise ValueError(f"Unknown initialization mode: {mode}")

            # Initialize biases to zeros
            b = np.zeros((1, size_out))

            # Append the weight matrix and bias vector to their respective lists
            weights.append(W)
            biases.append(b)

        return weights, biases

    def forward(self, X):
        """
        Perform a forward pass through the neural network.

        Parameters:
        - X (np.ndarray): Input data with shape (num_samples, input_size).

        Returns:
        - activations (dict): Dictionary containing activations for each layer.
        - pre_activations (dict): Dictionary containing pre-activations (z) for each layer.
        """
        activations = {}  # Store activations (a) for each layer
        pre_activations = {}  # Store pre-activations (z) for each layer

        # Initialize input layer activations
        activations[0] = X

        # Iterate through each layer
        for i in range(len(self.weights)):
            # Retrieve weights and biases for the current layer
            W = self.weights[i]
            b = self.biases[i]

            # Compute pre-activation (z = a_prev @ W + b)
            z = np.dot(activations[i], W) + b

            # Clip pre-activation values to avoid numerical overflow
            z = np.clip(z, -500, 500)
            pre_activations[i + 1] = z

            # Compute activation (a)
            if i < len(self.weights) - 1:
                # Hidden layers use the specified activation function
                a = self.activation_function(z)
            else:
                # Output layer uses linear activation
                a = z
            activations[i + 1] = a

        return activations, pre_activations


    def activation_function(self, z):
        """
        Compute the activation for the specified activation function.

        Parameters:
        - z (np.ndarray): Pre-activation values.

        Returns:
        - np.ndarray: Activations computed using the specified activation function.
        """
        # Clip pre-activation values to prevent numerical overflow
        safe_z = np.clip(z, -500, 500)

        if self.activation == 'relu':
            # ReLU activation: max(0, z)
            return np.maximum(0, safe_z)
        elif self.activation == 'tanh':
            # Tanh activation
            return np.tanh(safe_z)
        elif self.activation == 'swish':
            # Swish activation: z * sigmoid(z)
            return safe_z * (1 / (1 + np.exp(-safe_z)))
        else:
            # Raise an error for unsupported activation functions
            raise ValueError(f"Unknown activation function: {self.activation}")


    def activation_derivative(self, z):
        """
        Compute the derivative of the activation function for backpropagation.

        Parameters:
        - z (np.ndarray): Pre-activation values.

        Returns:
        - np.ndarray: Derivatives computed for the specified activation function.
        """
        # Clip pre-activation values to prevent numerical overflow
        safe_z = np.clip(z, -500, 500)

        if self.activation == 'relu':
            # ReLU derivative: 1 if z > 0, else 0
            return (safe_z > 0).astype(float)
        elif self.activation == 'tanh':
            # Tanh derivative: 1 - tanh(z)^2
            return 1 - np.tanh(safe_z) ** 2
        elif self.activation == 'swish':
            # Swish derivative: sigmoid(z) + z * sigmoid(z) * (1 - sigmoid(z))
            sig_safe_z = 1 / (1 + np.exp(-safe_z))  # Sigmoid function
            return sig_safe_z + safe_z * sig_safe_z * (1 - sig_safe_z)
        else:
            # Raise an error for unsupported activation functions
            raise ValueError(f"Unknown activation function: {self.activation}")

    def compute_loss(self, activations, y):
        """
        Compute the total loss for the neural network, including MSE and L2 regularization.

        Parameters:
        - activations (dict): Dictionary of activations from forward pass.
        - y (np.ndarray): Target values with shape (num_samples, 1).

        Returns:
        - total_loss (float): Combined loss (MSE + L2 regularization).
        """
        # Ensure target values are in column vector format
        y = y.reshape(-1, 1)

        # Get the output layer activations (predictions)
        output_activations = activations[len(self.weights)]

        # Compute Mean Squared Error (MSE) loss
        mse_loss = np.mean((output_activations - y) ** 2)

        # Compute L2 regularization loss
        l2_loss = self.lambda_reg * sum(np.sum(W ** 2) for W in self.weights)

        # Combine losses
        total_loss = mse_loss + l2_loss

        return total_loss

    def compute_gradients(self, X, y, activations, pre_activations, max_norm=None):
        """
        Compute gradients of the loss with respect to weights and biases using backpropagation.

        Parameters:
        - X (np.ndarray): Input data, shape (num_samples, input_size).
        - y (np.ndarray): Target values, shape (num_samples, 1).
        - activations (dict): Activations from forward pass.
        - pre_activations (dict): Pre-activations from forward pass.
        - max_norm (float, optional): Maximum gradient norm for clipping.

        Returns:
        - gradients_W (list of np.ndarray): Gradients for weights.
        - gradients_b (list of np.ndarray): Gradients for biases.
        """
        gradients_W = []  # Store gradients of weights
        gradients_b = []  # Store gradients of biases
        m = X.shape[0]  # Number of training examples
        y = y.reshape(-1, 1)  # Ensure y is a column vector

        # Total number of layers
        L = len(self.weights)

        # Initialize gradient for the output layer
        output_layer = L
        aL = activations[output_layer]  # Output activations
        dA = (2 / m) * (aL - y)  # Gradient of MSE loss with respect to output activations

        # Backpropagation loop (from output layer to input layer)
        for i in reversed(range(1, L + 1)):
            if i == L:
                # Output layer: dZ = dA (linear activation, derivative is 1)
                dZ = dA
            else:
                # Hidden layers: dZ = dA * activation_derivative(z)
                z = pre_activations[i]
                da_dz = self.activation_derivative(z)  # Activation derivative
                dZ = dA * da_dz

            # Compute gradients for weights and biases
            a_prev = activations[i - 1]  # Activations from previous layer
            reg_term = 2 * self.lambda_reg * self.weights[i - 1]  # L2 regularization term
            dW = np.dot(a_prev.T, dZ) + reg_term  # Gradient for weights
            db = np.sum(dZ, axis=0, keepdims=True)  # Gradient for biases
            gradients_W.insert(0, dW)  # Insert weight gradients at the beginning
            gradients_b.insert(0, db)  # Insert bias gradients at the beginning

            if i > 1:
                # Update dA for the next layer (dA = dZ * W.T)
                W = self.weights[i - 1]
                dA = np.dot(dZ, W.T)

        # Compute the global norm of all gradients
        total_norm = np.sqrt(sum(np.sum(np.square(g)) for g in gradients_W + gradients_b))
#         print(f"Total gradient norm: {total_norm}") #I should remember to comment this line*****

        # Apply gradient clipping if max_norm is specified
        if max_norm is not None and total_norm > max_norm:
            scale_factor = max_norm / (total_norm + 1e-6)
            gradients_W = [g * scale_factor for g in gradients_W]
            gradients_b = [g * scale_factor for g in gradients_b]
#             print(f"Applied gradient clipping with scale factor: {scale_factor}") #I should remember to comment this line*****

        return gradients_W, gradients_b


def flatten_parameters(model):
    """
    Flatten the weights and biases of a neural network into a single 1D vector.

    Parameters:
    - model: Neural network model with weights and biases.

    Returns:
    - np.ndarray: Flattened parameter vector.
    """
    # Flatten all weight matrices and bias vectors, then concatenate them
    flat_weights = np.concatenate([W.flatten() for W in model.weights])
    flat_biases = np.concatenate([b.flatten() for b in model.biases])
    return np.concatenate([flat_weights, flat_biases])


def reshape_parameters(flat_params, model):
    """
    Reshape a flat parameter vector back into the original structure of weights and biases.

    Parameters:
    - flat_params (np.ndarray): Flattened parameter vector.
    - model: Neural network model with defined architecture.

    Returns:
    - tuple: (reshaped_weights, reshaped_biases), where each is a list of numpy arrays.
    """
    idx = 0  # Index to track the current position in the flat parameter vector
    reshaped_weights = []  # List to store reshaped weight matrices
    reshaped_biases = []   # List to store reshaped bias vectors

    # Determine layer sizes from the model
    layer_sizes = [model.input_size] + model.hidden_layer_sizes + [1]

    # Reshape weights
    for size_in, size_out in zip(layer_sizes[:-1], layer_sizes[1:]):
        size = size_in * size_out  # Total elements in the weight matrix
        reshaped_weights.append(flat_params[idx:idx + size].reshape(size_in, size_out))
        idx += size

    # Reshape biases
    for size_out in layer_sizes[1:]:
        size = size_out  # Total elements in the bias vector
        reshaped_biases.append(flat_params[idx:idx + size].reshape(1, size))
        idx += size

    return reshaped_weights, reshaped_biases



def loss_function(flat_params, X_train, y_train, model, max_norm=None):
    """
    Compute the loss and gradients for the model with respect to flattened parameters.

    Parameters:
    - flat_params (np.ndarray): Flattened parameter vector.
    - X_train (np.ndarray): Training input data.
    - y_train (np.ndarray): Training target data.
    - model: Neural network model.
    - max_norm (float, optional): Maximum gradient norm for clipping.

    Returns:
    - loss (float): Total loss (MSE + L2 regularization).
    - flat_gradients (np.ndarray): Flattened gradient vector.
    """
    # Reshape flat parameters into weights and biases
    model.weights, model.biases = reshape_parameters(flat_params, model)

    # Forward pass to compute activations and pre-activations
    activations, pre_activations = model.forward(X_train)

    # Compute loss (MSE + regularization)
    loss = model.compute_loss(activations, y_train)

    # Compute gradients
    gradients_W, gradients_b = model.compute_gradients(
        X_train, y_train, activations, pre_activations, max_norm=max_norm)

    # Flatten gradients for weights and biases
    flat_gradients_W = np.concatenate([grad.flatten() for grad in gradients_W])
    flat_gradients_b = np.concatenate([grad.flatten() for grad in gradients_b])
    flat_gradients = np.concatenate([flat_gradients_W, flat_gradients_b])

    # Record the loss value for monitoring
    model.loss_history.append(loss)
#     print(f'loss: {loss} \n') # I should comment this******

    return loss, flat_gradients


def train_nn(X_train, y_train, model, maxiter=200, function_tolerance=1e-4, max_norm=5.0):
    """
    Train a neural network model using the L-BFGS-B optimizer.

    Parameters:
    - X_train (np.ndarray): Training input data.
    - y_train (np.ndarray): Training target data.
    - model: Neural network model to be trained.
    - maxiter (int): Maximum number of iterations for the optimizer.
    - function_tolerance (float): Tolerance for stopping the optimization.
    - max_norm (float): Maximum gradient norm for clipping.

    Returns:
    - model: Trained model with updated weights and biases.
    - optimization_results (dict): A dictionary containing optimization details
    """
#     print("Training the model using L-BFGS-B optimizer...")

    loss_history = []  # Tracks all function evaluations (for debugging or fine-grained analysis)
    
    iteration_losses = []  # Tracks loss values for completed optimizer iterations


    def loss_func(wb):
         # Compute loss and gradients for the current parameters
        loss, grads = loss_function(wb, X_train, y_train, model, max_norm=max_norm)
        loss_history.append(loss)           
        return loss, grads

    # Flatten the initial weights and biases and Compute initial loss before optimization
    initial_params = flatten_parameters(model)
    initial_loss, _ = loss_function(initial_params, X_train, y_train, model)
    
    # Use a mutable integer to keep track of optimizer iterations in the callback
    # The callback is called once per iteration, ensuring alignment with result.nit
    iteration_count = [0]  # Using a list for closure property

    def iteration_callback(xk):
        # The callback is called after each optimizer iteration is completed
        iteration_count[0] += 1
        # Print the last computed loss. The final evaluation of this iteration
        # is the last element in loss_history.
        current_loss = loss_history[-1]
        iteration_losses.append(current_loss)
#         print(f"Iteration {iteration_count[0]}: Loss = {current_loss}")
    
    
    # Start timing before calling the optimizer function
    start_optimization_time = time.time()  
    
    # Call the optimizer with the callback
    result = minimize(
        fun=loss_func,
        x0=initial_params,
        method='L-BFGS-B',
        jac=True,  # Indicates that gradients are provided
        options={'maxiter': maxiter, 'ftol': function_tolerance},
        callback=iteration_callback  # This ensures iteration alignment
    )
    
    # End timing
    end_optimization_time = time.time()     
    optimization_time = end_optimization_time - start_optimization_time

    # Output optimization results
#     print("Optimization Success:", result.success)
#     print(f"Optimization message: {result.message}")
#     print(f"Reported Iterations (result.nit): {result.nit}")
#     print("Final Loss:", result.fun)
    
    # Document the optimization results for further info
    optimization_results = {
    'Optimization_Success': result.success,
    'Iteration_number': result.nit,
    'Optimization_message': result.message,
    'initial_loss': initial_loss,
    'Final_Loss': result.fun,
    'optimization_time_seconds': optimization_time}


    # Reshape optimized parameters back into model weights and biases
    model.weights, model.biases = reshape_parameters(result.x, model)
    
    # Append losses aligned with optimizer iterations
    model.loss_history = iteration_losses  # Use the iteration-level loss history

    return model , optimization_results



def mean_absolute_percentage_error(y_true, y_pred):
    """
    Calculate Mean Absolute Percentage Error (MAPE) between y_true and y_pred,
    excluding any zero values in y_true.

    Parameters:
    - y_true (array-like): Actual values.
    - y_pred (array-like): Predicted values.

    Returns:
    - float: MAPE value as a percentage.
    """
    # Convert inputs to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Mask to exclude zero values in y_true
    non_zero_mask = y_true != 0
    y_true_non_zero = y_true[non_zero_mask]
    y_pred_non_zero = y_pred[non_zero_mask]

    # Check if there are non-zero y_true values
    if len(y_true_non_zero) == 0:
        raise ValueError("All y_true values are zero. MAPE is undefined.")

    # Compute MAPE
    mape = np.mean(np.abs((y_true_non_zero - y_pred_non_zero) / y_true_non_zero)) * 100
    return mape
