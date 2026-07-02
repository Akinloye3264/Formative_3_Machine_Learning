import numpy as np
import matplotlib.pyplot as plt
from scipy.differentiate import derivative
 
# Data  (identical to the manual worksheet)
X = np.array([[1, 3], [4, 10]], dtype=float)   
Y = np.array([[5], [6]], dtype=float)          
 
m = np.array([[-1.0], [2.0]])                  
b = 1.0                                        
 
LEARNING_RATE = 0.001
NUM_ITERATIONS = 4
 
 
# Model and cost
def predict(X, m, b):
    """Return y_hat = X @ m + b.  Scalar b broadcasts to every row."""
    return X @ m + b
 
 
def mse(Y, X, m, b):
    """Mean Squared Error: (1/n) * sum( (y - y_hat)^2 )."""
    return np.mean((Y - predict(X, m, b)) ** 2)
 
 
# Gradients via SciPy's numerical derivative (assignment requirement)
def scipy_derivative(func, at):
    return derivative(func, at).df
 
 
def gradient_wrt_m(Y, X, m, b, j):
    def cost_of_mj(mj_values):
        mj_values = np.asarray(mj_values, dtype=float)
        costs = np.empty_like(mj_values)
        for idx, value in np.ndenumerate(mj_values):
            m_temp = m.copy()
            m_temp[j, 0] = value
            costs[idx] = mse(Y, X, m_temp, b)
        return costs
    return scipy_derivative(cost_of_mj, float(m[j, 0]))
 
 
def gradient_wrt_b(Y, X, m, b):
    """d(MSE)/d(b): treat MSE as a function of the scalar b, differentiate.
    Vectorised the same way so the returned shape matches SciPy's input."""
    def cost_of_b(b_values):
        b_values = np.asarray(b_values, dtype=float)
        costs = np.empty_like(b_values)
        for idx, value in np.ndenumerate(b_values):
            costs[idx] = mse(Y, X, m, float(value))
        return costs
    return scipy_derivative(cost_of_b, float(b))
 
 
# Training loop
m_history, b_history, error_history = [], [], []
 
m_history.append(m.flatten().copy())
b_history.append(b)
error_history.append(mse(Y, X, m, b))
print(f"Iteration 0 (start): "
      f"m = [{m[0,0]:.4f}, {m[1,0]:.4f}], "
      f"b = {b:.4f}, MSE = {error_history[-1]:.4f}")
 
for iteration in range(NUM_ITERATIONS):
    # 1) gradients at the current parameters
    grad_m = np.array([[gradient_wrt_m(Y, X, m, b, 0)],
                       [gradient_wrt_m(Y, X, m, b, 1)]])
    grad_b = gradient_wrt_b(Y, X, m, b)
 
    # 2) simultaneous update:  param = param - alpha * gradient
    m = m - LEARNING_RATE * grad_m
    b = b - LEARNING_RATE * grad_b
 
    # 3) record for plotting / inspection
    m_history.append(m.flatten().copy())
    b_history.append(b)
    error_history.append(mse(Y, X, m, b))
 
    print(f"Iteration {iteration + 1}: "
          f"m = [{m[0,0]:.4f}, {m[1,0]:.4f}], "
          f"b = {b:.4f}, MSE = {error_history[-1]:.4f}")
 
 
# Final results
print("\n Final result")
print("Final m:", np.round(m.flatten(), 4))
print("Final b:", round(b, 4))
print("Final MSE:", round(mse(Y, X, m, b), 4))
print("Predictions:", np.round(predict(X, m, b).flatten(), 4))
 
 
# Plots
m_history = np.array(m_history)
iterations = range(0, NUM_ITERATIONS + 1)   # 0 = start, then 1..4
 
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
 
# Plot 1: how m1, m2, and b change over iterations
axes[0].plot(iterations, m_history[:, 0], marker='o', label='m1', color='blue')
axes[0].plot(iterations, m_history[:, 1], marker='o', label='m2', color='orange')
axes[0].plot(iterations, b_history, marker='s', label='b',
             color='green', linestyle='--')
axes[0].set_title('m and b over Iterations')
axes[0].set_xlabel('Iteration')
axes[0].set_ylabel('Value')
axes[0].set_xticks(list(iterations))
axes[0].legend()
axes[0].grid(True)
 
# Plot 2: how the MSE error changes over iterations
axes[1].plot(iterations, error_history, marker='o', color='purple')
axes[1].set_title('MSE Error over Iterations')
axes[1].set_xlabel('Iteration')
axes[1].set_ylabel('MSE')
axes[1].set_xticks(list(iterations))
axes[1].grid(True)
 
plt.tight_layout()
plt.show()