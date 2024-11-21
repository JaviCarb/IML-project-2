import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

from data import load_wine_quality

# Dataset preparation
X, y = load_wine_quality()
X_pool, X_test, y_pool, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Parameters
N = 250  # Training size
R = 50   # Number of repetitions

# Define models and their complexity parameters
models = {
    "lasso": {
        "model": Lasso,
        "params": [0.001, 0.005, 0.01, 0.05, 0.1, 1, 10],  # Regularization strength (alpha in scikit-learn)
        "param_name": "alpha"
    },
    "knn": {
        "model": KNeighborsRegressor,
        "params": [1, 2, 3, 5, 10, 30, 50, 100, 150, 200, 250],  # Number of neighbors
        "param_name": "n_neighbors"
    },
    "tree": {
        "model": DecisionTreeRegressor,
        "params": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, None],  # Max tree depth
        "param_name": "max_depth"
    }
}

# Storage for results
results = {method: {"bias2": [], "variance": [], "expected_error": []} for method in models.keys()}

for model_name, config in models.items():
    print(f"Processing {model_name}...")
    for param in config["params"]:
        predictions = np.zeros((len(X_test), R))
        
        for r in range(R):
            # Sample training set
            X_train, _, y_train, _ = train_test_split(X_pool, y_pool, train_size=N, random_state=r)
            
            # Initialize model with specific parameter
            model = config["model"](**{config["param_name"]: param})
            model.fit(X_train, y_train)
            
            # Predict on test set
            predictions[:, r] = model.predict(X_test)
        
        # Compute bias, variance, and expected error
        avg_predictions = predictions.mean(axis=1)
        bias2 = np.mean((avg_predictions - y_test) ** 2)
        variance = np.mean(np.var(predictions, axis=1))  
        expected_error = np.mean((predictions - y_test.reshape(-1, 1)) ** 2)  # Reshape for broadcasting

        # Store results
        results[model_name]["bias2"].append(bias2)
        results[model_name]["variance"].append(variance)
        results[model_name]["expected_error"].append(expected_error)


# Print results
for model_name, result in results.items():
    print(f"{model_name.capitalize()}:")
    for param, bias2, variance, expected_error in zip(models[model_name]["params"], result["bias2"], result["variance"], result["expected_error"]):
        print(f"  {models[model_name]['param_name']}={param}: Bias^2={bias2:.2f}, Variance={variance:.2f}, Expected Error={expected_error:.2f}")


# Plot results
for model_name, result in results.items():
    params = models[model_name]["params"]
    
    plt.figure(figsize=(8, 6))
    plt.plot(params, result["bias2"], label="Bias^2", marker='o')
    plt.plot(params, result["variance"], label="Variance", marker='o')
    plt.plot(params, result["expected_error"], label="Expected Error", marker='o')
    plt.xlabel(models[model_name]["param_name"])
    plt.ylabel("Error")
    plt.title(f"{model_name.capitalize()} Bias-Variance Tradeoff")
    plt.legend()
    plt.grid()
    plt.show()


# Print the variance of the dataset (to explain why, in extreme cases, the error is 0.75)
print("Variance of the dataset:", np.var(y))