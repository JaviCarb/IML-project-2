import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

from data import load_wine_quality


# Dataset preparation
X, y = load_wine_quality()
X_pool, X_test, y_pool, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Parameters
training_sizes = [20, 50, 100, 150, 200, 250, 300, 500, 1000, 1500, 2000, 3000]
R = 50  # Number of repetitions
fixed_params = {
    "lasso": {"model": Lasso(alpha=0.01)},
    "knn": {"model": KNeighborsRegressor(n_neighbors=10)},
    "tree": {"model": DecisionTreeRegressor(max_depth=2)},
    "tree_unlimited": {"model": DecisionTreeRegressor(max_depth=None)}
}

# Storage for results
results = {method: {"bias2": [], "variance": [], "expected_error": []} for method in fixed_params.keys()}

for model_name, config in fixed_params.items():
    print(f"Processing {model_name}...")
    model = config["model"]
    
    for N in training_sizes:
        predictions = np.zeros((len(X_test), R))
        
        for r in range(R):
            # Sample training set
            X_train, _, y_train, _ = train_test_split(X_pool, y_pool, train_size=N, random_state=r)
            
            # Train model
            model.fit(X_train, y_train)
            
            # Predict on test set
            predictions[:, r] = model.predict(X_test)
        
        # Compute bias, variance, and expected error
        avg_predictions = predictions.mean(axis=1)
        bias2 = np.mean((avg_predictions - y_test) ** 2)
        variance = np.mean(np.var(predictions, axis=1))
        expected_error = np.mean((predictions - y_test.reshape(-1, 1)) ** 2)

        pointwise_variances = np.var(predictions, axis=1)
        print(f"Pointwise variances: {pointwise_variances}")
        print(f"Mean variance: {variance}")


        # Store results
        results[model_name]["bias2"].append(bias2)
        results[model_name]["variance"].append(variance)
        results[model_name]["expected_error"].append(expected_error)

# Plot results
for model_name, result in results.items():
    plt.figure(figsize=(8, 6))
    plt.plot(training_sizes, result["bias2"], label="Bias^2", marker='o')
    plt.plot(training_sizes, result["variance"], label="Variance", marker='o')
    plt.plot(training_sizes, result["expected_error"], label="Expected Error", marker='o')
    plt.xlabel("Training Sample Size (N)")
    plt.ylabel("Error")
    plt.title(f"{model_name.capitalize()} Error vs. Sample Size")
    plt.legend()
    plt.grid()
    plt.show()
