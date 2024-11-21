import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

from data import load_wine_quality

# Dataset preparation
X, y = load_wine_quality()
X_pool, X_test, y_pool, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Parameters
n_estimators_list = [1, 5, 10, 20, 50, 100]  # Number of estimators
R = 50  # Number of repetitions
tree_depths = [2, None]  # Tree complexities: fixed depth vs. fully grown

# Storage for results
results = {"bagging": {}, "boosting": {}}
for method in results:
    results[method] = {depth: {"bias2": [], "variance": [], "expected_error": []} for depth in tree_depths}

# Evaluate Bagging and Boosting
for method_name, ensemble_model in [("bagging", BaggingRegressor), ("boosting", AdaBoostRegressor)]:
    print(f"Processing {method_name.capitalize()}...")
    
    for depth in tree_depths:
        print(f"  Tree depth: {'Fixed (2)' if depth == 2 else 'Unlimited'}")
        for n_estimators in n_estimators_list:
            predictions = np.zeros((len(X_test), R))
            
            for r in range(R):
                # Sample training set
                X_train, _, y_train, _ = train_test_split(X_pool, y_pool, train_size=250, random_state=r)
                
                # Base learner
                base_learner = DecisionTreeRegressor(max_depth=depth)
                
                # Ensemble model
                model = ensemble_model(estimator=base_learner, n_estimators=n_estimators, random_state=r)
                model.fit(X_train, y_train)
                
                # Predict on test set
                predictions[:, r] = model.predict(X_test)
            
            # Compute bias, variance, and expected error
            avg_predictions = predictions.mean(axis=1)
            bias2 = np.mean((avg_predictions - y_test) ** 2)
            variance = np.mean(np.var(predictions, axis=1))
            expected_error = np.mean((predictions - y_test.reshape(-1, 1)) ** 2)

            # Store results
            results[method_name][depth]["bias2"].append(bias2)
            results[method_name][depth]["variance"].append(variance)
            results[method_name][depth]["expected_error"].append(expected_error)

# Plot results
for method_name, method_results in results.items():
    for depth, result in method_results.items():
        depth_label = "Fixed Depth (2)" if depth == 2 else "Unlimited Depth"
        
        plt.figure(figsize=(8, 6))
        plt.plot(n_estimators_list, result["bias2"], label="Bias^2", marker='o')
        plt.plot(n_estimators_list, result["variance"], label="Variance", marker='o')
        plt.plot(n_estimators_list, result["expected_error"], label="Expected Error", marker='o')
        plt.xlabel("Number of Estimators (n_estimators)")
        plt.ylabel("Error")
        plt.title(f"{method_name.capitalize()} with {depth_label}")
        plt.legend()
        plt.grid()
        plt.show()
