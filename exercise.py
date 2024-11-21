import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

from data import load_wine_quality


# Load dataset
X, y = load_wine_quality()

# Split into test and pool
X_pool, X_test, y_pool, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Parameters
N = 250  # Training size
R = 50   # Number of training repetitions
models = {
    "lasso": Lasso(alpha=0.1),
    "knn": KNeighborsRegressor(n_neighbors=5),
    "tree": DecisionTreeRegressor(max_depth=5)
}

# Storage variables
bias_squared = {}
variance = {}
expected_error = {}

for model_name, model in models.items():
    predictions = np.zeros((len(X_test), R))
    
    for r in range(R):
        # Sample training set
        X_train, _, y_train, _ = train_test_split(X_pool, y_pool, train_size=N, random_state=r)
        
        # Train model
        model.fit(X_train, y_train)
        
        # Predict on test set
        predictions[:, r] = model.predict(X_test)
    
    # Compute metrics
    avg_predictions = predictions.mean(axis=1)
    bias_squared[model_name] = np.mean((avg_predictions - y_test) ** 2)
    variance[model_name] = np.mean(np.var(predictions, axis=1))
    expected_error[model_name] = mean_squared_error(y_test, avg_predictions)

# Results
print("Bias^2:", bias_squared)
print("Variance:", variance)
print("Expected Error:", expected_error)
