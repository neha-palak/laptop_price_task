import pickle
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression

# Generate dummy regression data
X, y = make_regression(n_samples=100, n_features=1, noise=0.1, random_state=42)

# Train a model
model = LinearRegression()
model.fit(X, y)

# Save model into .pkl file
with open("regression_model1.pkl", "wb") as f:
    pickle.dump(model, f)

print("regression_model1.pkl created successfully!")
