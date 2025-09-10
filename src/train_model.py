import pandas as pd
import numpy as np
import pickle
import os

def add_bias(X):
    return np.c_[np.ones((X.shape[0], 1)), X]

def linear_regression(X, y):
    X_b = add_bias(X)
    theta = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y
    return theta

def polynomial_features(X, degree=2):
    poly = X.copy()
    for d in range(2, degree + 1):
        poly = np.c_[poly, X ** d]
    return poly

def ridge_regression(X, y, lam=1.0):
    X_b = add_bias(X)
    n = X_b.shape[1]
    I = np.eye(n)
    I[0, 0] = 0  # don’t regularize bias
    theta = np.linalg.pinv(X_b.T @ X_b + lam * I) @ X_b.T @ y
    return theta

def evaluate(X, y, theta):
    X_b = add_bias(X)
    preds = X_b @ theta
    mse = np.mean((y - preds) ** 2)
    rmse = np.sqrt(mse)
    ss_res = np.sum((y - preds) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot
    return mse, rmse, r2

if __name__ == "__main__":
    df = pd.read_csv('/Users/nehapalak/cs3410/Jahnavi_Neha_A1/laptop_price_task/data/processed_train.csv')
    X = df.drop("Price", axis=1).values
    y = df["Price"].values

    BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # project root
    MODELS_DIR = os.path.join(BASE_DIR, "models")
    os.makedirs(MODELS_DIR, exist_ok=True)

    # 1. Linear Regression
    theta_lin = linear_regression(X, y)
    with open(os.path.join(MODELS_DIR, "regression_model1.pkl"), "wb") as f:
        pickle.dump(theta_lin, f)

    # 2. Polynomial Regression
    X_poly = polynomial_features(X, degree=2)
    theta_poly = linear_regression(X_poly, y)
    with open(os.path.join(MODELS_DIR, "regression_model2.pkl"), "wb") as f:
        pickle.dump((theta_poly, 2), f)  # save degree too

    # 3. Ridge Regression
    theta_ridge = ridge_regression(X, y, lam=10)
    with open(os.path.join(MODELS_DIR, "regression_model3.pkl"), "wb") as f:
        pickle.dump(theta_ridge, f)

    # Pick best model (example: based on R²)
    mse1, rmse1, r21 = evaluate(X, y, theta_lin)
    mse2, rmse2, r22 = evaluate(X_poly, y, theta_poly)
    mse3, rmse3, r23 = evaluate(X, y, theta_ridge)

    best = max([(r21, "lin", theta_lin),
                (r22, "poly", (theta_poly, 2)),
                (r23, "ridge", theta_ridge)], key=lambda x: x[0])

    with open(os.path.join(MODELS_DIR, "regression_model_final.pkl"), "wb") as f:
        pickle.dump(best[2], f)

    print("Models trained and saved. Best:", best[1], "R² =", best[0])
