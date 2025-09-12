import pandas as pd
import numpy as np
import pickle
import os

def add_bias(X):
    return np.c_[np.ones((X.shape[0], 1)), X]

def predict(X, model):
    if isinstance(model, tuple):  # for polynomial model
        theta, degree = model
        X_poly = X.copy()
        for d in range(2, degree + 1):
            X_poly = np.c_[X_poly, X ** d]
        X_b = add_bias(X_poly)
    else:
        theta = model
        X_b = add_bias(X)
    return X_b @ theta

def evaluate(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot
    return mse, rmse, r2

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # project root

    # Default paths
    model_path = os.path.join(BASE_DIR, "models", "regression_model_final.pkl")
    data_path = os.path.join(BASE_DIR, "data", "processed_train.csv")
    metrics_output_path = os.path.join(BASE_DIR, "results", "train_metrics.txt")
    predictions_output_path = os.path.join(BASE_DIR, "results", "train_predictions.csv")

    # Load model
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Load data
    df = pd.read_csv(data_path)
    X = df.drop("Price", axis=1).values
    y = df["Price"].values

    # Predictions
    preds = predict(X, model)

    # Save predictions (single column, no header)
    os.makedirs(os.path.dirname(predictions_output_path), exist_ok=True)
    pd.DataFrame(preds).to_csv(predictions_output_path, index=False, header=False)

    # Save metrics
    mse, rmse, r2 = evaluate(y, preds)
    os.makedirs(os.path.dirname(metrics_output_path), exist_ok=True)
    with open(metrics_output_path, "w") as f:
        f.write("Regression Metrics:\n")
        f.write(f"Mean Squared Error (MSE): {mse:.2f}\n")
        f.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}\n")
        f.write(f"R-squared (R²) Score: {r2:.2f}\n")

    print("Evaluation complete.")
    print(f"Predictions saved to {predictions_output_path}")
    print(f"Metrics saved to {metrics_output_path}")
