import argparse
import pandas as pd
import numpy as np
import pickle

def add_bias(X):
    return np.c_[np.ones((X.shape[0], 1)), X]

def polynomial_features(X, degree=2):
    poly = X.copy()
    for d in range(2, degree + 1):
        poly = np.c_[poly, X ** d]
    return poly

def evaluate(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot
    return mse, rmse, r2

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="Path to saved model .pkl file")
    parser.add_argument("--data_path", required=True, help="Path to CSV with features + Price")
    parser.add_argument("--metrics_output_path", required=True, help="Path to save metrics .txt")
    parser.add_argument("--predictions_output_path", required=True, help="Path to save predictions .csv")
    args = parser.parse_args()

    # Load model
    with open(args.model_path, "rb") as f:
        model = pickle.load(f)

    # Load dataset
    df = pd.read_csv(args.data_path)
    X = df.drop("Price", axis=1).values
    y = df["Price"].values

    # Check if polynomial model
    if isinstance(model, tuple):  # (theta, degree)
        theta, degree = model
        X = polynomial_features(X, degree)
    else:
        theta = model

    # Add bias
    X_b = add_bias(X)
    y_pred = X_b @ theta

    # Save predictions (no header, one value per line)
    pd.DataFrame(y_pred).to_csv(args.predictions_output_path, index=False, header=False)

    # Evaluate
    mse, rmse, r2 = evaluate(y, y_pred)

    # Save metrics
    with open(args.metrics_output_path, "w") as f:
        f.write("Regression Metrics:\n")
        f.write(f"Mean Squared Error (MSE): {mse:.2f}\n")
        f.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}\n")
        f.write(f"R-squared (R²) Score: {r2:.2f}\n")



# run the following command to execute this program:

# python3 src/predict.py \
# --model_path models/regression_model_final.pkl \
# --data_path data/processed_train.csv \
# --metrics_output_path results/train_metrics.txt \
# --predictions_output_path results/train_predictions.csv