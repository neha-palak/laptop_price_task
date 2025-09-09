import pandas as pd

def preprocess_data(input_path, output_path):
    df = pd.read_csv(input_path)

    # Clean 'Ram' and 'Weight'
    df["Ram"] = df["Ram"].str.replace("GB", "").astype(int)
    df["Weight"] = df["Weight"].str.replace("kg", "").astype(float)

    # Encode categorical variables manually
    for col in ["Company", "TypeName", "ScreenResolution", "Cpu", "Memory", "Gpu", "OpSys"]:
        df[col] = pd.factorize(df[col])[0]

    # Save processed data
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    preprocess_data("data/Laptop Price Raw.csv", "data/processed_train.csv")

