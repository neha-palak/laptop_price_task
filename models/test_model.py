import pickle

# Load the model
with open('/Users/nehapalak/cs3410/Jahnavi_Neha_A1/laptop_price_task/models/regression_model_final.pkl', "rb") as f:
    model = pickle.load(f)

print(type(model))

print("Tuple length:", len(model))
print("Contents:", model)
