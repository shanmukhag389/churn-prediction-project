import joblib

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")

print("Model Type:")
print(type(model))

print("\nScaler Type:")
print(type(scaler))

print("\nNumber of Features:")
print(len(feature_columns))

print("\nFirst 20 Features:")
for i, column in enumerate(feature_columns, start=1):
    print(f"{i}. {column}")