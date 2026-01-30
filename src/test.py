import mlflow
print("Printing  tracking URL scheme below")
print(mlflow.get_tracking_uri())

print("\n")

mlflow.set_tracking_uri("http://127.0.0.1:5000")
print("Printing  tracking URL scheme below after setting the tracking URI")
print(mlflow.get_tracking_uri())
print("\n")


