# MLOPS_MLFLOW_TUTORIAL
This repo has a complete demonstration of performing experiment tracking using mlflow.



# MLflow Tutorial 📊

This guide shows how to use MLflow for experiment tracking.

---

## 1. Set Tracking URI
Tell MLflow where to store logs (local folder, remote server, or DB):
```python
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")  # or "file:/path/to/mlruns"
```

---

## 2. Set Experiment
Organize runs under a named experiment:
```python
mlflow.set_experiment("my_experiment")
```

---

## 3. Log Parameters
Record model or training parameters (like hyperparameters):
```python
mlflow.log_params({
    "n_estimators": 100,
    "max_depth": 5,
    "learning_rate": 0.01
})
```

---

## 4. Log Metrics
Track performance metrics (accuracy, loss, etc.):
```python
mlflow.log_metrics({
    "train_accuracy": 0.95,
    "val_accuracy": 0.92,
    "val_loss": 0.08
})
```

---

## 5. Example Run
```python
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y)

with mlflow.start_run():
    clf = RandomForestClassifier(n_estimators=100, max_depth=5)
    clf.fit(X_train, y_train)

    acc = clf.score(X_test, y_test)

    mlflow.log_params({"n_estimators": 100, "max_depth": 5})
    mlflow.log_metrics({"accuracy": acc})

    mlflow.sklearn.log_model(clf, "model")
```

---

## 6. Summary
- **`mlflow.set_tracking_uri()`** → choose where logs are stored  
- **`mlflow.set_experiment()`** → group runs under an experiment name  
- **`mlflow.log_params()`** → record hyperparameters/config values  
- **`mlflow.log_metrics()`** → record performance metrics

## 7. Example of Autolog()

import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Enable autologging
mlflow.sklearn.autolog()

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y)

with mlflow.start_run():
    clf = RandomForestClassifier(n_estimators=100, max_depth=5)
    clf.fit(X_train, y_train)
###🔎 Explanation
mlflow.sklearn.autolog() automatically logs:

Model parameters (like n_estimators, max_depth)

Training metrics (accuracy, loss, etc.)

Model artifacts (serialized model files)

Saves time and reduces boilerplate code.

Works across multiple ML libraries with their respective mlflow.<library>.autolog() calls.

