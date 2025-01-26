import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

np.random.seed(42)
scaler = StandardScaler()
df = pd.read_csv("iris_dataset.csv")

X = df.drop(columns=["species"])
X = scaler.fit_transform(X)
y = df["species"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

results = []
for k in range(1, 6):
    KNN = KNeighborsClassifier(n_neighbors=k)
    KNN.fit(X_train, y_train)
    y_pred = KNN.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1score = f1_score(y_test, y_pred, average="weighted")
    
    results.append({
        "k_value": k,
        "accuracy": np.round(accuracy * 100, 2),
        "precision": np.round(precision, 4),
        "recall": np.round(recall, 4),
        "f1_score": np.round(f1score, 4),
    })

results = json.dumps(results, indent=4)
with open("results.json", "w") as json_file:
    json_file.write(results)