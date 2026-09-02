import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("Crop_recommendation.csv")

print("Dataset shape:", df.shape)

print("\nColumns:")
print(df.columns)

# Input features
X = df[["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]]

# Target
y = df["label"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

# Feature scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Test different K values
k_values = [1, 3, 5, 7, 9, 11]
accuracies = []

for k in k_values:

    model = KNeighborsClassifier(n_neighbors=k)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    accuracies.append(accuracy * 100)

# Display accuracy for each K
print("\nAccuracy for different K values:")

for k, accuracy in zip(k_values, accuracies):
    print("K =", k, "Accuracy =", round(accuracy, 2), "%")

# Best K
best_index = accuracies.index(max(accuracies))
best_k = k_values[best_index]

print("\nBest K value:", best_k)
print("Best Accuracy:", round(accuracies[best_index], 2), "%")

# Train final KNN model
model = KNeighborsClassifier(n_neighbors=best_k)

model.fit(X_train, y_train)

# Predict test data
y_pred = model.predict(X_test)

# Final accuracy
final_accuracy = accuracy_score(y_test, y_pred)

print("\nFinal K-NN Accuracy:",
      round(final_accuracy * 100, 2), "%")

# Predict first 5 test samples
print("\nPredictions for first 5 test samples:")

for i in range(5):
    print(
        "Actual:", y_test.iloc[i],
        "Predicted:", y_pred[i]
    )

# Accuracy vs K graph
plt.figure(figsize=(7, 5))

plt.plot(k_values, accuracies, marker="o")

plt.title("K-NN Accuracy vs K")
plt.xlabel("K Value")
plt.ylabel("Accuracy (%)")

plt.xticks(k_values)
plt.grid(True)

plt.show()
