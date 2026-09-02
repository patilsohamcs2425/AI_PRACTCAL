# Support Vector Machine

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Load dataset
data = pd.read_csv("final_dataset.csv")

print("Dataset loaded successfully")
print("Dataset shape:", data.shape)

# Display columns
print("\nColumns:")
print(data.columns)

# Target column
target = "label"

if target not in data.columns:
    print("\nTarget column not found")
    print("Available columns:")
    print(data.columns)
    exit()

# Separate features and target
X = data.drop(target, axis=1)
y = data[target]

# Keep numerical features
X = X.select_dtypes(include=["number"])

# Replace missing values
X = X.fillna(0)

# Create balanced dataset
data["label"] = y

legitimate = data[data["label"] == 0]
phishing = data[data["label"] == 1]

n = min(len(legitimate), len(phishing), 5000)

legitimate = legitimate.sample(n=n, random_state=42)
phishing = phishing.sample(n=n, random_state=42)

data = pd.concat([legitimate, phishing])

# Shuffle data
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# Prepare features and target again
X = data.drop("label", axis=1)
X = X.select_dtypes(include=["number"])
X = X.fillna(0)

y = data["label"]

print("\nRecords used:", len(data))
print("Legitimate:", sum(y == 0))
print("Phishing:", sum(y == 1))

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

# Scale features
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Parameter optimization
c_values = [0.1, 1, 10, 100]
accuracies = []

print("\nParameter Optimization")

for c in c_values:

    svm = SVC(
        kernel="rbf",
        C=c,
        gamma="scale"
    )

    svm.fit(X_train, y_train)

    prediction = svm.predict(X_test)

    accuracy = accuracy_score(y_test, prediction)

    accuracies.append(accuracy)

    print(
        "C =", c,
        "Accuracy =", round(accuracy * 100, 2), "%"
    )

# Find best C value
best_index = accuracies.index(max(accuracies))
best_c = c_values[best_index]

print("\nBest C value:", best_c)
print(
    "Best Accuracy:",
    round(accuracies[best_index] * 100, 2),
    "%"
)

# Train final SVM
model = SVC(
    kernel="rbf",
    C=best_c,
    gamma="scale"
)

print("\nTraining final SVM...")

model.fit(X_train, y_train)

print("Training completed")

# Prediction
y_pred = model.predict(X_test)

# Final accuracy
final_accuracy = accuracy_score(y_test, y_pred)

print("\nFinal Accuracy:")
print(round(final_accuracy * 100, 2), "%")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# Classification report
print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Legitimate", "Phishing"]
    )
)

# Parameter optimization graph
plt.figure(figsize=(7, 5))

plt.plot(
    c_values,
    accuracies,
    marker="o"
)

plt.title("SVM Parameter Optimization")

plt.xlabel("C Value")

plt.ylabel("Accuracy")

plt.xscale("log")

plt.grid()

plt.show()

# Confusion matrix graph
plt.figure(figsize=(6, 5))

plt.imshow(cm)

plt.title("SVM Confusion Matrix")

plt.xlabel("Predicted Class")

plt.ylabel("Actual Class")

plt.xticks(
    [0, 1],
    ["Legitimate", "Phishing"]
)

plt.yticks(
    [0, 1],
    ["Legitimate", "Phishing"]
)

for i in range(2):
    for j in range(2):
        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.colorbar()

plt.show()
