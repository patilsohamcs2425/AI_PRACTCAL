import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("confidence_features.csv")

print("Dataset shape:", df.shape)

print("\nColumns:")
print(df.columns)

# Remove missing values
df = df.dropna()

# Target column
target = "confidence_label"

# Separate input and target
X = df.drop(columns=[target])
y = df[target]

# Convert categorical features into numbers
categorical_columns = X.select_dtypes(include=["object"]).columns

for column in categorical_columns:
    encoder = LabelEncoder()
    X[column] = encoder.fit_transform(X[column])

# Convert target labels into numbers
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y)

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

# Create Gaussian Naive Bayes model
model = GaussianNB()

# Train model
model.fit(X_train, y_train)

# Predict test data
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nNaive Bayes Accuracy:",
      round(accuracy * 100, 2), "%")

# Calculate class probabilities
probabilities = model.predict_proba(X_test)

# Display probabilities for first 5 test records
print("\nClass Probabilities:")

for i in range(5):

    print("\nTest Record", i + 1)

    for j in range(len(target_encoder.classes_)):
        print(
            target_encoder.classes_[j],
            ":",
            round(probabilities[i][j] * 100, 2),
            "%"
        )

# Classification report
print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=target_encoder.classes_,
        zero_division=0
    )
)

# Probability graph for first test record
first_probabilities = probabilities[0] * 100
class_names = target_encoder.classes_

plt.figure(figsize=(7, 5))

bars = plt.bar(
    class_names,
    first_probabilities
)

plt.title("Naive Bayes Class Probabilities")
plt.xlabel("Confidence Class")
plt.ylabel("Probability (%)")

plt.ylim(0, 100)

# Display probability values
for bar, value in zip(bars, first_probabilities):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 2,
        f"{value:.2f}%",
        ha="center"
    )

plt.show()
