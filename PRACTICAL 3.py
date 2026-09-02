# Decision Tree Learning
# AI Practical

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# 1. Load Dataset

data = pd.read_csv("Indian_Student_Placement_Dataset_2025.csv")

print("First 5 records:")
print(data.head())

print("\nDataset Information:")
print(data.info())

print("\nDataset Columns:")
print(data.columns)

# 2. Remove unnecessary ID column

if "Student_ID" in data.columns:
    data = data.drop("Student_ID", axis=1)

if "student_id" in data.columns:
    data = data.drop("student_id", axis=1)

# 3. Display missing values

print("\nMissing Values:")
print(data.isnull().sum())

# 4. Convert categorical columns into numbers
data = data.drop(["company_type", "package_lpa"], axis=1)

# 5. Convert categorical columns into numbers

from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

for column in data.columns:
    if data[column].dtype == "object":
        data[column] = encoder.fit_transform(data[column].astype(str))

# 6. Set Target Column

target = "placed"

# 7. Separate Features and Target

X = data.drop(target, axis=1)
y = data[target]

print("\nFeatures:")
print(X.columns)

print("\nTarget:")
print(target)

# 8. Split Dataset

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

# 9. Create Decision Tree

model = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=5,
    random_state=42
)

# 10. Train Model

model.fit(X_train, y_train)

print("\nDecision Tree successfully trained.")

# 11. Prediction

y_pred = model.predict(X_test)

print("\nPredicted Values:")
print(y_pred[:20])

# 12. Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nDecision Tree Accuracy:")
print(round(accuracy * 100, 2), "%")

# 13. Confusion Matrix

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# 14. Classification Report

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# 15. Visualize Decision Tree

plt.figure(figsize=(20, 10))

plot_tree(
    model,
    feature_names=X.columns,
    class_names=["Not Placed", "Placed"],
    filled=True,
    rounded=True,
    fontsize=9
)

plt.title("Decision Tree - Student Placement Prediction")

plt.show()
