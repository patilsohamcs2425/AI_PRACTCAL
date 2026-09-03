import pandas as pd
import matplotlib.pyplot as plt

from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

# Load dataset
df = pd.read_csv("saleshourly.csv")

print("Dataset shape:", df.shape)

print("\nColumns:")
print(df.columns)

# Drug columns
drug_columns = [
    "M01AB",
    "M01AE",
    "N02BA",
    "N02BE",
    "N05B",
    "N05C",
    "R03",
    "R06"
]

# Keep only drug columns
transactions = df[drug_columns].copy()

# Convert sales quantities into 0 and 1
# 1 means the drug was sold during that hour
# 0 means the drug was not sold
transactions = transactions > 0

# Remove hours where no drug was sold
transactions = transactions[transactions.sum(axis=1) > 0]

print("\nNumber of transactions:", len(transactions))

# Find frequent itemsets
frequent_itemsets = apriori(
    transactions,
    min_support=0.05,
    use_colnames=True
)

# Sort by support
frequent_itemsets = frequent_itemsets.sort_values(
    by="support",
    ascending=False
)

print("\nFrequent Itemsets:")
print(frequent_itemsets)

# Generate association rules
rules = association_rules(
    frequent_itemsets,
    metric="confidence",
    min_threshold=0.30
)

# Keep useful columns
rules = rules[
    [
        "antecedents",
        "consequents",
        "support",
        "confidence",
        "lift"
    ]
]

# Sort rules by confidence
rules = rules.sort_values(
    by="confidence",
    ascending=False
)

print("\nAssociation Rules:")
print(rules)

# Display rules in an easier format
print("\nTop Association Rules:")

for index, row in rules.head(10).iterrows():

    antecedent = ", ".join(row["antecedents"])
    consequent = ", ".join(row["consequents"])

    print(
        antecedent,
        "->",
        consequent,
        "| Support:",
        round(row["support"], 3),
        "| Confidence:",
        round(row["confidence"], 3),
        "| Lift:",
        round(row["lift"], 3)
    )

# Graph of top 10 rules by confidence
top_rules = rules.head(10).copy()

rule_names = []

for index, row in top_rules.iterrows():

    antecedent = ", ".join(row["antecedents"])
    consequent = ", ".join(row["consequents"])

    rule_names.append(
        antecedent + " -> " + consequent
    )

plt.figure(figsize=(9, 6))

plt.barh(
    rule_names[::-1],
    top_rules["confidence"].values[::-1]
)

plt.title("Top Association Rules by Confidence")
plt.xlabel("Confidence")
plt.ylabel("Association Rule")

plt.tight_layout()
plt.show()
