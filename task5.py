import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
df = pd.read_csv("heart.csv")

print("First 5 Rows:")
print(df.head())
print("\nDataset Info:")
print(df.info())
X = df.drop("target", axis=1)

y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
tree = DecisionTreeClassifier(random_state=42)

tree.fit(X_train, y_train)
y_pred = tree.predict(X_test)

print("\nDecision Tree Accuracy:")

print(accuracy_score(y_test, y_pred))
plt.figure(figsize=(20,10))

plot_tree(
    tree,
    feature_names=X.columns,
    class_names=["No Disease","Disease"],
    filled=True
)

plt.savefig("decision_tree.png")

plt.show()
tree_depth = DecisionTreeClassifier(
    max_depth=3,
    random_state=42
)

tree_depth.fit(X_train, y_train)

pred = tree_depth.predict(X_test)

print("\nDecision Tree Accuracy (max_depth=3):")

print(accuracy_score(y_test, pred))
max_depth=3
forest = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

forest.fit(X_train, y_train)
forest_pred = forest.predict(X_test)

print("\nRandom Forest Accuracy:")

print(accuracy_score(y_test, forest_pred))

importance = pd.Series(
    forest.feature_importances_,
    index=X.columns
)

print("\nFeature Importance")

print(importance.sort_values(ascending=False))

scores = cross_val_score(
    forest,
    X,
    y,
    cv=5
)

print("\nCross Validation Scores:")

print(scores)

print("Average Accuracy:", scores.mean())
