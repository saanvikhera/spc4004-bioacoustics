# =============================================================================
# Marine Bioacoustics Species Classification
# Dataset: SM_1_WMD_Features_and_Labels.csv
# Task: Multi-class classification of marine mammal species from acoustic features
# Initial version (AI-generated starting point)
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# ── 1. Load Data ──────────────────────────────────────────────────────────────
df = pd.read_csv("SM_1_WMD_Features_and_Labels.csv")

# Feature columns are named '0' through '127'
feature_cols = [str(i) for i in range(128)]
X = df[feature_cols]
y = df["SPECIES"]

# ── 2. Handle Missing Values ──────────────────────────────────────────────────
X = X.fillna(X.median())

# ── 3. Encode Labels ──────────────────────────────────────────────────────────
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# ── 4. Train/Test Split ───────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# ── 5. Feature Scaling ────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ── 6. PCA for Dimensionality Reduction (optional visualisation) ──────────────
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

plt.figure(figsize=(9, 6))
scatter = plt.scatter(
    X_train_pca[:, 0], X_train_pca[:, 1],
    c=y_train, cmap="tab10", alpha=0.4, s=10
)
plt.title("PCA – First Two Components (Training Set)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.colorbar(scatter, label="Species (encoded)")
plt.tight_layout()
plt.savefig("pca_visualisation.png", dpi=150)
plt.show()

# ── 7. Train and Evaluate Models ─────────────────────────────────────────────
models = {
    "Random Forest":      RandomForestClassifier(n_estimators=100, random_state=42),
    "KNN (k=5)":          KNeighborsClassifier(n_neighbors=5),
    "SVM (RBF)":          SVC(kernel="rbf", random_state=42),
    "Decision Tree":      DecisionTreeClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
}

results = {}
for name, model in models.items():
    # Use PCA-reduced features for KNN and SVM (speed); full features for tree-based
    if name in ["KNN (k=5)", "SVM (RBF)", "Logistic Regression"]:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    print(f"\n{'='*50}")
    print(f"  {name}  |  Accuracy: {acc:.4f}")
    print(f"{'='*50}")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

# ── 8. Accuracy Comparison Bar Chart ─────────────────────────────────────────
plt.figure(figsize=(9, 5))
bars = plt.bar(results.keys(), results.values(), color="steelblue", edgecolor="black")
plt.ylim(0, 1)
plt.ylabel("Test Accuracy")
plt.title("Model Accuracy Comparison – Marine Bioacoustics Species Classification")
plt.xticks(rotation=20, ha="right")
for bar, acc in zip(bars, results.values()):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
             f"{acc:.3f}", ha="center", va="bottom", fontsize=9)
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150)
plt.show()

# ── 9. Confusion Matrix for Best Model (Random Forest) ───────────────────────
rf = models["Random Forest"]
y_pred_rf = rf.predict(X_test)
cm = confusion_matrix(y_test, y_pred_rf)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
fig, ax = plt.subplots(figsize=(10, 8))
disp.plot(ax=ax, colorbar=True, cmap="Blues")
plt.title("Confusion Matrix – Random Forest")
plt.tight_layout()
plt.savefig("confusion_matrix_rf.png", dpi=150)
plt.show()

# ── 10. Feature Importance (Random Forest) ───────────────────────────────────
importances = rf.feature_importances_
top_n = 20
top_idx = np.argsort(importances)[::-1][:top_n]

plt.figure(figsize=(10, 5))
plt.bar(range(top_n), importances[top_idx], color="teal", edgecolor="black")
plt.xticks(range(top_n), [feature_cols[i] for i in top_idx], rotation=45, ha="right")
plt.title(f"Top {top_n} Feature Importances – Random Forest")
plt.ylabel("Importance")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
plt.show()

print("\nDone. Figures saved to working directory.")
