"""
Student Stress Detector — Exploratory Data Analysis & Model Insights
Run: python notebooks/eda_analysis.py
Generates analysis charts as PNG files in notebooks/charts/
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

DATA_DIR  = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
CHART_DIR = os.path.join(os.path.dirname(__file__), "charts")
os.makedirs(CHART_DIR, exist_ok=True)

COLORS = {"Low": "#1D9E75", "Moderate": "#EF9F27", "High": "#E24B4A"}
PALETTE = ["#1D9E75", "#EF9F27", "#E24B4A"]


def save(fig, name):
    path = os.path.join(CHART_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {name}")


# ─── Load data ───────────────────────────────────────────────

df3 = pd.read_csv(os.path.join(DATA_DIR, "StressLevelDataset.csv"))
df2 = pd.read_csv(os.path.join(DATA_DIR, "Stress_Dataset.csv"))
df1 = pd.read_csv(os.path.join(DATA_DIR, "Student_Mental_health.csv"))
importances = joblib.load(os.path.join(MODEL_DIR, "feature_importances.pkl"))
model = joblib.load(os.path.join(MODEL_DIR, "primary_model.pkl"))

df3["stress_label"] = df3["stress_level"].map({0:"Low", 1:"Moderate", 2:"High"})
label_order = ["Low", "Moderate", "High"]

print("\n" + "=" * 55)
print("  STUDENT STRESS DETECTOR — EDA & MODEL ANALYSIS")
print("=" * 55 + "\n")


# ── Chart 1: Stress level distribution ──────────────────────
print("[1/7] Stress distribution...")
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

counts = df3["stress_label"].value_counts().reindex(label_order)
bars = axes[0].bar(label_order, counts.values,
                   color=PALETTE, edgecolor="white", linewidth=0.8, width=0.6)
axes[0].set_title("Stress level distribution (primary dataset)", fontsize=12, pad=12)
axes[0].set_ylabel("Number of students")
axes[0].set_ylim(0, counts.max() * 1.15)
for bar, v in zip(bars, counts.values):
    axes[0].text(bar.get_x() + bar.get_width()/2, v + 8, str(v),
                 ha="center", va="bottom", fontsize=11, fontweight="bold")
axes[0].spines[["top","right"]].set_visible(False)

stress2 = df2["Which type of stress do you primarily experience?"].str.split(" - ").str[0]
c2 = stress2.value_counts()
wedge_colors = ["#1D9E75", "#EF9F27", "#E24B4A"]
axes[1].pie(c2.values, labels=[l.replace("(", "\n(") for l in c2.index],
            colors=wedge_colors[:len(c2)], autopct="%1.1f%%",
            startangle=140, pctdistance=0.8,
            wedgeprops={"edgecolor":"white","linewidth":1.5})
axes[1].set_title("Stress type distribution (survey dataset)", fontsize=12, pad=12)

fig.suptitle("Dataset Overview", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
save(fig, "1_stress_distribution.png")


# ── Chart 2: Feature importances ────────────────────────────
print("[2/7] Feature importances...")
top_n = importances.head(12)
fig, ax = plt.subplots(figsize=(10, 5))
colors = [PALETTE[2] if v > top_n.mean() else "#B4B2A9" for v in top_n.values]
bars = ax.barh(top_n.index[::-1], top_n.values[::-1],
               color=colors[::-1], edgecolor="white", linewidth=0.5)
ax.set_xlabel("Feature importance (Gini impurity reduction)")
ax.set_title("Top 12 features driving stress prediction", fontsize=13, pad=12)
ax.spines[["top","right","left"]].set_visible(False)
ax.tick_params(left=False)
for bar, v in zip(bars, top_n.values[::-1]):
    ax.text(v + 0.002, bar.get_y() + bar.get_height()/2,
            f"{v:.3f}", va="center", fontsize=9, color="#5F5E5A")
plt.tight_layout()
save(fig, "2_feature_importances.png")


# ── Chart 3: Key feature distributions by stress level ──────
print("[3/7] Feature vs stress level boxplots...")
key_features = ["anxiety_level", "sleep_quality", "depression",
                "social_support", "academic_performance", "study_load"]
fig, axes = plt.subplots(2, 3, figsize=(13, 7))
axes = axes.flatten()

for i, feat in enumerate(key_features):
    groups = [df3[df3["stress_level"] == lv][feat].values for lv in [0, 1, 2]]
    bp = axes[i].boxplot(groups, patch_artist=True, notch=False,
                          medianprops={"color":"white","linewidth":2},
                          whiskerprops={"color":"#888780"},
                          capprops={"color":"#888780"},
                          flierprops={"marker":"o","markersize":3,"alpha":0.4})
    for patch, color in zip(bp["boxes"], PALETTE):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)
    axes[i].set_xticklabels(label_order)
    axes[i].set_title(feat.replace("_", " ").title(), fontsize=11)
    axes[i].spines[["top","right"]].set_visible(False)

fig.suptitle("Feature distributions by stress level", fontsize=14,
             fontweight="bold", y=1.01)
legend_handles = [mpatches.Patch(color=c, label=l)
                  for c, l in zip(PALETTE, label_order)]
fig.legend(handles=legend_handles, loc="lower right", ncol=3, fontsize=10)
plt.tight_layout()
save(fig, "3_feature_by_stress.png")


# ── Chart 4: Correlation heatmap ────────────────────────────
print("[4/7] Correlation heatmap...")
corr_cols = ["anxiety_level","self_esteem","depression","sleep_quality",
             "social_support","academic_performance","study_load",
             "peer_pressure","bullying","stress_level"]
corr = df3[corr_cols].corr()

fig, ax = plt.subplots(figsize=(9, 7))
im = ax.imshow(corr.values, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
ax.set_xticks(range(len(corr_cols)))
ax.set_yticks(range(len(corr_cols)))
labels = [c.replace("_", "\n") for c in corr_cols]
ax.set_xticklabels(labels, fontsize=8)
ax.set_yticklabels(labels, fontsize=8)
for i in range(len(corr_cols)):
    for j in range(len(corr_cols)):
        v = corr.values[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                fontsize=7, color="black" if abs(v) < 0.5 else "white")
plt.colorbar(im, ax=ax, shrink=0.8)
ax.set_title("Feature correlation matrix", fontsize=13, pad=12)
plt.tight_layout()
save(fig, "4_correlation_heatmap.png")


# ── Chart 5: Confusion matrix ────────────────────────────────
print("[5/7] Confusion matrix...")
X = df3.drop(["stress_level","stress_label"], axis=1, errors="ignore")
y = df3["stress_level"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=["Low","Moderate","High"])
disp.plot(ax=ax, cmap="Greens", colorbar=False)
ax.set_title("Confusion matrix (ensemble model, test set)", fontsize=12, pad=12)
plt.tight_layout()
save(fig, "5_confusion_matrix.png")


# ── Chart 6: Dataset 1 — mental health breakdown ─────────────
print("[6/7] Student mental health breakdown...")
yn = {"Yes": 1, "No": 0}
df1["depression"] = df1["Do you have Depression?"].map(yn).fillna(0)
df1["anxiety"]    = df1["Do you have Anxiety?"].map(yn).fillna(0)
df1["panic"]      = df1["Do you have Panic attack?"].map(yn).fillna(0)
df1["gender"]     = df1["Choose your gender"]

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
conditions = [("depression", "Depression"), ("anxiety", "Anxiety"), ("panic", "Panic attacks")]

for i, (col, label) in enumerate(conditions):
    counts_m = df1[df1["gender"] == "Male"][col].value_counts().reindex([0, 1]).fillna(0)
    counts_f = df1[df1["gender"] == "Female"][col].value_counts().reindex([0, 1]).fillna(0)
    x = np.array([0, 1])
    width = 0.35
    axes[i].bar(x - width/2, counts_m.values, width, label="Male",
                color="#378ADD", alpha=0.85, edgecolor="white")
    axes[i].bar(x + width/2, counts_f.values, width, label="Female",
                color="#D4537E", alpha=0.85, edgecolor="white")
    axes[i].set_xticks(x)
    axes[i].set_xticklabels(["No", "Yes"])
    axes[i].set_title(label, fontsize=12)
    axes[i].set_ylabel("Students")
    axes[i].spines[["top","right"]].set_visible(False)
    if i == 0:
        axes[i].legend(fontsize=9)

fig.suptitle("Mental health indicators by gender (Dataset 1, n=101)",
             fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
save(fig, "6_mental_health_gender.png")


# ── Chart 7: Learning curve ───────────────────────────────────
print("[7/7] Learning curve...")
rf_simple = RandomForestClassifier(n_estimators=100, random_state=42)
train_sizes, train_scores, val_scores = learning_curve(
    rf_simple, X, y, cv=5, scoring="accuracy",
    train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1)

fig, ax = plt.subplots(figsize=(8, 5))
train_mean = train_scores.mean(axis=1)
train_std  = train_scores.std(axis=1)
val_mean   = val_scores.mean(axis=1)
val_std    = val_scores.std(axis=1)

ax.plot(train_sizes, train_mean, "o-", color="#1D9E75", label="Training accuracy")
ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                alpha=0.15, color="#1D9E75")
ax.plot(train_sizes, val_mean, "o-", color="#E24B4A", label="Validation accuracy")
ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                alpha=0.15, color="#E24B4A")
ax.set_xlabel("Training set size")
ax.set_ylabel("Accuracy")
ax.set_ylim(0.6, 1.05)
ax.set_title("Learning curve — Random Forest on StressLevelDataset", fontsize=12, pad=12)
ax.legend(fontsize=10)
ax.spines[["top","right"]].set_visible(False)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
save(fig, "7_learning_curve.png")


print(f"\n✓ All 7 charts saved to: {CHART_DIR}/")
print("\nKey findings:")
print(f"  Primary model test accuracy : {(y_pred == y_test.values).mean()*100:.1f}%")
print(f"  Top predictor               : {importances.index[0]}")
print(f"  Dataset sizes used          : 1100 + 843 + 101 = 2044 student records")
