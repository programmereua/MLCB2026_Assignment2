import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

# load data
df = pd.read_csv("data/heart_disease.csv", na_values=["?"])

target = "num"
continuous  = ["age", "trestbps", "chol", "thalach", "oldpeak"]
binary      = ["sex", "fbs", "exang"]
categorical = ["cp", "restecg", "slope", "thal"]
ordinal     = ["ca"]
features = continuous + binary + categorical + ordinal


# 1) Dataset Overview and Descriptive Statistics

print("Shape:", df.shape)
print(df.head())
print(df.dtypes)
print(df.describe())

# missing values
print("\nMissing values:")
print(df.isna().sum())

# duplicates
print("\nDuplicates:", df.duplicated().sum())

# class balance
print("\nClass balance:")
print(df[target].value_counts())

plt.figure()
df[target].value_counts().plot(kind="bar", color=["steelblue", "tomato"])
plt.title("Class balance")
plt.xlabel("num")
plt.ylabel("count")
plt.savefig("class_balance.png")
plt.close()


# 2) Feature Assessment and Visualization

# boxplots of continuous features by class
fig, axes = plt.subplots(1, 5, figsize=(20, 4))
for ax, col in zip(axes, continuous):
    sns.boxplot(x=target, y=col, data=df, ax=ax)
    ax.set_title(col)
plt.savefig("boxplots.png")
plt.close()

# histograms of continuous features by class
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()
for ax, col in zip(axes, continuous):
    df[df[target] == 0][col].hist(ax=ax, bins=20, alpha=0.6, label="0")
    df[df[target] == 1][col].hist(ax=ax, bins=20, alpha=0.6, label="1")
    ax.set_title(col)
    ax.legend()
plt.savefig("histograms.png")
plt.close()

# count plots for discrete features
fig, axes = plt.subplots(2, 4, figsize=(20, 8))
axes = axes.flatten()
for ax, col in zip(axes, binary + categorical + ordinal):
    sns.countplot(x=col, hue=target, data=df, ax=ax)
    ax.set_title(col)
plt.savefig("countplots.png")
plt.close()

# correlation heatmap (Spearman, handles ordinal + non-linear)
corr = df[features + [target]].corr(method="spearman")
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Spearman correlation")
plt.savefig("correlation.png")
plt.close()

print("\nCorrelation with target:")
print(corr[target].drop(target).abs().sort_values(ascending=False))

# PCA (impute + scale only for visualization)
X = SimpleImputer(strategy="median").fit_transform(df[features])
X = StandardScaler().fit_transform(X)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(7, 6))
for cls, color in zip([0, 1], ["steelblue", "tomato"]):
    mask = df[target] == cls
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], c=color, alpha=0.6, label=f"num={cls}")
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
plt.legend()
plt.title("PCA 2D")
plt.savefig("pca.png")
plt.close()

print("\nPC1+PC2 explained variance:", pca.explained_variance_ratio_.sum())
print("Done.")