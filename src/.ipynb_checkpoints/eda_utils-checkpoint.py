import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


# the features

target     = "num"
continuous  = ["age", "trestbps", "chol", "thalach", "oldpeak"]
binary      = ["sex", "fbs", "exang"]
categorical = ["cp", "restecg", "slope", "thal"]
ordinal     = ["ca"]
features    = continuous + binary + categorical + ordinal


# TASK 1
def load_data(path):
    return pd.read_csv(path, na_values=["?"])

def show_overview(df):
    print("Shape:", df.shape)
    print("\nDtypes:")
    print(df.dtypes)
    
    return df.describe()

def show_missing(df):
    
    return df.isna().sum()

def show_duplicates(df):
    
    return df.duplicated().sum()

def show_outliers(df): # The provided function effectively detects outliers using the Interquartile Range (IQR) method, which is robust for skewed data. It #identifies data points that fall below \(Q_1 - 1.5 \times \text{IQR}\) or above \(Q_3 + 1.5 \times \text{IQR}\).
    
    for col in continuous:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        n = ((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).sum()
        print(f"{col}: {n} outliers")

def plot_class_balance(df):
    
    counts = df[target].value_counts().sort_index()
    print(counts)
    plt.figure()
    counts.plot(kind="bar", color=["steelblue", "tomato"])
    plt.title("Class balance")
    plt.xlabel("num")
    plt.ylabel("count")
    plt.show()

def plot_boxplots(df):
    
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    for ax, col in zip(axes, continuous):
        sns.boxplot(x=target, y=col, data=df, ax=ax)
        ax.set_title(col)
    plt.show()

def plot_histograms(df):
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    for ax, col in zip(axes, continuous):
        df[df[target] == 0][col].hist(ax=ax, bins=20, alpha=0.6, label="0")
        df[df[target] == 1][col].hist(ax=ax, bins=20, alpha=0.6, label="1")
        ax.set_title(col)
        ax.legend()
    plt.show()

def plot_countplots(df):
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    axes = axes.flatten()
    
    for ax, col in zip(axes, binary + categorical + ordinal):
        sns.countplot(x=col, hue=target, data=df, ax=ax)
        ax.set_title(col)
    plt.show()

def plot_correlation(df):

    cols_to_check = features + [target]
    corr = df[cols_to_check].corr(method="spearman")

    # Draw the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corr,
        annot=True,      # show the numbers inside each cell
        fmt=".2f",      
        cmap="coolwarm", # blue-negative, red-positive correlation
        center=0         #  no correlation
    )

    plt.title("Spearman Heatmap")
    
    plt.show()

    return corr


# build_preprocessor lives here because plot_pca below needs it,
# and it is also imported by nested_cv.py for the modelling pipeline.
def build_preprocessor(cont_cols=None, cat_cols=None):
    if cont_cols is None:
        cont_cols = continuous
    if cat_cols is None:
        cat_cols = binary + categorical + ordinal

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
    ])

    preprocessor = ColumnTransformer([   # connect the features
        ("num", num_pipeline, cont_cols),
        ("cat", cat_pipeline, cat_cols)
    ])

    return preprocessor


# Reduce all our features down to just 2 numbers (PC1, PC2) so we can see if classes separate
def plot_pca(df):

    # Split data into inputs X and labels y
    X = df[features]
    y = df[target]

    # encode the features the same way we will do for for training
    preprocessor = build_preprocessor()
    X_ready = preprocessor.fit_transform(X)

    # Reduce to 2 dimensions with PCA
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_ready)

    # How much info each dimension kept 
    pc1_info = pca.explained_variance_ratio_[0] * 100  
    pc2_info = pca.explained_variance_ratio_[1] * 100

    # Draw one colour per class
    plt.figure(figsize=(7, 6))

    for class_label, color in zip([0, 1], ["steelblue", "tomato"]):
        
        is_this_class = (y == class_label)           # boolean mask
        plt.scatter(
            X_2d[is_this_class, 0],                  # x = PC1
            X_2d[is_this_class, 1],                  # y = PC2
            c=color,
            alpha=0.6,                               
            label=f"class {class_label}"
        )

    plt.xlabel(f"PC1  ({pc1_info:.1f}% of variance)")
    plt.ylabel(f"PC2  ({pc2_info:.1f}% of variance)")
    
    plt.title("PCA – 2D View of the Data")
    plt.legend()
    plt.show()

    return pca.explained_variance_ratio_
