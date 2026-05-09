# libraries
import os
import pickle
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
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    matthews_corrcoef, roc_auc_score, balanced_accuracy_score,
    f1_score, recall_score, precision_score, average_precision_score,
    confusion_matrix
)

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)


# the features
target     = "num"
continuous  = ["age", "trestbps", "chol", "thalach", "oldpeak"]
binary      = ["sex", "fbs", "exang"]
categorical = ["cp", "restecg", "slope", "thal"]
ordinal     = ["ca"]
features    = continuous + binary + categorical + ordinal


# =============================================================
# TASK 1 — EDA FUNCTIONS
# =============================================================

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


def show_outliers(df):
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
    corr = df[features + [target]].corr(method="spearman")
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Spearman correlation")
    plt.show()
    return corr


def plot_pca(df):
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
    plt.show()
    return pca.explained_variance_ratio_


# =============================================================
# TASK 2 — nCV INTUITION
# =============================================================

def explain_ncv():
    print("\n=== NESTED CROSS VALIDATION (nCV) ===\n")
    print("STEP 1: Split dataset into 5 outer folds (StratifiedKFold)")
    print("        Each fold reflects the overall class distribution.\n")
    print("   OUTER LOOP (k = 1..5):")
    print("   - 1 fold  = outer-test  (~48 patients)  -> LOCKED")
    print("   - 4 folds = outer-train (~194 patients)\n")
    print("   STEP 2: Inner Cross Validation on outer-train (3 folds)")
    print("   INNER LOOP (j = 1..3) -- Hyperparameter tuning via Optuna:")
    print("   - Preprocessing fitted on inner-train ONLY")
    print("   - Applied (not fitted) on inner-val")
    print("   - Score: MCC on inner-val\n")
    print("   STEP 3: Select BEST hyperparameters (best Optuna trial)\n")
    print("   STEP 4: Refit on FULL outer-train with best HP\n")
    print("   STEP 5: Evaluate on outer-test (unlocked for first time)\n")
    print("   STEP 6: Store metrics (MCC, AUC, BA, F1, Recall, ...)\n")
    print("REPEAT x5 folds x10 repetitions = 50 evaluations per algorithm\n")
    print("NO-LEAKAGE RULES:")
    print("  - Imputer  : fit on training fold only")
    print("  - Scaler   : fit on training fold only")
    print("  - Encoder  : fit on training fold only")
    print("  - Feat.sel.: fit on training fold only")
    print("  - Outer-test NEVER seen during tuning")


# =============================================================
# TASK 3 — PREPROCESSING + RepeatedNestedCV CLASS
# =============================================================

def build_preprocessor():
    # continuous: median imputation + standard scaling
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler())
    ])

    # categorical/binary/ordinal: most_frequent imputation + ordinal encoding
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1))
    ])

    # combined into one transformer; always fitted only on training data
    preprocessor = ColumnTransformer([
        ("num", num_pipeline, continuous),
        ("cat", cat_pipeline, binary + categorical + ordinal)
    ])

    return preprocessor


def compute_metrics(y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {
        "MCC":         matthews_corrcoef(y_true, y_pred),
        "AUC":         roc_auc_score(y_true, y_prob),
        "BA":          balanced_accuracy_score(y_true, y_pred),
        "F1":          f1_score(y_true, y_pred, zero_division=0),
        "Recall":      recall_score(y_true, y_pred, zero_division=0),
        "Specificity": specificity,
        "Precision":   precision_score(y_true, y_pred, zero_division=0),
        "PRAUC":       average_precision_score(y_true, y_prob),
    }


class RepeatedNestedCV:

    def __init__(self, estimators, param_spaces,
                 R=10, N=5, K=3, n_trials=50, random_state=42):
        self.estimators   = estimators
        self.param_spaces = param_spaces
        self.R            = R
        self.N            = N
        self.K            = K
        self.n_trials     = n_trials
        self.random_state = random_state
        self.results_     = {}
        self.baseline_    = {}

    def _build_pipeline(self, estimator):
        # preprocessing inside the pipeline -> fitted only on training data
        return Pipeline([
            ("preprocessor", build_preprocessor()),
            ("clf", estimator)
        ])

    def _inner_loop(self, X_train, y_train, name, seed):
        space_fn = self.param_spaces[name]
        base_estimator = self.estimators[name]

        def objective(trial):
            params   = space_fn(trial)
            clf      = clone(base_estimator).set_params(**params)
            pipeline = self._build_pipeline(clf)
            inner_cv = StratifiedKFold(
                n_splits=self.K, shuffle=True, random_state=seed)
            scores = cross_val_score(
                pipeline, X_train, y_train,
                cv=inner_cv, scoring="matthews_corrcoef", n_jobs=-1)
            return scores.mean()

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=seed))
        study.optimize(objective, n_trials=self.n_trials,
                       show_progress_bar=False)
        return study.best_params

    def _outer_fold(self, X, y, name, train_idx, test_idx, seed):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        best_params = self._inner_loop(X_train, y_train, name, seed)

        clf      = clone(self.estimators[name]).set_params(**best_params)
        pipeline = self._build_pipeline(clf)
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        return compute_metrics(y_test, y_pred, y_prob)

    def run_baseline(self, X, y):
        for name, estimator in self.estimators.items():
            print(f"\n[Baseline] {name}")
            records = []
            for r in range(self.R):
                seed = self.random_state + r
                cv   = StratifiedKFold(
                    n_splits=self.N, shuffle=True, random_state=seed)
                pipeline = self._build_pipeline(estimator)
                for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
                    pipeline.fit(X.iloc[train_idx], y.iloc[train_idx])
                    y_pred = pipeline.predict(X.iloc[test_idx])
                    y_prob = pipeline.predict_proba(X.iloc[test_idx])[:, 1]
                    m = compute_metrics(y.iloc[test_idx], y_pred, y_prob)
                    m["repetition"] = r + 1
                    m["fold"]       = fold + 1
                    records.append(m)
            self.baseline_[name] = pd.DataFrame(records)
        return self.baseline_

    def run(self, X, y):
        for name in self.estimators:
            print(f"\n[rnCV] {name}")
            records = []
            for r in range(self.R):
                seed     = self.random_state + r
                outer_cv = StratifiedKFold(
                    n_splits=self.N, shuffle=True, random_state=seed)
                for fold, (train_idx, test_idx) in \
                        enumerate(outer_cv.split(X, y)):
                    m = self._outer_fold(
                        X, y, name, train_idx, test_idx, seed)
                    m["repetition"] = r + 1
                    m["fold"]       = fold + 1
                    records.append(m)
                    print(f"  rep {r+1} / fold {fold+1} -- MCC={m['MCC']:.3f}")
            self.results_[name] = pd.DataFrame(records)
        return self.results_

    def summary(self, results=None):
        if results is None:
            results = self.results_
        metric_cols = ["MCC", "AUC", "BA", "F1",
                       "Recall", "Specificity", "Precision", "PRAUC"]
        rows = []
        for name, df in results.items():
            row = {"Algorithm": name}
            for m in metric_cols:
                vals  = df[m].values
                med   = np.median(vals)
                boots = np.array([
                    np.median(np.random.choice(vals, len(vals), replace=True))
                    for _ in range(2000)
                ])
                lo, hi = np.percentile(boots, [2.5, 97.5])
                row[f"{m}_median"] = round(med, 3)
                row[f"{m}_CI"]     = f"[{lo:.3f}, {hi:.3f}]"
            rows.append(row)
        return pd.DataFrame(rows).set_index("Algorithm")

    def save_model(self, pipeline, path="models/final_model.pkl"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(pipeline, f)
        print(f"Model saved to {path}")

