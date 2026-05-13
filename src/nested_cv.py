# Modeling code: rnCV, feature selection, final model, SHAP, error analysis

import os
import pickle
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    matthews_corrcoef, roc_auc_score, balanced_accuracy_score,
    f1_score, recall_score, precision_score, average_precision_score,
    confusion_matrix
)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from mrmr import mrmr_classif
import shap

# Reuse the features from eda_utils
from eda_utils import (
    target, continuous, binary, categorical, ordinal, features,
    build_preprocessor
)


# TASK 3
def compute_metrics(y_true, y_pred, y_prob):
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    if (tn + fp) > 0:
        specificity = tn / (tn + fp)
        
    else:
        specificity = 0.0

    metrics = {
        "MCC":         matthews_corrcoef(y_true, y_pred),
        "AUC":         roc_auc_score(y_true, y_prob),
        "BA":          balanced_accuracy_score(y_true, y_pred),
        "F1":          f1_score(y_true, y_pred, zero_division=0),
        "Recall":      recall_score(y_true, y_pred, zero_division=0),
        "Specificity": specificity,
        "Precision":   precision_score(y_true, y_pred, zero_division=0),
        "PRAUC":       average_precision_score(y_true, y_prob),
    }

    return metrics

class RepeatedNestedCV:

    def __init__(self, estimators, param_spaces,
                 R=10, N=5, K=3, n_trials=50, random_state=42,
                 inner_metric="matthews_corrcoef",
                 trial_selection="best_mean"):

        self.estimators = estimators
        self.param_spaces = param_spaces

        self.R = R
        self.N = N
        self.K = K

        self.n_trials = n_trials
        self.random_state = random_state

        # which metric Optuna optimizes in the inner loop
        self.inner_metric = inner_metric

        # how to pick the best trial: "best_mean" or "best_median"
        self.trial_selection = trial_selection

        # results go here later
        self.results_ = {}
        self.baseline_ = {}


    def _build_pipeline(self, est):
        # preprocessor + classifier
        my_pipe = Pipeline([
            ("preprocessor", build_preprocessor()),
            ("clf", est)
        ])
        return my_pipe


    def _inner_loop(self, X_tr, y_tr, name, seed):

       
        space_fn = self.param_spaces[name]  # the search space + estimator for this algorithm
        base_est = self.estimators[name]

        # save the fold scores of each trial (needed for "best_median")
        trial_scores = {}

        # print(f"  starting tuning for {name}")   

        def objective(trial):
            # Optuna 
            chosen = space_fn(trial)

            my_clf = clone(base_est).set_params(**chosen)
            my_pipe = self._build_pipeline(my_clf)

            # 3-fold CV inside the outer-train
            inner_cv = StratifiedKFold(n_splits=self.K, shuffle=True, random_state=seed)

            fold_scores = cross_val_score(my_pipe, X_tr, y_tr,
                                          cv=inner_cv,
                                          scoring=self.inner_metric,
                                          n_jobs=-1)

            # keep the per-fold scores
            trial_scores[trial.number] = fold_scores

            # print(f"    trial {trial.number}: mean={fold_scores.mean():.3f}")   # too noisy

            # Optuna optimizes the mean
            return fold_scores.mean()


        # set up the Optuna study and run trials
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=seed)
        )
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)

        # pick the best trial depending on the selection strategy
        if self.trial_selection == "best_median":

            # trial with the highest median across the inner folds
            best_id = max(trial_scores,
                          key=lambda n: np.median(trial_scores[n]))

            # print(f"  best (median) trial = #{best_id}")
            return study.trials[best_id].params

        else:
            # default: best mean
            return study.best_params


    def _outer_fold(self, X, y, name, train_idx, test_idx, seed):

        # split into outer-train / outer-test
        X_tr = X.iloc[train_idx]
        X_te = X.iloc[test_idx]

        y_tr = y.iloc[train_idx]
        y_te = y.iloc[test_idx]

        # tune on the outer-train only
        best = self._inner_loop(X_tr, y_tr, name, seed)

        # train final model with best params on full outer-train
        my_clf = clone(self.estimators[name]).set_params(**best)
        my_pipe = self._build_pipeline(my_clf)
        my_pipe.fit(X_tr, y_tr)

        # predict on the locked outer-test
        y_pred = my_pipe.predict(X_te)
        y_prob = my_pipe.predict_proba(X_te)[:, 1]

        # print(f"    test_size={len(y_te)}  positive_in_test={y_te.sum()}")

        return compute_metrics(y_te, y_pred, y_prob)


    def run_baseline(self, X, y):

        # baseline = no tuning, just default hyperparameters
        for name, est in self.estimators.items():

            print(f"\n[Baseline] {name}")
            rows = []

            # 10 repetitions
            for r in range(self.R):

                seed = self.random_state + r       # different splits each rep

                cv = StratifiedKFold(n_splits=self.N, shuffle=True, random_state=seed)

                my_pipe = self._build_pipeline(est)

                # 5 folds inside the rep
                for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):

                    my_pipe.fit(X.iloc[train_idx], y.iloc[train_idx])

                    y_pred = my_pipe.predict(X.iloc[test_idx])
                    y_prob = my_pipe.predict_proba(X.iloc[test_idx])[:, 1]

                    m = compute_metrics(y.iloc[test_idx], y_pred, y_prob)
                    m["repetition"] = r + 1
                    m["fold"] = fold + 1
                    rows.append(m)

                    # print(f"  rep {r+1} / fold {fold+1} -- MCC={m['MCC']:.3f}")

            # 50 rows per algorithm (10 reps x 5 folds)
            self.baseline_[name] = pd.DataFrame(rows)

        return self.baseline_


    def run(self, X, y):

      
        for name in self.estimators:   # full repeated nested CV with hyperparameter tuning

            print(f"\n[rnCV] {name}")

            rows = []

            # 10 repetitions
            for r in range(self.R):

                seed = self.random_state + r

                outer_cv = StratifiedKFold(
                    n_splits=self.N,
                    shuffle=True,
                    random_state=seed,
                )

                fold = 1
                for train_idx, test_idx in outer_cv.split(X, y):

                    # tune + evaluate on this outer fold
                    m = self._outer_fold(X, y, name, train_idx, test_idx, seed)

                    m["repetition"] = r + 1
                    m["fold"] = fold

                    rows.append(m)

                    # progress
                    print(f"  rep {r+1} / fold {fold} -- MCC={m['MCC']:.3f}")
                    # print(f"    AUC={m['AUC']:.3f}  F1={m['F1']:.3f}")   # too verbose

                    fold += 1

            # 50 evaluation points per algorithm
            self.results_[name] = pd.DataFrame(rows)

        return self.results_


    def summary(self, results=None):

        # if no results passed, use the tuned ones
        if results is None:
            results = self.results_

        metric_cols = ["MCC", "AUC", "BA", "F1",
                       "Recall", "Specificity", "Precision", "PRAUC"]

        # reproducible bootstrap
        rng = np.random.default_rng(self.random_state)

        rows = []
        for name, df in results.items():

            row = {"Algorithm": name}

            # for each metric - median + 95% bootstrap CI
            for m in metric_cols:

                vals = df[m].values             # 50 fold-level values
                med = np.median(vals)

                # 2000 bootstrap medians
                boots = []
                for _ in range(2000):
                    sample = rng.choice(vals, size=len(vals), replace=True)
                    boots.append(np.median(sample))
                boots = np.array(boots)

                # 2.5% and 97.5% percentiles 95% CI
                lo, hi = np.percentile(boots, [2.5, 97.5])

                row[f"{m}_median"] = round(med, 3)
                row[f"{m}_CI"] = f"[{lo:.3f}, {hi:.3f}]"

                # print(f"{name} {m}: median={med:.3f} CI=[{lo:.3f},{hi:.3f}]")

            rows.append(row)

        return pd.DataFrame(rows).set_index("Algorithm")


# Task 4

def _impute_for_mrmr(X_train, X_test=None):

    # find which cont/cat cols are actually in this slice of X
    cont_here = [c for c in continuous if c in X_train.columns]
    cat_here  = [c for c in (binary + categorical + ordinal) if c in X_train.columns]

    # one imputer for numeric, one for categorical
    num_imp = SimpleImputer(strategy="median")
    cat_imp = SimpleImputer(strategy="most_frequent")

    # fit + transform on X_train
    X_tr_imp = X_train.copy()
    if cont_here:
        X_tr_imp[cont_here] = num_imp.fit_transform(X_train[cont_here])
    if cat_here:
        X_tr_imp[cat_here] = cat_imp.fit_transform(X_train[cat_here])

    # print(f"  imputed train: {X_tr_imp.shape}, cont={len(cont_here)}, cat={len(cat_here)}")

    # if no test set passed return only the imputed train
    if X_test is None:
        return X_tr_imp

    # apply the same fitted imputers on X_test
    X_te_imp = X_test.copy()
    if cont_here:
        X_te_imp[cont_here] = num_imp.transform(X_test[cont_here])
    if cat_here:
        X_te_imp[cat_here] = cat_imp.transform(X_test[cat_here])

    return X_tr_imp, X_te_imp



# Task 4 

class RepeatedNestedCV_FS(RepeatedNestedCV):

    def __init__(self, estimators, param_spaces,
                 R=10, N=5, K=3, n_trials=50, random_state=42, k_features=8):

        # reuse parent constructor for everything else
        super().__init__(estimators, param_spaces,
                         R, N, K, n_trials, random_state)

        self.k_features = k_features

        # how many times each feature was selected 
        self.feature_counts_ = {}


        self._current_selected = None


    def _build_pipeline(self, est):

        picked = self._current_selected

      
        if picked is None:
            return super()._build_pipeline(est)

        # split the selected features into cont / cat groups
        sel_cont = [c for c in continuous if c in picked]
        sel_cat  = [c for c in (binary + categorical + ordinal) if c in picked]

        # print(f"    pipeline cols: cont={len(sel_cont)}, cat={len(sel_cat)}")

        my_pipe = Pipeline([
            ("preprocessor", build_preprocessor(sel_cont, sel_cat)),
            ("clf", est)
        ])
        return my_pipe


    def _outer_fold(self, X, y, name, train_idx, test_idx, seed):

        # split outer-train / outer-test
        X_tr = X.iloc[train_idx]
        X_te = X.iloc[test_idx]

        y_tr = y.iloc[train_idx]
        y_te = y.iloc[test_idx]


        # impute X_train (using X_train stats only) so mRMR can run
        X_tr_imp = _impute_for_mrmr(X_tr)


        #  mRMR feature selection on the outer-train only
        picked = list(mrmr_classif(
            X=X_tr_imp, y=y_tr, K=self.k_features, show_progress=False))

        # print(f"    fold picked: {picked}")

        # update stability counter
        for f in picked:
            self.feature_counts_[f] = self.feature_counts_.get(f, 0) + 1


        # 3. subset RAW data (the pipeline will impute again later)
        X_tr_sel = X_tr[picked]
        X_te_sel = X_te[picked]


        # 4. inner-loop tuning on the selected features
        # remember which columns to use in _build_pipeline
        self._current_selected = picked

        try:
            best = self._inner_loop(X_tr_sel, y_tr, name, seed)

            my_clf = clone(self.estimators[name]).set_params(**best)
            my_pipe = self._build_pipeline(my_clf)
            my_pipe.fit(X_tr_sel, y_tr)

            y_pred = my_pipe.predict(X_te_sel)
            y_prob = my_pipe.predict_proba(X_te_sel)[:, 1]

            # print(f"    fold MCC = {matthews_corrcoef(y_te, y_pred):.3f}")

            return compute_metrics(y_te, y_pred, y_prob)

        finally:
            # always reset so non-FS calls of _build_pipeline still work
            self._current_selected = None


    def feature_stability(self):

        # total number of folds across all reps (e.g. 10 reps x 5 folds = 50)
        total = self.R * self.N

        freq = pd.Series(self.feature_counts_).sort_values(ascending=False)

        # return as percentages
        return (freq / total * 100).round(1)



def choose_best_k(X, y, k_values=(5, 6, 7, 8, 9, 10), random_state=42):

    # impute once on the whole X so mRMR + LDA can run
    X_imp = _impute_for_mrmr(X)

    scores_by_k = {}

    for K in k_values:

        # mRMR picks the top K features
        picked = list(mrmr_classif(X=X_imp, y=y, K=K, show_progress=False))

        # print(f"K={K} picked features: {picked}")

        # LDA needs scaled data
        my_pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    LinearDiscriminantAnalysis())
        ])

        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)

        fold_scores = cross_val_score(my_pipe, X_imp[picked], y,
                                       cv=cv, scoring="matthews_corrcoef")

        scores_by_k[K] = round(float(fold_scores.mean()), 3)

        print(f"K={K} -> MCC={scores_by_k[K]:.3f}")


    # best K = the one with highest CV MCC
    best_k = max(scores_by_k, key=scores_by_k.get)
    print(f"\nBest K = {best_k}  (MCC = {scores_by_k[best_k]})")


    # quick plot
    plt.figure(figsize=(7, 4))
    plt.plot(list(scores_by_k.keys()), list(scores_by_k.values()),
             marker="o", color="steelblue")
    plt.xlabel("K (number of features)")
    plt.ylabel("MCC (3-fold CV, LDA)")
    plt.title("mRMR — How many features do we need?")
    plt.show()

    return best_k, scores_by_k



def plot_feature_stability(stab_series):

    # bar plot of selection frequency
    plt.figure(figsize=(10, 5))
    stab_series.sort_values().plot(kind="barh", color="steelblue")

    # 50% reference line
    plt.axvline(50, color="red", linestyle="--", label="50% threshold")

    plt.xlabel("Selection frequency (%)")
    plt.title("Feature stability across 50 folds (mRMR)")
    plt.legend()
    plt.show()



def compare_full_vs_selected(results_full, results_fs, algorithm="LDA"):

    metric_cols = ["MCC", "AUC", "BA", "F1",
                   "Recall", "Specificity", "Precision", "PRAUC"]

    # header
    print(f"\n{'Metric':<14} {'Full (13)':>12} {'Selected':>12} {'Diff':>8}")
    print("-" * 50)

    for m in metric_cols:
        full_val = results_full[algorithm][m].median()
        fs_val   = results_fs[algorithm][m].median()
        diff     = fs_val - full_val

        # print(f"  full vals: {results_full[algorithm][m].values[:3]}")   

        print(f"{m:<14} {full_val:>12.3f} {fs_val:>12.3f} {diff:>+8.3f}")




# TASK 5.1 

def find_best_params(X, y, estimator, space_fn,
                     n_trials=50, cv=5, random_state=42):

    def objective(trial):
        params   = space_fn(trial)
        clf      = clone(estimator).set_params(**params)
        # only pass the columns that actually exist in X (handles feature-selected subset)
        cont_here = [c for c in continuous if c in X.columns]
        cat_here  = [c for c in (binary + categorical + ordinal) if c in X.columns]
        pipeline = Pipeline([
            ("preprocessor", build_preprocessor(cont_here, cat_here)),
            ("clf", clf)
        ])
        kfold = StratifiedKFold(
            n_splits=cv, shuffle=True, random_state=random_state)
        scores = cross_val_score(
            pipeline, X, y, cv=kfold, scoring="matthews_corrcoef")
        return scores.mean()

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=random_state))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    print("Best hyperparameters:", study.best_params)
    return study.best_params


# TASK 5.2 

def save_final_model(X, y, best_params, path="models/final_model.pkl"):
    clf = LinearDiscriminantAnalysis(**best_params)
    # only pass the columns that actually exist in X (handles feature-selected subset)
    cont_here = [c for c in continuous if c in X.columns]
    cat_here  = [c for c in (binary + categorical + ordinal) if c in X.columns]
    final_pipeline = Pipeline([
        ("preprocessor", build_preprocessor(cont_here, cat_here)),
        ("clf", clf)
    ])
    final_pipeline.fit(X, y)
    print("Final model trained on all data.")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(final_pipeline, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("Model saved to", path)

    # round-trip sanity check
    with open(path, "rb") as f:
        _ = pickle.load(f)
    print("Model loaded back successfully.")
    return final_pipeline


# TASK 5.3 

def compute_shap(final_pipeline, X):
    # Transform X through the fitted preprocessor first (LDA sees the
    # processed numeric matrix, not the raw DataFrame).
    X_transformed = final_pipeline.named_steps["preprocessor"].transform(X)

    explainer  = shap.LinearExplainer(
        final_pipeline.named_steps["clf"], X_transformed)
    shap_vals  = explainer(X_transformed)

    # use the actual columns of X, not the global features list,
    # so this works correctly when a feature-selected subset is passed
    fnames = list(X.columns)

    # Beeswarm
    shap.summary_plot(shap_vals.values, X_transformed,
                      feature_names=fnames, show=True)
    # Bar
    shap.summary_plot(shap_vals.values, X_transformed,
                      feature_names=fnames, plot_type="bar", show=True)
    # Waterfall for one patient
    shap.plots.waterfall(shap_vals[0], show=True)
    return shap_vals, explainer


# BONUS 
def error_analysis(X, y, best_params, test_size=0.2, random_state=42):
    from sklearn.model_selection import train_test_split

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state)

    clf      = LinearDiscriminantAnalysis(**best_params)
    # only pass the columns that actually exist in X (handles feature-selected subset)
    cont_here = [c for c in continuous if c in X.columns]
    cat_here  = [c for c in (binary + categorical + ordinal) if c in X.columns]
    pipeline = Pipeline([
        ("preprocessor", build_preprocessor(cont_here, cat_here)),
        ("clf", clf)
    ])
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_val)
    y_prob = pipeline.predict_proba(X_val)[:, 1]

    # Build a DataFrame with predictions but keep original X_val intact
    diag = X_val.copy()
    diag["true"]      = y_val.values
    diag["predicted"] = y_pred
    diag["prob"]      = y_prob

    fp      = diag[(diag["true"] == 0) & (diag["predicted"] == 1)]
    fn      = diag[(diag["true"] == 1) & (diag["predicted"] == 0)]
    correct = diag[diag["true"] == diag["predicted"]]

    print(f"False Positives: {len(fp)}")
    print(f"False Negatives: {len(fn)}")
    print(f"Correct        : {len(correct)}")

    plt.figure(figsize=(10, 4))
    plt.hist(correct["age"], bins=15, alpha=0.6,
             label="Correct",        color="steelblue")
    plt.hist(fp["age"],      bins=15, alpha=0.6,
             label="False Positive", color="orange")
    plt.hist(fn["age"],      bins=15, alpha=0.6,
             label="False Negative", color="tomato")
    plt.xlabel("Age"); plt.ylabel("Count")
    plt.title("Age distribution by prediction outcome")
    plt.legend(); plt.show()

    # SHAP on the validation set
    X_val_transformed = pipeline.named_steps["preprocessor"].transform(X_val)
    explainer = shap.LinearExplainer(
        pipeline.named_steps["clf"], X_val_transformed)
    shap_vals = explainer(X_val_transformed)

    print("\nSHAP summary — validation set:")
    fnames_val = list(X_val.columns)
    shap.summary_plot(shap_vals.values, X_val_transformed,
                      feature_names=fnames_val, show=True)

    return fp, fn, correct, shap_vals
