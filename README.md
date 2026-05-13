# Heart Disease Classification using Repeated Nested Cross-Validation

**MLCB 2026 — Assignment #2**  
National and Kapodistrian University of Athens — Spring 2026  
Author: Evangelia Kourtzelli 

---

## Overview

A complete object-oriented machine-learning pipeline for binary classification of coronary artery disease (CAD) on the Cleveland subset of the UCI Heart Disease Dataset (242 patients, 13 clinical features). Covers:

- Exploratory Data Analysis (Task 1)
- Repeated Nested Cross-Validation pipeline (Task 3)
- Comparison of 7 classifiers — Elastic-Net LR, GNB, LDA, RF, LightGBM, XGBoost, CatBoost
- Model-agnostic feature selection with mRMR (Task 4)
- Final model training, pickling and SHAP interpretation (Task 5)
- Bonus error analysis on a held-out validation split

---

## Key results

| Algorithm        | Median MCC (tuned) | 95 % bootstrap CI |
|------------------|--------------------|-------------------|
| **LDA (winner)** | **0.664**          | [0.623, 0.706]    |
| GNB              | 0.654              | [0.610, 0.678]    |
| LR (Elastic Net) | 0.639              | [0.615, 0.674]    |
| RF               | 0.624              | [0.582, 0.659]    |
| CatBoost         | 0.586              | [0.554, 0.667]    |
| LightGBM         | 0.580              | [0.538, 0.602]    |
| XGBoost          | 0.549              | [0.507, 0.580]    |

mRMR feature selection (K = 6) raises LDA's median MCC to **0.673** using only `thal`, `ca`, `cp`, `sex`, `exang`, `thalach`.

---

## Repository structure
