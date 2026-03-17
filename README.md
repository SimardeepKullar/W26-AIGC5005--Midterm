# W26-AIGC5005--Midterm
## Payment Default Risk Classification

---

## Project Overview

This project builds a Logistic Regression model to classify online purchase orders as either high-risk or low-risk for payment default. The dataset contains 30,000 orders from an online retailer with 44 attributes per order.

**Goal:** Predict whether a new incoming order is likely to result in payment default (`yes` = high-risk, `no` = low-risk).

---

## Files

| File | Description |
|---|---|
| `midterm.ipynb` | Main notebook containing all code, outputs, and visualizations |
| `risk-train.txt` | Dataset — 30,000 orders, tab-separated, `?` used for missing values |

---

## How to Run

1. Make sure `risk-train.txt` is in the **same folder** as `midterm.ipynb`
2. Install the required libraries (see below)
3. Open `midterm.ipynb` in Jupyter Notebook or JupyterLab
4. Run all cells in order: **Kernel → Restart & Run All**

---

## Requirements

The following Python libraries are required:

```
pandas
numpy
scikit-learn
matplotlib
seaborn
```

Install them all at once with:

```
pip install pandas numpy scikit-learn matplotlib seaborn
```

---

## Notebook Structure

| Step | Description |
|---|---|
| Import Libraries | Loads all required packages |
| Load & Explore Data | Reads the dataset and plots the class distribution |
| Step 1 | Identifies columns with missing values |
| Step 2 | Drops irrelevant and high-missing columns |
| Step 3 | Encodes binary yes/no columns to 1/0 |
| Step 4 | Feature engineering — extracts order hour and customer age |
| Step 5 | One-hot encodes categorical columns (weekday, payment method) |
| Step 6 | Imputes remaining missing values using median |
| Step 7 | Verifies the cleaned dataset shape and missing value count |
| Step 8 | Feature analysis — correlation chart and high-risk rate plots |
| Step 9 | Separates features (X) and target label (y) |
| Step 10 | Stratified 80/20 train/test split |
| Step 11 | Scales features using StandardScaler |
| Step 12 | Trains Logistic Regression with balanced class weights |
| Step 13 | Generates predictions on the test set |
| Step 14 | Evaluates model using classification report |
| Step 15 | Plots the confusion matrix heatmap |

---

## Dataset Notes

- **Size:** 30,000 rows, 44 columns
- **Target column:** `CLASS` — `yes` (high-risk) or `no` (low-risk)
- **Class imbalance:** ~94.2% low-risk, ~5.8% high-risk
- **Missing values:** Several columns have 50%+ missing data and are dropped during preprocessing
- **Separator:** Tab (`\t`)
- **Missing value indicator:** `?`

---

## Results Overview

| Metric | no (low-risk) | yes (high-risk) |
|---|---|---|
| Precision | 0.97 | 0.11 |
| Recall | 0.65 | 0.72 |
| F1-Score | 0.78 | 0.19 |
| Support | 5,651 | 349 |

| | Value |
|---|---|
| Overall Accuracy | 65% |
| Test Set Size | 6,000 orders |

**Note:** The 70% F1-score target for the yes (high-risk) class was not reached. This is primarily due to severe class imbalance (94.2% / 5.8%) combined with weak individual feature correlations — the strongest predictor has a correlation of only ~0.10 with the target. Logistic Regression as a linear model cannot capture the non-linear feature interactions that would be needed to push performance higher.

---

## Model Details

- **Algorithm:** Logistic Regression
- **Class imbalance handling:** `class_weight="balanced"`
- **Feature scaling:** StandardScaler (fit on training data only)
- **Train/test split:** 80% training, 20% testing, stratified by class
- **Random state:** 42 (fixed for reproducibility)
