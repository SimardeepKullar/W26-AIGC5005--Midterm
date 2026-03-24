# W26-AIGC5005--Midterm
## Payment Default Risk Classification
## Group 4 Members

| Name | Student ID |
|---|---|
| Simardeep Kullar | n10008693 |
| Pravdeep Kullar | n01430968 |
| Udeme Akpausoh | n01495603 |
| Istikbal Turut | n01404444 |

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
| `Midterm_Project_Professional_Report.docx` | Word Document containing professional report |
| `Midterm_Project_Presentation.pptx` | PowerPoint containing slides for recording |
| `Midterm_Project_Recording.mp4` | Recording |

If the recording file does not work here is a link to the video
- https://teams.microsoft.com/l/meetingrecap?driveId=b%219z-2cGZVRk24rvnbtIfqgqx3BEm7ik5Kg8YIXoJV1UWSC6_u8kVHRKZUPVaWa068&driveItemId=01UH54QSVPCRBHABXKZBG23KH3KQJWQJQU&sitePath=https%3A%2F%2Fhumberital-my.sharepoint.com%2Fpersonal%2Fn01495603_humber_ca%2FDocuments%2FRecordings%2FMeeting+with+UDEME+AKPAUSOH-20260323_205351-Meeting+Recording.mp4&fileUrl=https%3A%2F%2Fhumberital-my.sharepoint.com%2Fpersonal%2Fn01495603_humber_ca%2FDocuments%2FRecordings%2FMeeting+with+UDEME+AKPAUSOH-20260323_205351-Meeting+Recording.mp4&threadId=19%3Ameeting_MDFjNzNkMzMtZjI2Ni00NjgzLTgyMGQtM2RiMGU3NTdkNGJi%40thread.v2&organizerId=9974aae5-f0e2-4375-85f6-9d22e39b2f96&tenantId=ca92071f-f342-40c7-8385-6997a60526cc&callId=340e09e2-743b-485b-b7b4-47fbe8583a53&threadType=meeting&meetingType=MeetNow&subType=RecapSharingLink_RecapCore

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
| Step 2 | Drops irrelevant columns (ORDER_ID, ANUMMER_02–10, Z_CARD_ART) |
| Step 3 | Encodes binary yes/no columns to 1/0 |
| Step 4 | Feature engineering — extracts order hour, customer age, dunning history (MAHN_AKT, MAHN_HOECHST), and recency features from DATE_LORDER |
| Step 5 | One-hot encodes categorical columns (weekday, payment method) |
| Step 6 | Imputes remaining missing values using median |
| Step 7 | Verifies the cleaned dataset shape and missing value count |
| Step 8 | Feature analysis — correlation chart and high-risk rate plots for top 3 features |
| Step 9 | Separates features (X) and target label (y) |
| Step 10 | Stratified 80/20 train/test split |
| Step 11 | Scales features using StandardScaler (fit on training set only) |
| Step 12 | Trains Logistic Regression with balanced class weights |
| Step 13 | Plots predicted probability distributions to visualise why threshold tuning is needed |
| Step 14 | Sweeps all thresholds from 0.01 to 0.99 and plots F1 vs threshold curve to find the optimal cutoff |
| Step 15 | Generates predictions using both the default (0.50) and optimal (0.68) thresholds |
| Step 16 | Evaluates both models using classification reports |
| Step 17 | Plots confusion matrices for both the default and tuned threshold |
| Step 18 | Plots ROC curve and reports AUC |

---

## Dataset Notes

- **Size:** 30,000 rows, 44 columns
- **Target column:** `CLASS` — `yes` (high-risk) or `no` (low-risk)
- **Class imbalance:** ~94.2% low-risk, ~5.8% high-risk
- **Missing values:** Several columns have 50%+ missing data — some are dropped, others are engineered into useful features
- **Separator:** Tab (`\t`)
- **Missing value indicator:** `?`
- **Features after preprocessing:** 40

---

## Results Overview

### Default Threshold (0.50) — No Tuning

| Metric | no (low-risk) | yes (high-risk) |
|---|---|---|
| Precision | 0.97 | 0.11 |
| Recall | 0.64 | 0.72 |
| F1-Score | 0.77 | 0.19 |
| Support | 5,651 | 349 |

| | Value |
|---|---|
| Overall Accuracy | 65% |
| Test Set Size | 6,000 orders |

### Optimal Threshold (0.68) — With Tuning

| Metric | no (low-risk) | yes (high-risk) |
|---|---|---|
| Precision | 0.96 | 0.22 |
| Recall | 0.91 | 0.40 |
| F1-Score | 0.94 | 0.28 |
| Support | 5,651 | 349 |

| | Value |
|---|---|
| Overall Accuracy | 88% |
| AUC | 0.74 |
| Test Set Size | 6,000 orders |

**Note:** The 70% F1-score target for the yes (high-risk) class was not reached. This is due to three compounding limitations: (1) severe class imbalance (94.2% / 5.8%), (2) weak individual feature correlations (the strongest predictor has a correlation of only ~0.10 with the target), and (3) logistic regression's linear decision boundary, which cannot capture the non-linear feature relationships present in this dataset. Threshold tuning improved the high-risk F1-score from 0.19 to 0.28 by raising the decision threshold from 0.50 to 0.68.

---

## Model Details

- **Algorithm:** Logistic Regression
- **Class imbalance handling:** `class_weight="balanced"`
- **Feature scaling:** StandardScaler (fit on training data only, applied to test data)
- **Train/test split:** 80% training (24,000 samples), 20% testing (6,000 samples), stratified by class
- **Threshold tuning:** Swept all values from 0.01 to 0.99, selected threshold that maximised F1 for the high-risk class
- **Optimal threshold:** 0.68
- **Random state:** 42 (fixed for reproducibility)
