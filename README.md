# Loan Default Prediction Model

This project focuses on building a machine learning model to **predict loan default** using borrower profiles, loan contract details, credit scores (CRB), account behavior, and loan performance snapshots. The objective is to help financial institutions **identify high-risk customers early** and reduce credit losses.

---

##  Problem Statement

Predict whether a customer will **default on a loan** based on:
- Application information (demographics, income, employment)
- Loan contract terms
- Historical performance (delinquencies, outstanding amounts)
- Credit bureau (CRB) data
- Banking behavior from savings/current accounts

---

##  Datasets Used

1. **`ApplicationData.csv`**  
   ➤ Demographics, employment, and income details per applicant

2. **`ContractsData.csv`**  
   ➤ Loan contract details including term, amount, and dates

3. **`ContractSnapshotData.csv`**  
   ➤ Time-based performance data like outstanding amount, days past due

4. **`CRB Data.csv`**  
   ➤ Credit scores and grades from bureau sources

5. **`Current and Savings Account Data.csv`**  
   ➤ Banking behavior data including balance trends and transaction patterns

---

##  Data Cleaning & Feature Engineering

 **Missing Values**  
- Imputed based on level of missingness (e.g., mode, median, flags for missing)

 **Outliers & Skewed Features**  
- Capped extreme values, applied `log1p` transformation to skewed distributions

 **Categorical Encoding**  
- One-hot encoding for low-cardinality features  
- Manual target encoding for high-cardinality fields

 **Feature Engineering**  
- `DTI Ratio = Loan Amount / Monthly Income`  
- `Employment Stability Score`  
- `Age Bins`  
- Flags for bounced cheques, dependents, etc.

 **Target Variable Creation**  
- Defined `Loan_Default` as:
  - Contract Status = “Default”, “WriteOff”, “NonAccrual” **OR**
  - Days Past Due > 90

---

## Data Merging Strategy

We merged data around the **contract level**, ensuring each row represents **one loan** with enriched features from:
- Application → via `Application_ID`
- Snapshot → via `Contract_ID`
- CRB & Banking → aggregated at applicant level then merged in

---

##  Exploratory Data Analysis (EDA)

- Visualized distributions of CRB Scores, Income, Outstanding Balances
- Compared default vs non-default groups using boxplots and histograms
- Identified income, CRB score, and due balances as strong differentiators

---

##  Modeling

###  Logistic Regression
- Simple, interpretable baseline
- Used `StandardScaler`, `SimpleImputer`, and logistic model in a pipeline
- Achieved perfect performance on synthetic balanced data (100% Accuracy, AUC = 1.0)

###  Random Forest Classifier
- Handles non-linear patterns and feature interactions
- Built using a pipeline with imputation
- Provided **feature importances**:
  - Most important: `Monthly_Income_Log`, `Outstanding_Amount_last`, `CRB Score`

---

##  Feature Importance Summary

```text
1. Monthly_Income_Log           → 29%
2. Outstanding_Amount_last      → 27%
3. CRB Score                    → 24%
4. Age_at_Application           → 20%
5. Employment_Stability_Score   → ~0%
