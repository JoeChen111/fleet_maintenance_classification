# Fleet Maintenance Classification

Comparing **Logistic Regression** and **Random Forest** to identify maintenance needs from vehicle records and explore their potential to support logistics maintenance decisions.

## Project Presentation

[View the full project presentation (PDF)](Fleet_Maintenance_Analysis.pdf)

## Project Objectives

- Compare two classification models using the same training and test data.
- Examine false positives and false negatives to understand their implications for maintenance decisions.
- Identify the variables that the models rely on most.

The intended application is to support maintenance planning and reduce unnecessary alerts or missed maintenance needs. Actual operational savings have not been measured.

## Dataset Overview

| Item | Description |
|---|---|
| Training records | 56,697 |
| Test records | 24,299 |
| Input features | 15: 9 numerical and 6 categorical |
| Target variable | `Need_Maintenance` |
| Target labels | 0 = Maintenance not required; 1 = Maintenance required |
| Class distribution | Approximately balanced in both datasets |

The existing training and test split was retained. No additional SMOTE or resampling was applied.

**The CSV datasets are not included in the current repository. They are required to run the analysis.**

## Methodology

1. Check missing values, duplicate records and data types.
2. Apply one-hot encoding to categorical features.
3. Standardize numerical features for Logistic Regression; retain their original scale for Random Forest.
4. Compare both models using identical stratified five-fold cross-validation splits.
5. Refit each model on the full training set and evaluate on the same test set.
6. Examine confusion matrices, ROC curves and permutation importance.

Preprocessing is fitted only on the training portion within each cross-validation fold. Both models use a classification threshold of **0.5** and a random seed of **42**. Model settings were fixed without test-set tuning.

## Results

### Cross-validation ROC-AUC

| Model | Mean ± standard deviation |
|---|---:|
| Logistic Regression | 0.8182 ± 0.0016 |
| Random Forest | **0.8945 ± 0.0017** |

Random Forest achieved higher ROC-AUC in every validation fold. Both models showed limited variation across the five folds.

### Test Performance

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Accuracy | 74.91% | **82.32%** |
| Precision | 76.73% | **90.69%** |
| Recall | 71.50% | **72.05%** |
| F1-score | 0.7402 | **0.8030** |
| ROC-AUC | 0.8193 | **0.8976** |

### False Positives and False Negatives

| Error type | Logistic Regression | Random Forest |
|---|---:|---:|
| False positives: unnecessary alerts | 2,635 | 899 |
| False negatives: missed maintenance needs | 3,462 | 3,396 |

Random Forest produced **1,736 fewer false positives** and **66 fewer false negatives**.

Its main advantage was reducing unnecessary alerts. The improvement in identifying records that actually required maintenance was relatively small.

### Feature Importance

Both models relied most on:

1. `Reported_Issues`
2. `Service_History`

Permutation importance measures model reliance on an input variable. It does not establish a causal relationship.

## Repository Files

| File | Description |
|---|---|
| `Fleet_Maintenance_Analysis.pdf` | Project presentation |
| `run_analysis.py` | Model training, evaluation and interpretation |
| `run_eda.py` | Exploratory data analysis |
| `requirements.txt` | Python dependencies |
| `README.md` | Project overview and usage instructions |

## How to Run

Download the repository and open its folder in VS Code or a terminal.

### 1. Create a Python environment

For macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### 2. Add the datasets

Create a `data` folder in the project directory and place the matching datasets inside it:

```text
data/
├── train_set.csv
└── test_set.csv
```

### 3. Run the exploratory analysis

```bash
python run_eda.py
```

EDA figures and tables are saved to `results/eda_train/`.

### 4. Run the model comparison

```bash
python run_analysis.py --train data/train_set.csv --test data/test_set.csv --output results/model_run --cv-folds 5 --trees 300 --seed 42
```

Model results are saved to `results/model_run/`.

## Limitations and Next Steps

- Results apply to the supplied balanced datasets. Real-world maintenance prevalence may differ.
- Dependencies from earlier sampling or data preparation cannot be ruled out.
- This project classifies maintenance needs; it does not predict the timing of future failures.
- Actual reductions in downtime, inspections or maintenance costs have not been measured.

Further work should validate the models on independent operational data and explore classification thresholds using training-side validation, considering the trade-off between missed needs and unnecessary alerts.
