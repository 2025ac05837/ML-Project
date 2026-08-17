# Breast Cancer Classification — ML Assignment 2

## a. Problem Statement

Breast cancer diagnosis relies on classifying tumor characteristics (derived
from digitized images of fine needle aspirate biopsies) as either **malignant**
or **benign**. The goal of this project is to build, evaluate, and deploy
multiple supervised classification models that predict the diagnosis from a
set of numeric tumor features, and to compare their performance using
standard classification metrics.

## b. Dataset Description

- **Source:** Breast Cancer Wisconsin (Diagnostic) dataset, available via
  `sklearn.datasets.load_breast_cancer` (originally from the UCI Machine
  Learning Repository).
- **Instances:** 569
- **Features:** 30 numeric features (mean, standard error, and "worst"
  values of 10 real-valued measurements per cell nucleus, e.g. radius,
  texture, perimeter, area, smoothness, compactness, concavity, concave
  points, symmetry, fractal dimension).
- **Target:** Binary — `0 = malignant`, `1 = benign`.
- **Class balance:** 212 malignant / 357 benign (moderately imbalanced,
  which is why AUC and MCC matter alongside accuracy).
- **Preprocessing:** 80/20 stratified train-test split; features standardized
  with `StandardScaler` (fit on train only, applied to both splits).

## c. GitHub Repository Link

`<< paste your GitHub repo URL here after you push the code >>`

## d. Models Used

All 5 models were trained on the same standardized train split and evaluated
on the same held-out test split (114 samples).

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree | 0.9211 | 0.9163 | 0.9565 | 0.9167 | 0.9362 | 0.8341 |
| kNN | 0.9737 | 0.9884 | 0.9600 | 1.0000 | 0.9796 | 0.9442 |
| Naive Bayes | 0.9298 | 0.9868 | 0.9444 | 0.9444 | 0.9444 | 0.8492 |
| Random Forest (Ensemble) | 0.9561 | 0.9937 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |

*(Exact numbers regenerate slightly if you change the random seed or resplit
the data — rerun `model/train_models.py` and refresh this table if so.)*

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Best overall performer here. The classes are close to linearly separable after standardization, so a simple linear decision boundary generalizes very well — highest accuracy, AUC, and MCC of all 5 models. |
| Decision Tree | Weakest of the five. A single tree (even depth-limited to 5) overfits patterns in the training split and doesn't generalize as smoothly as ensembles or linear models — lowest accuracy, AUC, and MCC. |
| kNN | Very strong, and had **perfect recall (1.0)** — it caught every malignant/benign case correctly on the positive class, though precision was slightly lower than Logistic Regression, meaning a few more false positives. Performance depends on the standardized feature scale, which was applied here. |
| Naive Bayes | Solid AUC (0.9868) despite the independence assumption between features being unrealistic for correlated tumor measurements (e.g. radius and area are highly correlated) — accuracy and F1 are middling because the independence assumption hurts calibration more than ranking ability. |
| Random Forest (Ensemble) | Strong and stable — very high AUC (0.9937), close to Logistic Regression. Averaging 200 trees smooths out the overfitting a single Decision Tree showed, at the cost of some interpretability. |
| **Overall Winner for your dataset?** | **Logistic Regression** — it topped 5 of the 6 metrics (Accuracy, AUC, Precision, Recall tied, F1, MCC). This makes sense given the dataset's features separate the two classes almost linearly after scaling. Random Forest is a close second and would likely be preferred if the dataset were larger or noisier. |

> Replace/expand these observations with your own phrasing and analysis before
> submitting — the assignment explicitly checks for original written
> observations, not templated text.

## App Features

The deployed Streamlit app (`app.py`) includes:
- CSV upload for test data (matching `test_data.csv` schema)
- Model selection dropdown (all 5 models)
- Live evaluation metrics (Accuracy, AUC, Precision, Recall, F1, MCC)
- Confusion matrix heatmap and full classification report
- A second tab showing the precomputed comparison table and a bar-chart
  comparison across models

## Project Structure

```
project-folder/
│-- app.py
│-- requirements.txt
│-- README.md
│-- test_data.csv
│-- model/
│   │-- train_models.py
│   │-- scaler.pkl
│   │-- meta.json
│   │-- metrics_comparison.csv
│   │-- logistic_regression.pkl
│   │-- decision_tree.pkl
│   │-- knn.pkl
│   │-- naive_bayes.pkl
│   └-- random_forest_ensemble.pkl
```

## How to Run Locally

```bash
pip install -r requirements.txt
python model/train_models.py   # regenerates models, scaler, and metrics
streamlit run app.py
```
