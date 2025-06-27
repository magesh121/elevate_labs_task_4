# 🧠 Breast Cancer Classification using Logistic Regression

## 🎯 Project Objective

Build a **binary classification model** using **logistic regression** to accurately predict breast cancer diagnoses (Malignant vs Benign) using the Breast Cancer Wisconsin dataset. The model should maximize **accuracy** while minimizing **false negatives**, which are critical in medical diagnostics.

---

## 📊 Dataset Overview

- **Source:** Breast Cancer Wisconsin Dataset (from Scikit-learn)
- **Samples:** 569 total
- **Classes:**
  - Malignant (0): 212 samples (37.3%)
  - Benign (1): 357 samples (62.7%)
- **Features:** 30 numerical features extracted from cell nuclei images, covering:
  - Mean
  - Standard Error
  - Worst-case measurements

---

## 🛠 Tools & Libraries Used

- `Python 3.x`
- `Scikit-learn` – Model & metrics
- `Pandas` – Data handling
- `NumPy` – Numeric operations
- `Seaborn` & `Matplotlib` – Visualization
- `Warnings` – Error handling

---

## 📁 Project Structure

```bash
logistic-regression-classification/
│
├── logistic_regression_classifier.py       # Main implementation
├── README.md                               # Project documentation
└── visualizations/                         # Generated charts and graphs
    ├── data_exploration.png
    ├── feature_scaling.png
    ├── sigmoid_function.png
    ├── confusion_matrix.png
    ├── performance_analysis.png
    └── feature_importance.png
