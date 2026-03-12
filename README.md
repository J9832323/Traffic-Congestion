# 🚦 Traffic Congestion Analysis | Python & Machine Learning

**Tools:** Python, scikit-learn, Pandas, Matplotlib, Seaborn  
**Skills:** Classification Modeling, GridSearchCV, Feature Engineering, Data Visualization

---

## Overview

This project analyzes urban traffic congestion patterns using weather, time, and environmental data to classify traffic volume as **High** (≥4,000 vehicles/hour) or **Low**.

Using a dataset of 33,750 hourly traffic observations, I built and compared five machine learning classifiers, applied feature engineering, tested the impact of scaling, and performed hyperparameter tuning to maximize accuracy.

---

## Models Compared

| Model | Accuracy (Unscaled) | Accuracy (Scaled) |
|-------|-------------------|------------------|
| Decision Tree | 69% | 69% |
| Random Forest | 66% | 66% |
| K-Nearest Neighbors | 90% | 57% |
| Logistic Regression | 57% | 58% |
| SVM (LinearSVC) | 57% | 59% |
| **KNN (Fine-Tuned)** | **93%** | — |

---

## Key Technical Details

- **Fine-tuning:** GridSearchCV with 5-fold cross-validation used to optimize KNN (best k=1) and SVM (best C=1, kernel=rbf)
- **Scaling:** StandardScaler applied to compare model performance with and without normalization — notably, scaling *hurt* KNN accuracy, an important finding about distance-based models
- **Feature engineering:** Dummy encoding for categorical weather variables (weather type, weather description, holiday flags); datetime converted to Unix timestamps
- **Evaluation:** Classification reports and confusion matrices for all models
- **Visualization:** Correlation heatmap, predicted vs. actual bar charts, histogram of class distribution

---

## Key Finding

KNN was the strongest performer at **93% accuracy** after fine-tuning. A notable insight: StandardScaler significantly *decreased* KNN performance (90% → 57%), illustrating how scaling assumptions can hurt distance-based algorithms when raw feature magnitudes carry meaningful signal.

---

## Files

- `Traffic_Congestion.ipynb` — Full Jupyter notebook with analysis and visualizations
- `TrafficVolumeData.csv` — Source dataset
