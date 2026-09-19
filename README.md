# Rainfall Prediction Classifier

A binary classification project predicting whether it will rain, built on real-world daily
weather observations from the Australian Bureau of Meteorology (Melbourne region,
2008–2017). The project covers a full applied machine learning workflow: data cleaning,
feature engineering, reasoning about data leakage, building a scikit-learn preprocessing +
model pipeline, hyperparameter tuning with grid search cross-validation, and evaluating
results on an imbalanced target with metrics beyond accuracy.

## Highlights

- Identified and resolved a data leakage issue by reframing the prediction target
  (predicting today's rain from yesterday's data, rather than using same-day features that
  wouldn't be available in advance)
- Engineered a `Season` feature from raw date data
- Built a `ColumnTransformer` + `Pipeline` handling numeric scaling and categorical
  one-hot encoding together
- Tuned a `RandomForestClassifier` with `GridSearchCV` and stratified 5-fold
  cross-validation
- Diagnosed class imbalance (~76% "No rain" / 24% "Rain") and evaluated the model with
  precision, recall, F1-score and a confusion matrix rather than accuracy alone
- Extracted and visualized feature importances, mapping one-hot encoded categorical
  features back to their original columns
- Compared Random Forest against Logistic Regression and discussed the precision/recall
  trade-off between the two

## Results

| Model               | Accuracy | Recall ("Rain") | Precision ("Rain") |
|---------------------|:--------:|:----------------:|:--------------------:|
| Random Forest        | ~84%     | ~50%             | ~76%                |
| Logistic Regression  | ~83%     | ~51%             | ~69%                |

Both models land around 83–84% accuracy, but since ~76% accuracy is achievable by simply
always predicting "no rain," accuracy alone is a weak signal here. The more telling metric
is recall on the "Rain" class: both models only catch about half of the days it actually
rains, so neither is yet a strongly reliable rainfall predictor. Random Forest edges ahead
overall on precision, meaning its rain predictions are more trustworthy when it does
predict rain — see the notebook's conclusion for the full discussion of the trade-offs.

`Humidity3pm` was the single most important feature for the Random Forest model.

## Getting started

```bash
git clone <this-repo>
cd <this-repo>
pip install -r requirements.txt
jupyter notebook rainfall_prediction_classifier.ipynb
```

The notebook pulls the dataset directly from a hosted CSV, so no manual download is needed.

## Data source

- Bureau of Meteorology: http://www.bom.gov.au/climate/dwo/
- Dataset (Kaggle): https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package

## Possible next steps

- Feature engineering from rolling weather trends
- Tuning the classification threshold to trade precision for recall
- Class-balancing techniques (oversampling, broader use of `class_weight='balanced'`)
- Expanding to more locations and a longer time window
