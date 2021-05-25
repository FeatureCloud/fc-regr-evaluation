# Regression Evaluation FeatureCloud App

## Description
A Regression Evaluation FeautureCloud app, allowing to evaluate your trained models with various regression metrics (e.g. Mean Squared Error).

## Input
- test.csv containing the actual test dataset
- pred.csv containing the predictions oof the model on the test dataset

## Output
- score.csv containing various evaluation metrics

## Workflows
Can be combined with the following apps:
- Pre: Various regression apps (e.g. Random Forest, Linear Regression, ...)

## Config
Use the config file to customize the evaluation. Just upload it together with your training data as `config.yml`
```
fc_regr_evaluation:
  input:
    y_true: "test.csv"
    y_pred: "pred.csv"
  format:
    sep: ","
  split:
    mode: directory # directory if cross validation was used before, else file
    dir: data # data if cross validation app was used before, else .
```
