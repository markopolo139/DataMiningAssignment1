# Titanic dataset presentation

## Introduction

- The topic of *Exploring Data (Techniques) on the Titanic Dataset*
- Explain the goal (briefly) -> Predicting survival on Titanic passengers using classifiers and feature selection

## Dataset

- Dataset structure - features briefly
- Our target - 'Survived?' (binary 0/1)

## Data preprocessing

- Missing values:

> dropped irrelevant columns (PassengerId, Name, Ticket, Cabin...)

- Encoding - Sex + Embarked (categorical -> numeric)

> Normalization/Standardization of categorical data is irrelevant

- Show piece of data

## Comparison of classifiers before preprocessing

- SVC, RandomForest, LogisticRegression
- Accuracy scores before - performance

## Preprocessing techniques

- Normalization

> MinMaxScaler

- Standardization

> StandardScaler

## Feature Selection

- RFE (Recursive Feature Elimination) with LR, RFC -> selected optimal feature
- Graphs
- Describe again selected features

## Feature extraction

- PCA (Principal Component Analysis)

> Reducing dimensionality + extracting

- Show data before and after PCA

## Classifiers comparison after selecting and extracting features

- SVC, RFC, LR after actions
- Compare scores

## Conclusion

- How preprocessing and selection/extraction improved performance
