# Data Mining Assignment 1
Raport: https://www.overleaf.com/project/66041cfce4b96f622a7f4140
## Data Titanic - Machine Learning from Disaster
The training set acts as the foundation for constructing the machine learning models. Within this set, we provide the 'ground truth' outcome for each passenger, which essentially means we disclose whether they survived or not. The model's development relies on a plethora of 'features,' such as passengers' gender, class, and potentially other characteristics like age or family size. Moreover, we have the flexibility to enhance our model's performance through feature engineering, a process where we create new, more informative features from the existing ones.

On the other hand, the test set serves a distinct purpose. It serves as a litmus test to evaluate how well our model generalizes to unseen data. Here, we deliberately conceal the ground truth for each passenger, leaving it up to our model to make predictions. Our task is to utilize the model trained on the training set to forecast whether each passenger in the test set survived the Titanic's tragic sinking or not. This evaluation is crucial as it assesses the real-world applicability and effectiveness of the model

#### Brief description of each data variable:

1) **Survival**: This variable indicates whether the passenger survived the sinking of the Titanic. It has two possible values:
        0: No (the passenger did not survive)
        1: Yes (the passenger survived)

2) **Pclass (Ticket class)**: This variable represents the ticket class, which serves as a proxy for socio-economic status (SES). It has three possible values:
   - 1: 1st class (Upper)
   - 2: 2nd class (Middle)
   - 3: 3rd class (Lower)

3) **Sex**: This variable denotes the gender of the passenger. It has two possible values:
    - Male
    - Female

4) **Age**: This variable indicates the age of the passenger in years. If the age is fractional, it typically represents a child's age, and if it's estimated, it's in the form of xx.5.

5) **SibSp**: This variable represents the number of siblings or spouses aboard the Titanic for each passenger. It captures family relations as follows:
   - Sibling: brother, sister, stepbrother, stepsister
   - Spouse: husband, wife (mistresses and fiancés were ignored)

6) **Parch**: This variable denotes the number of parents or children aboard the Titanic for each passenger. It defines family relations as follows:
   - Parent: mother, father
   - Child: daughter, son, stepdaughter, stepson
   - Some children traveled only with a nanny, so parch=0 for them.

7) **Ticket**: This variable represents the ticket number assigned to each passenger.

8) **Fare**: This variable indicates the fare paid by each passenger for their ticket.

9) **Cabin**: This variable denotes the cabin number assigned to each passenger.

10) **Embarked (Port of Embarkation)**: This variable represents the port of embarkation for each passenger. It has three possible values:
  - C: Cherbourg
  - Q: Queenstown
  - S: Southampton

## Libraries Used:
- **pandas**: For data manipulation and analysis. It provides data structures and functions to work with structured data, such as data frames.
- **numpy**: Numerical computing in Python. It provides support for large, multi-dimensional arrays and matrices, along with a collection of mathematical functions to operate on these arrays efficiently
- **MinMaxScaler, StandardScaler, OneHotEncoder, LabelEncoder** from sklearn.preprocessing : This submodule of scikit-learn provides various functions for preprocessing data before feeding it into machine learning algorithms.
    - MinMaxScaler: Used for scaling features to a specified range (usually between 0 and 1).
    - StandardScaler: Used for standardizing features by removing the mean and scaling to unit variance.
    - OneHotEncoder: Used for encoding categorical features as one-hot numeric arrays.
    - LabelEncoder: Used for encoding categorical target labels with numeric values.
- **SelectKBest, f_classif** from sklearn.feature_selection:  This submodule of scikit-learn provides functions for selecting features from the dataset before training machine learning models:
    - SelectKBest: Selects the top k features based on specified statistical tests.
    - f_classif: Computes the ANOVA F-value for the provided feature and target variable.
-**PCA** from sklearn.decomposition: This submodule of scikit-learn provides functions for dimensionality reduction techniques. 
    - PCA (Principal Component Analysis): Reduces the dimensionality of the dataset by projecting it onto a lower-dimensional subspace while preserving the most important features/variance.
- **copy**: Module provides functions for creating shallow and deep copies of objects in Python.
- **seaborn**: It is a statistical data visualization library based on Matplotlib. It provides a high-level interface for drawing attractive and informative statistical graphics.
- **matplotlib.pyplot**: Matplotlib is a plotting library for Python. The pyplot module provides a MATLAB-like interface for creating plots and visualizations.
- **RFE** from sklearn.feature_selection : (Recursive Feature Elimination): Eliminates the least important features recursively based on model performance.
- **LogisticRegression** from sklearn.linear_model:  A linear model for binary classification tasks.
- **cross_val_score** from sklearn.model_selection: Computes cross-validated scores for an estimator on multiple datasets.
- **RFECV** from sklearn.feature_selection: RFECV (Recursive Feature Elimination with Cross-Validation): Similar to RFE but uses cross-validation to determine the optimal number of features.
- **StratifiedKFold** from sklearn.model_selection: Provides stratified folds for cross-validation, ensuring that each fold contains approximately the same proportion of classes as the original dataset.
- **RandomForestClassifier** from sklearn.ensemble: A machine learning algorithm that fits a number of decision tree classifiers on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting.

## Installation of Libraries
**Pip freeze**: You can install the dependencies by running the following command in your terminal.

$ pip install -r Requirements.txt 

or 

$ pip freeze > Requirements.txt

## Brief descrition of what was done in the project
- **Data Loading**: Loaded the Titanic dataset consisting of passenger information such as survival status, ticket class, gender, age, family relations, ticket details, fare, cabin number, and port of embarkation.

- **Data Preprocessing**:
    - *Handling Missing Values*: Dropped columns with no relevant information and removed rows with missing values, considering the importance of features in the dataset.
    - *Encoding Categorical Variables*: Converted categorical variables like 'Sex' and 'Embarked' into numerical labels for machine learning algorithms to process.
    - *Normalization and Standardization*: Performed data scaling using MinMaxScaler and StandardScaler to ensure all features are on a similar scale.

- **Feature Selection**: Recursive Feature Elimination (RFE): Used RFE with Logistic Regression and Random Forest Classifier to select the most important features for prediction.

- **Feature Extraction**: Principal Component Analysis (PCA): Applied PCA to reduce the dimensionality of the dataset while retaining important information, creating new features.

- **Model Training and Evaluation**:
    - Trained machine learning models (Logistic Regression, SVC) on the training dataset with selected and extracted features.
    - Evaluated the models' performance on the validation set using metrics such as accuracy and classification report.
- **Prediction on Test Set**: Utilized the trained models to predict the survival outcomes of passengers in the test set, who did not have their survival status provided.
