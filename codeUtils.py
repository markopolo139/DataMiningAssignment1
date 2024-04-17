import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
import copy
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC



def load_data():
    titanic_train = pd.read_csv('data/train.csv')
    titanic_test = pd.read_csv('data/test.csv')
    y_test = pd.read_csv('data/gender_submission.csv')
    titanic_test["Survived"] = y_test["Survived"]
    return (titanic_test, titanic_train)


def drop_unnecessary_columns(table):
    table.drop(columns=['PassengerId', 'Name','Ticket', 'Cabin'], inplace=True)


def drop_na_rows(table):
    for col in table.columns:
        table.dropna(inplace=True)


def bin_sex_attributes(table):
    table['Sex'] = table['Sex'].map({'male': 0, 'female': 1})


def convert_embarked_into_numbers(table):
    label_encoder = LabelEncoder()
    table['Embarked'] = label_encoder.fit_transform(table['Embarked'])


def get_data_for_classification(train_data, test_data):
    y_train = train_data['Survived']
    y_test = test_data['Survived']
    x_train = train_data.drop(columns=['Survived'])
    x_test = test_data.drop(columns=['Survived'])
    return (x_train, y_train, x_test, y_test)


def test_accuracy_svc(x_train, y_train, x_test, y_test):
    classifier = SVC()
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    print(classification_report(y_test, y_pred))
    accuracy = accuracy_score(y_test, y_pred)

    print(f'The accuracy of the classifier is: {accuracy}')

def test_accuracy_random_forrest(x_train, y_train, x_test, y_test):
    classifier = RandomForestClassifier()
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    print(classification_report(y_test, y_pred))
    accuracy = accuracy_score(y_test, y_pred)

    print(f'The accuracy of the classifier  is: {accuracy}')


def test_accuracy_logistic_regression(x_train, y_train, x_test, y_test):
    classifier = LogisticRegression()
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    print(classification_report(y_test, y_pred))
    accuracy = accuracy_score(y_test, y_pred)

    print(f'The accuracy of the classifier is: {accuracy}')

def normalize(table, numerical_cols):
    norm_scaler = MinMaxScaler()
    table[numerical_cols] = norm_scaler.fit_transform(table[numerical_cols])


def standardize(table, numerical_cols):
    scaler_std = StandardScaler()
    table[numerical_cols] = scaler_std.fit_transform(table[numerical_cols])


def  features_selection_logistic_regression(table_train):
    X_train = table_train.drop(columns=['Survived'])
    y_train = table_train['Survived']
    number_of_features = range(1, len(X_train.columns) + 1)
    estimator = LogisticRegression()
    scores = []
    selected_features_list = []


    for i in number_of_features:
        rfe = RFE(estimator, n_features_to_select=i)
        rfe.fit(X_train, y_train)
        selected_features = X_train.columns[rfe.support_]
        selected_features_list.append(selected_features)
        scores.append(rfe.score(X_train, y_train))

    plt.figure(figsize=(10, 6))
    plt.plot(number_of_features, scores)
    plt.title('Number of Features vs. Cross-Validation Score')
    plt.xlabel('Number of Features')
    plt.ylabel('Cross-Validation Score')
    plt.xticks(number_of_features)
    plt.grid(True)
    plt.show()

    best_index = scores.index(max(scores))
    best_num_features = number_of_features[best_index]
    best_selected_features = selected_features_list[best_index]

    print("Optimal number of features:", best_num_features)
    print("Selected features:", best_selected_features)
    return best_selected_features


def  features_selection_random_forrest(table_train):
    X_train = table_train.drop(columns=['Survived'])
    y_train = table_train['Survived']

    estimator = RandomForestClassifier(n_estimators=100)
    rfecv = RFECV(estimator=estimator, step=1,
                cv=StratifiedKFold(10), scoring='accuracy')

    rfecv.fit(X_train, y_train)


    plt.figure(figsize=(10, 6))
    plt.xlabel("Number of features")
    plt.ylabel("Cross validation score")
    plt.plot(range(1, len(rfecv.cv_results_[
            'mean_test_score']) + 1), rfecv.cv_results_['mean_test_score'])
    plt.show()

    print("Optimal number of features: {}".format(rfecv.n_features_))
    selected_features = X_train.columns[rfecv.support_]
    print("Selected features:", selected_features)

    return selected_features


def extract_features(table, selected_features):
    features = list(selected_features)
    features.append("Survived")
    return table[features]
# X_test.drop(columns=['SibSp', 'Parch', 'Embarked', "Survived"],inplace=True)

# X_test


def pca_transform(train_data, test_data):
    x_train = train_data.drop(columns=['Survived'])
    x_test = test_data.drop(columns=['Survived'])
    pca = PCA(n_components=4)
    X_train_pca = pca.fit_transform(x_train)
    X_test_pca = pca.transform(x_test)
    return (X_test_pca, X_train_pca)