import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



def visualizations(X):
    #___ Visualization ___#
    # Pclass - Good
    # Sex - Good
    # Age - Average
    # Fare - Below Average
    # SibSp - Bad
    # Parch - Bad
    # Ticket - Bad
    # Embarked - Bad

    sns.barplot(x="Pclass", y="Survived", data=X); plt.show();
    sns.barplot(x="Sex", y="Survived", data=X); plt.show();
    t = make_bins(X, 'Age', 7.5); sns.barplot(x="Age", y="Survived", data=t); plt.show();
    g = sns.FacetGrid(X, col='Survived'); g.map(plt.hist, 'Age', bins=20); plt.show();
    sns.barplot(x="SibSp", y="Survived", data=X); plt.show();
    sns.barplot(x="Parch", y="Survived", data=X); plt.show();
    t = make_bins(X, 'Fare', 10); sns.barplot(x="Fare", y="Survived", data=t); plt.show();
    sns.barplot(x="Pclass", y="Fare", data=t); plt.show();
    sns.barplot(x="Embarked", y="Survived", data=X); plt.show();


def make_bins(d, col, factor=2):
    rounding = lambda x: np.around(x / factor)
    d[col] = d[col].apply(rounding)
    return d


def create_title(X):
    # From Names, create Title (eg. Mr.)

    X['Title'] = X.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    X['Title'] = X['Title'].replace(['Lady', 'Countess','Capt', 'Col', 'Don',\
                                                'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'],\
                                                'Rare')
    X['Title'] = X['Title'].replace('Mlle', 'Miss')
    X['Title'] = X['Title'].replace('Ms', 'Miss')
    X['Title'] = X['Title'].replace('Mme', 'Mrs')

    _, X['Title'] = np.unique(X['Title'], return_inverse=True) # Convert to numeric

    return X['Title']


def create_train(X):
    # print(train_x.isnull().sum()) # Show total null values
    
    X.drop(['SibSp', 'Parch', 'Ticket', 'Embarked', 'Name', 'Cabin', 'PassengerId', 'Fare', 'Age'], inplace=True, axis=1)
    X.dropna(inplace=True) # Drop rows with null values
    _, X['Sex'] = np.unique(X['Sex'], return_inverse=True)
    
    Y = np.ravel(X.Survived) # Make 1D
    X.drop(['Survived'], inplace=True, axis=1)

    return X, Y


def write_to_csv(test_x, predictions):
    submission = pd.DataFrame({
        "PassengerId": test_x["PassengerId"],
        "Survived": predictions
    })
    submission.to_csv('submission.csv', index=False)