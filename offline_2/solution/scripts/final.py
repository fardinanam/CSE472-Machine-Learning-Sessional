import pandas as pd
import numpy as np
from preprocessing import Preprocessor
from sklearn.model_selection import train_test_split
from mutual_classifier import info_gain_score, SelectKBest
from adaboost import AdaBoost
from logistic_regression import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_selection import mutual_info_classif

class TelcoChurnPreprocessor(Preprocessor):
    def __init__(self) -> None:
        data = pd.read_csv('../datasets/WA_Fn-UseC_-Telco-Customer-Churn.csv')
        self.y = data['Churn'].map({'Yes': 1, 'No': 0})
        data.drop(['customerID', 'Churn'], axis=1, inplace=True)

        super().__init__(data)

    def preprocess(self) -> list:
        self.fill_missing_values()
        self.normalize()
        self.encode_categorical_features()

        return self.data, self.y

class CreditCardPreprocessor(Preprocessor):
    def __init__(self) -> None:
        data = pd.read_csv('../datasets/creditcard.csv')
        self.y = data['Class']
        data.drop(['Class'], axis=1, inplace=True)

        super().__init__(data)
    
    def preprocess(self) -> list:
        self.fill_missing_values()
        self.normalize()

        return self.data, self.y

def select_k_features(X_train : np.ndarray, X_test : np.ndarray, Y_train : np.ndarray, k : int = 10) -> list:
    k = min(k, X_train.shape[1])

    selector = SelectKBest(score_func=mutual_info_classif, k=k)

    fit = selector.fit(X_train, Y_train)
    X_train = fit.transform(X_train)

    X_test = fit.transform(X_test)

    return X_train, X_test

if __name__ == "__main__":
    dataset = 0

    while dataset not in range(1, 3):
        print("Select dataset:")
        print("1. Telco Churn")
        print("2. Credit Card Fraud")
        print("0. to exit")
        dataset = int(input("Enter your choice: "))
        if dataset == 0:
            exit(0)

    preprocessor = None

    if dataset == 1:
        preprocessor = TelcoChurnPreprocessor()
    elif dataset == 2:
        preprocessor = CreditCardPreprocessor()

    print("Preprocessing...")
    preprocessor.preprocess()
    
    print("Splitting...")
    X_train, X_test, Y_train, Y_test = train_test_split(preprocessor.data, preprocessor.y, test_size=0.2, random_state=51)

    # convert to numpy arrays
    X_train = X_train.values
    X_test = X_test.values
    Y_train = Y_train.values
    Y_test = Y_test.values

    print(f"The dataset contains {X_train.shape[1]} features")
    k = -1

    while k not in range(1, X_train.shape[1] + 1):
        k = int(input(f"Enter the number of features to select: "))
        if k not in range(1, X_train.shape[1] + 1):
            print(f"Please enter a number between 1 and {X_train.shape[1]} or press Ctrl+C to exit")

    # select k best features
    print("Selecting features...")
    X_train, X_test = select_k_features(X_train, X_test, Y_train, k=k)

    # train the model
    print("Training...")
    model = AdaBoost(LogisticRegression, error_threshold=0.5)
    model.fit(X_train, Y_train, epochs=10)
    print("Done training!")

    y_pred = model.predict(X_test)
    
    print("Accuracy:", accuracy_score(Y_test, y_pred))
    print("Precision:", precision_score(Y_test, y_pred))
    print("Recall:", recall_score(Y_test, y_pred))
    print("F1:", f1_score(Y_test, y_pred))
