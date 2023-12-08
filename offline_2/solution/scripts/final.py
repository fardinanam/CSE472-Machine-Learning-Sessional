import pandas as pd
import numpy as np
from preprocessing import Preprocessor
from sklearn.model_selection import train_test_split
from mutual_classifier import info_gain_score, SelectKBest
from adaboost import AdaBoost
from logistic_regression import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.preprocessing import MinMaxScaler

class TelcoChurnPreprocessor(Preprocessor):
    def __init__(self) -> None:
        data = pd.read_csv('../datasets/WA_Fn-UseC_-Telco-Customer-Churn.csv')
        self.y = data['Churn'].map({'Yes': 1, 'No': 0})
        data.drop(['customerID', 'Churn'], axis=1, inplace=True)
        # convert TotalCharges to numeric
        data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')

        super().__init__(data)

    def preprocess(self) -> list:
        self.fill_missing_values()
        self.normalize()
        self.encode_categorical_features()

        return self.data, self.y
    
    def split(self) -> list:
        return train_test_split(self.data, self.y, test_size=0.2, random_state=42)

class CreditCardPreprocessor(Preprocessor):
    def __init__(self) -> None:
        data = pd.read_csv('../datasets/creditcard.csv')

        # only keep 20,000 of the negative examples and all other positive examples
        data = pd.concat([data[data['Class'] == 1], data[data['Class'] == 0].sample(n=20000, random_state=42)])
        self.y = data['Class']
        data.drop(['Class'], axis=1, inplace=True)

        super().__init__(data)

    def split(self) -> list:
        return train_test_split(self.data, self.y, test_size=0.2, random_state=42)
    
    def preprocess(self) -> list:
        self.fill_missing_values()
        self.normalize()

        return self.data, self.y

class AdultPreprocessor(Preprocessor):
    def __init__(self) -> None:
        # read train dataset
        data_train = pd.read_csv('../datasets/adult/adult.data', header=None)
        data_train.columns = ['age', 'workclass', 'fnlwgt', 'education', 
            'education-num', 'marital-status', 'occupation', 'relationship',
            'race', 'sex', 'capital-gain','capital-loss', 'hours-per-week',
            'native-country', 'income'
        ]

        # read test dataset
        data_test = pd.read_csv('../datasets/adult/adult.test', header=None, skiprows=1)

        data_test.columns = data_train.columns

        self.X_train = data_train[data_train.columns[:-1]]
        self.y_train = data_train[data_train.columns[-1]]

        self.X_test = data_test[data_test.columns[:-1]]
        self.y_test = data_test[data_test.columns[-1]]

        self.numeric_features = self.X_train.select_dtypes(include=np.number).columns
        self.categorical_features = list(set(self.X_train.columns) - set(self.numeric_features))

    def fill_missing_values(self) -> None:
        """
        First fill missing values in the train dataset and then fill missing values in the test dataset
        
        1. Fill missing values of numerical features with mean of the column of the train dataset
        2. Fill missing values of categorical features with "UNKNOWN_VALUE"
        """
        # replace whitespaced cells of numeric columns with NaN
        self.X_train.replace(r'^\s*$', np.nan, regex=True, inplace=True)
        self.X_test.replace(r'^\s*$', np.nan, regex=True, inplace=True)

        # find mean of numeric columns of train dataset
        numeric_features = self.numeric_features

        # fill missing values of numeric columns with mean of the column
        if len(numeric_features) > 0:
            imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
            self.X_train[numeric_features] = imputer.fit_transform(self.X_train[numeric_features])
            self.X_test[numeric_features] = imputer.transform(self.X_test[numeric_features])

        # fill missing values of categorical columns with "UNKNOWN_VALUE"
        categorical_features = self.categorical_features

        if len(categorical_features) > 0:
            self.X_train[categorical_features] = self.X_train[categorical_features].fillna("UNKNOWN_VALUE")
            self.X_test[categorical_features] = self.X_test[categorical_features].fillna("UNKNOWN_VALUE")

        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def encode_categorical_features(self) -> pd.DataFrame:
        """
        Encode categorical features of train and test datasets
        
        First encode categorical features of train dataset and then encode categorical features of test dataset keeping the same encoding
        """
        self.y_test = self.y_test.str.replace(".", "")

        encoder = LabelEncoder()
        encoder.fit(self.y_train)
        self.y_train = encoder.transform(self.y_train)
        self.y_test = encoder.transform(self.y_test)

        # convert to DataFrame
        self.y_train = pd.DataFrame(self.y_train)
        self.y_test = pd.DataFrame(self.y_test)
        
        # encode categorical features of train dataset
        categorical_features = self.categorical_features

        if len(categorical_features) > 0:
            encoder = OneHotEncoder(handle_unknown='ignore')
            encoder.fit(self.X_train[categorical_features])

            self.X_train = encoder.transform(self.X_train[categorical_features])
            self.X_test = encoder.transform(self.X_test[categorical_features])

        # convert to pandas dataframe
        self.X_train = pd.DataFrame(self.X_train.toarray())
        self.X_test = pd.DataFrame(self.X_test.toarray())

        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def normalize(self, scaler = MinMaxScaler) -> pd.DataFrame:    
        """
        Normalize numeric features of train and test datasets
        
        First normalize numeric features of train dataset and then normalize numeric features of test dataset keeping the same normalization
        """
        numeric_features = self.numeric_features

        if len(numeric_features) > 0:
            sc = scaler()
            sc.fit(self.X_train[numeric_features])

            self.X_train[numeric_features] = sc.transform(self.X_train[numeric_features])
            self.X_test[numeric_features] = sc.transform(self.X_test[numeric_features])

        return self.X_train, self.X_test, self.y_train, self.y_test

    def preprocess(self) -> list:
        self.fill_missing_values()
        self.normalize()
        self.encode_categorical_features()

        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def split(self) -> list:
        return self.X_train, self.X_test, self.y_train, self.y_test

def select_k_features(X_train : np.ndarray, X_test : np.ndarray, Y_train : np.ndarray, k : int = 10) -> list:
    k = min(k, X_train.shape[1])

    selector = SelectKBest(score_func=info_gain_score, k=k)

    fit = selector.fit(X_train, Y_train)
    X_train = fit.transform(X_train)

    X_test = fit.transform(X_test)

    return X_train, X_test

def specificity_score(x, y):
    """
    Calculate the specificity score of the confusion matrix
    """
    cm = confusion_matrix(x, y)
    tn, fp, fn, tp = cm.ravel()

    return tn / (tn + fp)

def accuracy_score(x, y):
    """
    Calculate the accuracy score of the confusion matrix
    """
    cm = confusion_matrix(x, y)
    tn, fp, fn, tp = cm.ravel()

    return (tp + tn) / (tp + tn + fp + fn)

def precision_score(x, y):
    """
    Calculate the precision score of the confusion matrix
    """
    cm = confusion_matrix(x, y)
    tn, fp, fn, tp = cm.ravel()

    return tp / (tp + fp)

def recall_score(x, y):
    """
    Calculate the recall score of the confusion matrix
    """
    cm = confusion_matrix(x, y)
    tn, fp, fn, tp = cm.ravel()

    return tp / (tp + fn)

def f1_score(x, y):
    """
    Calculate the f1 score of the confusion matrix
    """
    cm = confusion_matrix(x, y)
    tn, fp, fn, tp = cm.ravel()

    return (2 * tp) / (2 * tp + fp + fn)

def false_discovery_rate(x, y):
    """
    Calculate the false discovery rate of the confusion matrix
    """
    cm = confusion_matrix(x, y)
    tn, fp, fn, tp = cm.ravel()

    return fp / (tp + fp)

def report(y_train_pred, y_train, y_test_pred, y_test):
    # Calculate scores for the training set
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_recall = recall_score(y_train, y_train_pred)
    train_specificity = specificity_score(y_train, y_train_pred)
    train_precision = precision_score(y_train, y_train_pred)
    train_false_discovery_rate = false_discovery_rate(y_train, y_train_pred)
    train_f1 = f1_score(y_train, y_train_pred)

    # Calculate scores for the test set
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_recall = recall_score(y_test, y_test_pred)
    test_specificity = specificity_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred)
    test_false_discovery_rate = false_discovery_rate(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)

    # Create a DataFrame with the scores
    scores = pd.DataFrame({
        'Train Set': [train_accuracy, train_recall, train_specificity, train_precision, train_false_discovery_rate, train_f1],
        'Test Set': [test_accuracy, test_recall, test_specificity, test_precision, test_false_discovery_rate,  test_f1]
    }, index=['Accuracy', 'Recall', 'Specificity', 'Precision', 'False Discovery Rate', 'F1 Score'])

    return scores

if __name__ == "__main__":
    dataset = 0

    while dataset not in range(1, 4):
        print("Select dataset:")
        print("1. Telco Churn")
        print("2. Adult")
        print("3. Credit Card Fraud")
        print("0. to exit")

        dataset = int(input("Enter your choice: "))
        if dataset == 0:
            exit(0)

    preprocessor = None

    if dataset == 1:
        preprocessor = TelcoChurnPreprocessor()
    elif dataset == 2:
        preprocessor = AdultPreprocessor()
    elif dataset == 3:
        preprocessor = CreditCardPreprocessor()

    print("\nPreprocessing...")
    preprocessor.preprocess()
    
    print("Splitting...")
    X_train, X_test, Y_train, Y_test = preprocessor.split()

    # convert to numpy arrays
    X_train = X_train.values
    X_test = X_test.values
    Y_train = Y_train.values
    Y_test = Y_test.values

    print(f"\nThe dataset contains {X_train.shape[1]} features")
    k = -1

    while k not in range(1, X_train.shape[1] + 1):
        k = int(input(f"Enter the number of features to select: "))
        if k not in range(1, X_train.shape[1] + 1):
            print(f"Please enter a number between 1 and {X_train.shape[1]} or press Ctrl+C to exit")

    # select k best features
    print("\nSelecting features...")
    X_train, X_test = select_k_features(X_train, X_test, Y_train, k=k)

    k = int(input("\nEnter the number of epochs to train the model: "))
    # train the model
    print("\nTraining...")
    model = AdaBoost(LogisticRegression, error_threshold=0.5)
    model.fit(X_train, Y_train, epochs=k)
    print("Done training!")

    y_pred = model.predict(X_test)
    
    print("\nGenerating report...")
    scores = report(model.predict(X_train), Y_train, y_pred, Y_test)

    print(scores)

# if __name__ == "__main__":
#     # for every dataset, for epochs of  5, 10, 15 and 20 run adaboost and generate report to a csv file
#     datasets = [TelcoChurnPreprocessor, AdultPreprocessor, CreditCardPreprocessor]

#     epochs = [5, 10, 15, 20]

#     for dataset in datasets:
#         preprocessor = dataset()
#         preprocessor.preprocess()
#         X_train, X_test, Y_train, Y_test = preprocessor.split()

#         # convert to numpy arrays
#         X_train = X_train.values
#         X_test = X_test.values
#         Y_train = Y_train.values
#         Y_test = Y_test.values

#         for epoch in epochs:
#             # select k best features
#             X_train, X_test = select_k_features(X_train, X_test, Y_train, k=20)

#             # train the model
#             model = AdaBoost(LogisticRegression, error_threshold=0.5)
#             model.fit(X_train, Y_train, epochs=epoch)

#             y_pred = model.predict(X_test)
            
#             scores = report(model.predict(X_train), Y_train, y_pred, Y_test)

#             scores.to_csv(f"../results/{dataset.__name__}_{epoch}.csv")

#             print(f"Done for {dataset.__name__} with {epoch} epochs")