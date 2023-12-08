import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class Preprocessor:
    def __init__(self, data : pd.DataFrame) -> None:
        self.data = data
        self.numeric_features = data.select_dtypes(include=[np.number]).columns
        self.categorical_features = list(set(data.columns) - set(self.numeric_features))

    def fill_missing_values(self) -> pd.DataFrame:
        """
        Fill missing values in the data
        
        1. Fill missing values of numerical features with mean of the column
        2. Fill missing values of categorical features with "UNKNOWN_VALUE"
        """
        # replace whitespaced cells of numeric columns with NaN
        self.data.replace(r'^\s*$', np.nan, regex=True, inplace=True)

        # fill missing values of numeric columns with mean of the column
        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        self.data[self.numeric_features] = imputer.fit_transform(self.data[self.numeric_features])

        # consider all columns other than numeric columns as categorical columns
        categorical_cells = list(set(self.data.columns) - set(self.numeric_features))

        # fill missing values of categorical columns with "UNKNOWN_VALUE"
        self.data[categorical_cells] = self.data[categorical_cells].fillna("UNKNOWN_VALUE")

        return self.data

    def encode_categorical_features(self) -> pd.DataFrame:
        """
        Encode categorical features
        
        1. Encode to 0 or 1 if there are only two categories
        2. One-hot encode if there are more than two categories
        """
        # if no categorical features, return the dataframe as it is
        if not self.categorical_features:
            return self.data
        
        # one hot encode categorical features
        dummies = pd.get_dummies(self.data[self.categorical_features], drop_first=True)

        # drop the original categorical features from the dataframe
        self.data = self.data.drop(self.categorical_features, axis=1)
        
        # concat the encoded categorical features with the dataframe
        self.data = pd.concat([self.data, dummies.astype(int)], axis=1)

        return self.data
    
    def normalize(self, scaler = MinMaxScaler) -> pd.DataFrame:
        """
        Normalize the numeric features
        """
        # normalize numeric features
        scaler = scaler()
        self.data[self.numeric_features] = scaler.fit_transform(self.data[self.numeric_features])

        return self.data

if __name__ == "__main__":
    dataset = pd.read_csv('../datasets/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    X = dataset[dataset.columns[1:-1]]
    preprocessor = Preprocessor(X)
    X = preprocessor.fill_missing_values()
    X = preprocessor.encode_categorical_features()
    X = preprocessor.normalize()
    print (X.head())