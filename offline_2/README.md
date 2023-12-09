# Logistic Regression and AdaBoost for Classification

## Requirements

1. Python 3.9.1 or higher
1. Install all dependencies from requirements.txt using following command:
    ```
    pip install -r requirements.txt
    ```

## How to run

1. Download datasets from the links below and put them into `datasets` folder under `solution` directory.

    - [Telco Churn](https://www.kaggle.com/blastchar/telco-customer-churn)
    - [Adult](https://archive.ics.uci.edu/ml/datasets/adult)
    - [Credit Card Fraud](https://www.kaggle.com/mlg-ulb/creditcardfraud) 

1. Unizip the datasets. In case of `adult` dataset, make sure to keep the extracted files into a folder named `adult`.

    The directory structure should look like this:
    ```
    solution
    ├── datasets
    │   ├── adult
    │   │   ├── adult.data
    │   │   ├── adult.names
    │   │   ├── adult.test
    │   │   ├── Index
    │   │   └── old.adult.names
    │   ├── creditcard.csv
    │   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
    |── scripts
    |   ├── main.py
    |   ├...
    ```
1. Run the following command from `solution` directory:
    ```
    python main.py
    ```