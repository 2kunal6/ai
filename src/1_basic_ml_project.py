'''
TODO:
- sklearn's k-fold cross validation
- sklearn's grid search for hyperparameter finetuning, RandomizedSearchCV
- find out how confident the predictions are
'''


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

#########################################################
# set display option to show all columns
pd.set_option('display.max_columns', None)
#########################################################


def get_basic_info(df):
    print('#' * 25)
    print('#' * 25)
    print('INFO:\n')
    print(df.info())
    print('#' * 25)
    print(f'SOME SAMPLE VALUES:')
    print(df.head())
    print('#' * 25)
    print('NULL COLUMNS:')
    print(df.isnull().sum())
    print('#' * 25)
    print('#' * 25)


def handle_null_values(df):
    '''
    - No null values so removing them is not required.
    - choices for removal: 
        - remove rows
        - remove columns with a lot of nulls
        - imputation (replace null values with mean, median, embedding, etc.)
        - when there's a strong correlation between missing values and target label, mark it as MISSING to preserve the correlation
            - note: keeping 100% correlated column might be bad because this might overfit with those columns
    '''
    pass

def extract_label_column(df, label_column_name):
    y = df[label_column_name]
    X = df.drop(columns=[label_column_name])
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    return X, y_encoded

def convert_categorical_to_numerical_converter(X):
    categorical_cols = X.select_dtypes(include=["object", "category", "string"]).columns

    numerical_cols = X.select_dtypes(exclude=["object", "category", "string"]).columns

    # Preprocessing
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            # extend to standardize numerical columns
            ("num", "passthrough", numerical_cols)
        ]
    )

def get_model_pipeline(preprocessor):
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ))
    ])

def evaluate(y_test, y_pred):
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

def main():
    df = pd.read_csv("resources/AI_Impact_on_Jobs_2030.csv")
    get_basic_info(df)
    handle_null_values(df)
    X, y_encoded = extract_label_column(df, 'Hiring_Trend_2026')
    preprocessor = convert_categorical_to_numerical_converter(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    model_pipeline = get_model_pipeline(preprocessor)
    model_pipeline.fit(X_train, y_train)
    # using pipeline to predict automatically applies the same transformations to X_test; hence, no need to do transform X_test separately
    y_pred = model_pipeline.predict(X_test)

    evaluate(y_test, y_pred)


if __name__ == '__main__':
    main()