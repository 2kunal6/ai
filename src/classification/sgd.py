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
from sklearn.linear_model import SGDClassifier
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
    correlated_columns below are null for hallucination_label = 0 and notnull for hallucination_label = 1
    TODO: Remove these exactly correlated values
    '''
    correlated_columns = ['hallucination_type', 'hallucination_span', 'correct_information', 'correction_text', 'severity', 'intrinsic_or_extrinsic']
    for correlated_column in correlated_columns:
        df[correlated_column] = df[correlated_column].notna().astype(int)
    df['mitigation_strategy'] = df['mitigation_strategy'].fillna('none')
    df = df.drop(columns=['notes'])
    return df

def extract_label_column(df, label_column_name):
    y = df[label_column_name]
    X = df.drop(columns=[label_column_name])
    return X, y

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
        ("classifier", SGDClassifier(random_state=42))
    ])

def evaluate(y_test, y_pred):
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

def main():
    df = pd.read_csv("../resources/llm_hallucination_dataset_v1.csv")
    get_basic_info(df)
    df = handle_null_values(df)
    X, y = extract_label_column(df, 'hallucination_label')
    preprocessor = convert_categorical_to_numerical_converter(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_pipeline = get_model_pipeline(preprocessor)
    model_pipeline.fit(X_train, y_train)
    print(type(X_test))
    print(y_test.iloc[0])
    print(model_pipeline.predict(X_test.iloc[0:1]))
    # using pipeline to predict automatically applies the same transformations to X_test; hence, no need to do transform X_test separately
    y_pred = model_pipeline.predict(X_test)

    evaluate(y_test, y_pred)


if __name__ == '__main__':
    main()