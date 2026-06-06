import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer

#########################################################
# set display option to show all columns
pd.set_option('display.max_columns', None)
#########################################################

def drop_columns_from_train_and_test_data(df):
    '''drop because Name is null means there might be some problem with the entry
        how do we remove null values for Age?
            - if we use min/max/median then the data becomes skewed with higher frequency count for the min/max/median age
            - also there's no pattern of age to the target column - each age has balanced distributed target column True/False
        '''
    drop_columns_list = ['Name', 'Age']
    df = df.dropna(subset=[drop_columns_list])
    return df

def handle_null_values(df):
    # because most of the entries are non-VIP and it's unlikely that a VIP entry will have wrong entry
    update_with_max_frequent_value_columns = ['VIP']
    update_with_median_int_value_columns = []
    # maybe the user did not spend anything on these services and thus those values are 0
    update_with_0_int_value_columns = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    # no info on these, so just say "missing" for missing values
    update_with_missing_keyword_columns = ['HomePlanet', 'CryoSleep', 'Cabin', 'Destination']

    preprocessor = ColumnTransformer(
        transformers=[
            (
                'most_frequent',
                SimpleImputer(strategy='most_frequent'),
                update_with_most_frequent_value_columns
            ),
            (
                'median',
                SimpleImputer(strategy='median'),
                update_with_median_int_value_columns
            ),
            (
                'zero',
                SimpleImputer(strategy='constant', fill_value=0),
                update_with_0_int_value_columns
            ),
            (
                'missing',
                Pipeline([
                    ('imputer',
                     SimpleImputer(
                         strategy='constant',
                         fill_value='missing'
                     )),
                    ('encoder',
                     OneHotEncoder(handle_unknown='ignore'))
                ]),
                update_with_missing_columns
            )
        ],
        remainder='passthrough'
    )
    return preprocessor

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
    df = pd.read_csv("../resources/spaceship-titanic/train.csv")
    test_df = pd.read_csv("../resources/spaceship-titanic/test.csv")

    df = drop_columns_from_train_and_test_data(df)
    test_df = drop_columns_from_train_and_test_data(test_df)

    preprocessor = handle_null_values(df)
    X, y = extract_label_column(df, 'Transported')
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