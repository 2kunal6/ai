import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

from logistic_regression_preprocessing import get_column_transformers


#########################################################
# set display option to show all columns
pd.set_option('display.max_columns', None)
#########################################################

def drop_columns(df):
    '''
    drop because Name is null means there might be some problem with the entry
    how do we remove null values for Age?
        - if we use min/max/median then the data becomes skewed with higher frequency count for the min/max/median age
        - also there's no pattern of age to the target column - each age has balanced distributed target column True/False
    '''
    drop_columns_list = ['Name', 'Age', 'Cabin', 'CryoSleep']
    df = df.drop(columns=drop_columns_list)
    return df

def extract_label_column(df, label_column_name):
    y = LabelEncoder().fit_transform(df[label_column_name])
    X = df.drop(columns=[label_column_name])
    return X, y

def get_model_pipeline(preprocessor):
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(random_state=42))
    ])

def evaluate(y_test, y_pred):
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

def main():
    train_df = pd.read_csv("../resources/spaceship-titanic/train.csv")
    test_df = pd.read_csv("../resources/spaceship-titanic/test.csv")
    df = pd.concat([train_df, test_df])

    df = drop_columns(df)
    preprocessor = get_column_transformers()
    X, y = extract_label_column(df, 'Transported')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_pipeline = get_model_pipeline(preprocessor)
    model_pipeline.fit(X_train, y_train)
    #print(type(X_test))
    #print(y_test.iloc[0])
    #print(model_pipeline.predict(X_test.iloc[0:1]))
    # using pipeline to predict automatically applies the same transformations to X_test; hence, no need to do transform X_test separately
    y_pred = model_pipeline.predict(X_test)

    evaluate(y_test, y_pred)


if __name__ == '__main__':
    main()