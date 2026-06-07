import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_selection import SelectKBest

from util.preprocessing import get_column_transformers


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

PREPROCESSING_CONFIG = {
    # ---------- CATEGORICAL ----------
    "HomePlanet": {"imputer": {"constant": "missing"}, "encoder": "onehot"},
    "Destination": {"imputer": {"constant": "missing"}, "encoder": "onehot"},
    # because most of the entries are non-VIP and it's unlikely that a VIP entry will have wrong entry
    "VIP": {"imputer": {"inbuilt": "most_frequent"}, "encoder": "ordinal", "bool": True},
    # ---------- NUMERIC ----------
    # maybe the user did not spend anything on these services and thus those values are 0
    "RoomService": {"imputer": {"constant": 0}, "scale": True},
    "FoodCourt": {"imputer": {"constant": 0}, "scale": True},
    "ShoppingMall": {"imputer": {"constant": 0}, "scale": True},
    "Spa": {"imputer": {"constant": 0}, "scale": True},
    "VRDeck": {"imputer": {"constant": 0}, "scale": True}
}

def extract_label_column(df, label_column_name):
    y = LabelEncoder().fit_transform(df[label_column_name])
    X = df.drop(columns=[label_column_name])
    return X, y

def get_model_pipeline(preprocessor):
    return Pipeline([
        ("preprocessor", preprocessor),
        ('feature_selection', SelectKBest(k=100)),
        ("classifier", LogisticRegression(random_state=42))
    ])

def evaluate(y_test, y_pred):
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

def main():
    df = pd.read_csv("resources/spaceship-titanic/train.csv")
    test_df = pd.read_csv("resources/spaceship-titanic/test.csv")

    df = drop_columns(df)
    preprocessor = get_column_transformers(PREPROCESSING_CONFIG)
    X, y = extract_label_column(df, 'Transported')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_pipeline = get_model_pipeline(preprocessor)
    model_pipeline.fit(X_train, y_train)
    print(type(X_test))
    print(y_test[1])
    print(model_pipeline.predict(X_test.iloc[1:2]))
    # using pipeline to predict automatically applies the same transformations to X_test; hence, no need to do transform X_test separately
    y_pred = model_pipeline.predict(X_test)
    comparison_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
    print(comparison_df.head(20))
    evaluate(y_test, y_pred)

    test_ids = test_df["PassengerId"]
    test_df = drop_columns(test_df)
    y_pred_submission = model_pipeline.predict(test_df)
    y_pred_submission = y_pred_submission.astype(bool)

    '''
    param_grid = {
        'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100]
    }

    grid = GridSearchCV(
        estimator=model_pipeline,
        param_grid=param_grid,
        cv=5,
        scoring='f1',  # or 'accuracy', 'roc_auc'
        n_jobs=-1
    )

    grid.fit(X_train, y_train)
    best_pipeline = grid.best_estimator_

    y_pred = best_pipeline.predict(test_df)
    '''
    y_pred_submission = y_pred_submission.astype(bool)
    submission = pd.DataFrame({
        "PassengerId": test_ids,
        "Transported": y_pred_submission
    })
    submission.to_csv("predictions.csv", index=False)

if __name__ == '__main__':
    main()