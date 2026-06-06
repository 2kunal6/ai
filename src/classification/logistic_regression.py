import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline


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

def handle_null_values(df):
    column_configs = {'HomePlanet': ("constant", "onehot", "missing"),
                      'Destination': ("constant", "onehot", "missing"),
                      # because most of the entries are non-VIP and it's unlikely that a VIP entry will have wrong entry
                      'VIP': ("most_frequent", "ordinal"),
                      # maybe the user did not spend anything on these services and thus those values are 0
                      'RoomService': ("constant", "none", 0),
                      'FoodCourt': ("constant", "none", 0),
                      'ShoppingMall': ("constant", "none", 0),
                      'Spa': ("constant", "none", 0),
                      'VRDeck': ("constant", "none", 0),
                      }

    transformers = []

    for col_name, config in column_configs.items():
        strategy = config[0]
        encoder_type = config[1]

        if strategy == "constant":
            fill_val = config[2]
            imputer = SimpleImputer(strategy="constant", fill_value=fill_val)
        elif strategy in ["mean", "median", "most_frequent"]:
            imputer = SimpleImputer(strategy=strategy)

        if encoder_type == "onehot":
            encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        elif encoder_type == "ordinal":
            encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        else:
            encoder = "passthrough"

        pipeline_steps = [imputer, encoder] if encoder != "passthrough" else [imputer]

        transformers.append(
            (f"{col_name}_{strategy}_{encoder_type}", make_pipeline(*pipeline_steps), [col_name])
        )

    preprocessor = ColumnTransformer(transformers=transformers, remainder="passthrough")
    return preprocessor

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
    preprocessor = handle_null_values(df)
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