import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_selection import SelectKBest

from sklearn.model_selection import GridSearchCV

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
    #drop_columns_list = ['id', 'g', 'r', 'i']
    drop_columns_list = ['id']
    df = df.drop(columns=drop_columns_list)
    return df

PREPROCESSING_CONFIG = {
    # ---------- CATEGORICAL ----------
    "spectral_type": {"encoder": "onehot"},
    "galaxy_population": {"encoder": "onehot"},
    # ---------- NUMERIC ----------
    "alpha": {"scale": True},
    "delta": {"scale": True},
    "u": {"scale": True},
    "z": {"scale": True},
    "g": {"scale": True},
    "r": {"scale": True},
    "i": {"scale": True},
    "redshift": {"scale": True}
}

def get_model_pipeline(preprocessor):
    return Pipeline([
        ("preprocessor", preprocessor),
        #('classifier', SVC(kernel='linear')),
        #('classifier', SVC(kernel='rbf')),
        #('classifier', LinearSVC()),
        #('classifier', DecisionTreeClassifier())
        #('classifier', RandomForestClassifier())
        #('classifier', ExtraTreesClassifier())
        #('classifier', GradientBoostingClassifier())
        ('classifier', XGBClassifier())
        #('classifier', GaussianNB())
        #('classifier', KNeighborsClassifier())
        #('classifier', MLPClassifier())
        #('feature_selection', SelectKBest(k=100)),
        #('classifier', LogisticRegression(random_state=42))
    ])

def evaluate(y_test, y_pred):
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

def main():
    df = pd.read_csv("resources/playground-series-s6e6/train.csv")
    test_df = pd.read_csv("resources/playground-series-s6e6/test.csv")

    df = drop_columns(df)
    print('columns dropped')
    preprocessor = get_column_transformers(PREPROCESSING_CONFIG)
    target_label = 'class'
    submission_id_label = "id"

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df[target_label])
    X = df.drop(columns=[target_label])


    print('target label extracted')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print('train test splitted')

    model_pipeline = get_model_pipeline(preprocessor)
    print('fitting model')
    model_pipeline.fit(X_train, y_train)
    print(type(X_test))
    print(y_test[1])
    print(model_pipeline.predict(X_test.iloc[1:2]))
    # using pipeline to predict automatically applies the same transformations to X_test; hence, no need to do transform X_test separately
    print('predicting')
    y_pred = model_pipeline.predict(X_test)
    comparison_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
    print(comparison_df.head(20))
    evaluate(y_test, y_pred)

    test_ids = test_df[submission_id_label]
    test_df = drop_columns(test_df)
    y_pred_submission = model_pipeline.predict(test_df)

    scoring = 'f1'
    param_grid = {
        'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100]
    }
    if(str(model_pipeline.steps[1][1]) == 'MLPClassifier()'):
        param_grid = {
            'classifier__hidden_layer_sizes': [(50,), (100,), (50, 50)],
            'classifier__activation': ['relu', 'tanh'],
            'classifier__alpha': [0.0001, 0.001, 0.01],
            'classifier__learning_rate_init': [0.001, 0.01]
        }
    if(str(model_pipeline.steps[1][1]) == 'XGBClassifier()'):
        param_grid = {
            'classifier__max_depth': [3, 4, 5, 6, 8, 10],
            'classifier__learning_rate': [0.01, 0.03, 0.05, 0.1],
            'classifier__n_estimators': [200, 500, 1000],
            'classifier__subsample': [0.6, 0.8, 1.0],
            'classifier__colsample_bytree': [0.6, 0.8, 1.0]
        }
        scoring = 'f1_weighted'

    grid = GridSearchCV(
        estimator=model_pipeline,
        param_grid=param_grid,
        cv=5,
        scoring=scoring,  # or 'accuracy', 'roc_auc'
        n_jobs=-1
    )

    grid.fit(X_train, y_train)
    best_pipeline = grid.best_estimator_

    y_pred_submission = best_pipeline.predict(test_df)

    y_pred_submission = label_encoder.inverse_transform(y_pred_submission)
    submission = pd.DataFrame({
        submission_id_label: test_ids,
        target_label: y_pred_submission
    })
    submission.to_csv("predictions.csv", index=False)

if __name__ == '__main__':
    main()