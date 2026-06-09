import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

from preprocessing import get_column_transformers


#########################################################
# config
TRAIN_DATASET = 'resources/nlp-getting-started/train.csv'
TEST_DATASET = 'resources/nlp-getting-started/test.csv'
TARGET_LABEL = 'target'
TEXT_COLUMNS = ['text', 'location', 'keyword']
COMBINED_TEXT_COLUMN = 'combined_text_column'
DROP_COLUMNS = ['id']
PREPROCESSING_CONFIG = {
    COMBINED_TEXT_COLUMN: {"encoder": "tfidf"}
}
#########################################################


#########################################################
# set display option to show all columns
pd.set_option('display.max_columns', None)
#########################################################


def get_model_pipeline(preprocessor):
    return Pipeline([
        ("preprocessor", preprocessor),
        ('classifier', LinearSVC()),
        #('classifier', XGBRegressor())
        #('classifier', RandomForestRegressor())
        #('classifier', MLPRegressor())
        #('classifier', Ridge())
        #('classifier', SVR())
    ])

def main():
    df = pd.read_csv(TRAIN_DATASET)
    test_df = pd.read_csv(TEST_DATASET)

    df[COMBINED_TEXT_COLUMN] = (df[TEXT_COLUMNS].fillna("").agg(" [SEP] ".join, axis=1))
    df = df.drop(columns=TEXT_COLUMNS)

    df = df.drop(columns=DROP_COLUMNS)
    print('columns dropped')

    preprocessor = get_column_transformers(PREPROCESSING_CONFIG)

    y = df[TARGET_LABEL]
    X = df.drop(columns=[TARGET_LABEL])
    print('target label extracted')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print('train test splitted')

    model_pipeline = get_model_pipeline(preprocessor)
    print('fitting model')
    model_pipeline.fit(X_train, y_train)

    print(model_pipeline.predict(X_test.iloc[1:2]))
    # using pipeline to predict automatically applies the same transformations to X_test; hence, no need to do transform X_test separately
    print('predicting')
    y_pred = model_pipeline.predict(X_test)
    comparison_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
    print(comparison_df.head(10))
    print(classification_report(y_test, y_pred))

    submission_id_label = "id"
    test_ids = test_df[submission_id_label]

    test_df[COMBINED_TEXT_COLUMN] = (test_df[TEXT_COLUMNS].fillna("").agg(" [SEP] ".join, axis=1))
    test_df = test_df.drop(columns=TEXT_COLUMNS)

    test_df = test_df.drop(columns=DROP_COLUMNS)
    y_pred_submission = model_pipeline.predict(test_df)

    '''scoring = 'f1'
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

    y_pred_submission = best_pipeline.predict(test_df)'''

    submission = pd.DataFrame({
        submission_id_label: test_ids,
        TARGET_LABEL: y_pred_submission
    })
    submission.to_csv("predictions.csv", index=False)

if __name__ == '__main__':
    main()