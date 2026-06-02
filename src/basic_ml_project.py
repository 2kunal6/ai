import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


pd.set_option('display.max_columns', None)

df = pd.read_csv("AI_Impact_on_Jobs_2030.csv")
print(df.info())
print(df.head())
print('\n\nnull values:')
print(df.isnull().sum())
print('\n\n')

# No null values so removing them is not required.
# choices for removal: remove rows, remove columns with a lot of nulls, imputation (replace null values with mean, median, embedding, etc.)

train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

categorical_cols = df.select_dtypes(
    include=["object", "category"]
).columns.tolist()


preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols),
    ]
)

df_numerical = preprocessor.fit_transform(df)
print(df_numerical)