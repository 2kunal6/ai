from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.pipeline import Pipeline


def get_imputer(strategy):
    if "inbuilt" in strategy:
        return SimpleImputer(strategy=strategy["inbuilt"])
    return SimpleImputer(strategy="constant", fill_value=strategy['constant'])

def get_encoder(strategy):
    if strategy == "onehot":
        return OneHotEncoder(handle_unknown="ignore")
    if strategy == "ordinal":
        if("bool" in strategy and strategy["bool"] is True):
            return OrdinalEncoder(categories = [[False, True]], handle_unknown="use_encoded_value", unknown_value=-1)
        return OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

def get_column_transformers(PREPROCESSING_CONFIG):
    transformers = []
    for column, config in PREPROCESSING_CONFIG.items():
        steps = []
        if ("imputer" in config):
            steps.append(("imputer", get_imputer(config["imputer"])))
        if("encoder" in config):
            steps.append(("encoder", get_encoder(config.get("encoder"))))
        if("scale" in config):
            if config["scale"] == True:
                steps.append(("scaler", StandardScaler()))

        transformers.append((column, Pipeline(steps), [column]))

    return ColumnTransformer(transformers=transformers, remainder="drop")