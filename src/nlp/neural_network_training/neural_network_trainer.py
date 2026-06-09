import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score

from datasets import Dataset

from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    BertTokenizerFast,
    BertForSequenceClassification,
    TrainingArguments,
    Trainer
)

import numpy as np

#########################################################
# config
TRAIN_DATASET = 'nlp-getting-started/train.csv'
TEST_DATASET = 'nlp-getting-started/test.csv'
TARGET_LABEL = 'target'
NUM_LABELS = 2
TEXT_COLUMNS = ['text', 'location', 'keyword']
COMBINED_TEXT_COLUMN = 'combined_text_column'
DROP_COLUMNS = ['id']
PREPROCESSING_CONFIG = {
    COMBINED_TEXT_COLUMN: {"encoder": "tfidf"}
}
MODEL_NAME = 'bert-base-uncased'
# MODEL_NAME = 'distilbert-base-uncased'
#########################################################


#########################################################
# set display option to show all columns
pd.set_option('display.max_columns', None)
#########################################################


def get_model_pipeline(preprocessor):
    return Pipeline([
        ("preprocessor", preprocessor),
        #('classifier', LinearSVC()),
        #('classifier', SGDClassifier())
        #('classifier', MultinomialNB())
        #('classifier', RandomForestClassifier())
        #('classifier', Ridge())
        #('classifier', SVR())
    ])

def convert_to_hf_datasets(train_df, test_df):
    train_dataset = Dataset.from_pandas(train_df[[COMBINED_TEXT_COLUMN, TARGET_LABEL]])
    test_dataset = Dataset.from_pandas(test_df[[COMBINED_TEXT_COLUMN, TARGET_LABEL]])
    train_dataset = train_dataset.rename_column(TARGET_LABEL, "labels")
    test_dataset = test_dataset.rename_column(TARGET_LABEL, "labels")
    return train_dataset, test_dataset


def tokenize(train_dataset, test_dataset):
    #tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    def tokenize_function(examples):
        return tokenizer(examples[COMBINED_TEXT_COLUMN], truncation=True, padding="max_length", max_length=256)

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function,batched=True)

    return train_dataset, test_dataset

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)

    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1_macro": f1_score(labels, predictions, average="macro")
    }

def train(train_dataset, test_dataset):
    #model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
    training_args = TrainingArguments(output_dir=f"./{MODEL_NAME}", eval_strategy="epoch", save_strategy="epoch", num_train_epochs=2, per_device_train_batch_size=8, per_device_eval_batch_size=16, weight_decay=0.01, logging_steps=100, load_best_model_at_end=True)
    trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=test_dataset, compute_metrics=compute_metrics)
    trainer.train(resume_from_checkpoint=True)
    return trainer

def main():
    df = pd.read_csv(TRAIN_DATASET)
    eval_df = pd.read_csv(TEST_DATASET)

    df[COMBINED_TEXT_COLUMN] = (df[TEXT_COLUMNS].fillna("").agg(" [SEP] ".join, axis=1))
    df = df.drop(columns=TEXT_COLUMNS)

    df = df.drop(columns=DROP_COLUMNS)
    print('columns dropped')

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    print('train test splitted')

    train_df, test_df = convert_to_hf_datasets(train_df, test_df)
    print('converted to hf datasets')

    train_df, test_df = tokenize(train_df, test_df)
    print('data tokenized')
    trainer = train(train_df, test_df)
    results = trainer.evaluate(test_df)
    print(results)

    submission_id_label = "id"
    submission_ids = eval_df[submission_id_label]
    eval_df[COMBINED_TEXT_COLUMN] = (eval_df[TEXT_COLUMNS].fillna("").agg(" [SEP] ".join, axis=1))
    eval_df = eval_df.drop(columns=TEXT_COLUMNS)
    eval_df = eval_df.drop(columns=DROP_COLUMNS)
    eval_df = Dataset.from_pandas(eval_df[[COMBINED_TEXT_COLUMN]])
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    def tokenize_function(examples):
        return tokenizer(examples[COMBINED_TEXT_COLUMN], truncation=True, padding="max_length", max_length=256)

    eval_df = eval_df.map(tokenize_function, batched=True)
    predictions = trainer.predict(eval_df)
    y_pred = np.argmax(
        predictions.predictions,
        axis=1
    )
    print(y_pred)
    submission = pd.DataFrame({
        submission_id_label: submission_ids,
        TARGET_LABEL: y_pred
    })
    submission.to_csv("predictions.csv", index=False)


if __name__ == '__main__':
    main()