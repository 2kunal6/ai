import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


df = pd.read_csv('store-sales-time-series-forecasting/train.csv')
test_df = pd.read_csv('store-sales-time-series-forecasting/test.csv')
df['time'] = np.arange(len(df.index))
test_df['time'] = np.arange(len(test_df.index))


# Training data
X = df.loc[:, ['time']]  # features
y = df.loc[:, 'sales']  # target

# Train the model
model = LinearRegression()
model.fit(X, y)


submission_id_label = "id"
test_ids = test_df[submission_id_label]
y_pred = pd.Series(model.predict(test_df.loc[:, ['time']]))

submission = pd.DataFrame({
    submission_id_label: test_ids,
    'sales': y_pred
})
submission.to_csv("predictions.csv", index=False)