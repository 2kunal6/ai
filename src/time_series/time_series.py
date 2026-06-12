import pandas as pd
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.deterministic import DeterministicProcess


df = pd.read_csv('store-sales-time-series-forecasting/train.csv')
test_df = pd.read_csv('store-sales-time-series-forecasting/test.csv')

'''
using DeterministicProcess 
df['time'] = np.arange(len(df.index))
test_df['time'] = np.arange(len(test_df.index))
'''

dp = DeterministicProcess(
    index=df.index,  # dates from the training data
    constant=True,       # dummy feature for the bias (y_intercept)
    order=1,             # the time dummy (trend)
    drop=True,           # drop terms if necessary to avoid collinearity
)
X = dp.in_sample()
y = df["sales"]   # target

# Train the model
# The intercept is the same as the `const` feature from
# DeterministicProcess. LinearRegression behaves badly with duplicated
# features, so we need to be sure to exclude it here.
model = LinearRegression(fit_intercept=False)
model.fit(X, y)


submission_id_label = "id"
test_ids = test_df[submission_id_label]
X_test = dp.out_of_sample(steps=len(test_df))
y_pred = pd.Series(model.predict(X_test))

submission = pd.DataFrame({
    submission_id_label: test_ids,
    'sales': y_pred
})
submission.to_csv("predictions.csv", index=False)