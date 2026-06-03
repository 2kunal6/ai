import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)

df = pd.read_csv("resources/AI_Impact_on_Jobs_2030.csv")

print(df.info())
print(df.describe()) # null values are ignored here
print(df.head(5))
print(df['Hiring_Trend_2026'].unique())
print(df.groupby("Hiring_Trend_2026")["Job_Title"].value_counts())

ct = pd.crosstab(df["Job_Title"], df['Hiring_Trend_2026'])


matplotlib.use("TkAgg")
#df.hist() # gives histogram of all numerical columns
#plt.show()

print(df.corr(numeric_only=True))
#scatter_matrix(df)
#plt.show()
df.plot(kind="hist", x='Years_Experience', y='AI_Replacement_Risk')
plt.show()