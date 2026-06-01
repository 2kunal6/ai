import pandas as pd
import matplotlib
import matplotlib.pyplot as plt


df = pd.read_csv("AI_Impact_on_Jobs_2030.csv")

pd.set_option('display.max_columns', None)

print(df.head(5))
print(df['Hiring_Trend_2026'].unique())
print(df.groupby("Hiring_Trend_2026")["Job_Title"].value_counts())

ct = pd.crosstab(df["Job_Title"], df['Hiring_Trend_2026'])


matplotlib.use("TkAgg")
ct.plot(kind="bar", stacked=True)
plt.show()