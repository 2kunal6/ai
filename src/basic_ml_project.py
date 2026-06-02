import pandas as pd
from sklearn.model_selection import train_test_split

pd.set_option('display.max_columns', None)



df = pd.read_csv("AI_Impact_on_Jobs_2030.csv")


train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
print(len(train_set))
print(len(test_set))


