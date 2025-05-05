import pandas as pd

df = pd.read_csv("data/train.csv")
df.head()

for col in df.columns:
    print(f"{col} Statistics:")
    print(df[col].describe())
    print()