import pandas as pd

df = pd.read_csv("./evaluation_results/pairs_comparison.csv")

print(df.values[:, 2][:17])
print(sum(df.values[:, 2][:17])/16)