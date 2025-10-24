# %%
# TESTING

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %%
gz2_hart = pd.read_csv("data/gz2_hart16.csv")
gz2_hart.info()

# %%
columns = gz2_hart.iloc[:, 0:9].columns
gz2_hart[columns].head()

# %%
# len(gz2_hart["gz2_class"].unique())
gz2_hart["gz2_class"].unique()

# %%
fig, ax = plt.subplots(figsize=(18, 12), dpi=200)

counts = gz2_hart["gz2_class"].value_counts()
ax.bar(counts.index, np.array(counts.values))

plt.show()

# %%
counts.head(30)

# %%
import re

# %%
short = gz2_hart["gz2_class"].unique()

# %%
types = {"S": 0, "E": 0, "A": 0}
for s in short:
    types[s[0]] += 1
print(types)

# %%
