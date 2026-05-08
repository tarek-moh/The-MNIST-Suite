# %% Cell 1
from sklearn.datasets import fetch_openml
print("Downloading MNIST dataset...")
mnist = fetch_openml('mnist_784', version=1, as_frame=True)
df = mnist.frame

print("MNIST dataset downloaded successfully.")
print(df.shape)
print(df.head())

# %%
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

X = mnist.data

plt.imshow(X.iloc[0].values.reshape(28,28), cmap='gray')
plt.show()
