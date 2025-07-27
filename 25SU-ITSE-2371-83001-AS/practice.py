# import libraries
import pandas as pd
import matplotlib as plt
import numpy as np
from sklearn.impute import SimpleImputer
# import the dataset
df= pd.read_csv("review_dataset.csv")
# Print the dataset shape (rows/columns)
print("The shape of the dataset is:", df.shape)
print(df.info())
print(df.describe())
print("take off")