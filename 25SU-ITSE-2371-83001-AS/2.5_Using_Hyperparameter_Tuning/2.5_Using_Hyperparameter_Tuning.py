##########
import re, string
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import randint
import Stemmer
##########


##########
df= pd.read_csv("../2_shared_data/review_dataset.csv")
##########



##########
print(f"The shape of the dataset (df) is:\n{df.shape}\n")
print(f"DataFrame (df), head:\n{df.head(10)}")
##########



##########
# Numerical features
numerical_features = ["Age upon Intake Days", "Age upon Outcome Days"]

# Drop the ID features: RescuerID and PetID
categorical_features = ["Sex upon Outcome", "Intake Type", "Intake Condition", "Pet Type", "Sex upon Intake", "Breed", "Color"]

# Based on exploratory data analysis (EDA), select the text features
text_features = ["Found Location"]

model_features = numerical_features + categorical_features + text_features
model_target = "Outcome Type"
##########



##########
print(model_features)
##########



##########
df[categorical_features + text_features]= df[categorical_features + text_features].astype("str")
##########



##########
##########



##########
##########



##########
##########



##########
##########



##########
##########



##########
##########



##########
##########



##########
##########



##########
##########



##########
##########