# import libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.impute import SimpleImputer
# import the dataset csv file with the required data
df= pd.read_csv("review_dataset.csv")


# analyzing Austin animal shelter data
print("\nAustin Animal Center Shelter Intakes and Outcomes dataset Analysis\n")


# number of rows (95485) and columns (13), respectively
print("The shape of the dataset is:", df.shape, "\n")


# I want to see the data type and non-null values for each column
print(df.info(), "\n")


# summary of statistics or the numerical columns, "std"= standard deviation from the mean, "mean" is average, "min" is...
#...minimum value in the dataset, "max" is the maximum value in the dataset.
print(df.describe(), "\n")


# printing the name of all the columns
print("\nColumns for dataframe:", df.columns)


# making a list of the features that I want to use and set it to a variable I can later call.
numerical_values= ["Age upon Intake Days", "Age upon Outcome Days"] # using the exact name of column name to call the data
model_target= "Outcome Type"


# I want a visual data graph, ## ONLY "Age upon Intake Days" ##
ageIntake= "Age upon Intake Days"
df[ageIntake].plot.hist(bins=5)
plt.title("Distribution of Age upon Intake Days")
plt.xlabel("Age (Days)")
plt.ylabel("Frequency")
plt.savefig(f"{ageIntake.replace(' ', '_')}_single.png")
#plt.show() # I want to avoid opening them while running, saving the data as a png file to view after


# looping through each numerical value index ## ("Age upon Intake Days" & "Age upon Outcome Days") ##
for c in numerical_values:
    (df[c]/365).plot.hist(bins=20)
    plt.title(c)
    plt.xlabel("Age (years)")
    plt.ylabel("Frequency")
    plt.savefig(f"{c.replace(' ', '_')}.png")
    plt.clf()
    #plt.show() # I want to avoid opening them while running, saving the data as a png file to view after


# I want to see the max and min for each index in numerical value (intake & outcome)
# looping through numerical_values where, intake & outcome data is saved
for c in numerical_values:
    print("\n")
    print(c, "min & max age (in years)")
    print("min:", df[c].min()/365, "max:", df[c].max()/365)