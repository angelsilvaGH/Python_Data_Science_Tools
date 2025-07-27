# import libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
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
print("\nColumns from dataset:", df.columns, "\n")


# making a list of the features that I want to use and set it to a variable I can later call.
numerical_features= ["Age upon Intake Days", "Age upon Outcome Days"] # using the exact name of column name to call the data
model_target= "Outcome Type"


# I want a visual data graph, ## ONLY "Age upon Intake Days" ## TEST/practice
ageIntake= "Age upon Intake Days"
(df[ageIntake]/365).plot.hist(bins=25)
plt.title("Distribution of Age upon Intake years")
plt.xlabel("Age (years)")
plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=10))
plt.ylabel("Frequency")
plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=10))
plt.savefig(f"{ageIntake.replace(' ', '_')}_single.png")
plt.clf()
#plt.show() # I want to avoid opening data while running, saving the data as a png file to view after


# looping through each numerical value index ## ("Age upon Intake Days" & "Age upon Outcome Days") ##
for c in numerical_features:
    (df[c]/365).plot.hist(bins=25)

    title= c.replace("Days", "in Years")
    plt.title(f"Distribution of {title}")

    plt.xlabel("Age (years)")
    plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=10))

    plt.ylabel("Frequency")
    plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=10))

    plt.savefig(f"{c.replace(' ', '_')}.png")
    plt.clf()
    #plt.show() # I want to avoid opening data while running, saving the data as a png file to view after


# I want to see the max and min for each index in numerical value (intake & outcome)
# looping through numerical_values where, intake & outcome data is saved
for c in numerical_features:
    title= c.replace("Days", "in Years")
    print("\n", title, ", min & max age:")
    print("min:", df[c].min()/365, "max:", df[c].max()/365, "\n")

# using value_counts to see numerical data range
for c in numerical_features:
    title= c.replace("Days", "in Years")
    print("\n", title)
    data_years = df[c]/365
    bins = np.linspace(0, 25, 30)
    counts = data_years.value_counts(bins=bins, sort=False)

    for interval, count in counts.items():
        print(f"{interval}: {count}")


#
for c in numerical_features:
    title= c.replace("Days", "in Years")
    print("\nDropped from feature:", title)
    # Calculate the upper and lower quartile values
    Q1= df[c].quantile(0.25)
    Q3= df[c].quantile(0.75)

    # Calculate the IQR
    IQR = Q3 - Q1
    print("Q1: ", Q1, ", Q3: ", Q3, ", IQR: ", IQR)

    print(Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)

    # Drop values below Q1 - 1.5 IQR and beyond Q3 + 1.5 IQR
    dropIndexes = df[df[c] > Q3 + 1.5 * IQR].index
    df.drop(dropIndexes, inplace=True)
    dropIndexes = df[df[c] < Q1 - 1.5 * IQR].index
    df.drop(dropIndexes, inplace=True)