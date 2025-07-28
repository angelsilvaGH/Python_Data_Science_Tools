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
print(f"The shape of the dataset is: {df.shape}\n")


# I want to see the data type and non-null values for each column
print("Info function:")
print(f"{df.info()}\n")


# summary of statistics or the numerical columns, "std"= standard deviation from the mean, "mean" is average, "min" is...
#...minimum value in the dataset, "max" is the maximum value in the dataset.
print(f"{df.describe()}\n")


# printing the name of all the columns
print(f"\nColumns from dataset: {df.columns}\n")


# making a list of the features that I want to use and set it to a variable I can later call.
numerical_features= ["Age upon Intake Days", "Age upon Outcome Days"] # using the exact name of column name to call the data
model_target= "Outcome Type"


# I want a visual data graph, ## ONLY "Age upon Intake Days" ## TEST/practice
intakeAge= "Age upon Intake Days"
(df[intakeAge]/365).plot.hist(bins=25)
plt.title("Distribution of Age upon Intake years")
plt.xlabel("Age (years)")
plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=10))
plt.ylabel("Frequency")
plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=10))
plt.savefig(f"{intakeAge.replace(' ', '_')}_single.png")
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
# looping through numerical_values where, where intake & outcome data is saved
for c in numerical_features:
    title= c.replace("Days", "in Years")
    print(f"\n{title}, min & max age:")
    print(f"min: {df[c].min()/365} max: {df[c].max()/365}")
# showcasing the maximum/minimum values for numerical features in years


# using value_counts to see numerical data range intervals
for c in numerical_features:
    title= c.replace("Days", "in Years")
    print(f"\n{title}, (interval) : (amount of data in the interval)")
    print((df[c]/365).value_counts(bins=30, sort=False))
########


# want to see where percentage of data falls under, plus removing outliers
for c in numerical_features:
    print("\nDropped from feature:", c)
    # Calculate the upper and lower quartile values
    Q1= df[c].quantile(0.25)
    Q3= df[c].quantile(0.75)

    # Calculate the IQR
    IQR= Q3 - Q1
    print(f"Q1 (25% of data falls below): {Q1} Days\nQ3 (75% of data falls below): {Q3} Days or {(Q3/365):.2f} Year(s)\nIQR: {IQR} Days")

    lower_bound= max(0, Q1-1.5*IQR)    # age cannot be negative, if below 0, it will return 0.
    upper_bound= Q3+1.5*IQR
    print(f"Outlier Range must fall outside: ({lower_bound} Days, ({upper_bound} Days or {(upper_bound/365):.2f} years))")

    # Drop values below Q1 - 1.5*IQR and beyond Q3 + 1.5*IQR
    outliers = df[(df[c]>upper_bound) | (df[c]<lower_bound)].index
    df.drop(outliers, inplace=True)
########


# dropping values in the upper 10%
for c in numerical_features:
    # Drop values beyond 90% of max()
    threshold= df[c].quantile(0.90)
    outliers = df[df[c] > threshold].index
    df.drop(outliers, inplace=True)
########


# re-calculating value_counts()
for c in numerical_features:
    print(f"\n{c}, (interval) : (amount of data in the interval)")
    print(df[c].value_counts(bins=10, sort=False))
#######


# Plot updated histograms with 100 bins for each numerical feature (UPDATED)
for c in numerical_features:
    (df[c]).plot.hist(bins=100)
    plt.title(c)
    plt.xlabel("Age (days)")
    #plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=10))

    plt.ylabel("Frequency")
    #plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=10))

    plt.savefig(f"{c.replace(' ', '_')}_updated.png")
    plt.clf()
    #plt.show()
######


# Generate random data and make a scatterplot of it
# Generate random data
x = np.random.rand(500)
y = np.random.rand(500)

# Plot the data
plt.scatter(x, y)
plt.title("ScatterPlot")
plt.xlabel("input")
    #plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=10))

plt.ylabel("output")
    #plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=10))

plt.savefig("scatterplot.png")
plt.clf()
#plt.show()
######