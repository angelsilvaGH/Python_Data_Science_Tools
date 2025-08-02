# Load libraries
import re, string
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import Stemmer
from matplotlib.ticker import MaxNLocator
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer


######## 1.3. DATA PROCESSING
# Read the dataset into a DataFrame and look at the data
df= pd.read_csv("../2_shared_data/review_dataset.csv")
# reading the dataset and saving it as df to call it throughout the program
######## DONE


########
print(f"The Shape of the Data Set is (columns, rows): {df.shape}\n")
######## DONE


########
# Print the first ten rows of the dataset
print(f"First 10 rows of data:\n{df.head(10)}\n")
######## DONE


########
print("Number of Columns their name, datatype, and non-null:")
print(f"{df.info()}\n")
######## DONE


########
print(f"Common statistics for data:\n{df.describe()}\n")
######## DONE


########
print(f"All the dataset columns:\n{df.columns}\n")
######## DONE


########
# Identify the numerical, categorical, and text features along with the target feature
numerical_features= ["Age upon Intake Days", "Age upon Outcome Days"]
categorical_features= ["Sex upon Outcome", "Intake Type", "Intake Condition", "Pet Type", "Sex upon Intake"]
text_features= ["Name", "Found Location", "Breed", "Color"]
model_target= "Outcome Type"
######## DONE


######## Value Counts
for c in numerical_features:
    # Print the name of the feature
    print(f"(Column) Data name:\n{c}\n")
    # Print the value counts in 20 bins for each feature
    print(f"Values Counts:\n{df[c].value_counts(bins=20, sort=False)}\n")
    # Plot bar charts based on value_counts (alternative plot method)
    df[c].value_counts(bins=20, sort=False).plot(kind="bar", alpha=1.0, rot=45, color="#CC5500")
    #title = c.replace("Days", "in Years")
    plt.tight_layout()
    plt.title(c)

    plt.xlabel("Age (years)")
    plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=10))

    plt.ylabel("Frequency")
    plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=10))

    plt.savefig(f"2.3{c.replace(' ', '_')}.png")
    plt.clf()
    #plt.show()
######## DONE


########
for c in numerical_features:
    # Drop values beyond 90% of max()
    dropIndexes = df[df[c] > df[c].max() * 9/10].index
    df.drop(dropIndexes, inplace=True)
######## DONE


######## UPDATED value counts
for c in numerical_features:
    print(f"(Column) Data name UPDATED:\n{c},")
    print(f"Values Counts:\n{df[c].value_counts(bins=20, sort=False)}\n")
    df[c].value_counts(bins=20, sort=False).plot(kind="bar", alpha=1.0, rot=45, color="#CC5500")
    # title = c.replace("Days", "in Years")
    plt.tight_layout()
    plt.title(c)

    plt.xlabel("Age (years)")
    plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=10))

    plt.ylabel("Frequency")
    plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=10))

    plt.savefig(f"2.3{c.replace(' ', '_')}_updated.png")
    plt.clf()
    # plt.show()
######## DONE


######## 1.3.4. Drop Rows with Missing Values
# Drop all rows that are missing values
df_missing_dropped = df.dropna()
print(df_missing_dropped.shape)
# Checking if missing values were dropped: print the total number of rows for each feature that has missing values
print(df_missing_dropped.isna().sum())
# dropping NaN values
######## DONE


##### 1.4. Feature Scaling
# Display the numerical features before feature scaling
print(f"Missing dropped, Head:\n{df_missing_dropped[numerical_features].head()}\n")
# Define the scaler
feature_scaler= MinMaxScaler()
scaled_features= feature_scaler.fit_transform(df_missing_dropped[numerical_features])
# Scale the features
df_scaled_numerical_features= pd.DataFrame(scaled_features, columns=numerical_features)
# Display the numerical features after feature scaling
print(f"Scaled numerical feat. 'head':\n{df_scaled_numerical_features.head()}\n")
###### DONE


###### 1.5. Encoding Categorical's
# Show the categorical features before encoding
print(f"Missing dropped 'head':\n{df_missing_dropped[categorical_features].head()}\n")
# Show the shape of the categorical features in the DataFrame before encoding
print(f"Missing dropped 'shape':\n{df_missing_dropped[categorical_features].shape}\n")
###### DONE


###### define the encoder
categorical_encoder= OneHotEncoder(handle_unknown="ignore")
# handle_unknown tells the function to ignore any value that was not present in the initial training set
encoded_categoricals= categorical_encoder.fit_transform(df_missing_dropped[categorical_features])
# Show the shape of the categorical features in the DataFrame after encoding
print(f"The shape of the categorical feature set: {encoded_categoricals.shape}")
###### DONE


###### 1.6. Text Preprocessing
# Define a list with the stop words: a, an, the, this, that, is, it, to, and, in
stop_words= ['a', 'an', 'the', 'this', 'is', 'it', 'to', 'and', 'in']
# Define the stemmer and language to use
stemmer= Stemmer.Stemmer('english')
# Define a function to remove white space, HTML, punctuation, and numbers
def preProcessText(text):
    # Lowercase text, and strip leading and trailing white space
    text= text.lower().strip()

    # Remove HTML tags
    text= re.compile("<.*?>").sub("", text)

    # Remove punctuation
    text= re.compile("[%s]" % re.escape(string.punctuation)).sub(" ", text)

    # Remove extra white space
    text= re.sub(r"\s+", " ", text)

    # Remove numbers
    text= re.sub(r"[0-9]", "", text)

    return text


# Define a function to remove stop words and stem the words
def lexiconProcess(text, stop_words, stemmer):
    filtered_sentence = []
    words = text.split(" ")
    for w in words:
        if w not in stop_words:
            filtered_sentence.append(stemmer.stemWord(w))
    text= " ".join(filtered_sentence)

    return text


# Define a function to clean a sentence
def cleanSentence(text, stop_words, stemmer):
    return lexiconProcess(preProcessText(text), stop_words, stemmer)
###### DONE


######
# Create a variable to use for cleaning
example_text = "   This is a message to be cleaned. 31 It may involve some things like: <br>, ?, :, ''  adjacent spaces and tabs     .  "
# Process and clean the example_text
processed_text = cleanSentence(example_text, stop_words, stemmer)

print("Original: " + example_text + "\n")
print("Processed: " + processed_text)
###### DONE


###### 1.7. Text Vectorization
# Display the text features before vectorization
print(df_missing_dropped[text_features].head())
# Show the shape of the text features before text vectorization
print(df_missing_dropped[text_features].shape)
# Loop through each row of the "Found Location" feature in the DataFrame and
# use the cleaning functions that you defined earlier in this lab
cleaned_text_feature = [
    cleanSentence(item, stop_words, stemmer)
    for item in df_missing_dropped["Breed"] #"Found Location", question 1, change to "Color", then "Breed"
]

# Define the count vectorizer
countVectorizer= CountVectorizer(binary=True, max_features=100)

# Vectorize the data
text_feature_vectorized= countVectorizer.fit_transform(cleaned_text_feature)
# Show the shape of the DataFrame after vectorization
print(text_feature_vectorized.shape)
print("Vocabulary: \n", countVectorizer.vocabulary_)
###### DONE