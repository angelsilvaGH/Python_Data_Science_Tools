##########
# importing libraries
import re, string
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import Stemmer
from sklearn import set_config
from matplotlib.ticker import MaxNLocator
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
##########



########## 1.3. Data processing¶
df= pd.read_csv("../2_shared_data/review_dataset.csv")

print("The shape of the dataset is:", df.shape)
##########



##########
print(f"\n{df.columns}")
##########



##########
numerical_features= ["Age upon Intake Days", "Age upon Outcome Days"]
categorical_features= ["Sex upon Outcome", "Intake Type", "Intake Condition", "Pet Type", "Sex upon Intake"]
text_features= ["Found Location", "Breed", "Color"]
model_features= numerical_features+categorical_features+text_features
model_target= "Outcome Type"
##########



########## 1.3.2. Clean numerical features
for c in numerical_features:

    print(f"\nValue_counts:\n{df[c].value_counts(bins=10, sort=False)}")
    df[c].value_counts(bins=10, sort=False).plot(kind="bar", alpha=0.75, rot=90)
    plt.tight_layout()
    plt.title(c)

    plt.xlabel("Age (years)")
    plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=10))

    plt.ylabel("Frequency")
    plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=10))

    plt.savefig(f"2.3{c.replace(' ', '_')}.png")
    plt.clf()
    #plt.show()
##########



##########
for c in numerical_features:
    # Drop values beyond 90% of max()
    dropIndexes = df[df[c] > df[c].max() * 9/10].index
    df.drop(dropIndexes, inplace=True)
##########



##########
for c in numerical_features:
    print(df[c].value_counts(bins=10, sort=False))
    df[c].value_counts(bins=10, sort=False).plot(kind="bar", alpha=0.75, rot=90)
    plt.show()
##########



########## 1.3.3. Clean text features
# Prepare cleaning functions
import re, string

stop_words = ["a", "an", "the", "this", "that", "is", "it", "to", "and", "in"]

stemmer = Stemmer.Stemmer('english')
############### CODE HERE ###############

############## END OF CODE ##############
##########


##########
# Clean the text features
for c in text_features:
    print("Text cleaning: ", c)
    df[c] = [cleanSentence(item, stop_words, stemmer) for item in df[c].values]
##########



########## 1.3.5. Create training, test, and validation datasets
from sklearn.model_selection import train_test_split

############### CODE HERE ###############



############## END OF CODE ##############

# Print the shapes of the training, validation, and test datasets
print(
    "Train - Test - Validation dataset shapes: ",
    train_data.shape,
    val_data.shape,
    test_data.shape,
)
##########



########## 1.3.7. Process the data with a pipeline and ColumnTransformer
### COLUMN_TRANSFORMER ###
##########################

# Preprocess the numerical features
numerical_processor = Pipeline(
    [
        ("num_imputer", SimpleImputer(strategy="mean")),
        (
            "num_scaler",

            MinMaxScaler(),
        ),  # Shown in case it is needed. Not a must with decision trees.
    ]
)

# Preprocess the categorical features
categorical_processor = Pipeline(
    [
        (
            "cat_imputer",
            SimpleImputer(strategy="constant", fill_value="missing"),
        ),  # Shown in case it is needed. No effect because you already imputed with 'nan' strings.
        (
            "cat_encoder",
            OneHotEncoder(handle_unknown="ignore"),
        ),  # handle_unknown tells it to ignore (rather than throw an error for) any value that was not present in the initial training set.
    ]
)

# Preprocess first text feature
text_processor_0 = Pipeline(
    [("text_vectorizer_0", CountVectorizer(binary=True, max_features=50))]
)

# Preprocess second text feature
text_processor_1 = Pipeline(
    [("text_vectorizer_1", CountVectorizer(binary=True, max_features=50))]
)

# Preprocess third text feature
text_processor_2 = Pipeline(
    [("text_vectorizer_2", CountVectorizer(binary=True, max_features=50))]
)

# Combine all data preprocessors (add more if you choose to define more)
# For each processor/step, specify: a name, the actual process, and the features to be processed.
data_processor = ColumnTransformer(
    [
        ("numerical_processing", numerical_processor, numerical_features),
        ("categorical_processing", categorical_processor, categorical_features),
        ("text_processing_0", text_processor_0, text_features[0]),
        ("text_processing_1", text_processor_1, text_features[1]),
        ("text_processing_2", text_processor_2, text_features[2]),
    ]
)

# Visualize the data processing pipeline
set_config(display="diagram")
data_processor
##########



########## 1.4. Train a classifier
### PIPELINE ###
################

# Pipeline with all desired data transformers, along with an estimator
# Later, you can set/reach the parameters by using the names issued - for hyperparameter tuning, for example
pipeline = Pipeline(
    [
        ("data_processing", data_processor),
        (
            "lg",
            LogisticRegression(
                solver="liblinear", penalty="l1", C=0.001, class_weight={0: 1, 1: 20}
            ),
        ),
    ]
)

# Visualize the pipeline
# This will be helpful especially when building more complex pipelines, stringing together multiple preprocessing steps
set_config(display="diagram")
pipeline
##########



########## 1.4.3. Model training
# Get training data to train the classifier
X_train = train_data[model_features]
y_train = train_data[model_target]

# Fit the classifier to the training data
# Training data going through the pipeline is imputed (with means from the training data),
#   scaled (with the min/max from the training data),
#   and finally used to fit the model
pipeline.fit(X_train, y_train)
##########



########## 1.5. Test the classifier
# Get validation data to validate the classifier
X_test = test_data[model_features]
y_test = test_data[model_target]

# Use the fitted model to make predictions on the test dataset
# Testing data going through the pipeline is imputed (with means from the training data),
#   scaled (with the min/max from the training data),
#   and finally used to make predictions
############### CODE HERE ###############

############## END OF CODE ##############

print("Model performance on the test set:")
print(confusion_matrix(y_test, test_predictions))
print(classification_report(y_test, test_predictions))
print("Test accuracy:", accuracy_score(y_test, test_predictions))
##########



########## 1.6.1. Comparing binary predictions and probability predictions
pipeline.predict(X_test)[0:5]
pipeline.predict_proba(X_test)[0:5]
##########



########## 1.6.2. Threshold calibration to improve model accuracy
# Calculate the accuracy by using different values for the classification threshold,
# and select the threshold that results in the highest accuracy.
highest_accuracy = 0
threshold_highest_accuracy = 0

thresholds = np.arange(0, 1, 0.01)
scores = []
for t in thresholds:
    # Set threshold to 't' instead of 0.5
    y_test_other = (pipeline.predict_proba(X_test)[:, 1] >= t).astype(float)
    score = accuracy_score(y_test, y_test_other)
    scores.append(score)
    if score > highest_accuracy:
        highest_accuracy = score
        threshold_highest_accuracy = t
print(
    "Highest accuracy on test:",
    highest_accuracy,
    ", Threshold for the highest accuracy:",
    threshold_highest_accuracy,
)

# Plot the accuracy against the threshold choices
plt.rcParams["figure.figsize"] = (8, 5)
plt.plot(
    [0.5, 0.5],
    [np.min(scores), np.max(scores)],
    linestyle="--",
    color="blue",
    label="Default threshold (0.5)",
)
plt.plot(
    [threshold_highest_accuracy, threshold_highest_accuracy],
    [np.min(scores), np.max(scores)],
    linestyle="--",
    color="green",
    label="Threshold for highest accuracy ({})".format(threshold_highest_accuracy),
)
plt.plot(thresholds, scores, marker=".", color="orange")
plt.title("Accuracy Compared to Threshold Choices")
plt.xlabel("Threshold")
plt.ylabel("Accuracy")
plt.legend(loc="upper left", bbox_to_anchor=(1, 1), fancybox=True, shadow=True, ncol=1)
plt.show()
##########
