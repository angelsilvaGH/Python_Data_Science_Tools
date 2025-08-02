##########
import re, string
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import set_config
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


########## 1.3. Features and the decision tree model
df= pd.read_csv("../2_shared_data/review_dataset.csv")
##########



##########
print(f"The shape of the dataset (df) is: {df.shape}\n")
print(f"DataFrame (df), head function:\n{df.head(5)}\n")
##########



##########
# Numerical features
numerical_features= ["Age upon Intake Days", "Age upon Outcome Days"]

# Drop the ID features: RescuerID and PetID
categorical_features= ["Sex upon Outcome", "Intake Type", "Intake Condition", "Pet Type", "Sex upon Intake", "Breed", "Color"]

# Based on exploratory data analysis (EDA), select the text features
text_features= ["Found Location"]

model_features= numerical_features + categorical_features + text_features
model_target= "Outcome Type"
##########



##########
print(f"testing:\n{df[model_features].head()}\n")
##########



##########
# converting all rows under categorical and text features to string values
df[categorical_features + text_features]= df[categorical_features + text_features].astype("str")
##########



##########
print(f"isna & sum of categorical & text feat.:\n{df[categorical_features + text_features].isna().sum()}")
##########



########## 1.3.1. Cleaning the text fields
# Prepare cleaning functions

stop_words = ["a", "an", "the", "this", "that", "is", "it", "to", "and"]

stemmer = Stemmer.Stemmer('english')

def preProcessText(text):
    # Lowercase text, and strip leading and trailing white space
    text = text.lower().strip()

    # Remove HTML tags
    text = re.compile("<.*?>").sub("", text)

    # Remove punctuation
    text = re.compile("[%s]" % re.escape(string.punctuation)).sub(" ", text)

    # Remove extra white space
    text = re.sub("\s+", " ", text)

    return text


def lexiconProcess(text, stop_words, stemmer):
    filtered_sentence = []
    words = text.split(" ")
    for w in words:
        if w not in stop_words:
            filtered_sentence.append(stemmer.stemWord(w))
    text = " ".join(filtered_sentence)

    return text


def cleanSentence(text, stop_words, stemmer):
    return lexiconProcess(preProcessText(text), stop_words, stemmer)


# Clean the text features
for c in text_features:
    print("Text cleaning: ", c)
    df[c] = [cleanSentence(item, stop_words, stemmer) for item in df[c].values]
##########



########## 1.3.2. Create training and test datasets
train_data, test_data = train_test_split(df, test_size=0.1, shuffle=True, random_state=23)
##########



########## 1.3.3. Process the data with a pipeline and ColumnTransformer
### COLUMN_TRANSFORMER ###
##########################

# Preprocess the numerical features
numerical_processor = Pipeline(
    [
        (
            "num_scaler",
            MinMaxScaler(),
        )  # Shown in case it is needed. Not a must with decision trees.
    ]
)

# Preprocess the categorical features
# handle_unknown tells it to ignore (rather than throw an error for) any value
# that was not present in the initial training set.
categorical_processor = Pipeline(
    [("cat_encoder", OneHotEncoder(handle_unknown="ignore"))]
)

# Preprocess the text feature
# This text processor uses max_features=150
text_processor_0 = Pipeline(
    [("text_vect_0", CountVectorizer(binary=True, max_features=150))]
)

# Combine all data preprocessors (add more if you choose to define more)
# For each processor/step, specify: a name, the actual process, and the features to be processed
data_preprocessor = ColumnTransformer(
    [
        ("numerical_pre", numerical_processor, numerical_features),
        ("categorical_pre", categorical_processor, categorical_features),
        ("text_pre_0", text_processor_0, text_features[0]),
    ]
)

### PIPELINE ###
################
# Pipeline with all desired data transformers, along with an estimator
# Later, you can set/reach the parameters by using the names issued - for hyperparameter tuning, for example
pipeline = Pipeline(
    [
        ("data_preprocessing", data_preprocessor),
        ("dt", DecisionTreeClassifier(max_depth=5)),
    ]
)  # The initial value is chosen as max_depth=5

# Visualize the pipeline
# This will be helpful especially when building more complex pipelines,
# stringing together multiple preprocessing steps
set_config(display="diagram")
pipeline
##########



##########
# Get training data to train the pipeline
X_train = train_data[model_features]
y_train = train_data[model_target]

# Fit the pipeline to the training data
pipeline.fit(X_train, y_train)

# Use the fitted pipeline to make predictions on the training dataset
train_predictions = pipeline.predict(X_train)
print(confusion_matrix(y_train, train_predictions))
print(classification_report(y_train, train_predictions))
print("Accuracy (training):", accuracy_score(y_train, train_predictions))

# Get testing data to test the pipeline
X_test = test_data[model_features]
y_test = test_data[model_target]

# Use the fitted pipeline to make predictions on the testing dataset
test_predictions = pipeline.predict(X_test)
print(confusion_matrix(y_test, test_predictions))
print(classification_report(y_test, test_predictions))
print("Accuracy (test):", accuracy_score(y_test, test_predictions))
##########



########## 1.4. Grid search
### PIPELINE GRID_SEARCH ###
############################

# Parameter grid for GridSearch
param_grid = {
    "dt__max_depth": [100, 200, 300],
    "dt__min_samples_leaf": [5, 10, 15],
}

grid_search = GridSearchCV(
    pipeline,  # Base model
    param_grid,  # Parameters to try
    cv=5,  # Apply 5-fold cross validation
    verbose=1,  # Print summary
    n_jobs=-1,  # Use all available processors
)

# Fit the GridSearch to the training data
grid_search.fit(X_train, y_train)
##########



##########
print(grid_search.best_params_)
print(grid_search.best_score_)
##########



##########
# Get the best model out of GridSearchCV
classifier = grid_search.best_estimator_

# Fit the best model to the training data once more
classifier.fit(X_train, y_train)
##########



##########
# Get testing data to test the classifier
X_test = test_data[model_features]
y_test = test_data[model_target]

# Use the fitted model to make predictions on the testing dataset
# Testing data going through the pipeline is first imputed
# (with means from the training set), scaled (with the min/max from the training data),
# and finally used to make predictions.
test_predictions = classifier.predict(X_test)

print("Model performance on the test set:")
print(confusion_matrix(y_test, test_predictions))
print(classification_report(y_test, test_predictions))
print("Test accuracy:", accuracy_score(y_test, test_predictions))
##########



########## 1.5. Randomized search
### PIPELINE RANDOMIZED_SEARCH ###
############################
from scipy.stats import randint
# Parameter grid for GridSearch
param_grid = {
    "dt__max_depth": [100, 200, 300],
    'dt__min_samples_leaf' :randint(15, 35)
    #"dt__min_samples_leaf": [5, 10, 15]
}

randomized_search = RandomizedSearchCV(
    pipeline,  # Base model
    param_grid,  # Parameters to try
    cv=5,  # Apply 5-fold cross validation
    verbose=1,  # Print summary
    n_jobs=-1,  # Use all available processors
)

# Fit the RandomizedSearch to the training data
randomized_search.fit(X_train, y_train)
##########



##########
print(randomized_search.best_params_)
print(randomized_search.best_score_)
##########



##########
# Get the best model out of GridSearchCV
classifier = randomized_search.best_estimator_

# Fit the best model to the training data once more
classifier.fit(X_train, y_train)
##########



##########
# Get testing data to test the classifier
X_test = test_data[model_features]
y_test = test_data[model_target]

# Use the fitted model to make predictions on the testing dataset
# Testing data going through the pipeline is first imputed
# (with means from the training set), scaled (with the min/max from the training data),
# and finally used to make predictions
test_predictions = classifier.predict(X_test)

print("Model performance on the test set:")
print(confusion_matrix(y_test, test_predictions))
print(classification_report(y_test, test_predictions))
print("Test accuracy:", accuracy_score(y_test, test_predictions))
##########



##########
# Parameter grid for randomized search

############### CODE HERE ###############



############## END OF CODE ##############

# Fit the RandomizedSearch to the training data
randomized_search.fit(X_train, y_train)
##########



##########
print(randomized_search.best_params_)
print(randomized_search.best_score_)

# Get the best model out of GridSearchCV
classifier = randomized_search.best_estimator_

# Fit the best model to the training data once more
classifier.fit(X_train, y_train)
##########



##########
# Get testing data to test the classifier
X_test = test_data[model_features]
y_test = test_data[model_target]

# Use the fitted model to make predictions on the testing dataset
# Testing data going through the pipeline is first imputed
# (with means from the training set), scaled (with the min/max from the training data),
# and finally used to make predictions
test_predictions = classifier.predict(X_test)

print("Model performance on the test set:")
print(confusion_matrix(y_test, test_predictions))
print(classification_report(y_test, test_predictions))
print("Test accuracy:", accuracy_score(y_test, test_predictions))
##########