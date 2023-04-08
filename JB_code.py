# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 19:42:57 2023

@author: Jbaru
"""

# Import necessary libraries
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix

# Load the data into a pandas data frame
data = pd.read_csv(r"C:\Users\Jbaru\OneDrive\Documents\MEGA\Centennial\Semester1\AI\reference_proyect\Youtube01-Psy.csv")


# Basic data exploration
print(data.head())
print(data.info())

# Select the relevant columns
data = data[['CONTENT', 'CLASS']]

# Preprocess the text data
def preprocess_text(text):
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token.lower() not in stop_words]
    return ' '.join(filtered_tokens)

data['CONTENT'] = data['CONTENT'].apply(preprocess_text)

# Vectorize the text data using the Bag of Words model
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data['CONTENT'])
y = data['CLASS']

# Present initial features and data shape
print("Initial feature shape:", X.shape)

# Downscale the data using tf-idf
tfidf_transformer = TfidfTransformer()
X_tfidf = tfidf_transformer.fit_transform(X)

# Present final features and data shape
print("Final feature shape:", X_tfidf.shape)

# Shuffle the dataset
data_shuffled = data.sample(frac=1, random_state=42)

# Split the dataset into training and testing sets
train_size = int(0.75 * len(data_shuffled))
train_data = data_shuffled.iloc[:train_size]
test_data = data_shuffled.iloc[train_size:]

X_train = vectorizer.transform(train_data['CONTENT'])
X_test = vectorizer.transform(test_data['CONTENT'])
y_train = train_data['CLASS']
y_test = test_data['CLASS']

# Train a Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

# Cross-validate the model using 5-fold cross-validation
cv_scores = cross_val_score(classifier, X_train, y_train, cv=5)
print("Cross-validation mean accuracy:", np.mean(cv_scores))

# Test the model on the test data
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)
print("Test accuracy:", accuracy)
print("Confusion matrix:\n", confusion)

# Test the model on new comments
new_comments = ["Great video! I love it!", "Very informative, thank you!", "Check out my channel for similar content!", "This helped me a lot, thanks!", "Get thousands of views with our service!", "Click this link to win a free iPhone!"]
new_comments_processed = [preprocess_text(comment) for comment in new_comments]
new_comments_vectorized = vectorizer.transform(new_comments_processed)
new_pred = classifier.predict(new_comments_vectorized)
print("Predictions for new comments:", new_pred)
