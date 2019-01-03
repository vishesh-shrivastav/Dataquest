# URL for data - https://github.com/arnauddri/hn
import pandas as pd
submissions = pd.read_csv("sel_hn_stories.csv")
submissions.columns = ["submission_time", "upvotes", "url", "headline"]
submissions = submissions.dropna()

# Tokenization - convert to bag of words model
tokenized_headlines = []

for headline in submissions["headline"]:
    tokenized_headlines.append(headline.split())

# Remove punctuations
punctuation = [",", ":", ";", ".", "'", '"', "’", "?", "/", "-", "+", "&", "(", ")"]
clean_tokenized = []

for headline in tokenized_headlines:
    tokens = []
    for token in headline:
        token = token.lower()
        for punc in punctuation:
            token = token.replace(punc, "")
        tokens.append(token)
    clean_tokenized.append(tokens)

# Find unique tokens
# Use two lists - unique_tokens contains all unique tokens that occur more than once
# single_tokens has all tokens
import numpy as np
unique_tokens = []
single_tokens = []

for tokens in clean_tokenized:
    for token in tokens:
        if token not in single_tokens:
            single_tokens.append(token)
        elif token in single_tokens and token not in unique_tokens:
            unique_tokens.append(token)

# Create dataframe with columns as unique_tokens and
# initialise all values to 0
counts = pd.DataFrame(0, index=np.arange(len(clean_tokenized)), columns=unique_tokens)

# Set correct counts in dataframe for each token
# We've already loaded in clean_tokenized and counts
for (i, tokens) in enumerate(clean_tokenized):
    for token in tokens:
        if token in unique_tokens:
            counts.iloc[i][token] += 1

# Remove words that occur less than 5 or more than 100 times
# We've already loaded in clean_tokenized and counts
word_counts = counts.sum(axis=0) # vector of word counts for each column
counts = counts.loc[:,(word_counts >= 5) & (word_counts <= 100)]

# Split into train and test
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(counts, submissions["upvotes"], test_size=0.2, random_state=1)

# Fit linear regression model
from sklearn.linear_model import LinearRegression

clf = LinearRegression()
clf.fit(X_train, y_train)

# Make predictions on number of upvotes for test data
predictions = clf.predict(X_test)

# Calculate Mean Squared Error for predictions
mse = sum((predictions - y_test) ** 2) / len(predictions)
# mse = 2181

# Not a great model - try to improve