import pandas as pd
import numpy as np
import pickle
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.stem.porter import *
import string
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as VS
from textstat.textstat import *
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

stopwords = stopwords = nltk.corpus.stopwords.words("english")

other_exclusions = ["#ff", "ff", "rt"]
stopwords.extend(other_exclusions)

stemmer = PorterStemmer()
df = pd.read_csv("data/labeled_data.csv")

tweets = df.tweet


def preprocess(text_string):
    """
    Accepts a text string and replaces:
    1) urls with URLHERE
    2) lots of whitespace with one instance
    3) mentions with MENTIONHERE

    This allows us to get standardized counts of urls and mentions
    Without caring about specific people mentioned
    """
    space_pattern = '\s+'
    giant_url_regex = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
                       '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    mention_regex = '@[\w\-]+'
    parsed_text = re.sub(space_pattern, ' ', text_string)
    parsed_text = re.sub(giant_url_regex, '', parsed_text)
    parsed_text = re.sub(mention_regex, '', parsed_text)
    return parsed_text


def tokenize(tweet):
    """Removes punctuation & excess whitespace, sets to lowercase,
    and stems tweets. Returns a list of stemmed tokens."""
    tweet = " ".join(re.split("[^a-zA-Z]*", tweet.lower())).strip()
    tokens = [stemmer.stem(t) for t in tweet.split()]
    return tokens


def basic_tokenize(tweet):
    """Same as tokenize but without the stemming"""
    tweet = " ".join(re.split("[^a-zA-Z.,!?]*", tweet.lower())).strip()
    return tweet.split()


vectorizer = TfidfVectorizer(
    tokenizer=tokenize,
    preprocessor=preprocess,
    ngram_range=(1, 3),
    stop_words=stopwords,
    use_idf=True,
    smooth_idf=False,
    norm=None,
    decode_error='replace',
    max_features=100,
    min_df=0,
    max_df=0.75
)

# Construct tfidf matrix and get relevant scores
tfidf = vectorizer.fit_transform(tweets).toarray()

print(tfidf.shape)
vocab = {v: i for i, v in enumerate(vectorizer.get_feature_names_out())}
idf_vals = vectorizer.idf_
idf_dict = {i: idf_vals[i] for i in vocab.values()}  # keys are indices; values are IDF scores
tweet_tags = []
# if the tweet_tags.txt exists in the data folder open it and assign it to the tweet_tags variable, else run the for loop
try:
    with open('data/tweet_tags.txt', 'r') as f:
        tweet_tags = f.read().splitlines()
except:
    # Get POS tags for tweets and save as a string
    for t in tweets:
        tokens = basic_tokenize(preprocess(t))
        tags = nltk.pos_tag(tokens)
        tag_list = [x[1] for x in tags]
        tag_str = " ".join(tag_list)
        tweet_tags.append(tag_str)
    # save the tweet tags to a file for later use
    with open("data/tweet_tags.txt", "w") as f:
        for t in tweet_tags:
            f.write(t + "\n")

# We can use the TFIDF vectorizer to get a token matrix for the POS tags
pos_vectorizer = TfidfVectorizer(
    tokenizer=None,
    lowercase=False,
    preprocessor=None,
    ngram_range=(1, 3),
    stop_words=None,
    use_idf=False,
    smooth_idf=False,
    norm=None,
    decode_error='replace',
    max_features=100,
    min_df=0,
    max_df=0.75,
)

# Construct POS TF matrix and get vocab dict
pos = pos_vectorizer.fit_transform(pd.Series(tweet_tags)).toarray()
pos_vocab = {v: i for i, v in enumerate(pos_vectorizer.get_feature_names_out())}

# Now get other features
sentiment_analyzer = VS()


def count_twitter_objs(text_string):
    """
    Accepts a text string and replaces:
    1) urls with URLHERE
    2) lots of whitespace with one instance
    3) mentions with MENTIONHERE
    4) hashtags with HASHTAGHERE

    This allows us to get standardized counts of urls and mentions
    Without caring about specific people mentioned.

    Returns counts of urls, mentions, and hashtags.
    """
    space_pattern = '\s+'
    giant_url_regex = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
                       '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    mention_regex = '@[\w\-]+'
    hashtag_regex = '#[\w\-]+'
    parsed_text = re.sub(space_pattern, ' ', text_string)
    parsed_text = re.sub(giant_url_regex, 'URLHERE', parsed_text)
    parsed_text = re.sub(mention_regex, 'MENTIONHERE', parsed_text)
    parsed_text = re.sub(hashtag_regex, 'HASHTAGHERE', parsed_text)
    return (parsed_text.count('URLHERE'), parsed_text.count('MENTIONHERE'), parsed_text.count('HASHTAGHERE'))


def other_features(tweet):
    """This function takes a string and returns a list of features.
    These include Sentiment scores, Text and Readability scores,
    as well as Twitter specific features"""
    sentiment = sentiment_analyzer.polarity_scores(tweet)

    words = preprocess(tweet)  # Get text only

    syllables = textstat.syllable_count(words)
    num_chars = sum(len(w) for w in words)
    num_chars_total = len(tweet)
    num_terms = len(tweet.split())
    num_words = len(words.split())
    avg_syl = round(float((syllables + 0.001)) / float(num_words + 0.001), 4)
    num_unique_terms = len(set(words.split()))

    ###Modified FK grade, where avg words per sentence is just num words/1
    FKRA = round(float(0.39 * float(num_words) / 1.0) + float(11.8 * avg_syl) - 15.59, 1)
    ##Modified FRE score, where sentence fixed to 1
    FRE = round(206.835 - 1.015 * (float(num_words) / 1.0) - (84.6 * float(avg_syl)), 2)

    twitter_objs = count_twitter_objs(tweet)
    retweet = 0
    if "rt" in words:
        retweet = 1
    features = [FKRA, FRE, syllables, avg_syl, num_chars, num_chars_total, num_terms, num_words,
                num_unique_terms, sentiment['neg'], sentiment['pos'], sentiment['neu'], sentiment['compound'],
                twitter_objs[2], twitter_objs[1],
                twitter_objs[0], retweet]
    # features = pandas.DataFrame(features)
    return features


def get_feature_array(tweets):
    feats = []
    for t in tweets:
        feats.append(other_features(t))
    return np.array(feats)


feats = get_feature_array(tweets)

# Now join them all up
M = np.concatenate([tfidf, pos, feats], axis=1)

# Finally get a list of variable names
variables = [''] * len(vocab)
for k, v in vocab.items():
    variables[v] = k

pos_variables = [''] * len(pos_vocab)
for k, v in pos_vocab.items():
    pos_variables[v] = k

X = pd.DataFrame(M)
y = df['class'].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2, test_size=0.2)

pipe = Pipeline(
    [('select', SelectFromModel(LogisticRegression(class_weight='balanced',
                                                   penalty="l2", C=0.01, solver='liblinear'))),
     ('model', LogisticRegression(class_weight='balanced', penalty='l2'))])

param_grid = [{}]  # Optionally add parameters here

grid_search = GridSearchCV(pipe, param_grid, cv=StratifiedKFold(n_splits=5, random_state=2, shuffle=True)
                           .split(X_train, y_train), verbose=2)

model = grid_search.fit(X_train, y_train)

y_preds = model.predict(X_test)


# create a function to predict the class of a string of text
def predict_class(text):
    feats = get_feature_array(text)
    text_df = pd.DataFrame(text)
    tfidf = vectorizer.fit_transform(text).toarray()
    tweet_tags = []
    for t in text:
        tokens = basic_tokenize(preprocess(t))
        tags = nltk.pos_tag(tokens)
        tag_list = [x[1] for x in tags]
        tag_str = " ".join(tag_list)
        tweet_tags.append(tag_str)
    pos = pos_vectorizer.fit_transform(pd.Series(tweet_tags)).toarray()
    M = np.concatenate([tfidf, pos, feats], axis=1)
    final = pd.DataFrame(M)
    # check which columns are expected by the model, but not exist in the inference dataframe
    not_existing_cols = [c for c in X.columns.tolist() if c not in final]
    # add this columns to the data frame
    final = final.reindex(final.columns.tolist() + not_existing_cols, axis=1)
    # new columns dont have values, replace null by 0
    final.fillna(0, inplace=True)
    # use the original X structure as mask for the new inference dataframe
    final = final[X.columns.tolist()]
    final.dropna(inplace=True)
    # predict the class of the new dataframe
    preds = model.predict(final)
    print("******************************************************************************************")
    print("text is:",text)
    print("The prediction is:",preds)
    print("the prediction probability: \n ",model.predict_proba(final))
    print("******************************************************************************************")
    # if the probably of the prediction is higher than 0.7 and 0.4 respectavility, return the prediction
    if preds[0] == 1 and model.predict_proba(final)[0][1] > 0.7:
        return preds[0]
        pass
    elif preds[0] == 0 and model.predict_proba(final)[0][0] > 0.4:
        return preds[0]
        pass
    else:
        pass
