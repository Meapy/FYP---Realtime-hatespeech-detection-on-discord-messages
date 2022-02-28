import pandas as pd
import numpy as np
import re

df = pd.read_csv('data/neither.csv')


def split_data(df):
    """
    split the data frame into based on the value in the "class" column
    if the value is 0 put the "tweet" column into its own csv file called data/hatespeech.csv
    if the value is 1 put the "tweet" column into its own csv file called data/offensive.csv
    if the value is 2 put the "tweet" column into its own csv file called data/neither.csv
    save the csv files to the data folder
    """
    #if the value is 0 put the "tweet" column into its own csv file called data/hatespeech.csv
    df.loc[df['class'] == 0, 'tweet'].to_csv('data/hatespeech.csv', index=False)
    #if the value is 1 put the "tweet" column into its own csv file called data/offensive.csv
    df.loc[df['class'] == 1, 'tweet'].to_csv('data/offensive.csv', index=False)
    #if the value is 2 put the "tweet" column into its own csv file called data/neither.csv
    df.loc[df['class'] == 2, 'tweet'].to_csv('data/neither.csv', index=False)
#split_data(df)

def process_tweets():
    """
    Accepts a text string and replaces:
    1) urls with URLHERE
    2) lots of whitespace with one instance
    3) mentions with MENTIONHERE
    4) hashtags with HASHTAGHERE

    This allows us to get standardized counts of urls and mentions
    Without caring about specific people mentioned.

    Returns counts of urls, mentions, and hashtags.
    :return:
    """
    space_pattern = '\s+'
    giant_url_regex = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
        '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    mention_regex = '@[\w\-]+'
    hashtag_regex = '#[\w\-]+'
    parsed_text = []
    df = pd.read_csv('data/offensive.csv')
    for i in range(len(df)):
        text = df['tweet'][i]
        # 1) url
        text = re.sub(giant_url_regex, 'URLHERE', text)
        # 2) lots of whitespace
        text = re.sub(space_pattern, ' ', text)
        # 3) mentions
        text = re.sub(mention_regex, 'MENTIONHERE', text)
        # 4) hashtags
        text = re.sub(hashtag_regex, 'HASHTAGHERE', text)
        parsed_text.append(text)
    df['tweet'] = parsed_text
    df.to_csv('data/offensive.csv', index=False)

#process_tweets()

#create a function to delete random rows from the dataframe to make sure it has around 1500 rows
def delete_random_rows(df):
    """
    Accepts a dataframe and deletes random rows to make sure it has around 1500 rows
    :param df:
    :return:
    """
    df = df.sample(frac=1).reset_index(drop=True)
    df = df.drop(df.index[1500:])
    df.to_csv('data/neither.csv', index=False)

delete_random_rows(df)