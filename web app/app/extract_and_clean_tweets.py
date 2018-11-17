import tweepy
import sys
import unicodedata
from unidecode import unidecode
import re
import nltk
from nltk.corpus import stopwords as stpw
import string


# set encoding
# reload(sys)
# sys.setdefaultencoding('utf8')

key = 'C3rDXiF7xn0Ng7zvkMQbMhMr5'
secret = 'tlUcNSwx1YwyirF1p41cU478nHI7iPrflXkMmv8akRgEYbcxUZ'
access_token = '1445640739-Dw49RlSvZymgDss3t3HoFwiKrtgAC9EvQ7jNEZY'
access_token_secret = 'C6hAhadwQDRR1KDQEZzWd18tjR3MGCDWY01yGcsD6XJm5'

# authenticate to Twitter API
auth = tweepy.OAuthHandler(key, secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

# extract tweets of a handle


def deEmojify(inputString):
    returnString = ""
    for character in inputString:
        try:
            character.encode("ascii")
            returnString += character
        except UnicodeEncodeError:
            returnString += ''
    return returnString


def extract_and_clean_user_tweets(user_handle):
    tweets = extract_tweets_lazy(user_handle)
    tweets = get_only_chars(tweets)
    return tweets

def extract_tweets_lazy(user_handle, number_of_tweets = 200):
    try:
        tweets = api.user_timeline(
            screen_name=user_handle, count=number_of_tweets)
        tweets_to_csv = [tweet.text.replace(',', '').replace('\t', '').replace(
            '\n', '').replace('@', '') for tweet in tweets if (not tweet.retweeted) and ('RT @' not in tweet.text)]
        # tweets_to_csv = [tweet.text for tweet in tweets]  # create csv file
    except:
        print(user_handle)
        tweets_to_csv = None
    return tweets_to_csv


def get_only_chars(tweets, regrex=",|\|\|\||\s+"):
    """
    :param messages: A list of list of text with punctuations, numbers, special characters for cleaning
    :return: A list of messages with all punctuations and numbers removed for example
    """
    # print(tweets)
    output = ''
    stop_words = get_stopwords()
    
    # print(tweets)
    #tweets = replace_links(tweets)
    table = str.maketrans({key: None for key in string.punctuation})
    for tweet in tweets:
        tweet = tweet.lower()
        tweet = tweet.strip()
        tweet = deEmojify(tweet)
        words = re.split(regrex, tweet)
        words = remove_stopword2(words, stop_words)
        words = replace_links2(words)
        for word in words:
            word = word.translate(table)
            if word.isalpha():
                #word = word.translate(string.punctuation)
                output += word + ' '
    return output

def remove_stopword2(messages, stopwords):
    output = []
    for i in messages:
        if i not in stopwords:
            output.append(i)
    return output

def replace_links2(messages):
    output = []
    for i in messages:
        if 't.co/' in i:
            pass
        else:
            i = i.replace('://youtube.com', 'youtube ')
            i = i.replace('http://', 'url ')
            i = i.replace('https://', 'url ')
            output.append(i)
    return output

def remove_stopwords(messages, stopwords):
    """
    This method removes all words which are not expressive semantically.
    Example of such words are 'a', 'are', 'wasn't' etc. While these words are often the most frequently
    occurring words, they do not have much information about the content of a message and thus not very
    discriminative across different examples.
    :param messages: The messages to remove the stop words from. This is a list of list with
    the inner list being the individual words for each example. The outer list is the examples.
    :param stopwords: The group of words considered as stopwords.
    :return: 'string' Messages without stop words. Also a list of list as messages
    """
    stopwords_free = []
    for i in range(len(messages)):
        example = messages[i]
        stopwords_free.append(
            [token for token in example if token not in stopwords])

    return stopwords_free


def replace_links(messages):
    """
    This function removes urls in the messages and replace them with a common term.
    All youtube links are replaced by the dummy word youtube. All other urls are replaced by 'url'

    :param messages: string. A list of list of strings. First level of links represent distinct examples. Second
        nesting represent the individual messages for a given example
    :return: string, a list of list of strings with the links encoded into different words
    """
    for i in range(len(messages)):
        row = messages[i]
        for j in range(len(row)):
            text = row[j]
            if text.find("://youtube.com") > -1:
                row[j] = "youtube"
            elif text.find("http://") > -1 or text.find("https://") > -1:
                row[j] = "url"
    return messages


def get_stopwords():
    """
    This function retrieves the set of stop words consider as being removed.
    The stop words are retrieved from the nltk package: https://www.nltk.org/api/nltk.html

    We further modify the set by adding words such as 'theyre', representing 'they're'. That is,
    whereas only 'they're' appears in nltk stopwords set, our stopwords set contains both 'theyre'
    and "they're"
    :return: string: a set of words considered not very informative.
    """
    # download stopwords if they not already downloaded.
    try:
        stopwords = set(stpw.words("english"))
    except LookupError:
        print("Stop words not in your computer yet. \n Downloading stop words...........")
        nltk.download()
        stopwords = set(stpw.words("english"))
    without_contractions = set()
    for stopword in stopwords:
        without_contractions.add("".join([c for c in stopword if c.isalpha()]))

    # include contracted forms with punctuations removed too.
    stopwords = stopwords.union(without_contractions)
    return stopwords


if __name__ == '__main__':
    handles_file = '../handles.csv'
    output_file = '../mbti_1.txt'
    #output_file = 'test.txt'
    number_of_tweets = 250
    index = 0
    all_tweets = []
    handles = []

    with open(handles_file) as f:
        rows = f.readlines()
        output = ''
        for handle in rows:
            handle, personality = handle.split(',')
            if handle:
                tweets = extract_tweets_lazy(handle, number_of_tweets)
                if tweets:
                    personality = personality.rstrip()
                    output += '__label__' + personality.lower() + ' '
                    output += get_only_chars(tweets)
            output += '\n'
    with open(output_file, 'w') as o:
        o.write(output)
