import re

import nltk
from nltk.corpus import stopwords as stpw

from evaluate.utils import checkFileExist


def read_data(filepath,  regrex=",|\|\|\||\s+"):
    """
    This function loads the raw csv data. Some of the post are separated by '|||' - not purely csv.
    :param filepath: string, the path to the file containing the data.
    :return:

    labels: the mbti personality types per each example
    messages: a list of list of the raw post of each person.
    """

    filepath = checkFileExist(filepath)

    with open(filepath) as data:
        labels = []
        messages = []
        for row in data:
            vals = re.split(regrex, row.lower())
            labels.append(vals[0])
            messages.append(vals[1:])

    return labels, messages


def get_only_chars(messages):
    """
    :param messages: A list of list of text with punctuations, numbers, special characters for cleaning
    :return: A list of messages with all punctuations and numbers removed for example
    """
    cleaned_text = []
    for i in range(len(messages)):
        row = messages[i]
        clean_row = []
        for j in range(len(row)):
            word = row[j]
            text = [c for c in word if c.isalpha()]

            clean_row.append("".join(text))
        cleaned_text.append(clean_row)
    return cleaned_text


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
        stopwords_free.append([token for token in example if token not in stopwords])

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


def main(filename="../mbti_1.csv"):
    """
    This module calls all the other functions, cleans the text and writes documents to root directory.
    Before calling this function, ensure
     * the file "mbti_1.csv" is present a folder up of the folder where this file is located.
     * or give a file name
     * there is write access to the folder where this file is located
     * or modify the code below to reflect your preferences.

    :return: The types and their corresponding texts.
    """

    labels, messages = read_data(filename)

    # ignore row zero, it's the headers 'types' and post.
    labels = labels[1:]
    messages = messages[1:]

    messages = replace_links(messages)
    messages = get_only_chars(messages)
    stopwords = get_stopwords()
    valid_words = remove_stopwords(messages, stopwords)

    # write to file for later use.
    with open("valid_words.csv", 'w') as stopwords_free, \
            open("types.txt", 'wt')as types, \
            open("word_counts.csv", 'wt') as counts:

        for i in range(len(valid_words)):
            types.write(labels[i] + "\n")
            # append __label__to labels so fast text can recognize them as labels
            counts.write(labels[i] + ", " + str(len(valid_words[i])) + "\n")
            stopwords_free.write("__label__" + labels[i] + "\t" + " ".join(valid_words[i]) + "\n")


if __name__=='__main__':
    main()
