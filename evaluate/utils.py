from __future__ import print_function
from os.path import isfile
from sys import exit


def checkFileExist(filename):
    if not isfile(filename):
        print("Error: ", filename, " does not exist")
        filename = input("Enter filename or press 0 to exit: ")
        if filename == '0':
            exit("Aborting program. Source: user")

        checkFileExist(filename)

    return filename


# ../fasttext supervised -input valid_words.csv -output mbti_pretrained_300d -lr 1.0
#  -epoch 50 -wordNgrams 2 -bucket 200000 -dim 300 -loss hs
# -pretrainedVectors wiki-news-300d-1M.vec


