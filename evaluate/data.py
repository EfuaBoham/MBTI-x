import numpy as np
from gensim.models import KeyedVectors as keyvecs

from evaluate.clean import main
from evaluate.personalitytypes import types
from evaluate.utils import checkFileExist


def get_paravectors(post, model):
    """
    Get the wordvectors for an individual post in the training data
    :param post: The textual data to get the wordvec representation
    :param model: The word embeddings model to use. We use fasttext word embeddings model

    :return:
        para_vector: A vector of word embeddings or this file which is the average of the individual word
        vectors in post
    """
    para_vector = np.zeros(model.vector_size)
    num_successful = 0

    for word in post:
        try:
            para_vector = para_vector + model[word]
            num_successful += 1
        except:
            pass
    if num_successful > 0:
        para_vector = para_vector/num_successful
    return para_vector


def get_full_vectorrep(posts, model):
    """
    Get the full vector representation of all posts in training data
    :param posts: The set of posts to get vector representations
    :param model: The fasttext model to use for word embeddings

    :return:
     embeddings: A matrix of with each row entry specifying the word vector representation
    of the corresponding post in posts
    """
    embeddings = np.zeros((len(posts), model.vector_size))

    for i in range(len(posts)):
        embeddings[i] = get_paravectors(posts[i], model)

    return embeddings


def loaddata(filename="valid_words.csv"):
    """
    This function helps us to read cleaned data without having to do all the pre-processing
    :param filename: the path to the file to read data
    :return:
        Labels: the mbti types
        Messages: the post of participants

        Refer to read_data for details
    """

    filename = checkFileExist(filename)

    labels = []
    posts = []
    with open(filename) as data:
        for row in data:
            if row == "\n":
                continue

            val_tokens = row.split()
            mbti_cate = val_tokens[0]
            labels.append(types.index(mbti_cate[-4:]))
            posts.append(val_tokens[1:])

    return labels, posts


def word_embeddings(filename):
    """
    This function loads the wordembeddings file for getting wordvec of features
    :param filename: The path to the vector file indicating vector model to use. Typically has the
    extension .vec
    :return:
        embedding_model: A wordembeddings model built using fasttext
    """
    filename = checkFileExist(filename)
    embedding_model = keyvecs.load_word2vec_format(filename, binary=False)

    return embedding_model


def get_data(file_path="valid_words.csv",
             model_path="mbti_model.vec",
             features_file="features.csv",
             cleaned=True,
             features=True,
             split_ratio=0.7):
    """
    This function loads the model, the posts and the labels. It then does word embeddings on
    on the post and returns train examples and validate examples
    :param file_path: The path to the cleaned textual training data. Relative paths are accepted
    :param model_path: The path to the fasttext vec weights
    :param features_file: String, the path to features file to use for the model
    :param cleaned: A parameter indicating whether the data has been cleaned. If False, main() from clean
    is called before the function proceeds. Default is True
    :param features: A boolean. If set to True the model loads numeric features from features file.
    Otherwise, the file will attempt to load textual data for file_path for feature generation.
    :param split_ratio: The ratio for splitting into train & validation. The value is a decimal greater
    than 0. and less than or equal to 1

    :return:
     trainX: A matrix of size (M, n) containing the wordvec representation of examples in training data
     trainY: A one-hot encoded vector of size(M, NUMCLASSES) indicating the label of the class
     validateX: a matrix of size(M*, n) where M* + M = m containing examples for validation
     validateY: a one-hot encoded vector of size(M*, NUMCLASSES) indicating the labels of the validation set
    """

    # ensure split is a valid entry
    assert split_ratio > 0
    assert split_ratio <= 1

    if not cleaned:
        # call functions to clean & write to directory
        main(file_path)

    if not features:
        organize_data(file_path, model_path, features_file)

    # shuffle the data points. m denotes number of examples
    labels, weights = load_features(features_file)
    m = labels.shape[0]
    np.random.seed(0)
    permute_indices = np.random.permutation(m) - 1
    weights = weights[permute_indices]
    labels = labels[permute_indices]

    # split into train & validation according to ratio
    split_idex = int(split_ratio * m)
    trainX = weights[0:split_idex]
    validateX = weights[split_idex:]
    trainY = labels[0:split_idex]
    validateY = labels[split_idex:]

    return trainX, trainY, validateX, validateY


def save_features(X, Y, filename='features.csv'):
    """
    Save the training set to a directory
    :param X:  The training examples
    :param Y:  the labels
    :param filename: The name of the file to save the examples too
    :return:
    """
    with open(filename, 'wt') as weights:
        for i in range(X.shape[0]):
            weights.write("{} \t {} \n".format(Y[i], " ".join((str(d) for d in X[i]))))


def organize_data(file_path="valid_words.csv",
                  model_path="mbti_model.vec",
                  output_file="features.csv"):
    """
    A utility method to organize the data. Calls other methods and writes the computed vectors
    to a file to be used latter.
    :param file_path: The file path of the textual data
    :param model_path: the filepath of the fasttext model
    :param output_file: The path to write the computed averages too.
    :return:

    Write outputs to the directory containing this.
    """

    # call function for loading post & labels. Also get word vector model
    labels, posts = loaddata(file_path)
    model = word_embeddings(model_path)

    # full vectors according to these posts
    wordvectors = get_full_vectorrep(posts, model)

    # save the features to text so they can be loaded without the word vector model
    # in future training runs
    save_features(wordvectors,labels, output_file)


def load_features(filename="features.csv"):
    """
    Load weights which have already been computed. No need to load the word vec model
    :param filename: The file name with the weights to initialize the model
    :return:
     labels: The labels of the weights.
     weights: the weights to initialize the model with
    """
    labels = []
    weights = []
    with open(filename) as weights_data:
        for row in weights_data:
            row = row.split()
            labels.append(int(row[0]))
            weights.append([float(val) for val in row[1:]])

    labels = np.array(labels)
    weights = np.array(weights)
    return labels, weights


def compute_test_features(test_data_path, model_path, output_file):
    """
    This function computes the test features of the twitter data.
    :param test_data_path: String, the path to the cleaned textual data from
    :param model_path:
    :param output_file:
    :return:
    """
    organize_data(test_data_path, model_path, output_file)


if __name__ == "__main__":
    compute_test_features("../test_data.csv", "../mbti_300d.vec", "test_features_kaggle.csv")