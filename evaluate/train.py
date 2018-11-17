from __future__ import print_function

import argparse

import torch
from torch import autograd, nn
from torch import optim
from torch.nn import functional as F
import os
from evaluate.data import get_data, word_embeddings, get_paravectors
from evaluate.personalitytypes import NUMCLASSES
from evaluate.personalitytypes import types


class MbtiTrain(nn.Module):
    def __init__(self, input_size, hidden_size=512, num_classes=NUMCLASSES):
        """
        A declaration of the model
        :param input_size: int, the dimensions of the features for each layer
        :param hidden_size: int, the number of neurons each hidden layer
        :param num_classes: int, the number of classes in the data. Default is 16
        """
        super().__init__()

        # metrics
        self._losses = []
        self._acc = []
        self._val_losses = []
        self._val_acc = []

        # Network specifications
        self._input_layer = nn.Linear(input_size, hidden_size)
        self._hidden_layer = nn.Linear(hidden_size, hidden_size)
        self._output_layer = nn.Linear(hidden_size, num_classes)

    def forward(self, input):
        """
        This overrides Pytorch Neural Network forward pass. It defines the model architecture
        :param input: The input to use for training.

        :return:
            output: The output layer of the network
        """
        output = F.relu(self._input_layer(input))
        output = F.relu(self._hidden_layer(output))
        output = F.relu(self._hidden_layer(output))
        output = F.relu(self._hidden_layer(output))
        output = F.relu(self._hidden_layer(output))
        output = F.relu(self._hidden_layer(output))
        output = F.relu(self._hidden_layer(output))
        output = F.relu(self._hidden_layer(output))
        output = F.dropout(output)
        output = F.relu(self._hidden_layer(output))
        output = F.relu(self._hidden_layer(output))
        output = F.softmax(self._output_layer(output))

        return output

    def train_model(self, data, labels, val_data, val_labels, lr=1e-3, epochs=2000, verbose=0):
        """
        This function trains the network using Adam and Cross entropy loss
        :param data:  A Pytorch autograd variable of type float. The input data to train network
        :param labels: A Pytorch autograd variable of type long. The corresponding labels of the input data
        :param val_data:  A Pytorch autograd variable. The data to use for validating the model
        :param val_labels: A Pytorch autograd variable of type long. The labels of the validation data
        :param lr: The learning rate
        :param epochs:  The number of epochs to trained model for
        :param verbose:  An integer. If 1, the progress of the training is displayed. All other integers
            make the model silent
        :return:
         : The training losses for this training run
        """
        # clear losses before new training
        self._losses = []
        self._val_losses = []
        self._acc = []
        self._val_acc = []

        # Optimizer and objective functions
        adam_optimizer = optim.Adam(params=self.parameters(), lr=lr)
        objective = nn.CrossEntropyLoss()

        # train on full batch since the data is not that massive
        for i in range(epochs):
            output = self(data)
            pred = torch.max(output.data, 1)[1]
            acc = 100 * (pred == labels.data).sum() / len(labels.data)

            # compute loss
            self.train()
            loss = objective(output, labels)
            self.zero_grad()
            l = loss.data[0]
            loss.backward()  # backpropagate
            adam_optimizer.step()

            # conduct validations
            self.eval()
            output = self.forward(val_data)
            val_l = objective(output, val_labels)
            val_acc = self.evaluate(val_data, val_labels.data)
            val_acc = val_acc * 100
            val_l = val_l.data[0]

            # print out metrics if verbose is set to 1
            if verbose == 1:
                print("Epoch {}/{}: loss {:2.03f} acc {:2.03f} val_loss {:2.02f} val_acc {:2.02f}".format((i + 1), epochs, l, acc, val_l, val_acc))

            # record metrics for this epoch
            self._losses.append(l)
            self._acc.append(acc)
            self._val_acc.append(val_acc)
            self._val_losses.append(val_l)

    def evaluate(self, validate_input, true_labels):
        """
        Make predictions on the current model given valid data and labels
        :param validate_input: The data to make predictions on
        :param true_labels: The ground truth labels of the inputs
        :return:
            acc 'float', the fraction of samples in the validation set which were correctly classified
        """

        outputs = self(validate_input)
        prediction = torch.max(outputs.data, 1)[1]
        correct_predictions = (prediction == true_labels).sum()
        acc = correct_predictions / len(true_labels)

        return acc

    def get_metrics(self):
        """
        :return:
            : Retrieve the metrics of a given training run. If no training has happen,
            returns the empty list
        """
        return self._losses, self._acc, self._val_losses, self._val_acc

    def save_metrics(self, directory="./"):
        """
        This method saves the training metrics to a directory. This model should only be called after
        a training has been conducted.
        :param directory: The directory to save the metrics to.
        :return:
        """

        if not os.path.exists(directory):
            os.system("mkdir {}".format(directory))

        # prepare files for opening
        tr_acc_file = "{}/tr_acc.txt".format(directory)
        val_acc_file = "{}/val_acc.txt".format(directory)
        tr_loss_file = "{}/tr_loss.txt".format(directory)
        val_loss_file = "{}/val_loss.txt".format(directory)

        # save the metrics
        with open(tr_acc_file, "wt") as tr_acc, open(val_acc_file, "wt") as val_acc:
            tr_acc.write(", ".join([str(tr_a) for tr_a in self._acc]))
            val_acc.write(", ".join([str(v_a) for v_a in self._val_acc]))

        with open(tr_loss_file, "wt") as tr_loss, open(val_loss_file, "wt") as val_loss:
            tr_loss.write(", ".join([str(tr_l) for tr_l in self._losses]))
            val_loss.write(", ".join([str(v_l) for v_l in self._val_losses]))

    def save(self, file_path):
        """
        Save the entire model after training
        :param file_path: The name to save the model.
        :return:
        """
        torch.save(self, file_path)

    def test_accuracy(self, model_path, test_data, test_labels):
        """
        Pytorch requires that a save model be loaded with its entire structure to enable
        predictions. This is a bit inconvenient because it means separate test files cannot be
        created where only a saved model is loaded for testing. Thus, instead of redefining the entire class
        in order file to make predictions, we instead conduct the predictions here.
        :param model_path: A string, the path to the saved model
        :param test_data: A Pytorch autograd variable of type float. The data to use for testing
        :param test_labels: A Pytorch autograd variable of type long. The corresponding labels of
        the data. Useful for accuracy computations.
        :return:
        """
        model = torch.load(model_path)
        model.eval()
        output = model(test_data)
        pred = torch.max(output.data, 1)[1]
        acc = 100 * (pred == test_labels.data).sum() / len(test_labels.data)
        return acc

    def predict(self, features, trained_model_path="trained_model.pt"):
        """

        :param features:
        :param trained_model_path:
        :return:
        """
        model = torch.load(trained_model_path)
        output = model(features)
        prediction = torch.max(output.data, 1)[1]

        personality = types[prediction[0]]

        return personality


def predict_personality(cleaned_tweets,
                        word_vec_model_path,
                        trained_model_path="trained_model.pt"):
    """

    :param cleaned_tweets:
    :param word_vec_model_path:
    :param trained_model_path:
    :return:
    """
    word_embeddings_model = word_embeddings(word_vec_model_path)
    features = get_paravectors(cleaned_tweets, word_embeddings_model)
    features = autograd.Variable(torch.from_numpy(features).float())

    model = MbtiTrain(features.size(1))
    personality = model.predict(features, trained_model_path)
    return personality


def main(file_path,
         model_path,
         features_path,
         cleaned,
         features,
         split_ratio,
         num_classes,
         hidden_size,
         lr, epochs,
         verbose=0,
         mode="train"
         ):
    """
    Bundles all the other functions into one method for convenience
    :param file_path: The path to the file containing the textual data.
    Only useful if features below is set to False
    :param model_path: A string, The path to the word vector model. Only useful if features below
    is set to False
    :param features_path: A string. The path to the features file to use for training. Should still
    be set even if features below is set to False so they can be saved after generation
    :param cleaned: A boolean: Indicates whether the text is cleaned or still raw. Only useful
    if features below is set to False
    :param features: A boolean: Indicates whether actual features has been generated for each class.
    Default is True
    :param split_ratio: float between 0 and 1: Indicates proportion of data to use for training:
    Only useful if mode is set to train.
    :param num_classes: int, the number of classes the data is made up of
    :param hidden_size: int, the number of neurons in the input layer
    :param lr: float, the learning rate to use in making the gradient updates
    :param epochs: int, indicating the number of iterations to run the model for
    :param verbose: int
    :param mode: 'String', the mode to run the model under: Can take only two values, train(default) and
    test.
    :return:
    """
    if mode == 'train':

        print("Loading data...............")
        train_input, train_labels, valid_input, valid_labels = get_data(file_path,
                                                                        model_path,
                                                                        features_path,
                                                                        cleaned,
                                                                        features,
                                                                        split_ratio)
        print("Finished loading data ")
        print("Train on {} samples".format(len(train_input)))
        print("Validate on {} samples".format(len(valid_input)))

        # convert to pytorch autograd variables so gradients can be computed
        train_input = autograd.Variable(torch.from_numpy(train_input).float())
        train_labels = autograd.Variable(torch.from_numpy(train_labels).long())
        valid_input = autograd.Variable(torch.from_numpy(valid_input).float())
        valid_labels = autograd.Variable(torch.from_numpy(valid_labels).long())

        # create model and train
        input_dim = train_input.size(1)
        model = MbtiTrain(input_dim, hidden_size, num_classes)

        model.train_model(train_input, train_labels, valid_input, valid_labels, lr, epochs, verbose)

        print("Finished Training, starting predictions on validation set")

        acc = model.evaluate(valid_input, valid_labels.data)

        print("Model Accuracy on {} unseen samples: {}%".format(valid_labels.size(0), acc * 100))

        print("Saving Training Metrics: ...............")
        model.save_metrics()

        print("Now saving the trained model..............")
        model.save("trained_model.pt")

    elif mode == "test":
        print("Loading test data.......")
        test_data, test_labels, _, _ = get_data(file_path,
                                                model_path,
                                                features_path,
                                                cleaned,
                                                features,
                                                split_ratio=1
                                                )
        print("Testing the model on {} examples from twitter".format(test_data.shape[0]))

        test_data = autograd.Variable(torch.from_numpy(test_data).float())
        test_labels = autograd.Variable(torch.from_numpy(test_labels).long())

        model = MbtiTrain(test_data.size(1), hidden_size, num_classes)
        acc = model.test_accuracy("trained_model_kaggle.pt", test_data, test_labels)

        print("Final Testing Accuracy :{:2.02f} ".format(acc))
    else:
        print("Unrecognized mode: Accepted modes are train and test")
        exit()


if __name__ == '__main__':

    # Parse command line arguments
    # All parameters have default values
    arg_passer = argparse.ArgumentParser()

    arg_passer.add_argument("--file_path", type=str, default="valid_words.csv",
                            help="""
                            The path to the raw data file. When cleaned is true, this
                            file should contain cleaned data. Otherwise, it should be the 
                            raw uncleaned data
                            """)
    arg_passer.add_argument("--model_path", type=str, default="../wiki-news-300d.vec",
                            help="""
                            The path to the word embeddings, required only when
                            features is set to false
                            """)
    arg_passer.add_argument("--features_path", type=str, default="wiki_models_features_3d.csv",
                            help="""
                            The path to the file to save or load the initial features 
                            """)
    arg_passer.add_argument("--cleaned", type=bool, default=True,
                            help="""
                            Indicate whether raw text is already cleaned or not. Setting to false will
                            cause the model to cleaned the data. 
                            """)
    arg_passer.add_argument("--features", type=bool, default=True,
                            help="""
                                 Whether there are initial weights. 
                                 Set to false if binary vector has been loaded before
                                 """)
    arg_passer.add_argument("--split_ratio", type=float, default=0.8,
                            help="""
                                 The split ratio for training and validation.
                                 Should be between 0.1 and 1 
                                 """)
    arg_passer.add_argument("--num_classes", type=int, default=NUMCLASSES,
                            help="Number of distinct personality types in training data")
    arg_passer.add_argument("--hidden_size", type=int, default=512,
                            help="The size of all hidden layers")
    arg_passer.add_argument("--lr", type=float, default=1e-4,
                            help="The learning rate for gradient updates")
    arg_passer.add_argument("--epochs", type=int, default=2000,
                            help="The number of training iterations to perform")
    arg_passer.add_argument("--verbose", type=int, default=1,
                            help="""
                                 1 to show progress after every training epoch. 
                                 0 to remain silent during training
                                 """)
    arg_passer.add_argument("--mode", type=str, default="train",
                            help="""
                                This parameter indicates whether to run the model under a training mode or under a 
                                test mode. 
                                    """)

    valid_args, _ = arg_passer.parse_known_args()

    # Parse arguments to main function
    main(**valid_args.__dict__)

