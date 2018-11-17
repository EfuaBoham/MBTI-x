from __future__ import print_function

import argparse
import torch
from torch import autograd, nn
from data import load_weights
from train import MbtiTrain


class Test(nn.Module):
    def __init__(self):
        """

        :param model_name: A string. the path to the saved model to be used for predictions
        """
        super().__init__()
        # self._model = torch.load(model_name)



    # def test_accuracy(self, test_data, test_labels):
    #     """
    #
    #     :param test_data:
    #     :param test_labels:
    #     :return:
    #     """
    #
    #     self._model.eval()
    #     output = self(test_data)
    #     pred = torch.max(output.data, 1)[1]
    #     acc = 100 * (pred == test_labels.data).sum() / len(test_labels.data)
    #
    #     return acc

    def load_data(self, test_data_path):
        test_labels, test_data = load_weights(test_data_path)
        return test_labels, test_data


def main_evaluate(model_path, test_data_path):

    print("Loading trained model...........")
    model = Test()

    print("Now loading test data.............")
    # load data
    test_labels, test_data = model.load_data(test_data_path)

    print("Now testing the model on {} samples from tweeter".format(test_data.shape[0]))
    model = MbtiTrain(300, 700)
    test_data = autograd.Variable(torch.from_numpy(test_data).float())
    acc = model.test_accuracy(model_path, test_data, test_labels)

    return acc


if __name__ == "__main__":
    arg_passer = argparse.ArgumentParser()

    arg_passer.add_argument("--model_path", type=str, default="trained_model.pt",
                            help="The path to the trained model to use for evaluation")
    arg_passer.add_argument("--test_data_path", type=str, default="initial_weights.csv",
                            help="The path to test data from twitter for. This data should be the features")

    valid_args, _ = arg_passer.parse_known_args()

    acc = main_evaluate(**valid_args.__dict__)

    print("Final Testing Accuracy :{:2.02f} ".format(acc))

