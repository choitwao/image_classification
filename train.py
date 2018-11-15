from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os


class Train:

    def __init__(self, dataset_name):
        self.train_x = None
        self.train_y = None
        self.val_x = None
        self.val_y = None
        try:
            print('Loading data set ' + dataset_name + ' ......')
            train_set = pd.read_csv('data/' + dataset_name + '/' + dataset_name + 'Train.csv', header=None)
            self.train_x = train_set.iloc[:, :-1]
            self.train_y = train_set.iloc[:, -1]
            val_set = pd.read_csv('data/' + dataset_name + '/' + dataset_name + 'Val.csv', header=None)
            self.val_x = val_set.iloc[:, :-1]
            self.val_y = val_set.iloc[:, -1]
            print('Data set ' + dataset_name + ' is loaded.')
            print('Instance: ', len(train_set.axes[0]))
            print('Feature: ', len(train_set.axes[1]) - 1)
        except FileNotFoundError:
            print('The data set cannot be found in the data directory, please double check.')
            os._exit(1)

    def train(self, algorithm):
        if algorithm == 'DT':
            self.__train_dt__()
        elif algorithm == 'NB':
            self.__train_nb__()
        elif algorithm == 'PH':
            self.__train_ph__()
        else:
            print('Unsupported algorithm, please try again.')

    def __train_dt__(self):
        pass

    def __train_nb__(self):
        pass

    def __train_ph__(self):
        # placeholder
        pass


