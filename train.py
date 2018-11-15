from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import pickle


class Train:

    def __init__(self, dataset_name):
        self.train_x = None
        self.train_y = None
        self.val_x = None
        self.val_y = None
        try:
            # load training set and validation set, split them into features and output
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
        # load previous model if any, or create a new model
        classifier = None
        try:
            with open('model/' + algorithm + '.pkl', 'rb') as model:
                classifier = pickle.load(model)
                print('Saved ' + algorithm + ' classification model is loaded.')
        except FileNotFoundError:
            if algorithm == 'dt':
                print('Training Decision Tree Classifier model with new data ......')
                classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
            elif algorithm == 'nb':
                print('Training Naive Bayes (Gaussian) Classifier model with new data ......')
                classifier = GaussianNB()
            elif algorithm == 'ph':
                pass
            else:
                print('Unsupported algorithm, please try again.')
                os._exit(1)
        # fit training set to classifier model
        classifier.fit(self.train_x, self.train_y)
        print('Training finished.')
        # cross-validation
        print('Cross-validating the model ......')
        prediction = classifier.predict(self.val_x)
        # show score metrics
        self.__show_score__(prediction)
        # show graphical result of training set
        self.__show_learning_curve__(classifier)


    def __show_score__(self, prediction):
        # print metrics
        print('Accuracy', metrics.accuracy_score(self.val_y, prediction))
        print('precision_score (micro)', metrics.precision_score(self.val_y, prediction, average='micro'))
        print('recall_score (micro)', metrics.recall_score(self.val_y, prediction, average='micro'))
        print('f1_score (micro)', metrics.f1_score(self.val_y, prediction, average='micro'))

    def __show_learning_curve__(self, model):
        # Create CV training and test scores for various training set sizes
        train_sizes, train_scores, test_scores = learning_curve(model,
                                                                self.train_x,
                                                                self.train_y,
                                                                # K-fold cross-validation where k set to 5
                                                                cv=10,
                                                                # use f1 score (micro-average) metric
                                                                scoring='f1_micro',
                                                                # use all system threads
                                                                n_jobs=-1,
                                                                # set batch size of the training set
                                                                train_sizes=np.linspace(0.01, 1.0, 5))
        # Create means and standard deviations of training set scores
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        # Create means and standard deviations of test set scores
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)
        # Draw lines
        plt.plot(train_sizes, train_mean, color="#330000", label="Training score")
        plt.plot(train_sizes, test_mean, color="#4d94ff", label="Cross-validation score")
        # Draw bands
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="#DDDDDD")
        plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="#DDDDDD")
        # Create plot
        plt.title("Learning Curve")
        plt.xlabel("Training Set Size"), plt.ylabel("F1 Score"), plt.legend(loc="best")
        plt.tight_layout()
        plt.show()




