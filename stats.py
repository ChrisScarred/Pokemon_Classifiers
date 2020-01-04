# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 20:35:40 2020

@author: Chris Vajdik s1018903
"""

import itertools
import numpy as np
from classifiers import Classifiers

class Stats:
    def __init__ (self, data_X, data_Y, names, splits):
        # get all two attributes combinations of the 9 possible attributes
        iters = itertools.combinations([i for i in range(9)],2)
        # initialise
        self.accuracies = []
        for itr in iters:
            # get index of the two attributes
            a = itr[0]
            b = itr[1]
            # shape the X data to only contain the two attributes
            dx = np.asarray([[x,y] for x, y in zip(data_X[:,a], data_X[:,b])])
            # the names only containt the names of the two attributes
            n = [names[a],names[b]]
            # create two classifiers
            c = Classifiers(dx,data_Y,splits,n)
            # get accuracies
            c.get_accuracy()
            # append average accuracies along with the classifiers info
            self.accuracies.append((c.get_AVG_accuracies(),c))
    
    # sort by test accuracy of linear regression classifier    
    def sort_LR(self):
        def key_LR(val):
            return val[0][1]
        self.accuracies.sort(key = key_LR)
    
    # sort by test accuracy of decision tree classifier    
    def sort_DT(self):
        def key_DT(val):
            return val[0][3]
        self.accuracies.sort(key = key_DT)
    
    # get the best linear regression classifier
    def get_best_LR(self):
        self.sort_LR()
        return self.accuracies[-1]
    
    # get the best decision tree classifier
    def get_best_DT(self):
        self.sort_DT()
        return self.accuracies[-1]