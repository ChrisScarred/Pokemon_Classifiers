# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 14:46:57 2020

@author: Chris Vajdik s1018903
"""

import pandas as pd
import numpy as np

class DataGetter:
    def __init__ (self):
        # read useful parts
        data = pd.read_csv('Data/Pokemon.csv',index_col=['Name'],usecols=[i+1 for i in range(12)])
        # get Y and X
        self.data_Y = np.asarray(data['Type 1'])
        self.data_X = np.asarray(data.drop('Type 1', axis=1))
        # get names of the attributes of X
        self.names = list(data.drop('Type 1', axis=1).columns)
        # get Ys in int shape
        self.data_Y = np.array([self.translate_type(y) for y in self.data_Y])
        # get Xs in int shape
        self.data_X[:,0] = np.array([self.translate_type(x) for x in self.data_X[:,0]])
        self.data_X[:,-1] = np.array([self.translate_legendary(x) for x in self.data_X[:,-1]])
    
    # cast Pokemon types to ints
    def translate_type(self, s):    
        if s=='Fire':
            return 1
        elif s=='Water':
            return 2
        elif s=='Grass':
            return 3
        elif s=='Electric':
            return 4
        elif s=='Psychic':
            return 5
        elif s=='Steel':
            return 6
        elif s=='Normal':
            return 7
        elif s=='Fairy':
            return 8
        elif s=='Dark':
            return 9
        elif s=='Flying':
            return 10
        elif s=='Ghost':
            return 11
        elif s=='Poison':
            return 12
        elif s=='Ice':
            return 13
        elif s=='Ground':
            return 14
        elif s=='Rock':
            return 15
        elif s=='Dragon':
            return 16
        elif s=='Fighting':
            return 17
        elif s=='Bug':
            return 18
        else:
            return 0

    # cast bool to int
    def translate_legendary(self, s):
        if s:
            return 1
        else:
            return 0        
    
    # return correctly shaped data
    def get_data(self):
        return self.data_X, self.data_Y, self.names