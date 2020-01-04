# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 13:59:42 2020

@author: Chris Vajdik s1018903
"""

from dataGetter import DataGetter
from classifiers import Classifiers
from stats import Stats

# load data
dg = DataGetter()
# get data in correct shape
data_X, data_Y, names = dg.get_data()

# initialise number of splits
splits = 3

# create nine attribute classifiers
cl = Classifiers(data_X,data_Y,splits,names)

# get nine attribute classifiers accuracy and print
cl.get_accuracy()
print('\n\n------\n\nCLASSIFIERS WITH 9 ATTRIBUTES\n\n------\n\n')
cl.print_stats()

# use Stats class to create classifiers for all possible 2-attribute combinations and get the best ones
stats = Stats(data_X,data_Y,names,splits)
bestLR = stats.get_best_LR()
bestDT = stats.get_best_DT()

# print their stats
print('\n\n------\n\nBEST LR WITH 2 ATTRIBUTES\n\n------\n\n')
bestLR[1].print_stats()
print('\n\n------\n\nBEST DT WITH 2 ATTRIBUTES\n\n------\n\n')
bestDT[1].print_stats()


#uncomment for tweaking results
"""
from tweaking import Tweaking

Tweaking()
"""