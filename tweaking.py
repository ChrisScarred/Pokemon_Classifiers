# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 20:56:53 2020

@author: Chris Vajdik s1018903
"""

from dataGetter import DataGetter
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

class Tweaking:
    def __init__(self):
        dg = DataGetter()
        self.X, self.y, names = dg.get_data()
        splits = 3        
        self.kf = KFold(n_splits=splits)
        self.kf.get_n_splits(self.X)
        # generate tweaking info
        self.tweaking()
        
    def tweaking(self):
        # obtain and plot tweaking info about max_depth
        self.max_depth()
        # obtain and plot tweaking info about min_samples_split
        self.min_samples_split()
        # obtain and plot tweaking info about min_samples_leaf
        self.min_samples_leaf()
        # obtain and plot tweaking info about max_features
        self.max_features()
    
    # obtain and plot tweaking info about max_depth    
    def max_depth(self):
        # set the attribute values to realistic possibilities
        max_depths = np.linspace(1, 32, 32, endpoint=True)
        train_results = []
        test_results = []
        
        # iterate through all possibilities
        for max_depth in max_depths:
            train_r = []
            test_r = []
            # iterate through all splits
            for train_index, test_index in self.kf.split(self.X):
                # get data
                X_train, X_test = self.X[train_index], self.X[test_index]
                y_train, y_test = self.y[train_index], self.y[test_index]
                # get decision tree and fit it
                dt = DecisionTreeClassifier(max_depth=max_depth)
                dt.fit(X_train, y_train)
                # get train and test predictions
                train_pred = dt.predict(X_train)
                test_pred = dt.predict(X_test)
                # get accuracy scores and append to the results of this feature
                train_r.append(accuracy_score(y_train,train_pred))
                test_r.append(accuracy_score(y_test,test_pred))
            # append the average accuracy score of the splits for this feature 
            # to the array of results
            train_results.append(np.average(train_r))
            test_results.append(np.average(test_r))
            
        # plot results
        plt.plot(max_depths,train_results)
        plt.plot(max_depths,test_results)
        plt.legend(labels=('train','test'))
        plt.title('The relation of accuracy and max_depth')
        plt.ylabel('Accuracy')
        plt.xlabel('Tree depth')
        plt.show()
    
    # obtain and plot tweaking info about min_samples_split
    def min_samples_split(self):
        # set the attribute values to realistic possibilities
        min_samples_splits = np.linspace(0.1, 1.0, 10, endpoint=True)
        train_results = []
        test_results = []
        
        # iterate through all possibilities
        for splt in min_samples_splits:
            train_r = []
            test_r = []
            # iterate through all splits
            for train_index, test_index in self.kf.split(self.X):
                # get data
                X_train, X_test = self.X[train_index], self.X[test_index]
                y_train, y_test = self.y[train_index], self.y[test_index]
                # get decision tree and fit it
                dt = DecisionTreeClassifier(min_samples_split=splt)
                dt.fit(X_train, y_train)
                # get train and test predictions
                train_pred = dt.predict(X_train)
                test_pred = dt.predict(X_test)
                # get accuracy scores and append to the results of this feature
                train_r.append(accuracy_score(y_train,train_pred))
                test_r.append(accuracy_score(y_test,test_pred))
            # append the average accuracy score of the splits for this feature 
            # to the array of results
            train_results.append(np.average(train_r))
            test_results.append(np.average(test_r))
            
        # plot results
        plt.plot(min_samples_splits,train_results)
        plt.plot(min_samples_splits,test_results)
        plt.legend(labels=('train','test'))
        plt.title('The relation of accuracy and min_sample_split')
        plt.ylabel('Accuracy')
        plt.xlabel('Min samples split')
        plt.show()
    
    # obtain and plot tweaking info about min_samples_leaf
    def min_samples_leaf(self):
        # set the attribute values to realistic possibilities
        min_samples_leafs = np.linspace(0.1, 0.5, 5, endpoint=True)
        train_results = []
        test_results = []
        
        # iterate through all possibilities
        for leaf in min_samples_leafs:
            train_r = []
            test_r = []
            # iterate through all splits
            for train_index, test_index in self.kf.split(self.X):
                # get data
                X_train, X_test = self.X[train_index], self.X[test_index]
                y_train, y_test = self.y[train_index], self.y[test_index]
                # get decision tree and fit it
                dt = DecisionTreeClassifier(min_samples_leaf=leaf)
                dt.fit(X_train, y_train)
                # get train and test predictions
                train_pred = dt.predict(X_train)
                test_pred = dt.predict(X_test)
                # get accuracy scores and append to the results of this feature
                train_r.append(accuracy_score(y_train,train_pred))
                test_r.append(accuracy_score(y_test,test_pred))
            # append the average accuracy score of the splits for this feature 
            # to the array of results
            train_results.append(np.average(train_r))
            test_results.append(np.average(test_r))
            
        # plot results
        plt.plot(min_samples_leafs,train_results)
        plt.plot(min_samples_leafs,test_results)
        plt.legend(labels=('train','test'))
        plt.title('The relation of accuracy and min_samples_leaf')
        plt.ylabel('Accuracy')
        plt.xlabel('Min samples leaf')
        plt.show()
    
    # obtain and plot tweaking info about max_features
    def max_features(self):
        # set the attribute values to realistic possibilities
        max_features = list(range(1,self.X.shape[1]))
        train_results = []
        test_results = []
        
        # iterate through all possibilities
        for feature in max_features:
            train_r = []
            test_r = []
            # iterate through all splits
            for train_index, test_index in self.kf.split(self.X):
                # get data
                X_train, X_test = self.X[train_index], self.X[test_index]
                y_train, y_test = self.y[train_index], self.y[test_index]
                # get decision tree and fit it
                dt = DecisionTreeClassifier(max_features=feature)
                dt.fit(X_train, y_train)
                # get train and test predictions
                train_pred = dt.predict(X_train)
                test_pred = dt.predict(X_test)
                # get accuracy scores and append to the results of this feature
                train_r.append(accuracy_score(y_train,train_pred))
                test_r.append(accuracy_score(y_test,test_pred))
            # append the average accuracy score of the splits for this feature 
            # to the array of results
            train_results.append(np.average(train_r))
            test_results.append(np.average(test_r))
            
        # plot results
        plt.plot(max_features,train_results)
        plt.plot(max_features,test_results)
        plt.legend(labels=('train','test'))
        plt.title('The relation of accuracy and max_features')
        plt.ylabel('Accuracy')
        plt.xlabel('Max features')        
        plt.show()
        