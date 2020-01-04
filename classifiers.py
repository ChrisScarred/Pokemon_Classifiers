# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 17:13:19 2020

@author: Chris Vajdik s1018903
"""

from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
import numpy as np

class Classifiers:
    def __init__ (self, data_X, data_Y, splits, names):
        self.X = data_X
        self.y = data_Y
        self.kf = KFold(n_splits=splits)
        self.kf.get_n_splits(data_X)

        self.LR = LogisticRegression(solver='lbfgs',multi_class='multinomial')
        # values obtained via class Tweaking
        self.DT = DecisionTreeClassifier(max_depth=5,min_samples_split=0.6,min_samples_leaf=0.3,max_features=min(3,self.X.shape[1]))
        # for normalisation
        self.min_max_scaler = MinMaxScaler()
        
        self.accu_train_LR = []
        self.accu_test_LR = []
        self.accu_train_DT = []
        self.accu_test_DT = []
        self.names = [str(name) for name in names]
        
    def get_accuracy(self):
        # repeat for all splits
        for train_index, test_index in self.kf.split(self.X):
            X_train, X_test = self.X[train_index], self.X[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]
            # normalise the data
            X_train_norm = self.min_max_scaler.fit_transform(X_train)       
            X_test_norm = self.min_max_scaler.fit_transform(X_test)   
            
            # fit and predict for linear regression
            self.LR = self.LR.fit(X_train_norm, y_train)
            train_pred_LR = self.LR.predict(X_train_norm)
            test_pred_LR = self.LR.predict(X_test_norm)  
            
            # fit and predict for decision trees
            self.DT = self.DT.fit(X_train_norm, y_train)
            train_pred_DT = self.DT.predict(X_train_norm)
            test_pred_DT = self.DT.predict(X_test_norm)  
            
            # keep track of the accuracies
            self.accu_train_LR.append(accuracy_score(y_train,train_pred_LR))
            self.accu_test_LR.append(accuracy_score(y_test,test_pred_LR))
            self.accu_train_DT.append(accuracy_score(y_train,train_pred_DT))
            self.accu_test_DT.append(accuracy_score(y_test,test_pred_DT))
         
    # return tuple of average accuracies
    def get_AVG_accuracies(self):
        return (np.average(self.accu_train_LR), np.average(self.accu_test_LR), np.average(self.accu_train_DT), np.average(self.accu_test_DT))
    
    # print stats of the two classifiers
    def print_stats(self):
        print('Classifiers with attributes: ',self.names)
        print("\n---\nLR\n---\n")    
        print("Accuracy in train set: ",self.accu_train_LR)
        print("AVG accuracy in train set: ",str(np.average(self.accu_train_LR)))
        print("Accuracy in test set: ",self.accu_test_LR)
        print("AVG Accuracy in test set: ",str(np.average(self.accu_test_LR)))

        print("\n---\nDT\n---\n")    
        print("Accuracy in train set: ",self.accu_train_DT)
        print("AVG accuracy in train set: ",str(np.average(self.accu_train_DT)))
        print("Accuracy in test set: ",self.accu_test_DT)
        print("AVG Accuracy in test set: ",str(np.average(self.accu_test_DT)))