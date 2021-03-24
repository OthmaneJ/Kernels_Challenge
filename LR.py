#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 20:05:17 2021


"""

# Librairies 
import os 
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class LinearLogisticRegression:
    """
    simple logistic regression 
    """
    def __init__(self, lr=0.05, num_iter=1000000, fit_intercept=True, verbose=False):
        """
        :param lr : the learning rate
        :param num_iter: the number of iteration 
        :param fit_intercept : whether or not use intercept
        :param verbose: whether or not use print information of the training
        """
        self.verbose = verbose
        self.lr = lr
        self.num_iter = num_iter
        self.fit_intercept = fit_intercept
    def __add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)
    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    def __loss(self, h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
    
    def fit(self, X, y):
        if self.fit_intercept:
            X = self.__add_intercept(X)

        # weights initialization
        self.theta = np.random.randn(X.shape[1]) #np.zeros(X.shape[1])

        for i in range(self.num_iter):
            z = np.dot(X, self.theta)
            h = self.__sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / y.size
            self.theta -= self.lr * gradient

            if(self.verbose == True and i % 10000 == 0):
                z = np.dot(X, self.theta)
                h = self.__sigmoid(z)
                print(f'loss: {self.__loss(h, y)} \t')
    def predict_prob(self, X):
        if self.fit_intercept:
            X = self.__add_intercept(X)

        return self.__sigmoid(np.dot(X, self.theta))
    
    def predict(self, X, threshold = 0.5):
        return self.predict_prob(X) >= threshold
    
if __name__ == '__main__':
    
    Xtr0_mat100 = pd.read_csv('Xtr0_mat100.csv', sep =" " , header = None)
    Xtr1_mat100 = pd.read_csv('Xtr1_mat100.csv', sep =" " , header = None)
    Xtr2_mat100 = pd.read_csv('Xtr2_mat100.csv', sep =" " , header = None)

    Ytr0 = pd.read_csv('Ytr0.csv', index_col= 'Id')
    Ytr1 = pd.read_csv('Ytr1.csv', index_col= 'Id')
    Ytr2 = pd.read_csv('Ytr2.csv', index_col= 'Id')
    
    Xte0_mat100 = pd.read_csv('Xte0_mat100.csv', sep =" " , header = None)
    Xte1_mat100 = pd.read_csv('Xte1_mat100.csv', sep =" " , header = None)
    Xte2_mat100 = pd.read_csv('Xte2_mat100.csv', sep =" " , header = None)


    len_tr = int(0.8*2000) ;# length of the training data 
    
    # training on first dataset 
    Xtr0_mat100_np = Xtr0_mat100.values
    Ytr0_np = Ytr0.values.ravel()
    Xtrain_0, Y_train0 = Xtr0_mat100_np[:len_tr] , Ytr0_np[:len_tr]
    Xval_0, Y_val0 = Xtr0_mat100_np[len_tr:] , Ytr0_np[len_tr:]
    Model_LLR0 = LinearLogisticRegression(verbose = False )
    %time Model_LLR0.fit(Xtrain_0, Y_train0 )
    Y_val0_pred = Model_LLR0.predict(Xval_0)
    _ = (Y_val0_pred  == Y_val0).mean() # 0.545
    
    # training on the second dataset 
    Xtr1_mat100_np = Xtr1_mat100.values
    Ytr1_np = Ytr1.values.ravel()
    Xtrain_1, Y_train1 = Xtr1_mat100_np[:len_tr] , Ytr1_np[:len_tr]
    Xval_1, Y_val1 = Xtr1_mat100_np[len_tr:] , Ytr1_np[len_tr:]
    Model_LLR1 = LinearLogisticRegression(verbose = False )
    %time Model_LLR1.fit(Xtrain_1, Y_train1 )
    Y_val1_pred = Model_LLR1.predict(Xval_1)
    _ = (Y_val1_pred  == Y_val1).mean() # 0.56
    
    # training of the third datased 
    Xtr2_mat100_np = Xtr2_mat100.values
    Ytr2_np = Ytr2.values.ravel()
    Xtrain_2, Y_train2 = Xtr2_mat100_np[:len_tr] , Ytr2_np[:len_tr]
    Xval_2, Y_val2 = Xtr2_mat100_np[len_tr:] , Ytr2_np[len_tr:]
    Model_LLR2 = LinearLogisticRegression(verbose = False )
    %time Model_LLR2.fit(Xtrain_2, Y_train2 )
    Y_val2_pred = Model_LLR2.predict(Xval_2)
    (Y_val2_pred  == Y_val2).mean() #0.68
    
    
    # Final prediction 
    Y_predict0 = Model_LLR0.predict(Xte0_mat100)
    Y_predict1 = Model_LLR1.predict(Xte1_mat100)
    Y_predict2 = Model_LLR2.predict(Xte2_mat100 )
    
    Y_precict0 = np.where( Y_predict0== True ,1,0)
    Y_precict1 = np.where( Y_predict1== True ,1,0)
    Y_precict2 = np.where( Y_predict2== True ,1,0)
    
    Id = list(np.arange(3000))
    Bound = np.array([Y_precict0 , Y_precict1 , Y_precict2 ]).ravel()

    Data = {"Id":Id, "Bound":Bound}
    Data = pd.DataFrame(Data)
    Data.to_csv('Yte.csv', index= False)