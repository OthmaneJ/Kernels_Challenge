#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kernel Logistic regression with the RBF Kernel 
# slides 105-114 of the course 
"""

import os 
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def K_rbf(x, y , gamma = 1000):
    return np.exp( -gamma*(np.linalg.norm(x-y)**2) )

def Gram_rbf(X, gamma = 0.1, Xte = []):
    
    n_X = X.shape[0] # number of observations in the training data 
    if len(Xte)== 0: # if no test data 
        Gram_matrix = np.zeros((n_X, n_X), dtype = np.float32)
        
        for i in range(n_X):
            for j in range(i, n_X):
                Gram_matrix[i, j] = K_rbf(x = X[i],y = X[j], gamma = gamma)
                Gram_matrix[j, i] = Gram_matrix[j, i]
        return Gram_matrix 
    else:
        n_Xte = Xte.shape[0]
        Gram_matrix = np.zeros((n_X, n_Xte), dtype = np.float32)
        
        for i in range(n_X):
            for j in range(n_Xte):
                Gram_matrix[i, j] = K_rbf(x = X[i],y = Xte[j], gamma = gamma)
        return Gram_matrix


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

#def eta(x, w):
#    return sigmoid( np.dot(W.T, x))

#def inv_sigmoid(x):
#    return np.log(1/x-1)


def m(p_K, p_alpha ):
    return p_K.dot(p_alpha)
def P(p_y, p_m):
    return np.diag(-sigmoid(-p_y*p_m)[:,0])

def W(p_m):
    diag = sigmoid(p_m)
    
    diag = diag*(1- diag)
    return np.diag(diag[:,0])

def Z(p_m, p_y):
    return p_m + p_y/sigmoid(-p_y*p_m)

def compute_alpha(p_K, p_W, p_Z, lambda_reg):
    W_sqrt = np.sqrt(p_W)
    n = p_K.shape[0]
    
    to_inv = W_sqrt.dot(p_K).dot(W_sqrt) + n*lambda_reg*np.eye(n)
    inv = np.linalg.inv(to_inv)
    alpha_ =  W_sqrt.dot(to_inv).dot( W_sqrt).dot(p_Z)
    return alpha_




class KLR:
    """
    Implement kernel logistique regression 
    """
    def __init__(self, init_coef = 0):
        if init_coef==0:
            self.alpha_ = 0
    # train 
    def fit(self, K_train, y, alpha_ = None, num_iter=100, tol = 1, lambda_reg = 0, verbose = False):
        """
        #sc = False 
        if y.ndim==1:
            y.resize([  y.shape[0]  , 1]) # reshape the label matrix 
            sc = True
        """
        if alpha_ == None:
        #if np.array((alpha_==None)).any():
            alpha_ = np.random.randn(K_train.shape[0], 1)
        for i in range(num_iter):
            if verbose:
                print(i)
            old_alpha_ = np.array(alpha_) # create a copy 

            # compute quantities
            m_c = m(K_train, alpha_)
            P_c =  P(y, m_c)     #np.nan_to_num(P(y, m_c))
            W_c = W(m_c) #np.nan_to_num(W(m_c))
            Z_c = Z(m_c, y)

            # upadate alpha 
            alpha_ = compute_alpha(K_train, W_c, Z_c, lambda_reg)

            """
            if sc:
                y.resize([  y.shape[0]  , 1])
            """
            if verbose:
                print(np.linalg.norm(alpha_ - old_alpha_))
            #if (np.linalg.norm(alpha_ - old_alpha_)> tol):
            #   self.fit(K_train, y, alpha_, tol, lambda_reg)
        
        self.alpha_ = alpha_
        
    def get_coef(self):
        return list(self.alpha_)
    def predict(self, K_test):
        y_pred = ( (self.alpha_.T.dot(K_test)).T).reshape(-1)
        y_pred = sigmoid(y_pred).reshape(-1)                        # predict the proba
        
        return np.array(y_pred >= 0.5 , dtype = int)