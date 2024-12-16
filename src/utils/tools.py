# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 10:07:51 2020

@author: yangm
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import random as rm
import scipy.sparse as sp
from scipy.sparse import coo_matrix

def get_max_indices(nums):

    max_of_nums = np.max(nums)
    tup = [(i, nums[i]) for i in range(len(nums))]
    return [i for i, n in tup if n == max_of_nums]

def get_reverse_sort(nums, ratio = 0.5, K = 5, is_padding = False):
    
    tup = [(i, nums[i]) for i in range(nums.size)]
    tup = [[item[0], item[1]] for item in sorted(tup, key = lambda x:x[1], reverse = True)]
    tup = np.array(tup)
    
    count = 0
    index = []
    
    for item in tup[0:K]:
        
        if(item[1] > 0):
            count = item[1] + count
        
            index.append(item[0])
        
        if(ratio <= count):
            break;
    
    if(True == is_padding):
        
        index = np.array(index, dtype=int)
        
        if(index.size < K):
        
            index = np.pad(index, [0, K - index.size], 'constant', constant_values = -1)
    
    return index

def get_mutual_information(A, bias = 1e-10):
    
    A = A.tocsr()
    
    N = A.shape[0]
    
    d = A.sum(axis = 1).A
    p_x = d / N
    
    p_x = np.squeeze(p_x, 1)
    p_inv_x = 1.0 - p_x
    
    p_x_y_row = []
    p_x_y_col = []
    p_x_y_data = []
    
    p_x_inv_y_row = []
    p_x_inv_y_col = []
    p_x_inv_y_data = []
    
    p_inv_x_y_row = []
    p_inv_x_y_col = []
    p_inv_x_y_data = []
    
    p_inv_x_inv_y_row = []
    p_inv_x_inv_y_col = []
    p_inv_x_inv_y_data = []
    
    for i in range(N):
        
        A_i = np.squeeze(A[i].toarray(), 0)
        
        for j in np.squeeze(np.argwhere(A_i > 0), 1): 
        
            if(i == j):
                
                continue;
                
            A_j = np.squeeze(A[j].toarray(), 0)
                
            t_i = (0 < A_i)
            t_j = (0 < A_j)
            f_i = (0 == A_i)
            f_j = (0 == A_j)
            
            t_i_t_j = (t_i * t_j)
            e = (t_i_t_j > 0).sum() / float(N)
            p_x_y_row.append(i)
            p_x_y_col.append(j)
            p_x_y_data.append(e * np.log(e / (p_x[i] * p_x[j] + bias) + bias))
            
            t_i_f_j = (t_i * f_j)
            e = (t_i_f_j > 0).sum() / float(N)
            p_x_inv_y_row.append(i)
            p_x_inv_y_col.append(j)
            p_x_inv_y_data.append(e * np.log(e / (p_x[i] * p_inv_x[j] + bias) + bias))
            
            f_i_t_j = (f_i * t_j)
            e = (f_i_t_j > 0).sum() / float(N)
            p_inv_x_y_row.append(i)
            p_inv_x_y_col.append(j)
            p_inv_x_y_data.append(e * np.log(e / (p_inv_x[i] * p_x[j] + bias) + bias))
            
            f_i_f_j = (f_i * f_j)
            e = (f_i_f_j > 0).sum() / float(N)
            p_inv_x_inv_y_row.append(i)
            p_inv_x_inv_y_col.append(j)
            p_inv_x_inv_y_data.append(e * np.log(e / (p_inv_x[i] * p_inv_x[j] + bias) + bias))
    
    I_x_y = sp.coo_matrix((p_x_y_data, (p_x_y_row, p_x_y_col)), shape = (N, N)).tocsr()
    I_x_inv_y = sp.coo_matrix((p_x_inv_y_data, (p_x_inv_y_row, p_x_inv_y_col)), shape = (N, N)).tocsr()
    I_inv_x_y = sp.coo_matrix((p_inv_x_y_data, (p_inv_x_y_row, p_inv_x_y_col)), shape = (N, N)).tocsr()
    I_inv_x_inv_y = sp.coo_matrix((p_inv_x_inv_y_data, (p_inv_x_inv_y_row, p_inv_x_inv_y_col)), shape = (N, N)).tocsr()
    
    I = I_x_y + I_x_inv_y + I_inv_x_y + I_inv_x_inv_y
    
    return I
    
def get_sim(I):
    
    true_sample_index = []
    
    N = I.shape[0]
    
    for i in range(N):
        
        I_i = np.squeeze(I[i].toarray(), 0)
        j = rm.choice(get_max_indices(I_i))
        true_sample_index.append(j)
            
    return true_sample_index

def get_mutli_sim(I, ratio = 0.5, K = 3, is_padding = False):
    
    true_sample_index = []
    
    N = I.shape[0]
    
    for i in range(N):
        
        I_i = np.squeeze(I[i].toarray(), 0)
        index = get_reverse_sort(I_i, ratio = ratio, K = K, is_padding = is_padding)
        true_sample_index.append(list(map(int, index)))
            
    return true_sample_index

def get_D(adj, n, is_self_loop = True):
    
    d = adj.sum(axis = 1).A
    d = np.squeeze(d)
    if(True == is_self_loop):
        d = d + 1
    d = d.astype(np.float32)
    
    D_index = list(range(d.shape[0]))
    D = coo_matrix((np.power(d, n), (D_index, D_index)), shape=(d.shape[0], d.shape[0]), dtype=np.float32)
    D = D.tocsr()
    
    return D

def get_L_DA(adj):
    
    D = get_D(adj, -1.0)
    I = sp.identity(adj.shape[0])
    A = adj + I
    L_DA = D.dot(A)
    
    return L_DA

def get_random(A, K):
    
    A = A.tocsr()
    
    N = A.shape[0]
    
    true_sample_index = []
    
    for i in range(N):
        
        A_i = np.squeeze(A[i].toarray(), 0)
        j = rm.sample(np.squeeze(np.argwhere(A_i > 0), 1).tolist(), K)
        true_sample_index.append(j)
        
    return true_sample_index