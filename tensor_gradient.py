import tensornetwork as tn
import numpy as np
from scipy.linalg import expm
from math import sqrt
import math
import random
import copy
from typing import List, Tuple
import itertools
import scipy.integrate as integrate
from scipy.optimize import minimize
import time
from numpy import random

# return the derivative of a tensor, with error O(4^n/T). n = 2, 4^2 = 16. 
def derivative_su4(T:int, para:list):
    #debug
    if len(para) != 15:
        raise TypeError("len(para) error")
    lamb = [
        np.array(
            [[1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
             [0.+0.j, -1.+0.j, 0.+0.j, 0.+0.j],
             [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
             [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j]]),
        (1/sqrt(3))*np.array(
            [[1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
             [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
             [0.+0.j, 0.+0.j, -2.+0.j, 0.+0.j],
             [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j]]),
        (1/sqrt(6))*np.array(
            [[1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
             [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
             [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
             [0.+0.j, 0.+0.j, 0.+0.j, -3.+0.j]]),
        np.array(
            [[0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
             [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
             [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
             [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j]]),
        np.array(
            [[0.+0.j, 0.-1.j, 0.+0.j, 0.+0.j],
             [0.+1.j, 0.+0.j, 0.+0.j, 0.+0.j],
             [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
             [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j]]),
        np.array(
            [[0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
             [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
             [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
             [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j]]),
        np.array(
            [[0.+0.j, 0.+0.j, 0.-1.j, 0.+0.j],
             [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
             [0.+1.j, 0.+0.j, 0.+0.j, 0.+0.j],
             [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j]]),
        np.array(
            [[0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
             [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
             [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
             [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j]]),
        np.array(
            [[0.+0.j, 0.+0.j, 0.+0.j, 0.-1.j],
             [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
             [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
             [0.+1.j, 0.+0.j, 0.+0.j, 0.+0.j]]),
        np.array(
            [[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
             [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
             [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
             [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j]]),
        np.array(
            [[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
             [0.+0.j, 0.+0.j, 0.-1.j, 0.+0.j],
             [0.+0.j, 0.+1.j, 0.+0.j, 0.+0.j],
             [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j]]),
        np.array(
            [[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
             [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
             [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
             [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j]]),
        np.array(
            [[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
             [0.+0.j, 0.+0.j, 0.+0.j, 0.-1.j],
             [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
             [0.+0.j, 0.+1.j, 0.+0.j, 0.+0.j]]),
        np.array(
            [[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
             [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
             [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
             [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j]]),
        np.array(
            [[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
             [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
             [0.+0.j, 0.+0.j, 0.+0.j, 0.-1.j],
             [0.+0.j, 0.+0.j, 0.+1.j, 0.+0.j]])
    ]
#     H = sum([para[i]*lamb[i] for i in range(15)])
#     U44 = expm(1.j * H)
    n_para = len(para)
    
    mat_mul = np.identity(4)
    H_local = [np.identity(4) for i in range(n_para)]
    mat_mul_front = [np.identity(4) for i in range(n_para)]
    mat_mul_back = [np.identity(4) for i in range(n_para)]
#     mat_mul = [np.matmul(mat_mul, lamb[j]) for j in range(len(lamb))]
#     H_local[0] = expm(1.j * para[0] * lamb[0]/T)
    H_local[n_para - 1] = expm(1.j * para[n_para-1] * lamb[n_para-1]/T)
     
    for i in range(n_para-1):
        H_local[i] = expm(1.j * para[i] * lamb[i]/T)
        mat_mul_front[i+1] = np.matmul(mat_mul_front[i], H_local[i])
    
    for i in range(n_para-1,0,-1):
#         H_local[j] = expm(1.j * para[j] * lamb[j])
        mat_mul_back[i-1] = np.matmul(H_local[i],mat_mul_back[i])
    
    mat_mul = np.matmul(mat_mul_front[n_para-1], H_local[n_para - 1])
#     mat_mul_com = np.identity(4)
#     mat_mul_com = np.matmul(H_local[0], mat_mul_back[0])
    
    mat_mul_exp = [np.identity(4) for i in range(T+1)]
    for i in range(1,T+1):
        mat_mul_exp[i] = np.matmul(mat_mul_exp[i-1], mat_mul)
    
    deri_mat = [np.array(4*[4*[0.+0.j]]) for i in range(n_para)]
    for j in range(n_para):
        if T == 1:
            deri_mat[j] = deri_mat[j] + 1.j/T * np.matmul(mat_mul_front[j], np.matmul(lamb[j], np.matmul(H_local[j], mat_mul_back[j])))
        else:
            for k in range(T):
                deri_mat[j] = deri_mat[j] + 1.j/T * np.matmul(mat_mul_exp[k-1], np.matmul(mat_mul_front[j], np.matmul(lamb[j], np.matmul(H_local[j],np.matmul(mat_mul_back[j],mat_mul_exp[T-k])))))
    
    return deri_mat

para = [random.uniform(0, 2*math.pi) for j in range(15)] #random.uniform(0,2*math.pi)
T = 10
deri_mat = derivative_su4(T, para)
print(deri_mat)
