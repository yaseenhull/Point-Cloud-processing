# -*- coding: utf-8 -*-
"""
Created on Sat May 12 13:48:57 2018

@author: Yaseen 
"""

import numpy as np
import math as m
import scipy.linalg
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
#from sklearn.decomposition import PCA


    
def lstsq(dlist):
    # best-fit linear plane
    A = np.c_[dlist[:,0], dlist[:,1], np.ones(dlist.shape[0])]
    C,_,_,_ = scipy.linalg.lstsq(A, dlist[:,2])    # coefficients
    # regular grid covering the domain of the data
   
   
    plt.figure()
    ax = plt.subplot(111,projection='3d')
    #ax.plot_surface(X, Y, rstride=1, cstride=1, alpha=0.2)
    ax.scatter(dlist[:,0], dlist[:,1], dlist[:,2], c='r')
    
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()    
    X,Y = np.meshgrid(np.arange(xlim[0],xlim[1]), np.arange(ylim[0], ylim[1]))
     
    Z = C[0]*X + C[1]*Y + C[2]
    
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.2)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()
    print(C[0],C[1],C[2])
    print('''''''''''''''''')
    print(Z)
    
def xyzrange(dlist):
    temp =[]    
    for i in range(5):
        temp.append(dlist[i][0])
    xMax = max(temp)
    xMin = min(temp)
    return xMax, xMin  
 
def eig(X_std):
    mean_vec = np.mean(X_std, axis=0)
    cov_mat = (X_std - mean_vec).T.dot((X_std - mean_vec)) / (X_std.shape[0]-1)    
    cov_mat = np.cov(X_std.T)

    eig_vals, eig_vecs = np.linalg.eig(cov_mat)
    
    return eig_vals, eig_vecs
    
def eigen(X):
    centered_matrix = X - X.mean(axis=1)[:, np.newaxis]
    cov = np.dot(centered_matrix, centered_matrix.T)
    eigvals, eigvecs = np.linalg.eig(cov)
    
    return eigvals, eigvecs
    
"""def PCA_(X):
    pca = PCA(n_components=3)
    pca.fit(X)
    pc = pca.components_
    
    #print(pca.components_)
    #print(pca.components_[:,2])
    
    
    Min = np.min(pc[:,2])

    #print(Min)
    return pc, Min"""
        
def mean(X):
    a = np.average(X)
    return a

def normalise(X,a,b): #needs editting   
    return (X-a)/(b)  

def Min(X):
    a = np.min(X)
    return a 

def Max(X):
    a = np.max(X)
    return a 

def PCA(points, correlation = False, sort = True):
    mean = np.mean(points, axis=0)

    points_adjust = points - mean

#: the points is transposed due to np.cov/corrcoef syntax
    if correlation:

        matrix = np.corrcoef(points_adjust.T)

    else:
        matrix = np.cov(points_adjust.T) 

        eigenvalues, eigenvectors = np.linalg.eig(matrix)

    if sort:
    #: sort eigenvalues and eigenvectors
        sort = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[sort]
        eigenvectors = eigenvectors[:,sort]
#        print(eigenvectors)
#        print(eigenvalues, eigenvectors)
    return eigenvalues, eigenvectors
        
def best_fitting_plane(points, equation=False):
    
    w, v = PCA(points)

#: the normal of the plane is the last eigenvector
    normal = v[:,2]
    #print(normal)
#: get a point from the plane
    point = np.mean(points, axis=0)


    if equation:
        a, b, c = normal
        d = -(np.dot(normal, point))
        return a, b, c, d

    else:
        return point, normal
        
def std_dev(dlist):
    n = len(dlist)
    mean = np.mean(dlist)
    sum1 = 0
    for i in range(len(dlist)):
        sum1 += (dlist[i] - mean)**2
    stdd = m.sqrt(sum1/n)
    
    return stdd

