# -*- coding: utf-8 -*-
"""
Created on Thu May 24 20:47:25 2018

@author: Yaseen

Point Cloud classification and operation
"""

""" Import data, classifier library and function """

import pandas as pd
from sklearn.neighbors import KDTree
import numpy as np
from Functions import  xyzrange, eigen, mean, normalise, Min, Max, best_fitting_plane, eig, std_dev, PCA
#from sklearn.decomposition import PCA
import random
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn import preprocessing
 
df = pd.read_csv('W55A_2.csv')

#r = random.sample(range(0,len(df)),8000) ramdomise data
#rows = df.iloc[r][['//X','Y','Z']]
all_rows = df.iloc[0:len(df)][['//X','Y','Z']]

#Xt = np.array(rows)
""" Create matrix of data """
Lt = np.array(all_rows)

kdt = KDTree(all_rows, leaf_size=50, metric='euclidean') # data structure for nearest neighbour
#dist, ind = kdt.query(Xt, k=10)

F = df.iloc[0:len(df)][['//X','Y','Z','R','G','B','Intensity']]
Ft = np.array(F)

#feature lists
f1 = []
f2 = []
f3 = []
f4 = []
ev = []
normies = []
#n_n = NearestNeighbors(n_neighbors=30, algorithm='kd_tree').fit(Xt)

"""Calculate attribute data to feed classifier """
for i in range(len(df)):

    dist, ind = kdt.query([Lt[i]], k=30)
    #distances, ind = n_n.kneighbors([Xt[i]])
    x = []
    y = []
    z = []
    Int = []
    R = []
    G = []
    B = []
    for k in range(len(ind[0])):
        indice = ind[0][k]
        
        x.append(Ft[indice][0]) #'''list of xyz -cords of indices'''
        y.append(Ft[indice][1])
        z.append(Ft[indice][2])
        Int.append(Ft[indice][6])
        R.append(Ft[indice][3])
        G.append(Ft[indice][4])
        B.append(Ft[indice][5])
    
    R_normalized = preprocessing.normalize([R], norm='l2') #normalisation of rgb, intensities
    G_normalized = preprocessing.normalize([G], norm='l2')
    B_normalized = preprocessing.normalize([B], norm='l2')
    Int_normalized = preprocessing.normalize([Int], norm='l2')
    Int_std = std_dev(Int_normalized.tolist()[0]) #standard deviation of intensities
    data = np.c_[x,y,z]
    pt, normal = best_fitting_plane(data)
    eigenv_x = PCA(data)[0][0]    
    eigenv_y = PCA(data)[0][1]    
    eigenv_z = PCA(data)[0][2]
    ev.append([eigenv_x,eigenv_y,eigenv_z])
  
    f4.append([np.mean(R_normalized.tolist()),np.mean(G_normalized.tolist()),np.mean(B_normalized.tolist())])
  
    f1.append([normal[0],normal[1],normal[2],np.mean(R_normalized.tolist()),np.mean(G_normalized.tolist()),np.mean(B_normalized.tolist())])
    normies.append(normal.tolist())

ex = preprocessing.normalize([np.array(ev)[:,0].tolist()]) #normalisation of eigenvalues
ey = preprocessing.normalize([np.array(ev)[:,1].tolist()])
ez = preprocessing.normalize([np.array(ev)[:,2].tolist()])   


for i in range(len(df)):
    f2.append([ex[0][i],ey[0][i],ez[0][i],normies[i][0],normies[i][1],normies[i][2]])
    f3.append([ex[0][i],ey[0][i],ez[0][i]])

# kmeans unsupervised classification
""" Classification """

kmeans1 = KMeans(n_clusters=150, random_state=0).fit(f1)
kmeans_labels = kmeans1.labels_
labels = {'X': Lt[:,0], 'Y': Lt[:,1], 'Z': Lt[:,2], 'kmeans labels': kmeans_labels}
labels_new = pd.DataFrame(data = labels)
labels_new.to_csv('norm_rgb.csv')

"""
kmeans2 = KMeans(n_clusters=150, random_state=0).fit(f2)
kmeans_labels2 = kmeans2.labels_
labels2 = {'X': Lt[:,0], 'Y': Lt[:,1], 'Z': Lt[:,2], 'kmeans labels': kmeans_labels2}
labels_new2 = pd.DataFrame(data = labels2)
labels_new2.to_csv('eig_norm1.csv')


kmeans3 = KMeans(n_clusters=150, random_state=0).fit(f3)
kmeans_labels3 = kmeans3.labels_
labels3 = {'X': Lt[:,0], 'Y': Lt[:,1], 'Z': Lt[:,2], 'kmeans labels': kmeans_labels3}
labels_new3 = pd.DataFrame(data = labels3)
labels_new3.to_csv('eig1.csv')

kmeans4 = KMeans(n_clusters=150, random_state=0).fit(f4)
kmeans_labels4 = kmeans4.labels_
labels4 = {'X': Lt[:,0], 'Y': Lt[:,1], 'Z': Lt[:,2], 'kmeans labels': kmeans_labels4}
labels_new4 = pd.DataFrame(data = labels4)
labels_new4.to_csv('rgb.csv')
"""