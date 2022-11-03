from matplotlib import pyplot as plt
import numpy as np
import copy
import time
import random
from numpy import diff
from numba import jit,njit, prange, vectorize, float64
from numba.typed import List

def gauss_2d(mu, sigma):
    x = random.gauss(mu, 5*sigma)
    y = random.gauss(mu, sigma)
    return np.array([x, y])

def unsupervised_loss(centroids,cluster_list):
    loss_unsup = 0
    for j in range(len(centroids)):
        centre = centroids[j]
        for i in range(len(np.array(cluster_list[j])[:,0])):
            loss_unsup += distance(centre,np.array(cluster_list[j])[i,:])
    return loss_unsup

@njit(nogil=True)
def distance(x1, x2, sigma=None, cov=None):
    if not cov:
        sigma = np.array([4,1])
    return np.sum(((x1-x2)/sigma)**2)**0.5

#@njit(nogil=True)
def distance_from_centres(centres,x_point,K,sigma=None,cov=None):
    distances_list = np.zeros(K)
    if not cov:
        sigma = None
        for j in range(len(centres)):
            c = centres[j]
            distances_list[j] = distance(c, x_point)#,sigma,cov)
    else: 
        cov = True
        for j in range(len(centres)):
            c = centres[j]
            ss= sigma[j]
            #print(ss)
            distances_list[j] = distance(c, x_point,ss,cov)
    return distances_list



@njit(nogil=True)
def distancee(x1, x2, sigma):
    return np.sum(((x1-x2))**2)**0.5


def kmeans(X, k, max_iters):
    centroids = X[np.random.choice(range(len(X)), k, replace=False)]
    converged = False    
    current_iter = 0
    #LOSS_WCSS = []
    while (not converged) and (current_iter < max_iters):
        cluster_list = [[] for i in range(len(centroids)+1)]
        for i in range(len(X)):  # Go through each data point
            x = X[i]

            distances_list = distance_from_centres(centroids,x,k)     

            # appendo x nella cluster list alla posizione argmin 
            cluster_list[int(np.argmin(distances_list))].append(x)

        cluster_list = list((filter(None, cluster_list)))

        if False:
            loss = unsupervised_loss(centroids,cluster_list)
            print('Loss unsupervised', loss)

        # Update centroids
        prev_centroids = centroids.copy()
        centroids = []

        for j in range(len(cluster_list)):
            centroids.append(np.mean(cluster_list[j], axis=0).astype(int))
        pattern = np.abs(np.sum(prev_centroids) - np.sum(centroids))

        if True:
            print('K-MEANS iter',current_iter,' delta centroids', pattern)

        converged = (pattern == 0)

        current_iter += 1
        clulist = cluster_list.copy()
        if current_iter==max_iters: print('Kmeans Alert: MAX ITER REACHED')

    return np.array(centroids), [[np.std(np.array(x)[:,0]),np.std(np.array(x)[:,1])] for x in cluster_list],  cluster_list








def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w



#####################################################################################
#                           Test Kmean
#####################################################################################

if False:
 ITER = 10
 N_data = 5000
 X_train = np.zeros((N_data,2))
 current_iter = 0
 for _ in range(1):
     if current_iter>0: loss_old = copy.copy(loss)
     rndnum = 1#random.gauss(1,0.3)
     for j in range(N_data):
         if j<int(N_data/4):
             mu = 45*rndnum
             sigma = 0.1
         if j<int(2*N_data/4) and j>int(N_data/4) :
             mu = 30*rndnum
             sigma = 0.1
         if j<int(3*N_data/4)and j>int(2*N_data/4):
            mu = 15*rndnum
            sigma = 0.1
         if j>int(3*N_data/4):
            mu = 1*rndnum
            sigma = 0.1
     
         X_train[j,:] = gauss_2d(mu,sigma)#.astype(int)
     
     if False:
        plt.plot(X_train[:,0],X_train[:,1], '.g')
        plt.show()
     #########
     # main
     #########
     i = time.time()
     ##########################################
     Centroids,sigma, Cluster_list = kmeans(X_train, 4, 1000)
     ##########################################
     f = time.time() - i
     print('Walltime k-mean: ',f)
     

     loss = unsupervised_loss(Centroids,Cluster_list)
     print('Loss unsupervised', round(loss,2))

     #########
     # plot
     #########
     if 1:
         plt.plot(Centroids[:,0],Centroids[:,1], '*k')
         for i in range(len(sigma)):     
                 clust_i = np.array(Cluster_list[i])
                 plt.plot(clust_i[:,0],clust_i[:,1],'.')
                 circle = plt.Circle((Centroids[i,0],Centroids[i,1]),
                                          2*sigma[i],fill= False, color='k')
                 plt.gca().add_patch(circle)
         plt.grid(True)
         plt.show()
     current_iter += 1


if True:
 ITER = 10
 N_data = 500
 X_train = np.zeros((N_data,2))
 current_iter = 0
 for _ in range(2):
     if current_iter>0: loss_old = copy.copy(loss)
     rndnum = random.gauss(1,0.3)
     for j in range(N_data):
         if j<int(N_data/4):
             mu = 20#*rndnum
             sigma = 1
         if j<int(2*N_data/4) and j>int(N_data/4) :
             mu = 13#*rndnum
             sigma = 1
         if j<int(3*N_data/4)and j>int(2*N_data/4):
            mu = 7#*rndnum
            sigma = 1
         if j>int(3*N_data/4):
            mu = 2#*rndnum
            sigma = 1
     
         X_train[j,:] = gauss_2d(mu,sigma)

     i = time.time()
     ##########################################
     Centroids,sigma, Cluster_list = kmeans(X_train, 4, 1000)
     ##########################################
     f = time.time() - i
     print('Walltime k-mean: ',f)

     loss = unsupervised_loss(Centroids,Cluster_list)
     print(len(sigma))
     print(sigma)

     if 0:     
         LOSSperCentre = []
         nrcentre = []
         for ncentre in range(20):
            Centroids,sigma,Cluster_list = kmeans(X_train, ncentre+1, 1000)
            loss = unsupervised_loss(Centroids,Cluster_list)
            if 0:       
                print(ncentre+1, 'centres, loss=', loss)
            LOSSperCentre.append(loss)
            nrcentre.append(ncentre+1)
         nrcentre = np.asarray(nrcentre)
         LOSSperCentre = np.asarray(LOSSperCentre)
         Transfer_function = (diff(LOSSperCentre)/diff(nrcentre))/np.max(LOSSperCentre)
         Transfer_function = moving_average(Transfer_function,4)
         plt.plot(Transfer_function)
         plt.title('Loss per nr centres')
         plt.grid(True)
         plt.show()
     if 1:
         print('Loss unsupervised', round(loss,2))
         plt.plot(Centroids[:,0],Centroids[:,1], '*k')
         for i in range(len(sigma)):     
                 clust_i = np.array(Cluster_list[i])
                 plt.plot(clust_i[:,0],clust_i[:,1],'.')
                 #circle = plt.Circle((Centroids[i,0],Centroids[i,1]),
                 #                     2*sigma[i],fill= False, color='k')
                 #plt.gca().add_patch(circle)
         plt.grid(True)
         plt.show()
         current_iter += 1
