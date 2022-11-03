from matplotlib import pyplot as plt
import numpy as np
import copy
import time
import random
from numba import jit,njit, prange, vectorize, float64
from sklearn.decomposition import PCA

from HashClass import HashTable


from Conv import activation_discrete, Convolution_2D, Convolution_3D, POOLING, RELU_tab, FCN, feedforward, feed



def unsupervised_loss(centroids,cluster_list):
    loss_unsup = 0
    for j in range(len(centroids)):
        centre = centroids[j]
        for i in range(len(np.array(cluster_list[j])[:,0])):
            loss_unsup += distance(centre,np.array(cluster_list[j])[i,:])
    return loss_unsup

def sup_loss(O,LABEL):
    losses = np.zeros((len(LABEL)), dtype=np.float64)
    for IDX in prange(len(LABEL)):
        label = LABEL[IDX]
        out = O[IDX]
        predict = np.zeros((len(out)))
        predict[label] = 1
        losses[IDX]= np.sum((predict-out)**2)
    return np.sqrt(np.mean(losses))

@njit(nogil=True)
def distance(x1, x2):
    return np.sum((x1-x2)**2)**0.5

@njit(nogil=True)
def distance_ell(x1, x2, s):
    return np.sum(((x1-x2)/s)**2)**0.5


def compute_l2_distance(x, centroid):
    # Compute the difference, following by raising to power 2 and summing
    dist = ((x - centroid) ** 2).sum(axis = x.ndim - 1)
    
    return dist

def get_closest_centroid(x, centroids):
    # Loop over each centroid and compute the distance from data point.
    dist = compute_l2_distance(x, centroids)

    # Get the index of the centroid with the smallest distance to the data point 
    closest_centroid_index =  np.argmin(dist, axis = 1)
    
    return closest_centroid_index


def compute_sse(data, centroids, assigned_centroids):
    # Initialise SSE 
    sse = 0
    # Compute SSE
    sse = compute_l2_distance(data, centroids[assigned_centroids]).sum() / len(data)
    return sse



def kmeans(data, k, max_iters, restart=False, oldcentres=None):
    # Number of dimensions in centroid
    num_centroid_dims = data.shape[1]

    # List to store SSE for each iteration 
    sse_list = []

    # Initialise centroids
    if restart==False:
        centroids = data[random.sample(range(data.shape[0]), k)]
    else: centroids = oldcentres

    # Create a list to store which centroid is assigned to each dataset
    assigned_centroids = np.zeros(len(data), dtype = np.int32)

    converged = False    
    current_iter = 0

    while (not converged) and (current_iter < max_iters):

        # Get closest centroids to each data point
        assigned_centroids = get_closest_centroid(data[:, None, :], centroids[None,:, :])    

        # Compute new centroids
        for c in range(centroids.shape[0]):

            # Get data points belonging to each cluster 
            cluster_members = data[assigned_centroids == c]
            
            # Compute the mean of the clusters
            #print(np.sum(cluster_members,axis=0))
            cluster_members = cluster_members.mean(axis = 0)
            prev_centroids = centroids.copy()

            # Update the centroids
            centroids[c] = cluster_members
    
        # Compute SSE
        sse = compute_sse(data.squeeze(), centroids.squeeze(),  assigned_centroids)
        sse_list.append(sse)

        pattern = np.abs(np.sum(prev_centroids) - np.sum(centroids))

        #####################################################################
        if False:
            print('K-MEANS iter', current_iter, ' delta centroids', pattern)
        #####################################################################

        converged = (pattern == 0)
        current_iter += 1
    
    cluster_list = [[] for _ in range(len(centroids))]
    for i in range(len(data)):

        cluster_list[assigned_centroids[i]].append(data[i,:])
    
    return np.array(centroids),[np.std(x) for x in cluster_list], cluster_list


def kmeans_incr(X, k, centroids, max_iters):
    converged = False    
    current_iter = 0
    while (not converged) and (current_iter < max_iters):
        cluster_list = [[] for i in range(len(centroids))]

        for i in range(len(X)):  # Go through each data point
            x = X[i]
            distances_list = np.zeros(len(centroids))

            for j in range(len(centroids)):
                c = centroids[j]
                distances_list[j] = distance(c, x)

            # appendo x nella cluster list alla posizione argmin 
            if int(np.argmin(distances_list))== len(cluster_list): 
                print(int(np.argmin(distances_list)))
                print(len(cluster_list))
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
        if False:
            print('K-MEANS iter',current_iter,' delta centroids', pattern)
        converged = (pattern == 0)
        current_iter += 1

        if current_iter==max_iters: print('MAX ITER REACHED')

    return np.array(centroids), [np.array([np.std(np.array(x)[:,0]),np.std(np.array(x)[:,1])]) for x in cluster_list], cluster_list



#####################################################################################
#                      RADIAL BASIS FUNCTION NETWORK
#####################################################################################
@njit(nogil=True)
def convert_to_one_hot( x, num_of_classes):
    arr = np.zeros((len(x), num_of_classes))
    for i in range(len(x)):
        c = int(x[i])
        arr[i][c] = 1
    return arr


def rbf(x, c, s,std_weight):
    dist = distance_ell(x, c, std_weight)
    return 1 / np.exp(-dist / s ** 2)

def rbf_list(DATASET, centroids, std_list,std_weight):
    RBF_list = []
    for x in DATASET:
        RBF_list.append([rbf(x, c, s, std_weight[cent,:]*10) for cent,(c, s) in enumerate(zip(centroids, std_list))])
    return np.array(RBF_list)

def RBF_Network( TRAIN_SET, TRAIN_LABEL, num_of_classes, k, std_weight, std_from_clusters=True): 
        # TEST_SET, TEST_LABEL
        '''
                   RBF NETWORK        
        
        TRAIN_SET   -> N array (vectorized img)
        TRAIN_LABEL -> N scalar from 0..9

        TEST_SET    -> M array
        TEST_LABEL  -> M scalar from 0..9

        num_of_classes = number of output (output neurons)
        k = number of centroids (hidden neurons)
        '''
        
        # K-MEANS ALGORITHM for estimating centres
        i = time.time()
        ##########################################
        centroids, std_list, clust_list = kmeans(TRAIN_SET, k, max_iters=1000)
        ##########################################
        f = time.time() - i
        print('Walltime k-mean: ',f)

        print(np.array(std_list).shape)
        ##########################################
        # INSIGHT ON CLUSTERING    
        if False:
          for n_centre in range(len(centroids)):
            r = []
            theta = []
            color = []
            for img in range(len(TRAIN_SET[:,0])):
                theta.append(2*np.pi*random.uniform(0, 1))
                r.append(distance(TRAIN_SET[img],centroids[n_centre]))
                color.append(round(TRAIN_LABEL[img]/9,2))
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='polar')
            c = ax.scatter(theta, r, c=color, cmap='hsv', alpha=0.75)
            plt.title('DISTANCE FROM CENTRE {}'.format(str(n_centre)))
            plt.show()
        ##########################################


        if 0:#not std_from_clusters:
            dMax = np.max([distance(c1, c2) for c1 in centroids for c2 in centroids])
            std_list = np.repeat(dMax / np.sqrt(2 * k), k)


        ##########################################
        # RBF weights evaluation
        std_weight = std_weight.copy() + 2
        RBF_X = rbf_list(TRAIN_SET, centroids, std_list, std_weight)
        print(RBF_X.shape)
        # Computation of the weights
        WEIGHTS = np.linalg.pinv(RBF_X.T @ RBF_X) @ RBF_X.T @\
                          convert_to_one_hot(TRAIN_LABEL,num_of_classes)

        #WEIGHTS = WEIGHTS #.astype(int)
        predictions = RBF_X @ WEIGHTS
        loss = sup_loss(predictions,TRAIN_LABEL)

        ##########################
        #  TEST SET and ACCURACY
        ##########################
        if 0:
            # RBF FEED on test set
            RBF_list_test = rbf_list(TEST_SET, centroids, std_list)
            # Linear combinations
            pred_ty = RBF_list_test @ WEIGHTS
            # Predictions
            pred_ty = np.array([np.argmax(x) for x in pred_ty])
    
            # ACCURACY
            diff = pred_ty - TEST_LABEL
            print('Accuracy: ', len(np.where(diff == 0)[0]) / len(diff))

        return centroids, WEIGHTS, clust_list, std_list, loss



def RBF_Net_incr( TRAIN_SET, TRAIN_LABEL, num_of_classes, k, centroids,
                                     std_weight, std_from_clusters=True):
        '''
                   RBF NETWORK        
        
        TRAIN_SET   -> N array (vectorized img)
        TRAIN_LABEL -> N scalar from 0..9

        TEST_SET    -> M array
        TEST_LABEL  -> M scalar from 0..9

        num_of_classes = number of output (output neurons)
        k = number of centroids (hidden neurons)
        '''
        
        # K-MEANS ALGORITHM for estimating centres
        #i = time.time()
        ##########################################
        centroids_new, std_list , clust_list = kmeans(TRAIN_SET, k, max_iters=1000, restart = True ,oldcentres=centroids )
        ########################################## (data, k, max_iters, restart=False, oldcentres=None):
        #f = time.time() - i
        #print('Walltime k-mean: ',f)

        if 0:# not std_from_clusters:
            dMax = np.max([distance(c1, c2) for c1 in centroids_new for c2 in centroids_new])
            std_list = np.repeat(dMax / np.sqrt(2 * k), k)


        ##########################################
        std_weight1 = std_weight.copy() + 2
        # RBF weights evaluation
        RBF_X = rbf_list(TRAIN_SET, centroids_new, std_list, std_weight1)

        # Computation of the weights
        WEIGHTS = np.linalg.pinv(RBF_X.T @ RBF_X) @ RBF_X.T @\
                          convert_to_one_hot(TRAIN_LABEL,num_of_classes)

        #WEIGHTS = WEIGHTS#.astype(int)
        predictions = RBF_X @ WEIGHTS
        loss = sup_loss(predictions,TRAIN_LABEL)
        
        ##########################
        #  TEST SET and ACCURACY
        ##########################
        if 0:
            # RBF FEED on test set
            RBF_list_test = rbf_list(TEST_SET, centroids, std_list)
            # Linear combinations
            pred_ty = RBF_list_test @ WEIGHTS
            # Predictions
            pred_ty = np.array([np.argmax(x) for x in pred_ty])

            # ACCURACY
            diff = pred_ty - TEST_LABEL
            print('Accuracy: ', len(np.where(diff == 0)[0]) / len(diff))

        return centroids_new, WEIGHTS, clust_list, std_list, loss

#####################################################################################
#####################################################################################
