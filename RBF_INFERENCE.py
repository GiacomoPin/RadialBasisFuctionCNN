from matplotlib import pyplot as plt
import numpy as np
import copy
import time
import random
from numba import jit,njit, prange, vectorize, float64
from sklearn.decomposition import PCA
from HashClass import HashTable

##################################################################################
#                       Convolutional Net
##################################################################################
def RELU(x):
    return np.maximum(0, x)
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))
def Sigmoid_tab(n_bit_in, n_bit_weights, N_input, n_bit_out):
    minW = -1 #np.power(2, n_bit_weights - 1)
    maxW =  2 #np.power(2, n_bit_weights - 1)

    OUT = []
    for _ in range(50000):
        kern   = np.random.randint(minW,maxW, size=(N_input))
        bias   = np.random.randint(minW,maxW, 1)
        subimg = np.random.randint(0,np.power(2, n_bit_in), size=(N_input)) 
        out = np.sum(kern*subimg)+bias*255#np.dot(kern,subimg) + bias
        OUT.append(out)
    OUT = np.array(OUT)
    zero  =  np.mean(OUT)

    minimum_value = np.min(OUT)
    maximum_value = np.max(OUT)#  int(zero+sigma)#zero+sigma
    sigma = abs(maximum_value-minimum_value)/2#sigma = 2*np.std(OUT)

    x = np.array([i for i in np.arange(-sigma,sigma)])
    normalization7_7 = sigma / 7
    x = x / normalization7_7
    y = sigmoid(x)
    sigm_tab = np.round(y * (np.power(2, n_bit_out)-1))

    return sigm_tab, sigma

#@njit(nogil=True)
def RELU_tab(n_bit_in, n_bit_weights, N_input, n_bit_out):
    minW = -1 
    maxW =  2

    OUT = []
    for _ in range(150000):
        kern   = np.random.randint(minW,maxW, size=(N_input))
        bias   = np.random.randint(minW,maxW, 1)
        subimg = np.random.randint(0,np.power(2, n_bit_in), size=(N_input)) 
        out = np.sum(kern*subimg) + bias*255 # np.dot(kern,subimg) + bias
        OUT.append(out)
    OUT = np.array(OUT)
    zero  =  np.mean(OUT)

    minimum_value = np.min(OUT)
    maximum_value = np.max(OUT)#  int(zero+sigma)#zero+sigma'''
    sigma = abs(maximum_value-minimum_value)/2#sigma = 2*np.std(OUT)

    x = np.array([i for i in np.arange(zero-sigma,zero+sigma)])
    normalization7_7 = sigma / 7
    x = (x) / normalization7_7
    y = RELU(x)
    y = y/np.max(y)
    relu_tab = np.round(y * (np.power(2, n_bit_out)-1))

    return relu_tab, minimum_value

@njit(nogil=True)
def activation_discrete(x, activation_tab, minimum_value):
    x = int(x + abs(int(minimum_value))-1)

    if   x>len(activation_tab): out = 255
    elif x<0: 
        out = 0
    else:     out = activation_tab[x]
    out = int(out)
    return out


##################################################################################

@njit(nogil=True)
def Convolution_2D(img,img_size, kernel, bias,Act_Tab_Ker, Min):
    img = img.reshape(img_size,img_size)
    '''
      - kernel_C1: (3, 5, 5) kernels
      - bias_C1:   (3, 2)   bias
      - img:       (28, 28) image
    '''
    N,k1_height,_ = kernel.shape
    _,biasdep     = bias.shape
    conv_height   = int( (img.shape[0]-k1_height) ) + 1

    # INITIALIZE CONVOLUTION
    C1    = np.zeros((N*biasdep, conv_height, conv_height), dtype=np.int32)  
    for n in range(N):
      for i in range(conv_height):
        for j in range(conv_height):
          region = img[i:(i + k1_height), j:(j + k1_height)]
          for mba in range(biasdep):
            S1_nij = np.sum(region * kernel[n]) + bias[n,mba]*500
            # Activation function
            C1[int(2*n+mba), i, j] = activation_discrete(int(S1_nij),Act_Tab_Ker, Min)

    return C1



@njit(nogil=True)
def Convolution_3D(img, kernel, bias, Act_Tab_Ker, Min):
    '''
      - kernel_C2: (6, 6, 5, 5) kernels
      - bias_C2:   (6, 2)    bias
      - img :      (6,12,12) image
    '''
    M,N,k2_height,_ = kernel.shape
    _,biasdep       = bias.shape
    conv2_height    = int( (img.shape[1]-k2_height) ) + 1

    # INITIALIZE CONVOLUTION
    C2 = np.zeros((M*biasdep, conv2_height, conv2_height), dtype = np.int32)
    S2 = np.zeros((M*biasdep, conv2_height, conv2_height), dtype = np.int32)   
    for m in range(M):
       for u in range(conv2_height):
         for v in range(conv2_height):
           region = img[0:N, u:(u + k2_height), v:(v + k2_height)]
           
           for mba in range(biasdep):
            S2_muv = np.sum(region * kernel[m]) + bias[m,mba]*500
            S2[int(2*m+mba),u,v] = S2_muv
            # Activation function
            # activation_discrete(int(S2_muv),Act_Tab_DEEP_Ker, DEEP_Min)
            C2[int(2*m+mba),u,v] = activation_discrete(int(S2_muv),Act_Tab_Ker, Min)
            # True if S2_muv>0 else False       
    return C2, S2


@njit(nogil=True)
def POOLING(img):
    N , C_height, _ = img.shape
    P_height = int(C_height / 2)
    # Initialize pooled feature map
    P = np.zeros((N, P_height, P_height), dtype=np.int32)
    for n in range(N):
      for i in range(P_height):
        for j in range(P_height):
          region = img[n, (2 * i):(2 * i + 2), (2 * j):(2 * j + 2)]
          P[n, i, j] = np.max(region)

    return P

@njit(nogil=True)
def FCN(Input, Weights, Bias, Act_Tab_Ker, Min):
    ''' Fully connected single layer MLP  '''
    output_o   = np.dot(Weights/1,Input/1) + 255*Bias/1 #np.dot(Weights/7,Input/255) + Bias/7
    output_on  = np.zeros(len(output_o))
    for i in range(len(output_on)):
        output_on[i] = activation_discrete(output_o[i],Act_Tab_Ker, Min)
    if np.sum(output_on)!=0: output_on = output_on/(np.sum(output_on))
    #print(output_on)
    exp_output = np.exp(output_on*10) # (output_o*10) 
    Norm       = np.sum(exp_output) 
    return exp_output/Norm, output_o

@njit(nogil=True)
def feedforward(img,img_size,k_C1,b_C1,k_C2,b_C2,Act_Tab_Ker1,Min1,Act_Tab_Ker2,Min2):#,Act_Tab_Ker3,Min3):
    #--------FEEDFORWARD----------------------------------------------#
    img = img.reshape(img_size,img_size)
    # CONVOLUTIONAL LAYER 1
    C1 = Convolution_2D(img, img_size, k_C1, b_C1, Act_Tab_Ker1, Min1)
    #print(np.max(C1))
    # POOLING (2,2) --> img = (12,12)
    P1 = POOLING(C1)
    p1 = P1.flatten() # (864,)
    # CONVOLUTIONAL LAYER 2 
    C2, S2 = Convolution_3D(P1,k_C2,b_C2, Act_Tab_Ker2, Min2)
    s2 = S2.flatten() # (768,)
    c2 = C2.flatten() # (768,)
    #print(np.max(C2))
    # POOLING (2,2) --> img = (4,4)
    P2 = POOLING(C2)
    #print(np.max(P2))
    # FCN --> f = (192), O = (10)
    f = P2.flatten()
    #print(np.max(f))

    return f, c2, s2, p1

@njit(parallel=True)
def feed(DATA,LABEL,k_C1,b_C1,k_C2,b_C2,Act_Tab_Ker1,Min1,Act_Tab_Ker2,Min2):#,Act_Tab_Ker3,Min3):
    
    Out_cnn  = np.zeros((len(LABEL),192))
    
    Pool1 = np.zeros((len(LABEL),864))   #  6*12x12
    Conv2 = np.zeros((len(LABEL),768))   # 12* 8x8
    S_conv2 = np.zeros((len(LABEL),768)) # 12* 8x8

    for IDX in prange(len(LABEL)):
       img   = DATA[IDX,:]
       label = LABEL[IDX]
       f, c2, s2, p1   =  feedforward(img,img_size,\
                            k_C1,b_C1,\
                            k_C2,b_C2,\
                            Act_Tab_Ker1,Min1,\
                            Act_Tab_Ker2,Min2)
                            
       Out_cnn[IDX,:] = f
       

       S_conv2[IDX,:] = s2
       Conv2[IDX,:]   = c2
       Pool1[IDX,:]   = p1

       if 0:
           losses = np.zeros((len(LABEL)), dtype=np.float64)
           predict = np.zeros((len(O)))
           predict[label] = 1
           losses[IDX]= np.sum((predict-O)**2)
    
    return Out_cnn, S_conv2, Conv2, Pool1


#####################################################################################
#                            K MEANS ALGORITHM
#####################################################################################

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


def kmeans(X, k, max_iters):
    centroids = X[np.random.choice(range(len(X)), k, replace=False)]
    converged = False    
    current_iter = 0
    while (not converged) and (current_iter < max_iters):
        cluster_list = [[] for i in range(len(centroids))]
        for i in range(len(X)):  # Go through each data point
            x = X[i]
            distances_list = np.zeros(k)
            for j in range(len(centroids)):
                c = centroids[j]
                distances_list[j] = distance(c, x)
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
        if current_iter==max_iters: print('Kmeans Alert: MAX ITER REACHED')

    return np.array(centroids), [np.std(x) for x in cluster_list], cluster_list


def kmeans_incr(X, k, centroids, max_iters):
    #centroids = X[np.random.choice(range(len(X)), k, replace=False)]
    converged = False    
    current_iter = 0
    while (not converged) and (current_iter < max_iters):
        cluster_list = [[] for i in range(len(centroids))]
        #print(len(cluster_list))
        for i in range(len(X)):  # Go through each data point
            x = X[i]
            distances_list = np.zeros(len(centroids))
            #print(len(distances_list))
            for j in range(len(centroids)):
                c = centroids[j]
                distances_list[j] = distance(c, x)
            # appendo x nella cluster list alla posizione argmin 
            #print(int(np.argmin(distances_list)))
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
        #print(len(cluster_list),len(centroids))
        pattern = np.abs(np.sum(prev_centroids) - np.sum(centroids))
        if False:
            print('K-MEANS iter',current_iter,' delta centroids', pattern)
        converged = (pattern == 0)
        current_iter += 1
        if current_iter==max_iters: print('Kmeans Alert: MAX ITER REACHED')

    return np.array(centroids), [np.std(x) for x in cluster_list], cluster_list

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

@njit(nogil=True)
def rbf(x, c, s):
    dist = distance(x, c)
    return 1 / np.exp(-dist / s ** 2)

def rbf_list(DATASET, centroids, std_list):
    RBF_list = []
    for x in DATASET:
        RBF_list.append([rbf(x, c, s) for (c, s) in zip(centroids, std_list)])
    return np.array(RBF_list)

def RBF_Network( TRAIN_SET, TRAIN_LABEL, num_of_classes, k,
                                    std_from_clusters=True): 
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


        if not std_from_clusters:
            dMax = np.max([distance(c1, c2) for c1 in centroids for c2 in centroids])
            std_list = np.repeat(dMax / np.sqrt(2 * k), k)


        ##########################################
        # RBF weights evaluation
        RBF_X = rbf_list(TRAIN_SET, centroids, std_list)

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



def RBF_Net_incr( TRAIN_SET, TRAIN_LABEL, num_of_classes, k, centroids, std_from_clusters=True):
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
        centroids_new, std_list, clust_list = kmeans_incr(TRAIN_SET, k, centroids, max_iters=1000)
        ##########################################
        #f = time.time() - i
        #print('Walltime k-mean: ',f)

        if not std_from_clusters:
            dMax = np.max([distance(c1, c2) for c1 in centroids_new for c2 in centroids_new])
            std_list = np.repeat(dMax / np.sqrt(2 * k), k)


        ##########################################
        # RBF weights evaluation
        RBF_X = rbf_list(TRAIN_SET, centroids_new, std_list)

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
##################################################################################
#                                IMPORTING DATA
##################################################################################
print('MNIST DIGIT RECOGNITION PROBLEM.')
print('Uploading data..')

train  = 1000
test   = 5000
##############
RESTART = True
##############
N_bit_weigh  = 2
N_bit_input  = 8
N_bit_output = 8

N_DATA    = 10000#int(train + test)
gDATA     = '/home/giacomo/Machine_Learning/Script/CNN/mnist_digit/'
cols      = np.array([ i for i in range(1,785) ])
full_DATASET   = np.loadtxt(gDATA+'mnist_train.csv', delimiter=',',\
                                      skiprows=1, max_rows= N_DATA,\
                                                     usecols = cols)
lab_cols = 0
full_DATASET_LABEL = np.loadtxt(gDATA+'mnist_train.csv', delimiter=',',\
                                          skiprows=1, max_rows= N_DATA,\
                                                     usecols = lab_cols)

DATASET_LABEL = full_DATASET_LABEL[1000:1000+train].astype(int)


DATASET = full_DATASET[1000:1000+train,:].astype(int)



DATASET_test = full_DATASET[1000+train:1000+train+test,:].astype(int)
DATASET_LABEL_test = full_DATASET_LABEL[1000+train:1000+train+test].astype(int)
print('Done!')
##################################################################################
#                           WEIGHTS INITIALIZATION
##################################################################################
if RESTART==True:
    print('... restarting from last config')
    gWEIGHTS = 'rbfcnn_weights/'
    # CONVOLUTIONAL LAYER 1
    kernels_C1 = np.round(np.loadtxt(gWEIGHTS+'kernC1.csv', delimiter=','))
    kernels_C1 = kernels_C1.reshape((3,5,5)).astype(int)
    bias_C1    = np.round(np.loadtxt(gWEIGHTS+'biasC1.csv', delimiter=','))
    bias_C1    = bias_C1.reshape((3,2)).astype(int)
    # CONVOLUTIONALE LAYER 2
    kernels_C2 = np.round(np.loadtxt(gWEIGHTS+'kernC2.csv', delimiter=','))
    kernels_C2 = kernels_C2.reshape(6,6,5,5).astype(int)
    bias_C2    = np.round(np.loadtxt(gWEIGHTS+'biasC2.csv', delimiter=','))
    bias_C2    = bias_C2.reshape((6,2)).astype(int)


    print(' Bias C1')
    print(bias_C1)
    print(' Bias C2')
    print(bias_C2)
    print(np.min(kernels_C1),np.max(kernels_C1))
    print(np.min(kernels_C2),np.max(kernels_C2))

    list_of_weights = [kernels_C1,bias_C1,kernels_C2,bias_C2]




##################################################################################
#                               MAIN
##################################################################################


img_size = 28

####################
RTS_iter = 50 

SAMPLE     =  50
batch_size = train
####################

####################
num_of_classes = 10
KMEAN = 100
####################

if True:
    sigm_tab1, minval1 = RELU_tab(N_bit_input,N_bit_weigh, 25,N_bit_input)
    sigm_tab2, minval2 = RELU_tab(N_bit_input,N_bit_weigh, 75,N_bit_input)
    out_cnn, S_conv2, Conv2, Pool1 = feed(DATASET,DATASET_LABEL,\
                          list_of_weights[0],list_of_weights[1],\
                          list_of_weights[2],list_of_weights[3],\
                           sigm_tab1, minval1,sigm_tab2, minval2) 

    Centroids,fin_weights,Clust_list,Std_list,_= RBF_Network(out_cnn,
                                DATASET_LABEL, num_of_classes, KMEAN,
                                              std_from_clusters=True)


    out_cnn_TEST,_,_,_ = feed(DATASET_test,\
                         DATASET_LABEL_test,\
                         list_of_weights[0],list_of_weights[1],\
                         list_of_weights[2],list_of_weights[3],\
                         sigm_tab1, minval1,sigm_tab2, minval2)
    # RBF FEED on test set
    RBF_list_test = rbf_list(out_cnn_TEST, Centroids, Std_list)
    # Linear combinations
    pred_ty = RBF_list_test @ fin_weights
    # Predictions
    pred_ty = np.array([np.argmax(x) for x in pred_ty])

    # ACCURACY
    diff = pred_ty - DATASET_LABEL_test
    print('Accuracy: ', len(np.where(diff == 0)[0]) / len(diff))
    Accuracy = len(np.where(diff == 0)[0]) / len(diff)
    


    ##########################################
    # INSIGHT ON CLUSTERING    
    if False:
        for n_centre in range(20):
            r = []
            theta = []
            color = []

            for img in range(len(out_cnn[:,0])):
                theta.append(2*np.pi*random.uniform(0, 1))
                r.append(distance(out_cnn[img],Centroids[n_centre]))
                color.append(round(DATASET_LABEL[img]/9,2))
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='polar')
            c = ax.scatter(theta, r, c=color, cmap='hsv', alpha=0.75)
            plt.title('DISTANCE FROM CENTRE {}'.format(str(n_centre)))
            plt.show()
    ##########################################



    if True:
        from collections import Counter
        ordered_centres = [[] for i in range(10)]

        N_centre = KMEAN
        for i in range(N_centre):
            RBF_list_test = rbf_list(Clust_list[i], Centroids, Std_list)

            pred = RBF_list_test @ fin_weights
            # Predictions
            pred = Counter([np.argmax(x) for x in pred])
            value, count = pred.most_common()[0]
            ordered_centres[value].append(Centroids[i])
            def distance(x1, x2):
                return np.sum((x1-x2)**2)**0.5


            ''' Confusion Matrix of distances '''
            CENTRES = [[] for i in range(N_centre)]
            j = 0
            print('Number of centres per digit:')
            if 1:
                for i in range(10):
                    print('Digit', i , 'centres:',len(ordered_centres[i]))
                    for ele in range(len(ordered_centres[i])):
                        CENTRES[j].append(ordered_centres[i][ele])
                        j += 1
    
                conf_matrix = np.zeros((N_centre,N_centre), dtype=float)
                for row in range(N_centre):
                    for col in range(N_centre):
                        conf_matrix[row,col]=distance(np.asarray(CENTRES[row]),
                                                np.asarray(CENTRES[col]))
                print('Min Max CONF MATRIX:',np.min(conf_matrix), np.max(conf_matrix))

                if 1:
                    plt.matshow(conf_matrix, cmap='PuBuGn')#, vmin=0., vmax=1000.)
                    plt.colorbar(label="Distance between centres")
                    import itertools
                    fmt = 'd'
                    l = 0
                    for i in range(10):    
                        for k, j in itertools.product(range(len(ordered_centres[i])),range(len(ordered_centres[i]))):
                                plt.text(l+j, l+k, format(i, fmt),
                             horizontalalignment="center")
                        l += len(ordered_centres[i])
                    plt.show()









