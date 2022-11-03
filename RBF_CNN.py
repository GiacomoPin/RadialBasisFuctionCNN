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
            S1_nij = np.sum(region * kernel[n]) + bias[n,mba]*255
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
            S2_muv = np.sum(region * kernel[m]) + bias[m,mba]*255
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
        print(RBF_X.shape)
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

train  = 50000
test   = 50
##############
RESTART = False
Save    = False
##############
N_bit_weigh  = 2
N_bit_input  = 8
N_bit_output = 8

N_DATA    = 20000 #int(train + test)
gDATA     = '/home/giacomo/Machine_Learning/Script/CNN/mnist_digit/'
cols      = np.array([ i for i in range(1,785) ])
full_DATASET   = np.loadtxt(gDATA+'mnist_train.csv', delimiter=',',\
                                      skiprows=1, max_rows= N_DATA,\
                                                     usecols = cols)
lab_cols = 0
full_DATASET_LABEL = np.loadtxt(gDATA+'mnist_train.csv', delimiter=',',\
                                          skiprows=1, max_rows= N_DATA,\
                                                     usecols = lab_cols)

DATASET_LABEL = full_DATASET_LABEL[0:train].astype(int)


DATASET = full_DATASET[0:train,:].astype(int)



DATASET_test = full_DATASET[train:train+test,:].astype(int)
DATASET_LABEL_test = full_DATASET_LABEL[train:train+test].astype(int)
print('Done!')
##################################################################################
#                           WEIGHTS INITIALIZATION
##################################################################################
if RESTART==False:
    wi_min = -1
    wi_max =  2
    #np.random.seed(123)
    # CONVOLUTIONAL LAYER 1
    kernels_C1 = np.random.randint(wi_min,wi_max, size=(3,5,5))
    bias_C1    = np.random.randint(wi_min,wi_max, size=(3,2))
    print(np.max(kernels_C1))
    # CONVOLUTIONALE LAYER 2
    kernels_C2 =  np.random.randint(wi_min,wi_max, size=(6,6,5,5))
    bias_C2    =  np.random.randint(wi_min,wi_max, size=(6,2))


    print(np.min(kernels_C1),np.max(kernels_C1))
    print(np.min(kernels_C2),np.max(kernels_C2))

if RESTART==True:
    print('... restarting from last config')
    gWEIGHTS = 'rbfcnn_weights3/'
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






##################################################################################
#                               MAIN
##################################################################################


img_size = 28

####################
RTS_iter = 2

SAMPLE     =  1
batch_size = train
####################

####################
num_of_classes = 10
KMEAN = 50
####################

LOSS        = []
ACCURACY    = []
TABU_LENGTH = []
list_of_weights = [ kernels_C1, bias_C1, kernels_C2, bias_C2 ]
print(list_of_weights)
stop = False
p_tot = 993

p1 =   75/p_tot    # 0
p2 =    6/p_tot    # 1
p3 =  900/p_tot    # 2
p4 =   12/p_tot    # 3

probabilities = [p1,p2,p3,p4,p5,p6] 



if 0:
    # DO NOT REPORT THIS... COMPILATION TIME IS INCLUDED IN THE EXECUTION TIME!
    print('Preparing CNN compilation..')
    DATASET_BATCH = DATASET[0:batch_size,:]
    DATASET_LABEL_BATCH = DATASET_LABEL[0:batch_size]
    sigm_tab1, minval1 = RELU_tab(N_bit_input,N_bit_weigh, 25,N_bit_input)
    sigm_tab2, minval2 = RELU_tab(N_bit_input,N_bit_weigh, 75,N_bit_input) # 150
    

    _,_,_,_, = feed(DATASET_BATCH,\
                       DATASET_LABEL_BATCH,\
                       list_of_weights[0],list_of_weights[1],\
                       list_of_weights[2],list_of_weights[3],\
                       sigm_tab1, minval1,sigm_tab2, minval2)
    _,_,_,_, = feedforward(DATASET_BATCH[0,:],28,\
                       list_of_weights[0],list_of_weights[1],\
                       list_of_weights[2],list_of_weights[3],\
                       sigm_tab1, minval1,sigm_tab2, minval2)


    start  = time.perf_counter()
    _,_,_,_= feed(DATASET_BATCH,\
                  DATASET_LABEL_BATCH,\
                  list_of_weights[0],list_of_weights[1],\
                  list_of_weights[2],list_of_weights[3],\
                  sigm_tab1, minval1,sigm_tab2, minval2) 
    finish = time.perf_counter()
    print(' I iteration done {} seconds'.format(str(round(finish-start,3))))

    print('Done!')
    print('-----')
    print(' DISCRETE CNN: Reactive Tabu Search.')
    print(' Train :', train , 'Batch size:', batch_size)
    print(' Test  :', test  )
    print(' RTS iteration per batch:' , RTS_iter)
    print(' SAMPLE:', SAMPLE)


for batch in range(int(len(DATASET[:,0])/batch_size)):
    # PERMUTATION ON DATASET
    permutation   = np.random.permutation(len(DATASET[:,0]))
    DATASET       = DATASET[permutation,:]
    DATASET_LABEL = DATASET_LABEL[permutation]

    # BATCH
    DATASET_BATCH = DATASET[0:batch_size,:]
    DATASET_LABEL_BATCH = DATASET_LABEL[0:batch_size]
    #break
    table = HashTable()
    REP       = 3
    CHAOS     = 3
    CYCLE_MAX = 2*(SAMPLE-1)

    step_since_last_size_change = 0
    moving_avg = 1
    escape     = 0
    ESCAPE = False

    tabu_list   = []
    tabu_length = 1
    print(' ##################')
    print(' RTS Started. Training CNN on BATCH: ',batch)

    # RTS ITERATION on Batch
    start_rts  = time.perf_counter()
    for iteration in range(RTS_iter):   

        if iteration%10==0 and iteration>1: print(' Iter nr:', iteration, 'Loss:', np.round(loss_old,3))

            
        #1) FIRST ITERATION
        if iteration>0 and iteration%1==0:  
            out_cnn, S_conv2, Conv2, Pool1 = feed(DATASET_BATCH,\
                                            DATASET_LABEL_BATCH,\
                                            list_of_weights[0],list_of_weights[1],\
                                            list_of_weights[2],list_of_weights[3],\
                                            sigm_tab1, minval1,sigm_tab2, minval2)
            Centroids,fin_weights,Clust_list,Std_list,loss_old = RBF_Net_incr(out_cnn,
                                                      DATASET_LABEL_BATCH, num_of_classes, KMEAN,
                                                      Centroids,std_from_clusters=True)
        ##############################################################
        #                   FEEDFORWARD + LOSS     
        ##############################################################
            
        if iteration==0: 
            start  = time.perf_counter()
            out_cnn, S_conv2, Conv2, Pool1 = feed(DATASET_BATCH,\
                                                DATASET_LABEL_BATCH,\
                              list_of_weights[0],list_of_weights[1],\
                              list_of_weights[2],list_of_weights[3],\
                              sigm_tab1, minval1,sigm_tab2, minval2) 
            #@if RESTART==False:
            finish = time.perf_counter()
            print(' I iteration done {} seconds'.format(str(round(finish-start,3))))
            Centroids,fin_weights,_,Std_list,loss_old= RBF_Network(out_cnn,
                                     DATASET_LABEL_BATCH, num_of_classes,KMEAN,
                                                        std_from_clusters=True)
            '''if RESTART==True:
                Centroids = np.loadtxt(gWEIGHTS+'centres.csv', delimiter=',')
                Centroids = np.asarray(Centroids).reshape(47,192)
                Centroids,fin_weights,Clust_list,Std_list,loss_old = RBF_Net_incr(out_cnn,
                                                      DATASET_LABEL_BATCH, num_of_classes, 
                                                      KMEAN, Centroids,std_from_clusters=True)'''
                
            finish = time.perf_counter()
            print(' I iteration done {} seconds'.format(str(round(finish-start,3))))
        ##############################################################
        #LOSS.append(loss_old)
        print(' Initial loss: ',loss_old)
        print(' N centres: ',len(Centroids))
# ----   ----   ----   ----   ----  ----  ----  ----  ----  ----  ----  ----  ---
        #3) REACTIVE TABU SEARCH
        print(' Neighbour exploration RTS')
        # Search in the hash structure for the best choosen config.
        # For each best config save: 
        #                   -) cycle_lenght = current_time - last_time 
        #                   -) Number of times I already visited a config
        #                   -) Time of visit
            # 1 Escape reaction
            # 2 Diversification
            # 3 Intensitification

        ''' Creo le liste in cui inseriso loss, coord e le mosse, scegliero poi 
        l'indice che corrisponde al valore piu basso di loss e di conseguenza la 
        mossa e la configurazione'''
        List_Loss_neigh  = []
        List_coord_neigh = []
        List_moves_neigh = []
        List_centres     = []
        '''DEFINITION OF THE NEIGHBOURHOOD'''
        #SAMPLE = 2
        # Scelgo SAMPLE elementi che andranno cambiati (neighbourhood)
        p_chose = np.random.choice(4,SAMPLE,p=probabilities)
        Sampled_weights = []
        for ele in p_chose:
            coord = []
            for dim in range(len(list_of_weights[ele].shape)):
                samp = np.random.choice(np.size(list_of_weights[ele],dim))
                coord.append(samp)
            Sampled_weights.append(coord)
          
        start_expl  = time.perf_counter()
        ''' Comincio l'esplorazione dell'intorno'''
        for ele,coord in zip(p_chose,Sampled_weights):

                ''' Converting int to a 4bit number
                Make a copy of weights on which perform a trial move.'''
                WEIGHTS_TRIAL     =  copy.deepcopy(list_of_weights)
                w1eleold = copy.deepcopy(WEIGHTS_TRIAL[ele][tuple(coord)])
                LIST = [-1,0,1]
                
                bit = int(w1eleold+1)
                LIST.remove(w1eleold)
                bit_new = random.choice(LIST) + 1
                #np.random.randint(N_bit_weigh)                
                MOVE = [ele,coord,bit] # INVERSE MOVE THAT WILL BE TABU
                MOVE_new = [ele,coord,bit_new] # TRIAL MOVE
                   
                while np.any([tabu==(MOVE_new) for tabu in tabu_list]): 

                    coord = []
                    for dim in range(len(list_of_weights[ele].shape)):
                        samp = np.random.choice(np.size(list_of_weights[ele]\
                                                                           ,dim))
                        coord.append(samp)
                    w1eleold = copy.deepcopy(WEIGHTS_TRIAL[ele][tuple(coord)])
                    LIST = [-1,0,1]
                    bit  = int(w1eleold+1)
                    # np.random.randint(N_bit_weigh)
                    LIST.remove(w1eleold)
                    bit_new = random.choice(LIST) + 1
                    MOVE = [ele,coord,bit]
                    MOVE_new = [ele,coord,bit_new]
                    print(' Tabu move at ', iteration)
                        

                w1elenew = random.choice(LIST)#LIST[bit]
                    
                    
                WEIGHTS_TRIAL[ele][tuple(coord)] = w1elenew
                delta_weights  = (w1elenew-w1eleold)
                #print(w1eleold,bit,w1elenew)
#------------------------------------------------------------------------------
                #######  HERE CALCULATE COST FUNCTION, IF BETTER: UPDATE!
                #########################################################
                #                   FEEDFORWARD + LOSS     
                #########################################################
                if ele==0 or ele==1:
                    out_cnn, _, _, _ = feed(DATASET_BATCH,\
                                                DATASET_LABEL_BATCH,\
                                  WEIGHTS_TRIAL[0],WEIGHTS_TRIAL[1],\
                                  WEIGHTS_TRIAL[2],WEIGHTS_TRIAL[3],\
                                  sigm_tab1, minval1,sigm_tab2, minval2) 

                    ''' out_cnn --> (batch_size,192)'''
                
                    Centroids,fin_weights,Clust_list,Std_list,losses_new= RBF_Net_incr(out_cnn,
                                                      DATASET_LABEL_BATCH, num_of_classes, KMEAN,
                                                      Centroids,std_from_clusters=True)
                    

                ########################### Incremental calculus
                
                if ele==2 or ele==3:
                    Pool1_trialf   = copy.deepcopy(Pool1)
                    S_conv2_trialf = copy.deepcopy(S_conv2)
                    Conv2_trialf   = copy.deepcopy(Conv2)
                    delta_weights  = (w1elenew-w1eleold)#/127
                    out_cnn = np.zeros((len(DATASET_BATCH[:,0]), 192))
                    for img in range(len(DATASET_BATCH[:,0])):
                        for i in range(8):
                            for j in range(8):
                                S_conv2_trial = S_conv2_trialf[img].reshape(12,8,8)
                                Conv2_trial   = Conv2_trialf[img].reshape(12,8,8)
                                Pool1_trial   = Pool1_trialf[img].reshape(6,12,12)


                                if   ele==2:
                                    for depbia in range(2):
                                        S_conv2_trial[2*coord[0]+depbia,i,j] = S_conv2_trial[2*coord[0]+depbia,i,j] + \
                                               Pool1_trial[coord[1],coord[2]+i,coord[3]+j]*delta_weights
                                        Conv2_trial[2*coord[0]+depbia,i,j]   = activation_discrete(int(S_conv2_trial[2*coord[0]+depbia,i,j]),sigm_tab2, minval2)
                                elif ele==3:
                                    S_conv2_trial[2*coord[0]+coord[1],i,j] = S_conv2_trial[2*coord[0]+coord[1],i,j] + \
                                                                               delta_weights*255 ####################### BIAAAAAAAAAS
                                    Conv2_trial[2*coord[0]+coord[1],i,j]   = activation_discrete(int(S_conv2_trial[2*coord[0]+coord[1],i,j]),sigm_tab2, minval2)
                                #True if S_conv2_trial[coord[0],i,j]>0 else False

                        Pool2 = POOLING(Conv2_trial)
                        ftrial = Pool2.flatten()
                        out_cnn[img,:] = ftrial
                    Centroids,fin_weights,Clust_list,Std_list,losses_new= RBF_Net_incr(out_cnn,
                                                      DATASET_LABEL_BATCH, num_of_classes, KMEAN,
                                                      Centroids,std_from_clusters=True)
                    #print('feed',losses_new)'''
                ###########################


                # Cost function with the new configuration.
                cost_new  = losses_new

                #######################################
                List_moves_neigh.append([ele,coord,bit])
                List_Loss_neigh.append(cost_new)
                List_coord_neigh.append([ele,coord,w1elenew])
                List_centres.append(Centroids)
                #######################################
        


        finish_expl = time.perf_counter()
        print(' EXPLORATION done {} seconds'.format(str(round(finish_expl-start_expl,3))))
        # SELECT THE BEST MOVE and associated config, according to lowest cost
        if ESCAPE == False : # Scegli la mossa migliore nell'intorno
            Loss   = np.asarray(List_Loss_neigh) 
            
            if len(Loss)==0: 
                print(' ERROR during Loss evalutation. Stopping CNN train.', Loss)  
                #stop = True
                continue
            index  = np.random.choice(np.asarray\
                                        (np.where(Loss == Loss.min()))[0])
            
            best_move   = List_moves_neigh[index]
            loss_old    = List_Loss_neigh[index]
            Centroids   = List_centres[index]
            print(' loss:',loss_old)
            coord_best  = copy.deepcopy([List_coord_neigh[index]])
            if 0:
                print(coord_best)
                print(coord_best[0][0])
                print(coord_best[0][1])
                print(coord_best[0][2])

            
            WEIGHTS_best = copy.deepcopy(list_of_weights)
            WEIGHTS_best[coord_best[0][0]][tuple(coord_best[0][1])] = coord_best[0][2]

        ''' Rimane da definire il meccanismo di escape !!!'''
        #1) ESCAPE MECHANISM ---> random walk through the config space
        '''Quando innesco il meccanismo di escape azzero la tavola hash e pulisco
        la memoria, quindi inizio un ciclo di steps mosse randomiche per uscire
        da una zona di minimo in cui ero bloccato'''
        if ESCAPE == True :
            table = HashTable()
            rand  = random.uniform(0,1) 
            steps = 1 + int((1+rand)*moving_avg/2)
            print(' Performing', steps,'rnd steps')
            if 0:
                print('performing rnd move')
                print(iteration+1)
            for escape_iter in range(steps):
                
                list_of_weights,tabu_list = Escape_mechanism(list_of_weights,
                                        probabilities,N_bit_weigh,tabu_list)

                #######  HERE CALCULATE COST FUNCTION
                loss_old,_,_,_,_,_ = feed(DATASET_BATCH,DATASET_LABEL_BATCH,\
                            list_of_weights[0],list_of_weights[1],\
                            list_of_weights[2],list_of_weights[3],\
                            list_of_weights[4],list_of_weights[5],\
                            sigm_tab1, minval1,sigm_tab2, minval2,sigm_tab3, minval3)


                table[loss_old] = [loss_old,iteration] #[WEIGHTS,iteration]
                #LOSS.append(loss_old)
                
            ESCAPE = False
            continue


        # New better configuration ---
        list_of_weights   = copy.deepcopy(WEIGHTS_best)
        step_since_last_size_change = step_since_last_size_change + 1

        ############################
        getitem = loss_old #[loss_old,WEIGHTS]
        ############################
        '''if iteration>1:
            comparison = loss_old==LOSS
            if comparison.any():    
                print(table[getitem])
                print(WEIGHTS)'''
        # HASH TABLE note:
        # __getitem__ requires a 2 value array as entry:
        # Example:       table[[loss=key,config=weights]]
        # Then will provide 3 output

        if table[getitem] != None:   
            ''' Aggiorno la memoria su una configurazione già visitata.
            La lunghezza del ciclo viene definita sottraendo il tempo attuale
            iteration al tempo in cui ho visitato in precedenza la
            configurazione.
            Poi aggiorno il numero di volte che ho visitato la configurazione
            includendo quella attuale.'''
            if 0:
                print('-----------------')
                print(' Already visited ')
                print(table[getitem][0]) # Nr of repetitions
                print(iteration)         # current.time
                print(table[getitem][1]) # length of cycle
            cycle_length    = iteration-copy.deepcopy(table[getitem][1])
            table[loss_old] = [loss_old,iteration]  
            repetitions     = copy.deepcopy(table[getitem][2])
            
            #1) INNESCO ESCAPE PER LA PROSSIMA ITERAZIONE
            if repetitions > REP:
                escape = escape + 1                
                if 0:
                    print('Configuration: {}'.format(str(table[getitem][0])))
                    print('Visited {} times '.format(str(table[getitem][2])))

                if escape > CHAOS :
                    '''Deciding whether activate or not the ESCAPE REACTION in
                    the next iteration.'''
                    ESCAPE = True
                    escape = 0      
                    if 1:   
                        print('')
                        print('ESCAPE Reaction at {}'.format(str(iteration)))
           
            #2) DIVERSIFICATION
            if cycle_length < CYCLE_MAX:
                '''Il meccanismo di diversificazione allunga la lunghezza della 
                lista tabu, permettendo di uscire da eventuali cicli e spostarsi
                in altre zone dello spazio config '''
                #print('DIVERSIFICATION', iteration)
                moving_avg = cycle_length*0.1 + moving_avg*0.9              
                tabu_length = copy.deepcopy(tabu_length) + 1
                step_since_last_size_change = 0

        else: # INSTALL THE CONFIGURATION
            '''La configurazione non risulta essere stata visitata in passato.'''
            table[loss_old] = [loss_old,iteration]
            ESCAPE = False

        #3) INTENSIFICATION
        if step_since_last_size_change > moving_avg:
            ''' In presenza di nuove configurazioni per moving_avg iterazioni, 
            la ricerca viene intensificata in una zona dello spazio config
            accorciando la lista di mosse tabu.'''
            tabu_length = max(1,copy.deepcopy(tabu_length) - 1)
            step_since_last_size_change = 0



        # AGGIORNAMENTO DELLA LISTA TABU
        ''' Inserisco la mossa applicata nella lista delle mosse tabu'''
        tabu_list.append(best_move)
        while len(tabu_list)>tabu_length: #while
           tabu_list.pop(0)
        
        # Setting the newconfig in the HASH table       
        # HASH SCHEME __getitem__ --> table[[loss,weights]][0]-> [weights matrix]
        #             __getitem__ --> table[[loss,weights]][1]-> time
        #             __getitem__ --> table[[loss,weights]][2]-> repetitions


        LOSS.append(loss_old)
        TABU_LENGTH.append(len(tabu_list))

    
    #################################################################################      

        ''' TEST ACCURACY '''
        if iteration%10==0 or iteration==RTS_iter-1:
                out_cnn, S_conv2, Conv2, Pool1 = feed(DATASET_BATCH,\
                                                DATASET_LABEL_BATCH,\
                              list_of_weights[0],list_of_weights[1],\
                              list_of_weights[2],list_of_weights[3],\
                              sigm_tab1, minval1,sigm_tab2, minval2) 
                
                Centroids,fin_weights,Clust_list,Std_list,_= RBF_Net_incr(out_cnn,
                                                      DATASET_LABEL_BATCH, num_of_classes, KMEAN,
                                                      Centroids,std_from_clusters=True)

                
                out_cnn_TEST,_,_,_ = feed(DATASET_test,\
                         DATASET_LABEL_test,\
                         list_of_weights[0],list_of_weights[1],\
                         list_of_weights[2],list_of_weights[3],\
                         sigm_tab1, minval1,sigm_tab2, minval2)

                '''
                #####################################################################
                cluster_list = [[] for i in range(len(Centroids))]
                #print(len(cluster_list))
                for i in range(len(out_cnn)):  # Go through each data point
                    x = out_cnn[i]
                    distances_list = np.zeros(len(Centroids))
                    for j in range(len(Centroids)):
                        c = Centroids[j]
                        distances_list[j] = distance(c, x)
                    # appendo x nella cluster list alla posizione argmin 
                    cluster_list[int(np.argmin(distances_list))].append(x)  
                Std_list = [np.std(x) for x in cluster_list]
                #####################################################################
                '''

                # RBF FEED on test set
                RBF_list_test = rbf_list(out_cnn_TEST, Centroids, Std_list)
    
                # Linear combinations
                pred_ty = RBF_list_test @ fin_weights

                # Predictions
                pred_ty = np.array([np.argmax(x) for x in pred_ty])

                # ACCURACY
                print(pred_ty[0:10],DATASET_LABEL_test[0:10])
                diff = pred_ty - DATASET_LABEL_test
                print('Accuracy: ', len(np.where(diff == 0)[0]) / len(diff))
                Accuracy = len(np.where(diff == 0)[0]) / len(diff)

                from collections import Counter
                ordered_centres = [[] for i in range(10)]

                N_centre = len(Centroids)
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
                for i in range(10):
                    #print('Digit', i , 'centres:',len(ordered_centres[i]))
                    for ele in range(len(ordered_centres[i])):
                        CENTRES[j].append(ordered_centres[i][ele])
                        j += 1
    
                conf_matrix = np.zeros((N_centre,N_centre), dtype=float)
                for row in range(N_centre):
                    for col in range(N_centre):
                        conf_matrix[row,col]=distance(np.asarray(CENTRES[row]),
                                                np.asarray(CENTRES[col]))
                print('Min Max CONF MATRIX:',np.min(conf_matrix), np.max(conf_matrix))

                if 0:
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
                #for i in range(5):



                if Save==True:
                    plt.imshow(conf_matrix, cmap='PuBuGn', vmin=0., vmax=220.)
                    plt.colorbar(label="Distance between centres")
                    plt.title('Iter {}, ACC = {}'.format(str(iteration),str(Accuracy)))
                    plt.savefig('conf_matrix3/{}'.format(str(iteration)))
                    plt.close()
                    print('Saving weights')
                    gW = 'rbfcnn_weights3/'
                    kernC1 = list_of_weights[0]
                    biasC1 = list_of_weights[1]
                    kernC2 = list_of_weights[2]
                    biasC2 = list_of_weights[3]
                    np.savetxt(gW+'centres.csv', Centroids, delimiter=',')
                    np.savetxt(gW+'kernC1.csv' , kernC1.reshape(kernC1.shape[0],-1),delimiter=',')
                    np.savetxt(gW+'biasC1.csv' , biasC1.reshape(biasC1.shape[0],-1),delimiter=',')
                    np.savetxt(gW+'kernC2.csv' , kernC2.reshape(kernC2.shape[0],-1),delimiter=',')
                    np.savetxt(gW+'biasC2.csv' , biasC2.reshape(biasC2.shape[0],-1),delimiter=',')

