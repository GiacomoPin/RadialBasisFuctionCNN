from matplotlib import pyplot as plt
import numpy as np
import copy
import time
import random
from numba import jit,njit, prange, vectorize, float64


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
def feedforward(img,img_size,k_C1,b_C1,k_C2,b_C2,w_FCN,b_FCN,Act_Tab_Ker1,Min1,Act_Tab_Ker2,Min2,Act_Tab_Ker3,Min3):
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
    O, o = FCN(f,w_FCN,b_FCN, Act_Tab_Ker3, Min3)
    #print(np.sum(O))
    return O, o, f, c2, s2, p1

@njit(parallel=True)
def feed(DATA,LABEL,k_C1,b_C1,k_C2,b_C2,w_FCN,b_FCN,Act_Tab_Ker1,Min1,Act_Tab_Ker2,Min2,Act_Tab_Ker3,Min3):
    losses = np.zeros((len(LABEL)), dtype=np.float64)
    Input_fcn  = np.zeros((len(LABEL),192))
    Out_fcn    = np.zeros((len(LABEL),10))

    Pool1 = np.zeros((len(LABEL),864))   #  6*12x12
    Conv2 = np.zeros((len(LABEL),768))   # 12* 8x8
    S_conv2 = np.zeros((len(LABEL),768)) # 12* 8x8

    for IDX in prange(len(LABEL)):
       img   = DATA[IDX,:]
       label = LABEL[IDX]
       O, o, f, c2, s2, p1   =  feedforward(img,img_size,\
                            k_C1,b_C1,\
                            k_C2,b_C2,\
                            w_FCN,b_FCN,\
                            Act_Tab_Ker1,Min1,\
                            Act_Tab_Ker2,Min2,\
                            Act_Tab_Ker3,Min3)
       Input_fcn[IDX,:]= f
       Out_fcn[IDX,:]  = o

       S_conv2[IDX,:] = s2
       Conv2[IDX,:]   = c2
       Pool1[IDX,:]= p1

       predict = np.zeros((len(O)))
       predict[label] = 1
       losses[IDX]= np.sum((predict-O)**2)
    
    return np.sqrt(np.mean(losses)), Input_fcn, Out_fcn, S_conv2, Conv2, Pool1





##################################################################################
##################################################################################

print('MNIST DIGIT RECOGNITION PROBLEM.')
print('Uploading data..')

train  = 30000
test   = 30000


N_DATA    = int(train + test)
gDATA     = '/home/giacomo/Machine_Learning/Script/CNN/mnist_digit/'
cols      = np.array([ i for i in range(1,785) ])
full_DATASET   = np.loadtxt(gDATA+'mnist_train.csv', delimiter=',',\
                                      skiprows=1, max_rows= N_DATA,\
                                                     usecols = cols)
lab_cols = 0
full_DATASET_LABEL = np.loadtxt(gDATA+'mnist_train.csv', delimiter=',',\
                                          skiprows=1, max_rows= N_DATA,\
                                                     usecols = lab_cols)

train_img = full_DATASET[0:0+train,:]
train_label = full_DATASET_LABEL[0:0+train].astype(int)


test_img = full_DATASET[train:train+test,:]
test_label = full_DATASET_LABEL[train:train+test].astype(int)

print('Uploaded')

CNN = True

####################################################################################
if CNN==True:
    N_bit_weigh  = 2
    N_bit_input  = 8
    N_bit_output = 8
    print('... restarting from last config')
    gWEIGHTS = '/home/giacomo/Desktop/PerformanceCNN/weights_MBA_RELU/'
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

    # FULLY CONNNECTED NETWORK
    weights_FCN = np.round(np.loadtxt(gWEIGHTS+'kernFCN.csv', delimiter=','))
    weights_FCN = weights_FCN.reshape(10,192).astype(int)
    bias_FCN    = np.round(np.loadtxt(gWEIGHTS+'biasFCN.csv', delimiter=',')).astype(int)
    bias_FCN    = bias_FCN.reshape(10).astype(int)
    print(' Bias C1')
    print(bias_C1)
    print(' Bias C2')
    print(bias_C2)
    print('min - max weights')
    print(np.min(kernels_C1),np.max(kernels_C1))
    print(np.min(kernels_C2),np.max(kernels_C2))
    print(np.min(weights_FCN),np.max(weights_FCN))

    # DO NOT REPORT THIS... COMPILATION TIME IS INCLUDED IN THE EXECUTION TIME!
    print('Preparing CNN compilation..')
    list_of_weights = [kernels_C1,bias_C1,kernels_C2,bias_C2,weights_FCN,bias_FCN]
    img_size=28
    sigm_tab1, minval1 = RELU_tab(N_bit_input,N_bit_weigh, 25,N_bit_input)
    sigm_tab2, minval2 = RELU_tab(N_bit_input,N_bit_weigh, 75,N_bit_input) # 150
    sigm_tab3, minval3 = Sigmoid_tab(N_bit_input,N_bit_weigh,192,N_bit_input) # 192

    _,_,_,_,_,_ = feed(train_img,\
                       train_label,\
                       list_of_weights[0],list_of_weights[1],\
                       list_of_weights[2],list_of_weights[3],\
                       list_of_weights[4],list_of_weights[5],\
                       sigm_tab1, minval1,sigm_tab2, minval2,sigm_tab3, minval3)
    _,_,_,_,_,_ = feedforward(train_img[0,:],28,\
                       list_of_weights[0],list_of_weights[1],\
                       list_of_weights[2],list_of_weights[3],\
                       list_of_weights[4],list_of_weights[5],\
                       sigm_tab1, minval1,sigm_tab2, minval2,sigm_tab3, minval3)
    print(' Done!')
    print(' -----')
    print(' DISCRETE CNN feature extraction')
    
####################################################################################


    _, train_img,_,_,_,_ = feed(train_img,\
                       train_label,\
                       list_of_weights[0],list_of_weights[1],\
                       list_of_weights[2],list_of_weights[3],\
                       list_of_weights[4],list_of_weights[5],\
                       sigm_tab1, minval1,sigm_tab2, minval2,sigm_tab3, minval3)
    _, test_img,_,_,_,_ = feed(test_img,\
                       test_label,\
                       list_of_weights[0],list_of_weights[1],\
                       list_of_weights[2],list_of_weights[3],\
                       list_of_weights[4],list_of_weights[5],\
                       sigm_tab1, minval1,sigm_tab2, minval2,sigm_tab3, minval3)
    print(np.size(train_img))
    print('Saving Extracted Input')
    gW = '/home/giacomo/Machine_Learning/Script/CNN/RBF_CNN/Input_postCNN/'

    np.savetxt(gW+'train_img.csv', train_img.reshape(train,192), delimiter=',')

    np.savetxt(gW+'test_img.csv', test_img.reshape(test,192), delimiter=',')
    

