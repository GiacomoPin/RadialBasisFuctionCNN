from matplotlib import pyplot as plt
import numpy as np
import copy
import time
import random
from numba import jit,njit, prange, vectorize, float64


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
    img_size = 28
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


