from sklearn.decomposition import PCA# Make an instance of the Model
from matplotlib import pyplot as plt
import numpy as np
import copy
import time
import random

print('MNIST DIGIT RECOGNITION PROBLEM.')
print('Uploading data..')

train  = 1000 # must 30000
test   = 1000 # must 30000


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
plt.imshow(test_img[0,:].reshape(28,28))
plt.show()
npca = 300
pca = PCA(n_components = npca)
pca.fit(train_img)
train_img = pca.transform(train_img)
test_img = pca.transform(test_img)
if True:
    STD = []
    for i in range(npca):
        std = np.std(train_img[:,i])
        STD.append(std)
    plt.plot(STD)
    plt.title('Variance in principal components')
    plt.xlabel('PComponent')
    plt.show()
