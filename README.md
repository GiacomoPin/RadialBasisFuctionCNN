# RadialBasisFuctionCNN

Initialization:  k-means to initialize cluster centres, rnd initialization of weights (output layer)

Dynamic Input Features: classical RBF has fixed input features, this assumption is not valid w.r.t CNN.
                        An optimization of k-means algorithm is needed.

Activation: new activation due to inefficient gradient flow.

# Model
![RBFCNN](https://user-images.githubusercontent.com/83760901/199760756-f81c6695-c244-42e6-a179-b5c78e77d3f8.png)


# Training
![training](https://user-images.githubusercontent.com/83760901/199762100-3510b242-2084-45b2-ac38-8cabd6fb767a.png)


# Confusion Matrix
![Screenshot from 2022-11-03 16-20-09](https://user-images.githubusercontent.com/83760901/199761520-f24e4b9f-c080-4249-8cc8-be91d6bad1e3.png)


# Parameters Setting

![elbowmet](https://user-images.githubusercontent.com/83760901/199762629-05f3c419-e7a0-4cd1-9655-8a34761cc91b.png)

![PcaKmeans](https://user-images.githubusercontent.com/83760901/199762705-9aa9a976-cc12-4dd7-932b-f57a55aac4ac.png)
