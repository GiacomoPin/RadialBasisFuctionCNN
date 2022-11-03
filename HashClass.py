''' Hash class for neural network
INPUT: matrix of weights
Hash key: value of loss function

'''
import numpy as np
class HashTable:
    def __init__(self):
        self.bit  = '111111111111111'
        self.MAX  = int(self.bit,2) + 1
        self.nbit = len(self.bit)
        self.arr  = [[] for i in range(self.MAX)]



    def DecTonBit(self,number,nbit):
        binary = bin(number).replace('0b','') #string
        x = binary[::-1]
        while len(x)<nbit:
            x+='0'
        binary = x[::-1]
        return binary

    def get_hash(self,loss):
        bit_out = self.nbit
        chiave   = int(loss * 3911289)
        x = self.DecTonBit(chiave,40)[::-1]
        binary   = x[0:bit_out][::-1]
        dec_hash = int(binary,2) 
        return dec_hash

    def __setitem__(self,key,val):
        h = self.get_hash(key)
        found = False
        for idx, element in enumerate(self.arr[h]):
            if len(element[1])==2 and element[0]==key:
                repetitions = element[2]+1
                self.arr[h][idx] = (key,val,repetitions)
                found = True
        if not found:
            repetitions = 1
            self.arr[h].append((key,val,repetitions))

    def __getitem__(self, key):
        h = self.get_hash(key)
        for idx, element in enumerate(self.arr[h]):
            if element[0] == key:
                return element[1][0],element[1][1],element[2]

    def __delitem__(self,key):
        h = self.get_hash(key,self.nbit)
        self.arr[h] = None
