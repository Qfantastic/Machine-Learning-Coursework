# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 15:18:51 2017

@author: Sreenivas
"""
import numpy as np
import pandas as pd
import sys
import math
import random
#class for layer
class Layer:
    def __init__(self):
        self.layerid=None
        self.nextlayer=None
        self.backlayer=None
        self.neuron=[]
        self.neuronvalues=[]
        self.outlayer=False
        self.inplayer=False
        self.layererror=[]
#class for neuron
class Neuron:
    def __init__(self):
        self.nodeid=None
        self.layerid=None
        self.weights=[]
        self.bias=False
        self.outneuron=False
#dot product of two list
def dot(X,W):
    dotval=[]
    if(len(X)!=len(W)):
        return 0
    dotval=[valX*valY for valX,valY in zip(X,W)]
    return sum(dotval)
#Activation function
def activation(net):
    return 1/(1+math.exp(-net))
#Softmax function
def softmax(net):
    return math.exp(net)
def forwardpass(Xdata,layer):
    #input layer initializing values to input neuron
    layer.neuronvalues=Xdata.values.tolist()
    layer.neuronvalues.insert(0,1)
    previouslayer=layer
    layer=layer.nextlayer
    while(layer!=None):
        layer.neuronvalues=[]
        for neuron in layer.neuron:
            if(neuron.bias==True):
                layer.neuronvalues.insert(0,1)
            else:
                net=dot(previouslayer.neuronvalues,neuron.weights)
                if(neuron.outneuron==True):
                    layer.neuronvalues.append(softmax(net))
                else:
                    layer.neuronvalues.append(activation(net))
        previouslayer=layer
        layer=layer.nextlayer
    outputlayer=previouslayer
    #Normalizing softmax output
    outputsum=sum(outputlayer.neuronvalues)
    outputlayer.neuronvalues=[value/outputsum for value in outputlayer.neuronvalues]
#Outputdelta
def outdelta(labelclass,layer):
    layer.layererror=[outval*(1-outval)*(targval-outval) for outval,targval in zip(layer.neuronvalues,labelclass)]
#calculateing product of delta and weight for delta calculation of hidden units
def weighterrorprod(previouslayer,nodeid):
    if(previouslayer.outlayer==True):
        neurons=previouslayer.neuron
    else:
        neurons=previouslayer.neuron[1:len(previouslayer.neuron)]
    weightsum=0
    for neuron in neurons:
        if(previouslayer.outlayer==True):
            weightsum=weightsum+((neuron.weights[nodeid])*(previouslayer.layererror[neuron.nodeid]))
        else:
            weightsum=weightsum+((neuron.weights[nodeid])*(previouslayer.layererror[neuron.nodeid-1]))
    return weightsum
#computing updated weights
def weightchange(outneuron,errorvalue,layer):
    learnrate=0.1
    for neuron in layer.neuron:
        outneuron.weights[neuron.nodeid]+=learnrate*errorvalue*layer.neuronvalues[neuron.nodeid]
#updating weights
def weightupdate(layer,previouslayer):
    errorvalue=previouslayer.layererror
    if(previouslayer.outlayer==True):
        neurons=previouslayer.neuron
    else:
        neurons=previouslayer.neuron[1:len(previouslayer.neuron)]
    for neuron in neurons:
        if(previouslayer.outlayer==True):
            errorval=errorvalue[neuron.nodeid]
        else:
            errorval=errorvalue[neuron.nodeid-1]
        weightchange(neuron,errorval,layer)
#hiddendelta
def hiddendelta(layer,previouslayer):
    #taking neuron excluding bias
    neurons=layer.neuron[1:len(layer.neuron)]
    layer.layererror=[]
    for neuron in neurons:
        neuronvalue=layer.neuronvalues[neuron.nodeid]
        sumweight=weighterrorprod(previouslayer,neuron.nodeid)
        layer.layererror.append(neuronvalue*(1-neuronvalue)*sumweight)
#Backward pass
def backwardpass(labelclass,outputlayer):
    outdelta(labelclass,outputlayer)
    previouslayer=outputlayer
    layer=outputlayer.backlayer
    while(layer.inplayer!=True):
        hiddendelta(layer,previouslayer)
        weightupdate(layer,previouslayer)
        previouslayer=layer
        layer=layer.backlayer
    #calculating weight updates coming from input layer
    weightupdate(layer,previouslayer)
#Backpropogation
def backpropogation(X,Y,inputlayer,outputlayer,iterations,labelclass):
    while(iterations>0):
        for i in range(0,X.shape[0]):
            forwardpass(X.iloc[i,:],inputlayer)
            labelval=labelclass.iloc[:,Y.iloc[i]].tolist()
            backwardpass(labelval,outputlayer)
        iterations=iterations-1
#print neuron weights
def printweights(count,layer):
    for i in range(0,count):
        print('Neuron',i,'weights:',' ')
        for neuron in layer.neuron:
            weight=neuron.weights[i]
            print(weight,end=' ')
        print('\n')
#Printing parameters
def printneural(layer):
    while(layer.outlayer!=True):
        count=len(layer.neuron)
        print('Layer',layer.layerid,':'+'\n')
        printweights(count,layer.nextlayer)
        layer=layer.nextlayer
#prediction
def prediction(X,Y,inputlayer,outputlayer):
    trainingerror=0
    for i in range(0,X.shape[0]):
        forwardpass(X.iloc[i,:],inputlayer)
        pred=np.argmax(outputlayer.neuronvalues)
        if(Y.iloc[i]!=pred):
            trainingerror+=1
    return trainingerror
#Main function
if __name__=="__main__":
    datapath=sys.argv[1]
    trainperc=sys.argv[2]
    maxiter=sys.argv[3]
    nhidden=sys.argv[4]
    #taking count of no arguments passed after the above four
    hiddenneuron=int(nhidden)
    #Taken first argument so count starts from five
    argno=5
    hiddenneurons=[]
    for i in range(0,hiddenneuron):
        hiddenneurons.append(int(sys.argv[argno]))
        argno+=1
    data=pd.read_csv(datapath,header=None)
    #plitting dataset into trainng set and test set and avoid random state after 
    traindata=data.sample(frac=0.8,random_state=200)
    testdata=data.sample(frac=0.2,random_state=200)
    #storing no of units in each layer in network
    Xtrain=traindata.iloc[:,:-1]
    Ytrain=traindata.iloc[:,-1]
    Xtest=testdata.iloc[:,:-1]
    Ytest=testdata.iloc[:,-1]
    #Creating the initial layer and and its units
    inputlayer=Layer()
    inputlayer.layerid=0
    inputlayer.inplayer=True
    #counting the number of features in data
    nfeature=len(Xtrain.iloc[:,:].columns)
    inputnodecounter=0
    #creating input units including bias so feature+1
    for i in range(0,nfeature+1):
        node=Neuron()
        node.layerid=0
        node.nodeid=inputnodecounter
        inputlayer.neuron.append(node)
        inputnodecounter+=1
        if(node.nodeid==0):
            node.bias=True
    #setting hidden layers and units
    hiddenlayerid=0
    #setting number of input weights coming from previous layer
    incomeweights=nfeature+1
    previouslayer=inputlayer
    for i in range(0,hiddenneuron):
        hiddenlayerid+=1
        hiddenlayer=Layer()
        previouslayer.nextlayer=hiddenlayer
        hiddenlayer.backlayer=previouslayer
        previouslayer=hiddenlayer
        hiddenlayer.layerid=hiddenlayerid
        hiddennodecounter=0
        #Creating no of input units in each layer 
        for j in range(0,hiddenneurons[i]+1):
            hiddennode=Neuron();
            hiddennode.layerid=hiddenlayerid
            hiddennode.nodeid=hiddennodecounter
            if(node.nodeid==0):
                hiddennode.bias=True
            else:
                hiddennode.weights=np.random.uniform(0,1,incomeweights).tolist()
            hiddenlayer.neuron.append(hiddennode)
            hiddennodecounter+=1
        incomeweights=hiddenneurons[i]+1
    #setting output layer and output unit
    outputlabelcount=len(np.unique(Ytrain))
    #setting dataframe  for ouput labe class in 1 ... 0 format
    labelarray=np.identity(outputlabelcount)
    labelarray.tolist()
    labelclass=pd.DataFrame(labelarray)
    outputlayer=Layer()
    outputlayer.outlayer=True
    previouslayer.nextlayer=outputlayer
    outputlayer.backlayer=previouslayer
    outputlayer.layerid=hiddenlayerid+1
    outnodecount=0
    for i in range(0,outputlabelcount):
        outputnode=Neuron()
        outputlayer.neuron.append(outputnode)
        outputnode.nodeid=outnodecount
        outputnode.outneuron=True
        outnodecount+=1
        outputnode.weights=np.random.uniform(0,1,incomeweights).tolist()
    #calling backpropogation algorithm
    
    backpropogation(Xtrain,Ytrain,inputlayer,outputlayer,int(maxiter),labelclass)
    layer=inputlayer
    trainingerror=prediction(Xtrain,Ytrain,inputlayer,outputlayer)
    printneural(inputlayer)
    print('\n')
    print('Total training error',trainingerror)
    testerror=prediction(Xtest,Ytest,inputlayer,outputlayer)
    print('\n')
    print('Total Testing error',testerror)

    
    
    
