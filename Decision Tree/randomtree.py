# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 19:09:50 2017

@author: Sreenivas
"""

import numpy as np
import pandas as pd
import sys,math
import random

class Node:
    def __init__(self):
        self.left=None
        self.right=None 
        self.leaf=False
        self.name=None
        self.value=None
        self.splitattribute=None
        self.depth=None
        self.leafvalue=None
        self.nodeid=None
        self.parent=None
        self.subpositiveleaf=0
        self.subnegativeleaf=0
    def nodeval(self,vid):
        self.nodeid=vid
#Class to create tree in decision tree
class Tree:
    def __init__(self):
        self.root=None
        self.totalnodes=0
        self.leafnodes=0
        self.depth=0
        self.sumdepth=0
    def insertroot(self):
        if self.root is None:
            newnode=Node()
            self.root=newnode
            self.totalnodes=self.totalnodes+1
            newnode.nodeval(1)
            return newnode
    def insert(self,parent,pos):
            newnode=Node()
            self.totalnodes=self.totalnodes+1
            if(pos=='left'):
                nodval=parent.nodeid*2
                newnode.nodeval(nodval)
            else:
                nodval=(parent.nodeid*2)+1
                newnode.nodeval(nodval)
            return newnode
        #function to calculate entropy
def entropy(classlabel):
    entr=0
    value,freq=np.unique(classlabel,return_counts=True)
    classprob=freq.astype(float)/len(classlabel)
    for val in classprob:
        if val!=0.0:
            entr-=val*np.log2(val)
    return entr
#Function to calculate information gain
def infogain(X,Y):
    gain=entropy(Y)
    value,freq=np.unique(X,return_counts=True)
    xprob=freq.astype(float)/len(X)
    for val,prob in zip(value,xprob):
        gain-=prob*entropy(Y[X==val])
    return gain
#Checking if output label is pure
def ispure(Y):
    boolval=len(np.unique(Y))==1
    return boolval
def id3decisiontree(X,Y,node,tree):
    if((ispure(Y)==False) and (not(X.empty))):
        '''Finding best attribute'''
        label=bestattribute(X,Y) 
        node.splitattribute=label
        '''Creating left and right child of a node'''
        leftchild=tree.insert(node,'left')
        rightchild=tree.insert(node,'right')
        node.left=leftchild
        node.right=rightchild
        leftchild.parent=node
        rightchild.parent=node
        leftchild.name=label
        rightchild.name=label
        leftchild.depth=node.depth+1
        rightchild.depth=node.depth+1
        leftchild.value=0
        rightchild.value=1
        '''Splitting dataframe based on splitting attribute'''
        Xleft=X[X[label]==0]
        Yleft=Y[X[label]==0]
        Xleft=Xleft.drop(label,1)
        Xright=X[X[label]==1]
        Yright=Y[X[label]==1]
        Xright=Xright.drop(label,1)
        '''Calling decision tree recursively'''
        id3decisiontree(Xleft,Yleft,leftchild,tree)
        id3decisiontree(Xright,Yright,rightchild,tree)
    else:
        node.leaf=True
        tree.sumdepth=tree.sumdepth+node.depth
        tree.leafnodes=tree.leafnodes+1
        if((ispure(Y)==True) and (not(Y.empty))):
            node.leafvalue=int(np.unique(Y))
        elif(Y.empty):
            node.leafvalue=random.choice([0,1])
        elif(not(ispure(Y))):
            node.leafvalue=Y.value_counts().index[0]
def bestattribute(X,Y):
    label=None
    maxgain=0
    for column in X:
        gain=infogain(X[column],Y)
        if(gain>=maxgain):
            label=column
            maxgain=gain
    return label
#Random decision tree
def randdecisiontree(X,Y,node,tree):
    if((ispure(Y)==False) and (not(X.empty))):
        '''Finding best attribute'''
        label=randattribute(X) 
        node.splitattribute=label
        '''Creating left and right child of a node'''
        leftchild=tree.insert(node,'left')
        rightchild=tree.insert(node,'right')
        node.left=leftchild
        node.right=rightchild
        leftchild.parent=node
        rightchild.parent=node
        leftchild.name=label
        rightchild.name=label
        leftchild.depth=node.depth+1
        rightchild.depth=node.depth+1
        leftchild.value=0
        rightchild.value=1
        '''Splitting dataframe based on splitting attribute'''
        Xleft=X[X[label]==0]
        Yleft=Y[X[label]==0]
        Xleft=Xleft.drop(label,1)
        Xright=X[X[label]==1]
        Yright=Y[X[label]==1]
        Xright=Xright.drop(label,1)
        '''Calling decision tree recursively'''
        randdecisiontree(Xleft,Yleft,leftchild,tree)
        randdecisiontree(Xright,Yright,rightchild,tree)
    else:
        node.leaf=True
        tree.sumdepth=tree.sumdepth+node.depth
        tree.leafnodes=tree.leafnodes+1
        if((ispure(Y)==True) and (not(Y.empty))):
            node.leafvalue=int(np.unique(Y))
        elif(Y.empty):
            node.leafvalue=random.choice([0,1])
        elif(not(ispure(Y))):
            node.leafvalue=Y.value_counts().index[0]
#Selecting a random attribute
def randattribute(X):
    label=None
    colabel=list(X)
    label=random.choice(colabel)
    return label
#Function to call prediction function one at a time and calculate accuracy and totalinstance
def predictiontree(node,feature,label):
    correctpred=0
    totalinstance=0
    for i in range(0,feature.shape[0]):
        predict=prediction(node,feature.iloc[i,:])
        if(predict==label.iloc[i]):
            correctpred+=1
            totalinstance+=1
        else:
            totalinstance+=1
    accuracy=(correctpred/totalinstance)*100
    return accuracy
#function to predict 
def prediction(node,feature):
    while(node.leaf==False):
        if(feature[node.splitattribute]==0):
            node=node.left
        else:
            node=node.right
    return node.leafvalue   
if __name__=="__main__":
    #Reading train,test and validation data sets from coomand line
    train=pd.read_csv(sys.argv[1])
    validation=pd.read_csv(sys.argv[2])
    test=pd.read_csv(sys.argv[3])
    #Splitting testing data by attribute dataframe and output dataframe
    trainfeature=train.iloc[:,:-1]
    trainlabel=train.iloc[:,-1]
    testfeature=test.iloc[:,:-1]
    testlabel=test.iloc[:,-1]
    validationfeature=validation.iloc[:,:-1]
    validationlabel=validation.iloc[:,-1]
    dtree=Tree()
    id3root=dtree.insertroot()
    id3root.name='Data'
    id3root.depth=0
    #creating decision tree
    id3decisiontree(trainfeature,trainlabel,id3root,dtree)
    #Summary of id3 decision tree
    sleafdepth=dtree.sumdepth
    nleaf=dtree.leafnodes
    avgid3=sleafdepth/nleaf
    #summary of random tree
    randtree=Tree()
    randroot=randtree.insertroot()
    randroot.name='Data'
    randroot.depth=0
    randdecisiontree(trainfeature,trainlabel,randroot,randtree)
    rleafdepth=randtree.sumdepth
    rleaf=randtree.leafnodes
    ravgid3=rleafdepth/rleaf
    data=[['Tree constructed using ID3',avgid3,dtree.totalnodes],['Tree using random attribute selection',ravgid3,randtree.totalnodes]]
    summarydf=pd.DataFrame(data,columns=['','Average Depth','Number of Nodes'])
    print('Tree parameters:','\n'*2)
    print(summarydf,'\n'*2)
    #Running the prediction tree for random decision tree 5 times
    randlist=list()
    for i in range(1,6):
        randtree=Tree()
        randroot=randtree.insertroot()
        randroot.name='Data'
        randroot.depth=0
        randdecisiontree(trainfeature,trainlabel,randroot,randtree)
        randaccuracy=predictiontree(randroot,testfeature,testlabel)
        randlist.append(randaccuracy)
    id3accuracy=predictiontree(id3root,testfeature,testlabel)
    res=[]
    counter=1
    for value in randlist:
        res.append((counter,value))
        counter=counter+1
    res=pd.DataFrame(res,columns=('Run#','Accuracy of tree constructed using random attribute selection'))
    print(res,'\n'*2)
    print('Accuracy of tree constructed using ID3:',id3accuracy)