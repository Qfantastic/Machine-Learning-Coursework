# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 16:19:39 2017

@author: Sreenivas
"""
import numpy as np
import pandas as pd
import sys,math
import random


#Class to create nodes in decision tree
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
#Creating Decision Tree
def decisiontree(X,Y,node,tree):
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
        decisiontree(Xleft,Yleft,leftchild,tree)
        decisiontree(Xright,Yright,rightchild,tree)
    else:
        node.leaf=True
        tree.leafnodes=tree.leafnodes+1
        if((ispure(Y)==True) and (not(Y.empty))):
            node.leafvalue=int(np.unique(Y))
        elif(Y.empty):
            node.leafvalue=random.choice([0,1])
        elif(not(ispure(Y))):
            node.leafvalue=Y.value_counts().index[0]
#Print tree
def printtree(node):
    if(node.leaf==False):
        for i in range(1,node.depth):
            print('|',end=" ")
        print(node.name,'=',end=" ")
        print(node.value,' ',':')
        printtree(node.left)
        printtree(node.right)
    else:
        for i in range(1,node.depth):
            print('|',end=" ")
        print(node.name,'=',end=" ")
        print(node.value,' ',':',end=" ")
        print(node.leafvalue)
#Finding the best attribute
def bestattribute(X,Y):
    label=None
    maxgain=0
    for column in X:
        gain=infogain(X[column],Y)
        if(gain>=maxgain):
            label=column
            maxgain=gain
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
    return accuracy,totalinstance
#function to predict 
def prediction(node,feature):
    while(node.leaf==False):
        if(feature[node.splitattribute]==0):
            node=node.left
        else:
            node=node.right
    return node.leafvalue    
#prune tree
def prunetree(prunevalue,node,totalnodes,tree):
    startnode=int(totalnodes-(prunevalue))
    prunenodes=random.sample(range(startnode,totalnodes),prunevalue)
    for value in prunenodes:
        retnode=prunetraverse(value,node)
        if(retnode!=None):
            positivevalue,negativevalue=pruneassign(retnode,tree)
            if(retnode.leaf==False):
                retnode.leaf=True
                tree.leafnodes=tree.leafnodes+1
            retnode.left=False
            retnode.right=False
            if(positivevalue>=negativevalue):
                retnode.leafvalue=1
            else:
                retnode.leafvalue=0
#Traversing tree for pruning
def prunetraverse(value,node):
    if(node.nodeid==value):
        return node
    elif(node.leaf==False):
        nodel=prunetraverse(value,node.left)
        if(nodel==None):
            noder=prunetraverse(value,node.right)
            if(noder==None):
                return None
            else:
                return noder
        else:
            return nodel
#Assigning label after removing pruned node
def pruneassign(node,tree):
    if(node.leaf==True):
        if(node.leafvalue==0):
            node.subnegativeleaf=node.subnegativeleaf+1
        else:
            node.subpositiveleaf=node.subpositiveleaf+1
        tree.leafnodes=tree.leafnodes-1
        tree.totalnodes=tree.totalnodes-1
        return node.subpositiveleaf,node.subnegativeleaf
    else:
        tree.totalnodes=tree.totalnodes-1
        positivevalue1,negativevalue1=pruneassign(node.left,tree)
        positivevalue2,negativevalue2=pruneassign(node.right,tree)
        positivevalue=positivevalue1+positivevalue2
        negativevalue=negativevalue1+negativevalue2
        return positivevalue,negativevalue
#Main Function
if __name__=="__main__":
    #Reading train,test and validation data sets from coomand line
    train=pd.read_csv(sys.argv[1])
    validation=pd.read_csv(sys.argv[2])
    test=pd.read_csv(sys.argv[3])
    prunefactor=sys.argv[4]
    #Splitting testing data by attribute dataframe and output dataframe
    trainfeature=train.iloc[:,:-1]
    trainlabel=train.iloc[:,-1]
    testfeature=test.iloc[:,:-1]
    testlabel=test.iloc[:,-1]
    validationfeature=validation.iloc[:,:-1]
    validationlabel=validation.iloc[:,-1]
    dtree=Tree()
    root=dtree.insertroot()
    root.name='Data'
    root.depth=0
    #creating decision tree
    decisiontree(trainfeature,trainlabel,root,dtree)
    #Summary of pre pruned accuracy
    trainaccuracy,traininstance=predictiontree(root,trainfeature,trainlabel)
    totalnodes=dtree.totalnodes
    leafnodes=dtree.leafnodes
    testaccuracy,testinstance=predictiontree(root,testfeature,testlabel)
    validationaccuracy,validationinstance=predictiontree(root,validationfeature,validationlabel)
    print('Pre-Pruned Acuuracy')
    print('...................')
    print('Number of training instances =',traininstance)
    print('Number of training attributes =', trainfeature.shape[1])
    print('Total number of nodes in the tree =',totalnodes)
    print('Total number of leaf nodes in the tree =',leafnodes)
    print('Accuracy of model on training dataset =',trainaccuracy,'\n'*2)
    print('Number of Validation instances =',validationinstance)
    print('Number of Validation attributes =',validationfeature.shape[1])
    print('Accuracy of model on validation dataset before pruning =',validationaccuracy,'\n'*2)
    print('Number of testing instances =',testinstance)
    print('Number of testing attributes -',testfeature.shape[1])
    print('Accuracy of model on testing Dataset =',testaccuracy)
    #printing decision tree
    print('Decision Tree','\n'*2)
    printtree(root)
    print('\n'*2)
    #Pruning the dataset and printing post pruned accuracy details
    prunevalue=int(0.3*totalnodes)
    prunetree(prunevalue,root,totalnodes,dtree)
    posttrainaccuracy,posttraininstance=predictiontree(root,trainfeature,trainlabel)
    totalnodes=dtree.totalnodes
    leafnodes=dtree.leafnodes
    posttestaccuracy,posttestinstance=predictiontree(root,testfeature,testlabel)
    postvalidationaccuracy,postvalidationinstance=predictiontree(root,validationfeature,validationlabel)
    print('Post-Pruned Acuuracy')
    print('...................')
    print('Number of training instances =',posttraininstance)
    print('Number of training attributes =', trainfeature.shape[1])
    print('Total number of nodes in the tree =',totalnodes)
    print('Total number of leaf nodes in the tree =',leafnodes)
    print('Accuracy of model on training dataset =',posttrainaccuracy,'\n'*2)
    print('Number of Validation instances =',postvalidationinstance)
    print('Number of Validation attributes =',validationfeature.shape[1])
    print('Accuracy of model on validation dataset before pruning =',postvalidationaccuracy,'\n'*2)
    print('Number of testing instances =',posttestinstance)
    print('Number of testing attributes -',testfeature.shape[1])
    print('Accuracy of model on testing Dataset =',posttestaccuracy)
    #printing decision tree
    print('Decision Tree after pruning','\n'*2)
    printtree(root)