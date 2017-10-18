# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 20:20:46 2017

@author: Sreenivas
"""
import pandas as pd
import numpy as np
import sys
#Main Function
if __name__=="__main__":
    #Taking URL
    inputurl=sys.argv[1]
    outputpath=sys.argv[2]
    data=pd.read_csv(inputurl,header=None)
    #removing rows having missing values
    data.dropna(axis=0,how='any')
    #Standarizing numeric values
    from sklearn.preprocessing import StandardScaler
    scalervar=StandardScaler()
    for i in range(0,len(data.iloc[:,:-1].columns)):
        if(data.iloc[:,i].dtype==np.float64 or data.iloc[:,i].dtype==np.int64):
            data.iloc[:,i]=scalervar.fit_transform(data.iloc[:,i].values.reshape(-1,1))
    #encoding categorical variables for features and output class
    from sklearn.preprocessing import LabelEncoder
    labelencodvar=LabelEncoder()
    for i in range(0,len(data.iloc[:,:].columns)):
        if(data.iloc[:,i].dtype==np.object or data.iloc[:,i].dtype.name=='category'):
            data.iloc[:,i]=labelencodvar.fit_transform(data.iloc[:,i])
    #writing to file specified in commandline
    data.to_csv(outputpath,mode='w',index=False,header=False)