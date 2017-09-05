import random
import numpy as np
#import sklearn-based class to perform pipeline process
from sklearn.base import BaseEstimator, TransformerMixin

class Uniform_Quantilization(BaseEstimator,TransformerMixin):
    """
    The data feed in should be in [nData * nDim] format to fit the unsupervised uniform quantilization method.
    """
    def __init__(self,nGroup):
        """
        :param nGroup: to specify how many groups to fit, the return numbers will be quantilized into the nGroup values
        """
        self.nGroup = nGroup
        self.thresold_values = np.zeros([nGroup])


    def fit_auto(self,data):
        """
        :param data: In [nData * nDim] shape
        :return: No explicit return, only determine the model parameter
        """
        #data_x is in [nData*nDim]
        nData = data.shape[0]
        nDim = data.shape[1]
        if nData<self.nGroup:
            raise ValueError("Cannot process data which amount is less than demanded quantilization groups!")
        data = np.reshape(data,[nData,nDim])
        if nDim==1:
            vec_norm = data
        else:
            vec_norm = np.linalg.norm(data,axis=1)      #[ndata,]
        max_ele = np.amax(vec_norm)
        min_ele = np.amin(vec_norm)
        self.thresold_values = np.linspace(min_ele,max_ele,self.nGroup+1)

    def fit_domain(self,minValue,maxValue):
        self.thresold_values = np.linspace(minValue, maxValue, self.nGroup + 1)


    def para_quantilize(self,data):
        """
        :param data: the input matrix data in [nData x nDim]
        :return: the quantilized data
        """
        nData = data.shape[0]
        nDim = data.shape[1]
        data = np.reshape(data, [nData, nDim])
        if nDim==1:
            data_norm = data
        else:
            data_norm = np.linalg.norm(data,axis=1)   #[nData,]
        #add a small noise to the largest and smallest so that it won't return execption
        data_norm[np.argmax(data_norm)] = np.amax(data_norm) - 1e-3
        data_norm[np.argmin(data_norm)] = np.amin(data_norm) + 1e-3
        #if the data is in one dimension, just use quick method
        if nDim==1:
            insert_ind = np.searchsorted(self.thresold_values,data_norm)
            data_rst = (self.thresold_values[insert_ind] + self.thresold_values[insert_ind-1])/2
            data_rst = np.reshape(data_rst,[nData,nDim])
        elif nDim>1:
            data_rst = np.zeros([nData,nDim])
            threshold_vec_value = np.zeros([self.nGroup+1,nDim])
            for i in range(self.nGroup+1):
                this_data_ind = np.asscalar(np.argwhere(data_norm == self.thresold_values[i]))
                threshold_vec_value[i,:] = data[this_data_ind,:]
            for cData in range(nData):
                insert_ind = np.searchsorted(self.thresold_values,data_norm[cData])
                data_rst[cData,:] = (threshold_vec_value[insert_ind,:] + threshold_vec_value[insert_ind-1,:])/2
        else:
            raise ValueError("Input Dimension of the data must be no less than 1!")

        return data_rst