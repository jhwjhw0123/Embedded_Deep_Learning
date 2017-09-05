import random
import numpy as np
#import sklearn-based class to perform pipeline process
from sklearn.base import BaseEstimator, TransformerMixin

class Logarithm_Quantilization(BaseEstimator,TransformerMixin):
    """
    The data feed in should be in [nData * nDim] format to fit the unsupervised logarithm quantilization method.
    Here we firstly divide the domains linearly, then we perform expotential algorithm that could forcus more on the
    diving of the massively distributed data
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
        min_abs_ele = np.amin(np.absolute(vec_norm))
        min_abs_log_value = np.log2(min_abs_ele+1e-5)
        if max_ele<=0:
            log_value = np.log2(abs(min_ele))
            if log_value>0:
                log_value = log_value
            else:
                log_value = log_value
            self.thresold_values = -np.power(2,np.linspace(min_abs_log_value,log_value,self.nGroup+1))
        elif min_ele>=0:
            log_value = np.log2(abs(max_ele))
            if log_value>0:
                log_value = log_value
            else:
                log_value = log_value
            # print(np.linspace(min_log_value, max_log_value, self.nGroup + 1))
            self.thresold_values = np.power(2, np.linspace(min_abs_log_value, log_value, self.nGroup + 1))
        else:
            if(abs(max_ele)>abs(min_ele)):
                log_value = np.log2(abs(max_ele))
            else:
                log_value = np.log2(abs(min_ele))
            if log_value>0:
                log_value = log_value
            else:
                log_value = log_value
            linear_threshold = np.linspace(min_abs_log_value, log_value, (self.nGroup // 2) + 1)
            thresold_values_pos = np.power(2,linear_threshold)
            thresold_values_neg = -np.power(2,linear_threshold)
            self.thresold_values = np.concatenate((thresold_values_neg,thresold_values_pos),axis=0)
            self.thresold_values = np.unique(self.thresold_values)
        self.thresold_values = np.sort(self.thresold_values,axis=0)

    def fit_domain(self,minValue,maxValue):
        max_ele = max(minValue,maxValue)
        min_ele = min(minValue,maxValue)
        min_abs_log_value = np.log2(1e-5)
        if max_ele <= 0:
            log_value = np.log2(abs(min_ele))
            if log_value > 0:
                log_value = log_value
            else:
                log_value = log_value
            self.thresold_values = -np.power(2, np.linspace(min_abs_log_value, log_value, self.nGroup + 1))
        elif min_ele >= 0:
            log_value = np.log2(abs(max_ele))
            if log_value > 0:
                log_value = log_value
            else:
                log_value = log_value
            # print(np.linspace(min_log_value, max_log_value, self.nGroup + 1))
            self.thresold_values = np.power(2, np.linspace(min_abs_log_value, log_value, self.nGroup + 1))
        else:
            if (abs(max_ele) > abs(min_ele)):
                log_value = np.log2(abs(max_ele))
            else:
                log_value = np.log2(abs(min_ele))
            if log_value > 0:
                log_value = log_value
            else:
                log_value = log_value
            linear_threshold = np.linspace(min_abs_log_value, log_value, (self.nGroup // 2) + 1)
            thresold_values_pos = np.power(2, linear_threshold)
            thresold_values_neg = -np.power(2, linear_threshold)
            self.thresold_values = np.concatenate((thresold_values_neg, thresold_values_pos), axis=0)
            self.thresold_values = np.unique(self.thresold_values)
        self.thresold_values = np.sort(self.thresold_values, axis=0)

    def para_quantilize(self,data):
        """
        :param data: the input matrix data in [nData x nDim]
        :return: the quantilized data
        """
        nData = data.shape[0]
        nDim = data.shape[1]
        data = np.reshape(data, [nData, nDim])
        if nDim == 1:
            data_norm = data
        else:
            np.linalg.norm(data,axis=1)   #[nData,]
        #add a small noise to the largest and smallest so that it won't return execption
        data_norm[np.argmax(data_norm)] = np.amax(data_norm) - 1e-3
        data_norm[np.argmin(data_norm)] = np.amin(data_norm) + 1e-3
        #if the data is in one dimension, just use quick method
        if nDim==1:
            insert_ind = np.searchsorted(self.thresold_values,data_norm)
            # insert_ind[insert_ind==0] = 1
            # insert_ind[insert_ind==self.thresold_values.shape[0]] = self.thresold_values.shape[0]-1
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
