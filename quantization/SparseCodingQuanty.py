import random
import numpy as np
from sklearn.linear_model import Lasso
#import sklearn-based class to perform pipeline process
from sklearn.base import BaseEstimator, TransformerMixin
from numpy import matlib

class Sparse_Programming_Quantilization(BaseEstimator,TransformerMixin):
    """
    The data feed in should be in [nData * nDim] format to fit the unsupervised uniform quantilization method.
    """
    def __init__(self,lambda_para):
        """
        :param lambda_para: The parameter to control the trade-off

        """
        self.para_lambda = lambda_para
        self.alpha = np.zeros([1,1])
        self.values = np.zeros([1,1])
        self.ref_values = np.zeros([1,1])


    def fit(self,data):
        """
        :param data: In [nData * nDim] shape
        :return: No explicit return, only determine the model parameter
        """
        #data_x is in [nData*nDim]
        nData = data.shape[0]
        nDim = data.shape[1]
        if (nDim>1):
            raise ValueError('The sparse programming method could only process data with dimension 1!')
        data_unique = np.unique(data)
        self.ref_values = data_unique
        nDim_unique = data_unique.shape[0]
        Np_value = np.sum(data_unique)
        Np_mat = np.tril(Np_value*np.ones((nDim_unique,nDim_unique)))
        Model_Regressor = Lasso(alpha=self.para_lambda, fit_intercept=False,
                                max_iter=1000, normalize=False, positive=False, precompute=False,
                                random_state=None, tol=1e-3, warm_start=False)
        data_unique_vec = np.reshape(data_unique,[nDim_unique,1])
        Model_Regressor.fit(Np_mat,data_unique_vec)

        self.alpha = Model_Regressor.coef_
        self.values = np.matmul(Np_mat,np.reshape(self.alpha,[nDim_unique,1]))
        self.values = np.reshape(self.values,[nDim_unique,])



    def para_quantilize(self,data):
        """
        :param data: the input matrix data in [nData x nDim]
        :return: the quantilized data
        """
        nData = data.shape[0]
        nDim = data.shape[1]
        data_quanty = np.reshape(data,[nData,])
        nUnique = self.values.shape[0]
        for value_ind in range(nUnique):
            this_value = self.ref_values[value_ind]
            data_quanty[data_quanty==this_value] = self.values[value_ind]
        data_quanty = np.reshape(data_quanty,[nData,nDim])

        return data_quanty

# test_data = np.random.randint(30,size=1000)
# test_data = np.reshape(test_data,[1000,1])
# test_input = test_data.astype('float32')
# sparse_quantilizer = Sparse_Programming_Quantilization(100)
# sparse_quantilizer.fit(test_input)
# test_output = sparse_quantilizer.para_quantilize(test_input)
# # print(test_data-test_output)
# print(sparse_quantilizer.alpha)
# print(np.unique(sparse_quantilizer.values).shape[0])
# print(np.linalg.norm(test_data-test_output))

# print(sparse_quantilizer.values)
# print(sparse_quantilizer.values)