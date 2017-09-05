import random
import numpy as np
#import sklearn-based class to perform pipeline process
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.mixture import GaussianMixture
from numpy import matlib

class MoG_Quantilization(BaseEstimator,TransformerMixin):
    """
    The data feed in should be in [nData * nDim] format to fit the unsupervised uniform quantilization method.
    """
    def __init__(self,nGroup,initialize_mode):
        """
        :param nGroup: to specify how many groups to fit, the return numbers will be quantilized into the nGroup values
        :param initialize_mode: specify the mode we use, currently support 'random', 'kmeans' and 'centroid' (Azimi et al. 2017 method)
        """
        self.nGroup = nGroup
        self.mode = initialize_mode
        self.thresold_values = np.zeros([nGroup])


    def fit(self,data):
        """
        :param data: In [nData * nDim] shape
        :return: No explicit return, only determine the model parameter
        """
        #data_x is in [nData*nDim]
        nData = data.shape[0]
        nDim = data.shape[1]
        #initialize the MoG Model
        if self.mode == 'random':
            self.MoGLearner = GaussianMixture(self.nGroup,init_params='random')
        elif self.mode == 'kmeans':
            self.MoGLearner = GaussianMixture(self.nGroup,init_params='kmeans')
        elif self.mode == 'centroid':
            self.MoGLearner = GaussianMixture(self.nGroup,means_init=self.centroid_initialization(data))
        else:
            raise ValueError('Input mode cannot be recognized!')
        self.MoGLearner.fit(data)
        self.log_likelihood = self.MoGLearner.lower_bound_
        self.max_conv = np.amax(np.reshape(self.MoGLearner.covariances_,[-1,1]),axis=0)


    def centroid_initialization(self,data):
        nData = data.shape[0]
        nDim = data.shape[1]
        #get the unique data with appearence times
        data_unique,counts = np.unique(data,return_counts=True)
        nDataNew = data_unique.shape[0]
        data_unique = np.reshape(data_unique,[nDataNew,nDim])
        counts = np.reshape(counts,[nDataNew,1])
        new_data = np.concatenate((data_unique,counts),axis=1)   #[data_new_amount * nDim+1]
        #sorting with norm accent
        data_norm = np.linalg.norm(new_data,axis=1)
        sort_ind = np.argsort(data_norm)
        sort_new_data = new_data[sort_ind]
        #split the data into nGroup data
        split_ind = np.linspace(0,nDataNew,num=self.nGroup+1)
        #define the center
        rst_center = np.zeros([self.nGroup,nDim])
        for i in range(self.nGroup):
            this_group_data = sort_new_data[(int)(split_ind[i]):(int)(split_ind[i+1])]    #nData_group * nDim
            nData_group = this_group_data.shape[0]
            weight_vec = np.zeros([nData_group])
            for cData in range(nData_group):
                norm_vec = np.linalg.norm(matlib.repmat(this_group_data[cData],nData_group,1)-this_group_data,axis=1)
                weight_vec[cData] = nData_group/np.sum(norm_vec)
            center_ind = np.argmax(weight_vec)
            this_center = this_group_data[center_ind,:-1]   #delete the appended time
            rst_center[i] = this_center

        return rst_center

    def centroid_initialization_vec(self,data):
        nData = data.shape[0]
        nDim = data.shape[1]
        #get the unique data with appearence times
        data_unique,counts = np.unique(data,return_counts=True)
        nDataNew = data_unique.shape[0]
        data_unique = np.reshape(data_unique,[nDataNew,nDim])
        counts = np.reshape(counts,[nDataNew,1])
        new_data = np.concatenate((data_unique,counts),axis=1)   #[data_new_amount * nDim+1]
        #sorting with norm accent
        data_norm = np.linalg.norm(new_data,axis=1)
        sort_ind = np.argsort(data_norm)
        sort_new_data = new_data[sort_ind]
        #split the data into nGroup data
        split_ind = np.linspace(0,nDataNew,num=self.nGroup+1)
        #define the center
        rst_center = np.zeros([self.nGroup,nDim])
        for i in range(self.nGroup):
            this_group_data = sort_new_data[(int)(split_ind[i]):(int)(split_ind[i+1])]    #nData_group * nDim
            nData_group = this_group_data.shape[0]
            this_group_data_rep = matlib.repmat(this_group_data,1,nData_group)
            rep_row = this_group_data_rep.shape[0]
            rep_col = this_group_data_rep.shape[1]
            this_group_data_rep = np.reshape(this_group_data_rep,[rep_row*rep_col])
            this_group_data_rep = np.reshape(this_group_data_rep,[-1,nDim+1])
            norm_vec = np.linalg.norm(this_group_data_rep - matlib.repmat(this_group_data,nData_group,1),axis=1)
            norm_vec_list = np.reshape(norm_vec,[nData_group,nData_group])
            norm_comp = np.sum(norm_vec_list,axis=1)
            norm_comp = nData_group/norm_comp
            center_ind = np.argmax(norm_comp)
            this_center = this_group_data[center_ind,:-1]   #delete the appended time
            rst_center[i] = this_center

        return rst_center


    def para_quantilize(self,data):
        """
        :param data: the input matrix data in [nData x nDim]
        :return: the quantilized data
        """
        nData = data.shape[0]
        nDim = data.shape[1]
        data = np.reshape(data, [nData, nDim])
        cluster_label = self.MoGLearner.predict(data)
        cluster_centers = self.MoGLearner.means_
        rst_mat = cluster_centers[cluster_label]

        return rst_mat
