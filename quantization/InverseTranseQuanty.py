import random
import numpy as np
#import sklearn-based class to perform pipeline process
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from numpy import matlib

class InverseTrans_Quantilization(BaseEstimator,TransformerMixin):
    """
    The data feed in should be in [nData * nDim] format to fit the unsupervised uniform quantilization method.
    This method will cost [n*log(n)*K*s] time complexity, where n is the amount of data, K is the number of clusters,
    and s is the number of steps.
    This is based on Azimi et al. 2017 paper, "A novel clustering algorithms based on data transformation approaches"
    """
    def __init__(self,nGroup,nStep):
        """
        :param nGroup: the number of clusters (groups) to specify
        :param intialize_mode: currently supporting kmeans++, random and centroid method (Azimi et al. 2017)
        """
        self.nGroup = nGroup
        self.nStep = nStep

    def fit(self,data):
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
        #get the clustering centers
        cluster_center = self.centroid_initialization(data)
        data_norm = np.linalg.norm(data,axis=1)
        #sort the data according to accent order of the norm
        sort_ind = np.argsort(data_norm)
        data_sorted = data[sort_ind]
        #generate the artificial data that are on the clustering center
        nClusters = cluster_center.shape[0]
        center_dist = np.zeros([nClusters,nData])
        for cCluster in range(nClusters):
            this_center = np.reshape(cluster_center[cCluster],[1,nDim])
            this_dist = np.linalg.norm(data_sorted - this_center, axis=1)
            center_dist[cCluster] = this_dist
        center_index = np.argmin(center_dist,axis=0)
        artificial_data = cluster_center[center_index]
        #compute the data step
        step_vec = (data_sorted - artificial_data)/self.nStep
        data_this_step = artificial_data
        center_this_step = cluster_center
        for cStep in range(self.nStep):
            data_this_step = data_this_step + step_vec
            self.Learner = KMeans(n_clusters=self.nGroup,init=center_this_step)
            #perform fit
            self.Learner.fit(data_this_step)
            center_this_step = self.Learner.cluster_centers_


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
        cluster_labels = np.reshape(self.Learner.predict(data),[nData])
        centers = self.Learner.cluster_centers_
        rst_mat = centers[cluster_labels]

        return rst_mat
