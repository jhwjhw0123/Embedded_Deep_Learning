#/usr/bin/env python3
import sys
sys.path.append('/Users/vcmo/Desktop/MasterProject/1.Prune_weight')
from HSIC_feature_selection import HSIC_Lasso
import os
import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns

layer_info_path = './weight_pruning/layer_info/'
weight_info_path = './weight_pruning/weight_info/'

layer_info_list = ['prev_layer.txt','next_layer.txt']

#read the weights of the layer
weight_file_name = weight_info_path + 'current_weight.txt'
with open(weight_file_name,'r') as weight_file_read:
    weight_info_list = weight_file_read.readlines()
weight_mat_list = []
for row in weight_info_list:
    this_row_para = row.split("\t")
    this_row_para = this_row_para[:-1]
    this_row_para = np.asarray(this_row_para,dtype=np.float)
    weight_mat_list.append(this_row_para)
weight_mat = np.asarray(weight_mat_list)    #[nDim_prev+1 * nNode]
#Processing parameters
print('Processing the weight pruning for the current layer...')
prev_layer_file_name = layer_info_path + layer_info_list[0]
next_layer_file_name = layer_info_path + layer_info_list[1]
#read and pre-process the information of the previous layer
with open(prev_layer_file_name,'r') as prev_layer_file_read:
    prev_info_list = prev_layer_file_read.readlines()
nData = len(prev_info_list)
prev_mat_list = []
for row in prev_info_list:
    this_row_para = row.split("\t")
    this_row_para = this_row_para[:-1]
    this_row_para = np.asarray(this_row_para,dtype=np.float)
    prev_mat_list.append(this_row_para)
prev_mat = np.asarray(prev_mat_list)    #[nData * nDim_prev]
#read and pre-process the information of the later layer
with open(next_layer_file_name,'r') as next_layer_file_read:
    next_info_list = next_layer_file_read.readlines()
if(len(next_info_list)!=nData):
    raise ValueError('The data amount of two files must be the same!')
next_mat_list = []
for row in next_info_list:
    this_row_para = row.split("\t")
    this_row_para = this_row_para[:-1]
    this_row_para = np.asarray(this_row_para,dtype=np.float)
    next_mat_list.append(this_row_para)
next_mat = np.asarray(next_mat_list)    #[nData * nDim_prev]
#Define the HSIC Lasso Regression Model
HSIC_regressor = HSIC_Lasso(para_lambda=1e-4,model_mode='regression',sigma_mode='auto',feature_mode='lambda_select')
#Now we prune the weights
nDim_prev = prev_mat.shape[1]
nNode = next_mat.shape[1]
for cNode in range(nNode):
    print('Pruning the weights of the Node',cNode+1,'of this layer...')
    # Add a small noise to avoid nan kernel result
    X_input = prev_mat + 0.1*np.random.normal(size=prev_mat.shape)
    if nData > 200:
        learn_ind = np.random.choice(nData, 200, replace=False)
    else:
        learn_ind = np.arange(nData)
    Y_output = np.reshape(next_mat[:,cNode],[nData,1])
    X_feed = X_input[learn_ind,:]
    Y_feed = Y_output[learn_ind,:]
    HSIC_regressor.fit(X_feed,Y_feed)
    _,remain_id = HSIC_regressor.data_feature_select(X_input)
    this_col_ind = np.arange(nDim_prev)
    ind_delete = np.setdiff1d(this_col_ind,remain_id)
    print(ind_delete.shape)
    weight_mat[ind_delete,cNode] = 0
target_file = weight_info_path + 'Weight_pruned.txt'
writefile = open(target_file, 'w')
print('Saving the weight file...')
for row in range(weight_mat.shape[0]):
    this_row = weight_mat[row,:]
    for col in range(nNode):
        this_ele = this_row[col]
        if col == nNode - 1:
            writefile.write("%s\n" % this_ele)
        else:
            writefile.write("%s\t" % this_ele)
