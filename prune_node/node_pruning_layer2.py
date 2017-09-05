import sys
sys.path.append('/Users/vcmo/Desktop/MasterProject/1.Prune_node')
from HSIC_feature_selection import HSIC_Lasso
import os
import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns

layer_info_path = './node_pruning/layer_info/'
weight_info_path = './node_pruning/weight_info/'

layer_info_list = ['layer.txt','output.txt']

#read the weights of the layer
weight_file_name = weight_info_path + 'Weight_this_layer.txt'
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
layer_file_name = layer_info_path + layer_info_list[0]    #The output of current layer
output_file_name = layer_info_path + layer_info_list[1]
#read and pre-process the information of the previous layer
with open(layer_file_name,'r') as layer_file_read:
    layer_info_list = layer_file_read.readlines()
nData = len(layer_info_list)
layer_mat_list = []
for row in layer_info_list:
    this_row_para = row.split("\t")
    this_row_para = this_row_para[:-1]
    this_row_para = np.asarray(this_row_para,dtype=np.float)
    layer_mat_list.append(this_row_para)
layer_mat = np.asarray(layer_mat_list)    #[nData * nDim_prev]
#read and pre-process the information of the outputs
with open(output_file_name,'r') as output_file_read:
    output_info_list = output_file_read.readlines()
if(len(output_info_list)!=nData):
    raise ValueError('The data amount of two files must be the same!')
output_mat_list = []
for row in output_info_list:
    this_row_para = np.asarray(row[:-1],dtype=np.int)
    output_mat_list.append(this_row_para)
output_mat = np.reshape(np.asarray(output_mat_list),[nData,1])
#Define the HSIC Lasso Regression Model
HSIC_regressor = HSIC_Lasso(para_lambda=1e-10,model_mode='classification',sigma_mode='auto',feature_mode='top_select',
                            feature_num=450)
#Now we prune the weights
nDim_prev = layer_mat.shape[1]
nNode = output_mat.shape[1]
#prune the nodes
print('HSIC running to prune...')
learn_ind = np.random.choice(nData, 500, replace=False)
X_input = layer_mat[learn_ind,:]
Y_output = output_mat[learn_ind,:]
HSIC_regressor.fit(X_input,Y_output)
_,remain_id = HSIC_regressor.data_feature_select(X_input)
this_col_ind = np.arange(nDim_prev)
ind_delete = np.setdiff1d(this_col_ind,remain_id)
print(ind_delete.shape)
# weight_mat_weights = weight_mat[:-1,:]
# weight_mat_bias = np.reshape(weight_mat[-1,:],[1,nDim_prev])
# weight_mat_weights[:,ind_delete] = 0
# weight_mat = np.concatenate((weight_mat_weights,weight_mat_bias),axis=0)
weight_mat[:,ind_delete]=0
target_file = weight_info_path + 'Weight_node_pruned.txt'
writefile = open(target_file, 'w')
print('Saving the weight file...')
for row in range(weight_mat.shape[0]):
    this_row = weight_mat[row,:]
    for col in range(nDim_prev):
        this_ele = this_row[col]
        if col == nDim_prev - 1:
            writefile.write("%s\n" % this_ele)
        else:
            writefile.write("%s\t" % this_ele)

