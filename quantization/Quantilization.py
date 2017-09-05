from uniformQuanty import Uniform_Quantilization
from LogaritmQuanty import Logarithm_Quantilization
from KmeansQuanty import Kmeans_Quantilization
from MoGQuanty import MoG_Quantilization
from InverseTranseQuanty import InverseTrans_Quantilization
from SparseCodingQuanty import Sparse_Programming_Quantilization
import os
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns

para_path_DNN = '../parameter/DNN/'
para_path_CNN = '../parameter/CNN/'

para_save_uniform_DNN = '../para_uniform/DNN/'
para_save_uniform_CNN = '../para_uniform/CNN/'

def para_mat_save(weight_mat,file_path,file_name):
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    nCol = weight_mat.shape[1]
    writefile = open(file_path+file_name, 'w')
    print('Saving the weight file...')
    for row in range(weight_mat.shape[0]):
        this_row = weight_mat[row, :]
        for col in range(nCol):
            this_ele = this_row[col]
            if col == nCol - 1:
                writefile.write("%s\n" % this_ele)
            else:
                writefile.write("%s\t" % this_ele)


para_DNN_list = os.listdir(para_path_DNN)

para_CNN_list = os.listdir(para_path_CNN)

for file in para_DNN_list:
    if file[-4:]!='.txt':
        para_DNN_list.remove(file)

for file in para_CNN_list:
    if file[-4:]!='.txt':
        para_CNN_list.remove(file)

print(para_DNN_list)
print(para_CNN_list)

#Define the quantilizers
uni_quantilizer = Uniform_Quantilization(16)
uni_quantilizer_minmax = Uniform_Quantilization(16)
log_quantilizer = Logarithm_Quantilization(16)
log_quantilizer_minmax = Logarithm_Quantilization(16)
Kmeans_quantilizer = Kmeans_Quantilization(16, 'kmeans++')
Kmeans_quantilizer_centriod = Kmeans_Quantilization(16, 'centroid')
MoG_quantilizer = MoG_Quantilization(16,'centroid')
Inv_trans_quantilizer = InverseTrans_Quantilization(16,10)      # clustering centers, steps
Sparse_opt_quantilizer = Sparse_Programming_Quantilization(0.1)     #Here the parameter is the value of parameter lambda

#Processing parameters and get the whole training vector
para_mat_all = np.zeros((1,1))
i=0
for file in para_DNN_list:
    print('Processing file parameter file '+file+'...')
    para_file_name = para_path_DNN + file
    with open(para_file_name,'r') as para_file_read:
        this_para_list = para_file_read.readlines()
    nRow = len(this_para_list)
    para_mat_list = []
    for row in this_para_list:
        this_row_para = row.split("\t")
        this_row_para = this_row_para[:-1]
        this_row_para = np.asarray(this_row_para,dtype=np.float)
        para_mat_list.append(this_row_para)
    #firstly remove the last line, this is the bias...
    bias_para = np.reshape(np.asarray(para_mat_list[-1]),[1,-1])
    para_mat_list = para_mat_list[:-1]
    para_mat = np.asarray(para_mat_list)
    nCol = para_mat.shape[1]
    this_data_flatten = np.reshape(para_mat,[(nRow-1)*nCol,1])
    if i==0:
        para_mat_all = this_data_flatten
    else:
        para_mat_all = np.concatenate((para_mat_all,this_data_flatten),axis=0)
    i = i + 1

print(para_mat_all.shape)
#fit the uniform domain
uni_quantilizer.fit_domain(-2,2)
#fit the logarithm domain
log_quantilizer.fit_domain(-2,2)
#fit the uniform min-max
uni_quantilizer_minmax.fit_auto(para_mat_all)
#fit the logarithm min-max
log_quantilizer_minmax.fit_auto(para_mat_all)
#fit kmeans++ method
t_kmean_pp = time.time()
Kmeans_quantilizer.fit(para_mat_all)
print('Keans++ fitting time is:',time.time() - t_kmean_pp)
#fit kmeans with centriod method
t_kmeans_centroid = time.time()
Kmeans_quantilizer_centriod.fit(para_mat_all)
print('Keans with centroid method fitting time is:',time.time() - t_kmeans_centroid)
#fitting with Inverse information transformation
t_inverse_trans = time.time()
Inv_trans_quantilizer.fit(para_mat_all)
print('Inverse Transformation Clustering fitting time is:',time.time() - t_inverse_trans)
#fitting MoG clustering
t_Mog = time.time()
MoG_quantilizer.fit(para_mat_all)
likelihood = MoG_quantilizer.log_likelihood
max_var = MoG_quantilizer.max_conv
print('Mixture of Gaussian Clustering fitting time is:',time.time() - t_Mog)
print('The log-likelihood is:',likelihood)
print('The maximum variance is:',max_var)

file_ind = 1
uniform_domain_norms = []
log_domain_norms = []
uniform_MinMax_norms = []
log_MinMax_norms = []
kmeans_pp_norms = []
kmeans_centroid_norms = []
inverse_trans_norms = []
MoG_cluster_norms = []
for file in para_DNN_list:
    print('Processing file parameter file '+file+'...')
    para_file_name = para_path_DNN + file
    with open(para_file_name,'r') as para_file_read:
        this_para_list = para_file_read.readlines()
    nRow = len(this_para_list)
    para_mat_list = []
    for row in this_para_list:
        this_row_para = row.split("\t")
        this_row_para = this_row_para[:-1]
        this_row_para = np.asarray(this_row_para,dtype=np.float)
        para_mat_list.append(this_row_para)
    #firstly remove the last line, this is the bias...
    bias_para = np.reshape(np.asarray(para_mat_list[-1]),[1,-1])
    para_mat_list = para_mat_list[:-1]
    para_mat = np.asarray(para_mat_list)
    nCol = para_mat.shape[1]
    this_data_flatten = np.reshape(para_mat,[(nRow-1)*nCol,1])
    '''Uniform Domain Quantization'''
    uni_quantized_para_domain = uni_quantilizer.para_quantilize(this_data_flatten)
    uni_domain_new_mat = np.concatenate((np.reshape(uni_quantized_para_domain,[(nRow-1),nCol]),bias_para),axis=0)
    para_mat_save(uni_domain_new_mat,'../parameter/Uniform_domain/','W_'+str(file_ind)+'.txt')
    uniform_domain_norms.append(np.linalg.norm(this_data_flatten-uni_quantized_para_domain))
    '''Logarithm Domain Quantization'''
    log_quantized_para_domain = log_quantilizer.para_quantilize(this_data_flatten)
    log_domain_new_mat = np.concatenate((np.reshape(log_quantized_para_domain, [(nRow - 1), nCol]), bias_para), axis=0)
    para_mat_save(log_domain_new_mat, '../parameter/Logarithm_domain/', 'W_' + str(file_ind) + '.txt')
    log_domain_norms.append(np.linalg.norm(this_data_flatten - log_quantized_para_domain))
    '''Uniform Min-Max Quantization'''
    uni_quantized_para_minmax = uni_quantilizer_minmax.para_quantilize(this_data_flatten)
    uni_minmax_new_mat = np.concatenate((np.reshape(uni_quantized_para_minmax, [(nRow - 1), nCol]), bias_para), axis=0)
    para_mat_save(uni_minmax_new_mat, '../parameter/Uniform_MinMax/', 'W_' + str(file_ind) + '.txt')
    uniform_MinMax_norms.append(np.linalg.norm(this_data_flatten - uni_quantized_para_minmax))
    '''Logarithm Min-Max Quantization'''
    log_quantized_para_minmax = log_quantilizer_minmax.para_quantilize(this_data_flatten)
    log_minmax_new_mat = np.concatenate((np.reshape(log_quantized_para_minmax, [(nRow - 1), nCol]), bias_para), axis=0)
    para_mat_save(log_minmax_new_mat, '../parameter/Logarithm_MinMax/', 'W_' + str(file_ind) + '.txt')
    log_MinMax_norms.append(np.linalg.norm(this_data_flatten - log_quantized_para_minmax))
    '''K-means ++ clustering method'''
    kmeans_quantized_para_pp = Kmeans_quantilizer.para_quantilize(this_data_flatten)
    log_kmeans_pp_new_mat = np.concatenate((np.reshape(kmeans_quantized_para_pp, [(nRow - 1), nCol]), bias_para), axis=0)
    para_mat_save(log_kmeans_pp_new_mat, '../parameter/Kmeans++/', 'W_' + str(file_ind) + '.txt')
    kmeans_pp_norms.append(np.linalg.norm(this_data_flatten - kmeans_quantized_para_pp))
    '''K-means centroid clustering method'''
    kmeans_quantized_para_centrioid = Kmeans_quantilizer_centriod.para_quantilize(this_data_flatten)
    log_kmeans_centroid_new_mat = np.concatenate((np.reshape(kmeans_quantized_para_centrioid, [(nRow - 1), nCol]), bias_para), axis=0)
    para_mat_save(log_kmeans_centroid_new_mat, '../parameter/Kmeans_centroid/', 'W_' + str(file_ind) + '.txt')
    kmeans_centroid_norms.append(np.linalg.norm(this_data_flatten - kmeans_quantized_para_centrioid))
    '''Inverse Transformation clustering method'''
    Inv_trans_quantized_para = Inv_trans_quantilizer.para_quantilize(this_data_flatten)
    Inv_trans_new_mat = np.concatenate((np.reshape(Inv_trans_quantized_para, [(nRow - 1), nCol]), bias_para), axis=0)
    para_mat_save(Inv_trans_new_mat, '../parameter/Inverse_Trans/', 'W_' + str(file_ind) + '.txt')
    inverse_trans_norms.append(np.linalg.norm(this_data_flatten - Inv_trans_quantized_para))
    '''MoG clustering method'''
    MoG_quantized_para = MoG_quantilizer.para_quantilize(this_data_flatten)
    MoG_new_mat = np.concatenate((np.reshape(MoG_quantized_para, [(nRow - 1), nCol]), bias_para), axis=0)
    para_mat_save(MoG_new_mat, '../parameter/MoG_quanty/', 'W_' + str(file_ind) + '.txt')
    MoG_cluster_norms.append(np.linalg.norm(this_data_flatten - MoG_quantized_para))
    '''File index increase'''
    file_ind = file_ind + 1
uniform_domain_norms = np.asarray(uniform_domain_norms)
log_domain_norms = np.asarray(log_domain_norms)
uniform_MinMax_norms = np.asarray(uniform_MinMax_norms)
log_MinMax_norms = np.asarray(log_MinMax_norms)
kmeans_pp_norms = np.asarray(kmeans_pp_norms)
kmeans_centroid_norms = np.asarray(kmeans_centroid_norms)
inverse_trans_norms = np.asarray(inverse_trans_norms)
MoG_cluster_norms = np.asarray(MoG_cluster_norms)
print('The average norm difference of Uniform quantization with domain fit is',np.mean(uniform_domain_norms))
print('The average norm difference of Logarithm quantization with domain fit is',np.mean(log_domain_norms))
print('The average norm difference of Uniform Min-Max fit is',np.mean(uniform_MinMax_norms))
print('The average norm difference of Logarithm quantization Min-Max fit is',np.mean(log_MinMax_norms))
print('The average norm difference of Kmeans++ fit is',np.mean(kmeans_pp_norms))
print('The average norm difference of Kmeans centroid method fit is',np.mean(kmeans_centroid_norms))
print('The average norm difference of inverse transformation fit is',np.mean(inverse_trans_norms))
print('The average norm difference of MoG fit is',np.mean(MoG_cluster_norms))

# sns.set(color_codes=True)
# sns.distplot(para_mat_all)
# plt.show()


# print(uni_quantilizer.thresold_values)
# log_quantilizer.fit_auto(this_data_flatten)


import sys
sys.exit()
# Sparse Optimization Quantilizer
Sparse_opt_quantilizer.fit(this_data_flatten)
quan_data_flatten_sparse = Sparse_opt_quantilizer.para_quantilize(this_data_flatten)
# print(log_quantilizer.thresold_values)
# Kmeans_quantilizer.fit(this_data_flatten)
# quan_data_flatten_k_means = Kmeans_quantilizer.para_quantilize(this_data_flatten)
# MoG_quantilizer.fit(this_data_flatten)
# quan_data_flatten_mog = MoG_quantilizer.para_quantilize(this_data_flatten)
#Inverstransform clustering
# Inv_trans_quantilizer.fit(this_data_flatten)
# quan_data_flatten_inv = Inv_trans_quantilizer.para_quantilize(this_data_flatten)
print(np.linalg.norm(this_data_flatten-quan_data_flatten_uni))
print(np.linalg.norm(this_data_flatten-quan_data_flatten_log))
print(np.linalg.norm(this_data_flatten-quan_data_flatten_sparse))
# print(np.linalg.norm(this_data_flatten-quan_data_flatten_mog))
# print(np.linalg.norm(this_data_flatten - quan_data_flatten_k_means))
# print(np.linalg.norm(this_data_flatten - quan_data_flatten_inv))
quan_data_mat = np.reshape(quan_data_flatten_uni,[(nRow-1),nCol])
quan_data_mat = np.concatenate((quan_data_mat, bias_para), axis=0)
print(quan_data_mat[:,1])

