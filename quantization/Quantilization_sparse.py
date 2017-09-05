from KmeansQuanty import Kmeans_Quantilization
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
#Define the Kmeans++ quantilizer for pre-process
kmeans_cluster_quantilizer = Kmeans_Quantilization(300,intialize_mode='kmeans++')
#Define the quantilizers
Sparse_opt_quantilizer = Sparse_Programming_Quantilization(100)     #Here the parameter is the value of parameter lambda

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

#Pre-process with kmeans clustering
kmeans_cluster_quantilizer.fit(para_mat_all)
processed_data = kmeans_cluster_quantilizer.para_quantilize(para_mat_all)
#fitting Sparse Programming clustering
t_initial = time.time()
Sparse_opt_quantilizer.fit(processed_data)
print('Quantization as sparse coding fitting time is:',time.time() - t_initial)
print(np.count_nonzero(Sparse_opt_quantilizer.alpha))
print(Sparse_opt_quantilizer.alpha)

file_ind = 1
sparse_coding_norms = []
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
    this_data_flatten_processed = kmeans_cluster_quantilizer.para_quantilize(this_data_flatten)
    '''Uniform Domain Quantization'''
    sparse_quantized_para = Sparse_opt_quantilizer.para_quantilize(this_data_flatten_processed)
    sparse_quantized_new_mat = np.concatenate((np.reshape(sparse_quantized_para,[(nRow-1),nCol]),bias_para),axis=0)
    para_mat_save(sparse_quantized_new_mat,'../parameter/Sparse_Quantization/','W_'+str(file_ind)+'.txt')
    sparse_coding_norms.append(np.linalg.norm(this_data_flatten_processed-sparse_quantized_para))
    '''File index increase'''
    file_ind = file_ind + 1
sparse_coding_norms = np.asarray(sparse_coding_norms)
print('The average norm difference of sparse coding norm fitting is',np.mean(sparse_coding_norms))

# sns.set(color_codes=True)
# sns.distplot(para_mat_all)
# plt.show()


# print(uni_quantilizer.thresold_values)
# log_quantilizer.fit_auto(this_data_flatten)

