import numpy as np
import tensorflow as tf
import os
from tensorflow.examples.tutorials.mnist import input_data
import random
import json

save_path = './parameter/'

mnist = input_data.read_data_sets("../data", one_hot=True)

#split into trainning and test data
train_x = mnist.train.images
train_y = mnist.train.labels
test_x = mnist.test.images
test_y = mnist.test.labels

#trainning data amount
train_amount = train_x.shape[0]

# Global hyper-parameters
n_Classes = 10
# Batch for Stochastic Gradient Descent
batch_size = 128
# Training Times
nEpochs = 100

# Number of Hidden Layer nodes
n_Nodes_Hidden_1 = 256
n_Nodes_Hidden_2 = 256

#define the dimension indicator
channel_dict = np.array([1,8,16])
conv_kernel_dict = np.array([[3,3],[3,3]])
conv_stride_dict = np.array([[1,1],[1,1]])
pool_kernel_dict = np.array([[2,2],[2,2]])
pool_stride_dict = np.array([[2,2],[2,2]])
dim_indic = np.array([7*7*16,256,128,10])
pool_method = ['max','max']
activation_method = ['relu','relu','none']

def save_parameter(parameter,name):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    parameter.tofile(save_path+name)

# Get the x input as float and reshape input
x = tf.placeholder("float", [None, 784])
y = tf.placeholder("float", [None, 10])

def convolution2D(Data,Weight,height_stride,width_stride):
    return tf.nn.conv2d(Data,Weight,strides = [1,height_stride,width_stride,1], padding = 'SAME')

def maxpooling(Data,height_stride,width_stride,height_kernel,width_kernel):
    return tf.nn.max_pool(Data, ksize=[1, height_kernel, width_kernel, 1], strides=[1, height_stride, width_stride, 1],padding='SAME')
    # 'ksize' controls the size of the pooling window, 'strides' controls how it moves

def averagepooling(Data,height_stride,width_stride,height_kernel,width_kernel):
    return tf.nn.avg_pool(Data,ksize=[1, height_kernel, width_kernel, 1], strides=[1, height_stride, width_stride, 1],padding='SAME')

# Define the Neural Network Model
def CNN_network(x,n_channels,conv_shape,conv_stride,pool_shape,pool_stride,layers_dim,pool_mode,act_mode,mode='run'):
    #####################################
    '''
    Parameters:
    #   x:
    #       the tensor of the input data
    #   n_channels:
    #       a numpy array that indicates channel numbers
    #   conv_shape:
    #       numpy array in [nlayer * 2], denoting the kernel size of convolutional weights
    #   conv_stride:
    #       numpy array in [nlayer * 2], denoting the stride size of convolution
    #   pool_shape:
    #       numpy array in [nlayer * 2], denoting the kernel size of pooling layer
    #   pool_stride:
    #       numpy array in [nlayer * 2], denoting the stride size of pooling layer
    #   layers_dim:
    #       the dimensions of all fully-conneted layers
    #       ***It is the user's resposibility to compute the dim of the first flattened layer***
    #   pool_mode:
    #       the list of string indicating pooling methods of each layer. support: 'avg', 'max', 'none'
    #   mode: 'run'(default): return the predicted value
    #         'para': return the parameters for storage
    '''
    #####################################
    conv_weight_list = []
    weight_list = []
    bias_list = []

    x_proceesed = tf.reshape(x,shape = [-1,28,28,1])
    for c_layer in range(len(conv_shape)):
        this_weight = tf.Variable(tf.random_normal([conv_shape[c_layer,0],conv_shape[c_layer,1],
                                                    n_channels[c_layer],n_channels[c_layer+1]]))
        conv_weight_list.append(this_weight)
    for c_layer in range(len(layers_dim)-1):
        this_weight = tf.Variable(tf.random_normal([layers_dim[c_layer],layers_dim[c_layer+1]]))
        weight_list.append(this_weight)
        this_bias = tf.Variable(tf.random_normal([layers_dim[c_layer+1]]))
        bias_list.append(this_bias)

    hidden_data = 0
    #compute the convolutional layers
    for c_layer in range(len(conv_weight_list)):
        if c_layer==0:
            hidden_data = convolution2D(x_proceesed,conv_weight_list[c_layer],
                                        conv_stride[c_layer,0],conv_stride[c_layer,1])
        else:
            hidden_data = convolution2D(hidden_data,conv_weight_list[c_layer],
                                        conv_stride[c_layer,0],conv_stride[c_layer,1])
        #Pooling
        if pool_mode[c_layer]=='avg':
            hidden_data = averagepooling(hidden_data,pool_shape[c_layer,0],pool_shape[c_layer,1],
                                         pool_stride[c_layer,0,],pool_stride[c_layer,1])
        elif pool_mode[c_layer]=='max':
            hidden_data = maxpooling(hidden_data, pool_shape[c_layer, 0], pool_shape[c_layer, 1],
                                         pool_stride[c_layer, 0,], pool_stride[c_layer, 1])
        elif pool_mode[c_layer]=='none':
            pass
        else:
            raise ValueError('Given input pooling method not recognized')
    #Compute the fully connected layers and output the result
    output_process_data = tf.reshape(hidden_data,[-1,layers_dim[0]])

    for c_layer in range(len(weight_list)):
        output_process_data = tf.add(tf.matmul(output_process_data,weight_list[c_layer]),bias_list[c_layer])
        if act_mode[c_layer] == 'relu':
            output_process_data = tf.nn.relu(output_process_data)
        elif act_mode[c_layer] == 'sigmoid':
            output_process_data = tf.nn.sigmoid(output_process_data)
        elif act_mode[c_layer] == 'none':
            pass
        else:
            raise ValueError('Given input activation method not recognized')

    if (mode=='run'):
        return output_process_data
    elif (mode=='para'):
        return conv_weight_list, weight_list, bias_list
    else:
        raise ValueError('Input mode unrecognized!')



def neural_network_storage(weight_conv,weight,bias):
    # define the dict to save the data
    key_name_list = []
    for c_layer in range(len(weight_conv)):
        current_conv_weight_name = 'W_conv_' + str(c_layer + 1)
        key_name_list.append(current_conv_weight_name)
    for c_layer in range(len(weight)):
        current_weight_name = 'W_' + str(c_layer + 1)
        current_bias_name = 'b_' + str(c_layer + 1)
        key_name_list.append(current_weight_name)
        key_name_list.append(current_bias_name)
    parameter_dict = dict.fromkeys(key_name_list, 0)
    # return the parameter list
    print('storing the parameters...')
    for c_layer in range(len(weight)):
        current_weight_name = 'W_' + str(c_layer + 1)
        current_bias_name = 'b_' + str(c_layer + 1)
        parameter_dict[current_weight_name] = weight[c_layer].eval().tolist()
        parameter_dict[current_bias_name] = bias[c_layer].eval().tolist()
    for c_layer in range(len(weight_conv)):
        current_conv_weight_name = 'W_conv_' + str(c_layer + 1)
        parameter_dict[current_conv_weight_name] = weight_conv[c_layer].eval().tolist()
    with open(save_path+'CNN_para.json', 'w') as fp:
            json.dump(parameter_dict, fp)


def train_CNN_neural_network(x):
    # Feed_Forward to get the prediction
    prediction = CNN_network(x,n_channels=channel_dict,conv_shape=conv_kernel_dict,
                             conv_stride=conv_stride_dict,pool_shape=pool_kernel_dict,pool_stride=pool_stride_dict,
                             layers_dim=dim_indic,pool_mode=pool_method,act_mode=activation_method,mode='run')
    #store the parameters
    conv_weight, weight, bias = CNN_network(x,n_channels=channel_dict,conv_shape=conv_kernel_dict,
                             conv_stride=conv_stride_dict,pool_shape=pool_kernel_dict,pool_stride=pool_stride_dict,
                             layers_dim=dim_indic,pool_mode=pool_method,act_mode=activation_method,mode='para')
    # Estimating the Loss
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    # defineing optimiser
    optimiser = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)
    # checking the accuracy
    correction_check = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    accuracy_test = tf.reduce_mean(tf.cast(correction_check, 'float'))
    previous_accuracy = 0
    with tf.Session().as_default() as sess:
        sess.run(tf.global_variables_initializer())
        for cEpoch in range(nEpochs):
            current_Epoch_Loss = 0
            # Get how many batches we need for each epoch
            random_index = random.sample(range(train_amount), train_amount)
            for i in range(int(train_amount // batch_size)):
                current_x = train_x[random_index[i*batch_size:(i+1)*batch_size]]
                current_y = train_y[random_index[i*batch_size:(i+1)*batch_size]]
                _, currentloss = sess.run([optimiser, loss], feed_dict={x: current_x, y: current_y})
                current_Epoch_Loss += currentloss
            print('The', cEpoch + 1, 'th out of ', nEpochs, 'Epochs in total has finished and loss in this epoch is',
                  current_Epoch_Loss)
            current_train_accuracy = accuracy_test.eval({x: train_x, y: train_y})
            current_test_accuracy = accuracy_test.eval({x: test_x, y: test_y})
            print('Train Accuracy=', current_train_accuracy)
            print('Test Accuracy=', current_test_accuracy)
            if (previous_accuracy > current_test_accuracy) and (current_test_accuracy>0.97):
                neural_network_storage(conv_weight,weight=weight,bias=bias)
                break
            previous_accuracy = current_test_accuracy
        print('Program finished!')


train_CNN_neural_network(x)