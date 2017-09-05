import numpy as np
import tensorflow as tf
import os
from tensorflow.examples.tutorials.mnist import input_data
import random
import json

save_path = '../parameter/'

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
dim_indic = np.array([784,n_Nodes_Hidden_1,n_Nodes_Hidden_2,n_Classes])
activation_method = ['relu','relu','none']

def save_parameter(parameter,name):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    parameter.tofile(save_path+name)


def save_model(session):
    if not os.path.exists('./double_nonlinear_model/'):
        os.mkdir('./double_nonlinear_model/')
    saver = tf.train.Saver()
    saver.save(session, './double_nonlinear_model/Non-linear_Tensorflow2.checkpoint')

# Get the x input as float and reshape input
x = tf.placeholder("float", [None, 784])
y = tf.placeholder("float", [None, 10])

# Define the Neural Network Model
def DNN_network(x,layers_dim,activation_mode,mode='run'):
    #####################################
    #Parameters:
    #   x:
    #       the tensor of the input data
    #   layers_dim:
    #       the dimensions of all layers, including input (data dim) and output dim
    #   activation_mode:
    #       the list of string indicating activations of eah layer. support: 'relu', 'sigmoid', 'none'
    #   mode: 'run'(default): return the predicted value
    #         'para': return the parameters for storage
    #####################################
    weight_list = []
    bias_list = []
    for c_layer in range(len(layers_dim)-1):
        this_weight = tf.Variable(tf.random_normal([layers_dim[c_layer],layers_dim[c_layer+1]]))
        weight_list.append(this_weight)
        this_bias = tf.Variable(tf.random_normal([layers_dim[c_layer+1]]))
        bias_list.append(this_bias)

    hidden_data = 0
    for c_layers in range(len(layers_dim)-1):
        if c_layers==0:
            hidden_data = tf.add(tf.matmul(x, weight_list[c_layers]), bias_list[c_layers])
        else:
            hidden_data = tf.add(tf.matmul(hidden_data, weight_list[c_layers]), bias_list[c_layers])
        if activation_mode[c_layers]=='relu':
            hidden_data = tf.nn.relu(hidden_data)
        elif activation_mode[c_layers]=='sigmoid':
            hidden_data = tf.nn.sigmoid(hidden_data)
        elif activation_mode[c_layers]=='none':
            pass
        else:
            raise ValueError('Given input activation method not recognized')

    if (mode=='run'):
        return hidden_data
    elif (mode=='para'):
        return weight_list, bias_list
    else:
        raise ValueError('Input mode unrecognized!')



def neural_network_storage(x,weight,bias):
    # define the dict to save the data
    key_name_list = []
    for c_layer in range(len(dim_indic) - 1):
        current_weight_name = 'W_' + str(c_layer + 1)
        current_bias_name = 'b_' + str(c_layer + 1)
        key_name_list.append(current_weight_name)
        key_name_list.append(current_bias_name)
    parameter_dict = dict.fromkeys(key_name_list, 0)
    # return the parameter list
    print('storing the parameters...')
    for c_layer in range(len(dim_indic) - 1):
        current_weight_name = 'W_' + str(c_layer + 1)
        current_bias_name = 'b_' + str(c_layer + 1)
        parameter_dict[current_weight_name] = weight[c_layer].eval().tolist()
        parameter_dict[current_bias_name] = bias[c_layer].eval().tolist()
    with open(save_path+'DNN_para.json', 'w') as fp:
            json.dump(parameter_dict, fp)


def train_DNN_neural_network(x):
    # Feed_Forward to get the prediction
    prediction = DNN_network(x,layers_dim=dim_indic,activation_mode=activation_method,mode='run')
    #store the parameters
    weight, bias = DNN_network(x, layers_dim=dim_indic, activation_mode=activation_method, mode='para')
    # Estimating the Loss
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    # defineing optimiser
    optimiser = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)
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
            if (previous_accuracy > current_test_accuracy) and (current_test_accuracy>0.96):
                neural_network_storage(x,weight=weight,bias=bias)
                break
            previous_accuracy = current_test_accuracy
        print('Program finished!')


train_DNN_neural_network(x)