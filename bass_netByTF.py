import numpy as np 
import scipy as sci 
import matplotlib.pyplot as plt
import scipy.io
from sklearn.metrics import confusion_matrix

#import helper
from random import randint

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('--data', type=str, default='Indian_pines')
parser.add_argument('--patch_size', type=int, default=3)
parser.add_argument('--library', type=str, default='tensorflow')
opt = parser.parse_args()

import tensorflow as tf
import keras

#Load MATLAB pre-processed image data
TRAIN = scipy.io.loadmat("./data/" + opt.data + "_Train_patch_" + str(opt.patch_size) + ".mat")
VALIDIATION = scipy.io.loadmat("./data/" + opt.data + "_Val_patch_" + str(opt.patch_size) + ".mat")
TEST = scipy.io.loadmat("./data/" + opt.data + "_Test_patch_" + str(opt.patch_size) + ".mat")

#Extract data and label from MATLAB file
training_data, training_label  = TRAIN['train_patch'], TRAIN['train_labels']
validiation_data, validiation_label = VALIDIATION['val_patch'], VALIDIATION['val_labels']
test_data, test_label = TEST['test_patch'], TEST['test_labels']

print('\nData input shape')

print('training_data shape' + str(training_data.shape) )
print('training_label shape' + str(training_label.shape) +'\n' )
print('testing_data shape' + str(test_data.shape) )
print('testing_label shape' + str(test_label.shape) +'\n' )

'''
Tensorflow Datagraph (Or neural network) helper functions 
'''
#Functions for creating new tensorflow variables
def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))

#Functions for creating new CNN
def new_conv_2dlayer(input,              # The previous layer.
                    num_input_channels, # Num. channels in prev. layer.
                    filter_size,        # Width and height of each filter.
                    num_filters,        #Number of filters
                    pooling=True):        # Number of filters. 

    shape = [filter_size, filter_size, num_input_channels, num_filters]
    weights = new_weights(shape=shape)
    biases = new_biases(length=num_filters)
    layer = tf.nn.conv2d(input=input, filter=weights, strides=[1, 1, 1, 1], padding='SAME')

    layer += biases
    
    if pooling:
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1, 3, 3, 1],
                               strides=[1, 1, 1, 1],
                               padding='VALID')

    layer = tf.nn.relu(layer)

    return layer, weights

#Function for flattening an array
def flatten_layer(layer):

    layer_shape = layer.get_shape() #layer = [num_images, img_height, img_width, num_channels]
    num_features = layer_shape[1:4].num_elements()
    layer_flat = tf.reshape(layer, [-1, num_features]) #-1 means total size of dimension is unchanged
    
    return layer_flat, num_features


#Function for creating new fully connected layer
def new_fully_connected_layer(input,
                            num_inputs,
                            num_outputs,
                            activation = None):
    
    weights = new_weights(shape = [num_inputs, num_outputs])
    biases = new_biases(length = num_outputs)

    layer = tf.matmul(input,weights) + biases

    if activation != None:
        if activation == 'relu':
            layer = tf.nn.relu(layer)

        elif activation == 'softmax':
            layer = tf.nn.softmax(layer)

    return layer


'''
Define Neural Network Model

'''
# Indian Pine image height, width, channel, number of class = 9

g = tf.Graph()
g1 = g.as_default()

img_entry = tf.placeholder(tf.float32, shape=[None,3,3,220] , name='img_entry')
img_label = tf.placeholder(tf.uint8, shape=[None,9], name='img_label')
image_true_class = tf.argmax(img_label, axis=1)

#Block 1 parameter
filter_size = 1      # Convolution filters are 5 x 5 pixels.
num_filters1 = 220 

#Block 1 
layer_blk1, weights_blk1 = new_conv_2dlayer(input = img_entry,
                                        num_input_channels= 220,
                                        filter_size = filter_size,
                                        num_filters = num_filters1,
                                        pooling = False )


#Fake Block 2 parameters
filter_size2 = 3
num_filters2 = 600

fake_block2, weights_conv2 = new_conv_2dlayer(input = layer_blk1,
                                            num_input_channels = num_filters1,
                                            filter_size=filter_size2,
                                            num_filters=num_filters2,
                                            pooling = True )
                                        

#Block 3 entry point 
block3_entry, num_features = flatten_layer(fake_block2)


#Fully connected layer from 600 -> 100
block3 = new_fully_connected_layer( input= block3_entry,
                                    num_inputs = num_features,
                                    num_outputs = 100,
                                    activation = 'relu')

#Fully connected layer from 100 -> 9
final_layer = new_fully_connected_layer( input = block3,
                                        num_inputs = 100,
                                        num_outputs = 9,
                                        activation = None)
                                    

#Result judgement by Softmax layer
#Predicted class
y_predict = tf.nn.softmax(final_layer)
y_predict_class = tf.argmax(y_predict, axis=1) 

#Summary
print('---------BASS-net structure Summary---------')
print(layer_blk1)
print(fake_block2)
print(block3_entry)
print(block3)
print(final_layer)
print(y_predict)
print('\n\n')


#Cost function to be optimised
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=final_layer,
                                                        labels=img_label)
cost = tf.reduce_mean(cross_entropy)

#Optimisation function
optimizer = tf.train.AdamOptimizer(learning_rate=0.0005).minimize(cost)

#Performance measure
correction = tf.equal(y_predict_class, image_true_class)
accuracy = tf.reduce_mean(tf.cast(correction, tf.float16))


'''
Train, test and Main function

'''
#When we test on test-image
'''
def print_confusion_matrix(true_class, predicted_class):
''' 


def train(num_iterations, train_batch_size = 200):   
    global total_iterations
    for i in range(total_iterations, total_iterations + num_iterations):
        
        idx = randint(1, 1400)
        train_batch = training_data[idx:idx+train_batch_size]
        train_batch_label = training_label[idx:idx+train_batch_size]
        
        feed_dict_train = {img_entry: train_batch, img_label:train_batch_label }
        
        session.run(optimizer, feed_dict=feed_dict_train)
        
        if i % 1 == 0:
            acc = session.run(accuracy, feed_dict=feed_dict_train)
            msg = "Optimization Iteration: {0:>6}, Training Accuracy: {1:>6.1%}"

            print(msg.format(i + 1, acc))
        
    total_iterations +=  num_iterations

def test(test_batch_size=500):
    print('\n -----Test----')
    idx = randint(1, 2000)
    test_img_batch = test_data[idx : idx+test_batch_size]
    test_img_label = test_label[idx: idx+test_batch_size]

    feed_dict_test = {img_entry: test_img_batch, img_label:test_img_label }
    class_pred = np.zeros(shape=test_batch_size, dtype=np.int)
    class_pred[:500] = session.run(y_predict_class, feed_dict=feed_dict_test)
    
    class_true = np.argmax(test_img_label, axis=1)

    correct = (class_true == class_pred).sum()
    accuracy_test = float(correct)/test_batch_size
    print('Accuracy at test: \t' + str(accuracy_test *100) +'%')
    
    #print_confusion_matrix(true_class =class_true, predicted_class=class_pred )
    print('Confusion matrix')
    con_mat = confusion_matrix(class_true, class_pred)
    print(con_mat)

saver = tf.train.Saver()
with tf.Session() as session:
    session.run(tf.global_variables_initializer())

    total_iterations = 0

    train(num_iterations = 6000, train_batch_size = 200)
    #saver.save(session, "SIP_trainedModel/")

    test()
    session.close()
    print('------- End -------')