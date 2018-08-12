import numpy as np
import scipy.io
from sklearn.metrics import confusion_matrix
from random import randint
import tensorflow as tf

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('--data', type=str, default='Indian_pines')
parser.add_argument('--patch_size', type=int, default=3)
parser.add_argument('--library', type=str, default='tensorflow')
opt = parser.parse_args()

import os
model_directory = os.path.join(os.getcwd(), 'BASSNET_Trained_model/')

# Load MATLAB pre-processed image data
try:
    TRAIN = scipy.io.loadmat("./data/" + opt.data + "_Train_patch_" + str(opt.patch_size) + ".mat")
    VALIDATION = scipy.io.loadmat("./data/" + opt.data + "_Val_patch_" + str(opt.patch_size) + ".mat")
    TEST = scipy.io.loadmat("./data/" + opt.data + "_Test_patch_" + str(opt.patch_size) + ".mat")

except NameError:
    raise print('--data options are: Indian_pines, Salinas, KSC, Botswana')


# Extract data and label from MATLAB file
training_data, training_label  = TRAIN['train_patch'], TRAIN['train_labels']
validation_data, validation_label = VALIDATION['val_patch'], VALIDATION['val_labels']
test_data, test_label = TEST['test_patch'], TEST['test_labels']

print('\nData input shape')
print('training_data shape' + str(training_data.shape) )
print('training_label shape' + str(training_label.shape) + '\n')
print('testing_data shape' + str(test_data.shape) )
print('testing_label shape' + str(test_label.shape) + '\n')


'''
TensorFlow data-flow graph (Or neural network) helper functions 
'''

# Functions for creating new tensorflow variables
def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))


def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))


# Functions for creating new CNN
def conv_2dlayer(input,
                 num_input_channels,
                 filter_size,
                 num_output_channel,
                 relu=True,
                 pooling=True):        # Number of filters.

    shape = [filter_size, filter_size, num_input_channels, num_output_channel]
    weights = new_weights(shape=shape)
    biases = new_biases(length=num_output_channel)

    # input = [batch, in_height, in_width, in_channels]
    # filter = [filter_height, filter_width, in_channels, out_channels]
    layer = tf.nn.conv2d(input=input, filter=weights, strides=[1, 1, 1, 1], padding='SAME')

    layer += biases
    
    if pooling:
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1, 3, 3, 1],
                               strides=[1, 1, 1, 1],
                               padding='VALID')

    if relu:
        layer = tf.nn.relu(layer)

    return layer, weights


# Special function for block2 configuration
def so_called_conv_1dlayer(name,
                           input,
                           filter_width,
                           filter_height,
                           num_output_channels,
                           num_input_channels=1,
                           relu=True):

    # input = [batch, height, in_width, channels]    #For example: ?,22,9,1
    # filter = [filter_height, filter_width, in_channels, out_channels]  For example 3,9,1,20
    # tf.Variable(tf.truncated_normal(shape, stddev=0.05))
    # tf.Variable(tf.constant(0.05, shape=[length]))

    shape = [filter_height, filter_width, num_input_channels, num_output_channels]
    weights = tf.get_variable(name=name+'w', shape=shape, initializer=tf.truncated_normal_initializer(stddev=0.05))
    biases = tf.get_variable(name=name+'b', shape=[num_output_channels], initializer=tf.constant_initializer(0.05))

    layer = tf.nn.conv2d(input=input, filter=weights, strides=[1,1,1,1], padding='VALID')

    out_height = input.shape[1] - filter_height + 1

    layer += biases

    layer = tf.reshape(layer, [-1, out_height, num_output_channels, 1])

    if relu:
        layer = tf.nn.relu(layer)

    return layer, weights


# Function for flattening an array
def flatten_layer(layer):
    layer_shape = layer.get_shape() #layer = [num_images, img_height, img_width, num_channels]
    num_features = layer_shape[1:4].num_elements() #Total number of elements in the network
    layer_flat = tf.reshape(layer, [-1, num_features]) #-1 means total size of dimension is unchanged
    
    return layer_flat, num_features


# Function for creating new fully connected layer
def fully_connected_layer(input,
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
-----------------------Define Neural Network Model -------------------------
'''
HEIGHT = training_data.shape[1]
WIDTH = training_data.shape[2]
BANDS = training_data.shape[3]
NUM_PARALLEL_BAND = 10
small_Bands = BANDS/10

NUM_CLASS = training_label.shape[1]

img_entry = tf.placeholder(tf.float32, shape=[None,HEIGHT,WIDTH,BANDS] , name='img_entry')
img_label = tf.placeholder(tf.uint8, shape=[None,NUM_CLASS], name='img_label')
image_true_class = tf.argmax(img_label, axis=1)


# Block 1 parameter
filter_size = 1
num_filters1 = 220 

# Block 1 2d-Convlution1
layer_blk1_p1, _ = conv_2dlayer(input=img_entry,
                                num_input_channels=BANDS,
                                filter_size=filter_size,
                                num_output_channel=num_filters1,
                                relu=True,
                                pooling=False)
# Block 2 2d-Covolution
layer_blk1_p2, _ = conv_2dlayer(input=layer_blk1_p1,
                                num_input_channels=BANDS,
                                filter_size=filter_size,
                                num_output_channel=220,
                                relu=True,
                                pooling=False)


block2_layer_prep1 = tf.reshape(layer_blk1_p2, [-1, 9, 220])
block2_layer_prep2 = tf.reshape(block2_layer_prep1, [-1, 9, 22, 10])


def condition(time, output_ta_l):
    return tf.less(time, 10)


def body(time, output_ta_l):

    block2_prep3 = block2_layer_prep2[:, :, :, time]

    block2_prep4 = tf.reshape(block2_prep3, (-1, 9, 22, 1))
    block2_prep5 = tf.transpose(block2_prep4, perm=[0, 2, 1, 3])

    block2_part1, _ = so_called_conv_1dlayer(name="b2_1",
                                             input=block2_prep5,
                                             filter_width=9,
                                             filter_height=3,
                                             num_output_channels=20,
                                             relu=True)

    block2_part2, _ = so_called_conv_1dlayer(name="b2_2",
                                             input=block2_part1,
                                             filter_width=20,
                                             filter_height=3,
                                             num_output_channels=10,
                                             relu=True)

    block2_part3, _ = so_called_conv_1dlayer(name="b2_3",
                                             input=block2_part2,
                                             filter_width=10,
                                             filter_height=3,
                                             num_output_channels=10,
                                             relu=True)

    block2_part4, _ = so_called_conv_1dlayer(name="b2_4",
                                             input=block2_part3,
                                             filter_width=10,
                                             filter_height=5,
                                             num_output_channels=5,
                                             relu=True)

    block2_final1, output_element = flatten_layer(block2_part4)
    output_ta_l = output_ta_l.write(time, block2_final1)

    return time + 1, output_ta_l


time = tf.constant(0)
block3_entry = tf.TensorArray(tf.float32, size=10)

time, block3_entry_e = tf.while_loop(condition, body, loop_vars=[time, block3_entry])

block3_entry2 = block3_entry_e.concat()
block3_entry3 = tf.reshape(block3_entry2,(-1,600))


# --------fake Block 2 parameters
filter_size2 = 3
num_filters2 = 600

fake_block2, weights_conv2 = conv_2dlayer(input=layer_blk1_p2,
                                          num_input_channels=num_filters1,
                                          filter_size=filter_size2,
                                          num_output_channel=num_filters2,
                                          relu=True,
                                          pooling=True)

fake_block3_entry, num_features = flatten_layer(fake_block2)
# ----------replace------------


# Fully connected layer from 600 -> 100
block3_mid = fully_connected_layer(input=block3_entry3,   #fake_block3_entry, block3_entry3
                               num_inputs=600,
                               num_outputs=100,
                               activation='relu')

# Fully connected layer from 100 -> 9
final_layer = fully_connected_layer(input=block3_mid,
                                    num_inputs=100,
                                    num_outputs=NUM_CLASS,
                                    activation=None)
                                    

y_predict = tf.nn.softmax(final_layer)
y_predict_class = tf.argmax(y_predict, axis=1) 

# Summary
print('-------------BASS-net structure Summary------------')
print('Block1_part1 ', layer_blk1_p1)
print('Block2_part2 ', layer_blk1_p2)
print('Block2 prep1 ', block2_layer_prep1)
print('Block2 prep2 ', block2_layer_prep2)
'''
print('Block2 prep3 ', block2_prep3)
print('Block2 part1 ', block2_part1)
print('Block2 part2 ', block2_part2)
print('Block2 part3 ', block2_part3)
print('Block2 part4 ', block2_part4)
print('block2 final 1 ', block2_final1)
print('Block2 final1 ', block2_final1)
'''
print('Block3_entry ', block3_entry)
print('Block3_Dense ', block3_mid)
print('Blcok3_Final ', final_layer)
print('Prediction', y_predict)
print('\n\n')
print('-------------End of Summary------------')

# Cost function to be optimised
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=final_layer,
                                                           labels=img_label)
cost = tf.reduce_mean(cross_entropy)

# Optimisation function
optimizer = tf.train.AdamOptimizer(learning_rate=0.0005).minimize(cost)

# Performance measure
correction = tf.equal(y_predict_class, image_true_class)
accuracy = tf.reduce_mean(tf.cast(correction, tf.float16))


'''
Train, test and Main function
'''

def train(num_iterations, train_batch_size = 1600):
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
    
    # print_confusion_matrix(true_class =class_true, predicted_class=class_pred )
    print('Confusion matrix')
    con_mat = confusion_matrix(class_true, class_pred)
    print(con_mat)


saver = tf.train.Saver()

# Run dataflow graph
with tf.Session() as session:

    writer = tf.summary.FileWriter("logs/", session.graph)
    # saver.restore(session, model_directory)
    session.run(tf.global_variables_initializer())
    total_iterations = 0

    train(num_iterations=200, train_batch_size=200)
    saver.save(session, model_directory)

    test()
    print(time.eval())
    session.close()
    print('------- End -------')



'''
block2_prep3 = block2_layer_prep2[:, :, :, 1]
block2_prep3 = tf.reshape(block2_prep3,(-1, 9, 22, 1))
block2_prep3 = tf.transpose(block2_prep3, perm=[0, 2, 1, 3])


block2_part1, _ = so_called_conv_1dlayer(name="b2_1",
                                         input=block2_prep3,
                                         filter_width=9,
                                         filter_height=3,
                                         num_output_channels=20,
                                         relu=True)

block2_part2, _ = so_called_conv_1dlayer(name="b2_2",
                                         input=block2_part1,
                                         filter_width=20,
                                         filter_height=3,
                                         num_output_channels=10,
                                         relu=True)

block2_part3, _ = so_called_conv_1dlayer(name="b2_3",
                                         input=block2_part2,
                                         filter_width=10,
                                         filter_height=3,
                                         num_output_channels=10,
                                         relu=True)

block2_part4, _ = so_called_conv_1dlayer(name="b2_4",
                                         input=block2_part3,
                                         filter_width=10,
                                         filter_height=5,
                                         num_output_channels=5,
                                         relu=True)

block2_final1, output_element = flatten_layer(block2_part4)
'''
