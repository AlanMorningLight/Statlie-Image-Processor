import numpy as np
import scipy.io
from sklearn.metrics import confusion_matrix
from random import randint, shuffle
from argparse import ArgumentParser
from helper import getValidDataset
import tensorflow as tf

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
training_data, training_label = TRAIN['train_patch'], TRAIN['train_labels']
validation_data, validation_label = VALIDATION['val_patch'], VALIDATION['val_labels']
test_data, test_label = TEST['test_patch'], TEST['test_labels']


getValidDataset(test_data, test_label)
print('\nData input shape')
print('training_data shape' + str(training_data.shape))
print('training_label shape' + str(training_label.shape) + '\n')
print('testing_data shape' + str(test_data.shape))
print('testing_label shape' + str(test_label.shape) + '\n')

SIZE = training_data.shape[0]
HEIGHT = training_data.shape[1]
WIDTH = training_data.shape[2]
BANDS = training_data.shape[3]
NUM_PARALLEL_BAND = 10
BAND_SIZE = BANDS / 10
NUM_CLASS = training_label.shape[1]


# Helper Functions
def create_conv_2dlayer(input,
                        num_input_channels,
                        filter_size,
                        num_output_channel,
                        relu=True,
                        pooling=True):  # Number of filters.

    shape = [filter_size, filter_size, num_input_channels, num_output_channel]
    weights = tf.get_variable('weights', shape=shape, initializer=tf.truncated_normal_initializer(stddev=0.05))
    biases = tf.get_variable('biases', shape=[num_output_channel], initializer=tf.constant_initializer(0.05))

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


def fully_connected_layer(input,
                          num_inputs,
                          num_outputs,
                          activation=None):

    weights = tf.get_variable('weights', shape=[num_inputs, num_outputs])
    biases = tf.get_variable('biases', shape=num_outputs)

    layer = tf.matmul(input, weights) + biases

    if activation is not None:
        if activation == 'relu':
            layer = tf.nn.relu(layer)

        elif activation == 'softmax':
            layer = tf.nn.softmax(layer)

    return layer


def flatten_layer(layer):
    layer_shape = layer.get_shape()  # layer = [num_images, img_height, img_width, num_channels]
    num_features = layer_shape[1:4].num_elements()  # Total number of elements in the network
    layer_flat = tf.reshape(layer, [-1, num_features])  # -1 means total size of dimension is unchanged

    return layer_flat, num_features


def specialized_conv1d(input,
                       filter_width,
                       filter_height,
                       num_output_channels,
                       num_input_channels = 1,
                       relu=True):

    shape = [filter_height, filter_width, num_input_channels, num_output_channels]
    weights = tf.get_variable(name='weights-1D', shape=shape, initializer=tf.truncated_normal_initializer(stddev=0.05))
    biases = tf.get_variable(name='biases-1D', shape=[num_output_channels], initializer=tf.constant_initializer(0.05))

    layer = tf.nn.conv2d(input=input, filter=weights, strides=[1,1,1,1], padding='VALID')
    out_height = input.shape[1] - filter_height + 1

    layer += biases
    layer = tf.reshape(layer, [-1, out_height, num_output_channels, 1])

    if relu:
        layer = tf.nn.relu(layer)

    return layer


def block2_parallel(model):

    layer = model['block2_preprocess']
    with tf.variable_scope('band1'):
        block2_prep = layer[0]
        block2_prep = tf.reshape(block2_prep, (-1, 9, 22, 1))
        block2_prep = tf.transpose(block2_prep, perm=[0, 2, 1, 3])
        with tf.variable_scope('block2_part1'):
            block2_part1 = specialized_conv1d(input=block2_prep, filter_width=9, filter_height=3,
                                              num_output_channels=20, relu=True)
        with tf.variable_scope('block2_part2'):
            block2_part2 = specialized_conv1d(input=block2_part1, filter_width=20, filter_height=3,
                                              num_output_channels=10, relu=True)
        with tf.variable_scope('block2_part3'):
            block2_part3 = specialized_conv1d(input=block2_part2, filter_width=10, filter_height=3,
                                              num_output_channels=10, relu=True)
        with tf.variable_scope('block2_part4'):
            block2_part4 = specialized_conv1d(input=block2_part3, filter_width=10, filter_height=5,
                                              num_output_channels=5, relu=True)
        with tf.variable_scope('block2_part5'):
            block2_part5, _ = flatten_layer(block2_part4)

        stack = tf.concat(block2_part5, axis=1)

    print(stack)
    with tf.variable_scope('band2'):
        block2_prep = layer[1]
        block2_prep = tf.reshape(block2_prep, (-1, 9, 22, 1))
        block2_prep = tf.transpose(block2_prep, perm=[0, 2, 1, 3])
        with tf.variable_scope('block2_part1'):
            block2_part1 = specialized_conv1d(input=block2_prep, filter_width=9, filter_height=3,
                                              num_output_channels=20, relu=True)
        with tf.variable_scope('block2_part2'):
            block2_part2 = specialized_conv1d(input=block2_part1, filter_width=20, filter_height=3,
                                              num_output_channels=10, relu=True)
        with tf.variable_scope('block2_part3'):
            block2_part3 = specialized_conv1d(input=block2_part2, filter_width=10, filter_height=3,
                                              num_output_channels=10, relu=True)
        with tf.variable_scope('block2_part4'):
            block2_part4 = specialized_conv1d(input=block2_part3, filter_width=10, filter_height=5,
                                              num_output_channels=5, relu=True)
        with tf.variable_scope('block2_part5'):
            block2_part5, _ = flatten_layer(block2_part4)
        stack = tf.concat([stack, block2_part5], axis=1)
    print(stack)

    with tf.variable_scope('band3'):
        block2_prep = layer[2]
        block2_prep = tf.reshape(block2_prep, (-1, 9, 22, 1))
        block2_prep = tf.transpose(block2_prep, perm=[0, 2, 1, 3])
        with tf.variable_scope('block2_part1'):
            block2_part1 = specialized_conv1d(input=block2_prep, filter_width=9, filter_height=3,
                                              num_output_channels=20, relu=True)
        with tf.variable_scope('block2_part2'):
            block2_part2 = specialized_conv1d(input=block2_part1, filter_width=20, filter_height=3,
                                              num_output_channels=10, relu=True)
        with tf.variable_scope('block2_part3'):
            block2_part3 = specialized_conv1d(input=block2_part2, filter_width=10, filter_height=3,
                                              num_output_channels=10, relu=True)
        with tf.variable_scope('block2_part4'):
            block2_part4 = specialized_conv1d(input=block2_part3, filter_width=10, filter_height=5,
                                              num_output_channels=5, relu=True)
        with tf.variable_scope('block2_part5'):
            block2_part5, _ = flatten_layer(block2_part4)
        stack = tf.concat([stack, block2_part5], axis=1)
    print(stack)

    with tf.variable_scope('band4'):
        block2_prep = layer[3]
        block2_prep = tf.reshape(block2_prep, (-1, 9, 22, 1))
        block2_prep = tf.transpose(block2_prep, perm=[0, 2, 1, 3])
        with tf.variable_scope('block2_part1'):
            block2_part1 = specialized_conv1d(input=block2_prep, filter_width=9, filter_height=3,
                                              num_output_channels=20, relu=True)
        with tf.variable_scope('block2_part2'):
            block2_part2 = specialized_conv1d(input=block2_part1, filter_width=20, filter_height=3,
                                              num_output_channels=10, relu=True)
        with tf.variable_scope('block2_part3'):
            block2_part3 = specialized_conv1d(input=block2_part2, filter_width=10, filter_height=3,
                                              num_output_channels=10, relu=True)
        with tf.variable_scope('block2_part4'):
            block2_part4 = specialized_conv1d(input=block2_part3, filter_width=10, filter_height=5,
                                              num_output_channels=5, relu=True)
        with tf.variable_scope('block2_part5'):
            block2_part5, _ = flatten_layer(block2_part4)
        stack = tf.concat([stack, block2_part5], axis=1)
    print(stack)

    with tf.variable_scope('band5'):
        block2_prep = layer[4]
        block2_prep = tf.reshape(block2_prep, (-1, 9, 22, 1))
        block2_prep = tf.transpose(block2_prep, perm=[0, 2, 1, 3])
        with tf.variable_scope('block2_part1'):
            block2_part1 = specialized_conv1d(input=block2_prep, filter_width=9, filter_height=3,
                                              num_output_channels=20, relu=True)
        with tf.variable_scope('block2_part2'):
            block2_part2 = specialized_conv1d(input=block2_part1, filter_width=20, filter_height=3,
                                              num_output_channels=10, relu=True)
        with tf.variable_scope('block2_part3'):
            block2_part3 = specialized_conv1d(input=block2_part2, filter_width=10, filter_height=3,
                                              num_output_channels=10, relu=True)
        with tf.variable_scope('block2_part4'):
            block2_part4 = specialized_conv1d(input=block2_part3, filter_width=10, filter_height=5,
                                              num_output_channels=5, relu=True)
        with tf.variable_scope('block2_part5'):
            block2_part5, _ = flatten_layer(block2_part4)
        stack = tf.concat([stack, block2_part5], axis=1)
    print(stack)

    with tf.variable_scope('band6'):
        block2_prep = layer[5]
        block2_prep = tf.reshape(block2_prep, (-1, 9, 22, 1))
        block2_prep = tf.transpose(block2_prep, perm=[0, 2, 1, 3])
        with tf.variable_scope('block2_part1'):
            block2_part1 = specialized_conv1d(input=block2_prep, filter_width=9, filter_height=3,
                                              num_output_channels=20, relu=True)
        with tf.variable_scope('block2_part2'):
            block2_part2 = specialized_conv1d(input=block2_part1, filter_width=20, filter_height=3,
                                              num_output_channels=10, relu=True)
        with tf.variable_scope('block2_part3'):
            block2_part3 = specialized_conv1d(input=block2_part2, filter_width=10, filter_height=3,
                                              num_output_channels=10, relu=True)
        with tf.variable_scope('block2_part4'):
            block2_part4 = specialized_conv1d(input=block2_part3, filter_width=10, filter_height=5,
                                              num_output_channels=5, relu=True)
        with tf.variable_scope('block2_part5'):
            block2_part5, _ = flatten_layer(block2_part4)
        stack = tf.concat([stack, block2_part5], axis=1)
    print(stack)

    with tf.variable_scope('band7'):
        block2_prep = layer[6]
        block2_prep = tf.reshape(block2_prep, (-1, 9, 22, 1))
        block2_prep = tf.transpose(block2_prep, perm=[0, 2, 1, 3])
        with tf.variable_scope('block2_part1'):
            block2_part1 = specialized_conv1d(input=block2_prep, filter_width=9, filter_height=3,
                                              num_output_channels=20, relu=True)
        with tf.variable_scope('block2_part2'):
            block2_part2 = specialized_conv1d(input=block2_part1, filter_width=20, filter_height=3,
                                              num_output_channels=10, relu=True)
        with tf.variable_scope('block2_part3'):
            block2_part3 = specialized_conv1d(input=block2_part2, filter_width=10, filter_height=3,
                                              num_output_channels=10, relu=True)
        with tf.variable_scope('block2_part4'):
            block2_part4 = specialized_conv1d(input=block2_part3, filter_width=10, filter_height=5,
                                              num_output_channels=5, relu=True)
        with tf.variable_scope('block2_part5'):
            block2_part5, _ = flatten_layer(block2_part4)
        stack = tf.concat([stack, block2_part5], axis=1)
    print(stack)

    with tf.variable_scope('band8'):
        block2_prep = layer[7]
        block2_prep = tf.reshape(block2_prep, (-1, 9, 22, 1))
        block2_prep = tf.transpose(block2_prep, perm=[0, 2, 1, 3])
        with tf.variable_scope('block2_part1'):
            block2_part1 = specialized_conv1d(input=block2_prep, filter_width=9, filter_height=3,
                                              num_output_channels=20, relu=True)
        with tf.variable_scope('block2_part2'):
            block2_part2 = specialized_conv1d(input=block2_part1, filter_width=20, filter_height=3,
                                              num_output_channels=10, relu=True)
        with tf.variable_scope('block2_part3'):
            block2_part3 = specialized_conv1d(input=block2_part2, filter_width=10, filter_height=3,
                                              num_output_channels=10, relu=True)
        with tf.variable_scope('block2_part4'):
            block2_part4 = specialized_conv1d(input=block2_part3, filter_width=10, filter_height=5,
                                              num_output_channels=5, relu=True)
        with tf.variable_scope('block2_part5'):
            block2_part5, _ = flatten_layer(block2_part4)
        stack = tf.concat([stack, block2_part5], axis=1)
    print(stack)

    with tf.variable_scope('band9'):
        block2_prep = layer[8]
        block2_prep = tf.reshape(block2_prep, (-1, 9, 22, 1))
        block2_prep = tf.transpose(block2_prep, perm=[0, 2, 1, 3])
        with tf.variable_scope('block2_part1'):
            block2_part1 = specialized_conv1d(input=block2_prep, filter_width=9, filter_height=3,
                                              num_output_channels=20, relu=True)
        with tf.variable_scope('block2_part2'):
            block2_part2 = specialized_conv1d(input=block2_part1, filter_width=20, filter_height=3,
                                              num_output_channels=10, relu=True)
        with tf.variable_scope('block2_part3'):
            block2_part3 = specialized_conv1d(input=block2_part2, filter_width=10, filter_height=3,
                                              num_output_channels=10, relu=True)
        with tf.variable_scope('block2_part4'):
            block2_part4 = specialized_conv1d(input=block2_part3, filter_width=10, filter_height=5,
                                              num_output_channels=5, relu=True)
        with tf.variable_scope('block2_part5'):
            block2_part5, _ = flatten_layer(block2_part4)
        stack = tf.concat([stack, block2_part5], axis=1)
    print(stack)

    with tf.variable_scope('band10'):
        block2_prep = layer[9]
        block2_prep = tf.reshape(block2_prep, (-1, 9, 22, 1))
        block2_prep = tf.transpose(block2_prep, perm=[0, 2, 1, 3])
        with tf.variable_scope('block2_part1'):
            block2_part1 = specialized_conv1d(input=block2_prep, filter_width=9, filter_height=3,
                                              num_output_channels=20, relu=True)
        with tf.variable_scope('block2_part2'):
            block2_part2 = specialized_conv1d(input=block2_part1, filter_width=20, filter_height=3,
                                              num_output_channels=10, relu=True)
        with tf.variable_scope('block2_part3'):
            block2_part3 = specialized_conv1d(input=block2_part2, filter_width=10, filter_height=3,
                                              num_output_channels=10, relu=True)
        with tf.variable_scope('block2_part4'):
            block2_part4 = specialized_conv1d(input=block2_part3, filter_width=10, filter_height=5,
                                              num_output_channels=5, relu=True)
        with tf.variable_scope('block2_part5'):
            block2_part5, _ = flatten_layer(block2_part4)

        stack = tf.concat([stack, block2_part5], axis=1)
    print(stack)
    return stack


# Define BASSNET archicture
def bassnet(statlieImg, prob):

    # Image_entry are images in format 3 x 3 x 220, Prob = Drop out probability ~ 0.5
    # return a dictionary contains all layer
    sequence = {}
    sequence['inputLayer'] = tf.reshape(statlieImg, [-1,3,3,220])

    with tf.variable_scope('block1_conv1'):
        layer = sequence['inputLayer']
        layer, weight = create_conv_2dlayer(input=layer,
                                            num_input_channels=BANDS,
                                            filter_size=1,
                                            num_output_channel=220,
                                            relu=True, pooling=False)
        sequence['block1_conv1'] = layer

    with tf.variable_scope('block1_conv2'):
        layer = sequence['block1_conv1']
        layer, weight = create_conv_2dlayer(input=layer,
                                            num_input_channels=BANDS,
                                            filter_size=1,
                                            num_output_channel=220,
                                            relu=True, pooling=False)
        sequence['block1_conv2'] = layer

# Block 2 Implementation
    with tf.variable_scope('block2_preprocess_GPU'):
        layer = sequence['block1_conv2']
        layer = tf.reshape(layer, [-1, 9, 220])

        container = tf.split(layer, num_or_size_splits=10, axis=2)
        sequence['block2_preprocess_GPU'] = container

        for i in range(10):
                scope = "BAND_"+str(i)
                with tf.variable_scope(scope):
                    print(tf.get_variable_scope())

    with tf.variable_scope('block2_preprocess'):
        layer = sequence['block1_conv2']
        layer = tf.reshape(layer, [-1, 9, 220])
        layer = tf.split(layer, num_or_size_splits=10, axis=2)
        sequence['block2_preprocess'] = layer

    with tf.variable_scope('block2_parallel'):
        parallel_model = block2_parallel(sequence)
        sequence['block2_end'] = parallel_model

    '''
    with tf.variable_scope('block2'):
        layer = sequence['block2_preprocess']
    
        def condition(time, output_ta_l):
            return time < 10
    
        def body(time, output_ta_l):
            block2_prep = layer[:, :, :, time]
            block2_prep = tf.reshape(block2_prep, (-1, 9, 22, 1))

            block2_prep = tf.transpose(block2_prep, perm=[0, 2, 1, 3])

            with tf.variable_scope('block2_part1'):
                block2_part1 = specialized_conv1d(input=block2_prep, filter_width=9, filter_height=3,
                                                  num_output_channels=20, relu=True)

            with tf.variable_scope('block2_part2'):
                block2_part2 = specialized_conv1d(input=block2_part1, filter_width=20, filter_height=3,
                                                  num_output_channels=10, relu=True)

            with tf.variable_scope('block2_part3'):
                block2_part3 = specialized_conv1d(input=block2_part2, filter_width=10, filter_height=3,
                                                  num_output_channels=10, relu=True)

            with tf.variable_scope('block2_part4'):
                block2_part4 = specialized_conv1d(input=block2_part3, filter_width=10, filter_height=5,
                                                  num_output_channels=5, relu=True)

            with tf.variable_scope('block2_part5'):
                block2_part5, _ = flatten_layer(block2_part4)

            output_ta_l = output_ta_l.write(time, block2_part5)
            return time+1, output_ta_l

        time = 0
        block3_entry = tf.TensorArray(tf.float32, size=10)
    
        _, block3_entry = tf.while_loop(condition, body, loop_vars=[time, block3_entry])

        block3_entry = block3_entry.concat()
        block3_entry3 = tf.reshape(block3_entry, (-1, 600))
        sequence['block3_entry_point'] = block3_entry3

# End of geniue block 2
    '''
# Begin of fake block 2
    with tf.variable_scope('block2_conv1_fake'):
        layer = sequence['block1_conv2']
        layer, weight = create_conv_2dlayer(input=layer,
                                            num_input_channels=220,
                                            filter_size=3,
                                            num_output_channel=600,
                                            relu=True, pooling=True)
        sequence['block2_conv1_fake'] = layer

    with tf.variable_scope('block2_exit_flatten'):
            layer = sequence['block2_conv1_fake']
            layer, number_features = flatten_layer(layer)
            sequence['block2_exit_flatten'] = layer
# End of fake block 2

# Final block 3 layer

    with tf.variable_scope('block3_dense1'):
        layer = sequence['block2_end']
        # layer = sequence['block3_entry_point']
        layer = fully_connected_layer(input=layer,
                                      num_inputs=number_features,
                                      num_outputs=100,
                                      activation='rely')
        layer = tf.nn.dropout(x=layer, keep_prob=prob)
        sequence['block3_dense1'] = layer

    with tf.variable_scope('block3_dense2'):
        layer = sequence['block3_dense1']
        layer = fully_connected_layer(input=layer,
                                      num_inputs=100,
                                      num_outputs=54)
        layer = tf.nn.dropout(x=layer, keep_prob=prob)
        sequence['block3_dense2'] = layer

    with tf.variable_scope('block3_dense3'):
        layer = sequence['block3_dense2']
        layer = fully_connected_layer(input=layer,
                                      num_inputs=54,
                                      num_outputs=9)
        layer = tf.nn.dropout(x=layer, keep_prob=prob)
        sequence['block3_dense3'] = layer

    y_predict = tf.nn.softmax(sequence['block3_dense3'])
    sequence['class_prediction'] = y_predict
    sequence['predict_class_number'] = tf.argmax(y_predict, axis=1)
    return sequence

a =8

graph = tf.Graph()
with graph.as_default():

    img_entry = tf.placeholder(tf.float32, shape=[None, HEIGHT, WIDTH, BANDS], name='img_entry')
    img_label = tf.placeholder(tf.uint8, shape=[None, NUM_CLASS], name='img_label')
    image_true_class = tf.argmax(img_label, axis=1, name="img_true_label")

    prob = tf.placeholder(tf.float32)

    model = bassnet(statlieImg=img_entry, prob=prob)
    final_layer = model['block3_dense3']

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=final_layer,
                                                               labels=img_label)
    cost = tf.reduce_mean(cross_entropy)
    # Optimisation function
    optimizer = tf.train.AdamOptimizer(learning_rate=0.0005).minimize(cost)

    predict_class = model['predict_class_number']
    correction = tf.equal( predict_class, image_true_class)
    accuracy = tf.reduce_mean(tf.cast(correction, tf.float32))

    saver = tf.train.Saver()


    with tf.Session(graph=graph) as session:
        writer = tf.summary.FileWriter("BASSNETlogs/", session.graph)

        if os.path.isdir(model_directory):
            saver.restore(session, 'BASSNET_Trained_model/')

        session.run(tf.global_variables_initializer())

        total_iterations = 0

        def train(num_iterations, train_batch_size=200, s=250, training_data=training_data, training_label=training_label, test_data=test_data, test_label=test_label, ):
            global total_iterations
            for i in range(total_iterations, total_iterations + num_iterations):
                idx = randint(1, 2550)
                for x in range(10):
                    train_batch = training_data[idx*x: idx*x + train_batch_size]
                    train_batch_label = training_label[idx*x:idx*x + train_batch_size]
                    feed_dict_train = {img_entry: train_batch, img_label: train_batch_label, prob: 0.2}
                    session.run(optimizer, feed_dict=feed_dict_train)

                print('Finished training an epoch...')
                if i % 10 == 0:
                    training_data, training_label, test_data, test_label = trainTestSwap(training_data, training_label, test_data, test_label, idx, size=s)
                    # val_x, val_y = getValidDataset(test_data, test_label)
                    val_x, val_y = test_data[:s], test_label[:s]
                    feed_dict_validate = {img_entry: val_x, img_label: val_y, prob: 1.0}
                    acc = session.run(accuracy, feed_dict=feed_dict_validate)
                    msg = "Optimization Iteration: {0:>6}, Training Accuracy: {1:>6.1%}"
                    print(msg.format(i + 1, acc))

            total_iterations += num_iterations


        def test(test_batch_size=validation_data.shape[0]):
            print('\n -----Test----')
            y_predict_class = model['predict_class_number']
            idx = randint(1, 2000)
            test_img_batch = test_data[idx: idx + test_batch_size]
            test_img_label = test_label[idx: idx + test_batch_size]

            feed_dict_test = {img_entry: validation_data, img_label: validation_label, prob: 1.0}
            class_pred = np.zeros(shape=test_batch_size, dtype=np.int)
            class_pred[:test_batch_size] = session.run(y_predict_class, feed_dict=feed_dict_test)

            class_true = np.argmax(validation_label, axis=1)

            correct = (class_true == class_pred).sum()
            accuracy_test = float(correct) / test_batch_size
            print('Accuracy at test: \t' + str(accuracy_test * 100) + '%')

            # print_confusion_matrix(true_class =class_true, predicted_class=class_pred )
            print('Confusion matrix')
            con_mat = confusion_matrix(class_true, class_pred)
            print(con_mat)

        def trainTestSwap(training_data, training_label, test_data, test_label, idx, size=250):
            a, b = test_data[:size], test_label[:size]
            c, d = training_data[idx: idx+size], training_label[idx: idx+size]

            test_data, test_label = test_data[size:], test_label[size:]
            test_data, test_label = np.concatenate((test_data, c), axis=0), np.concatenate((test_label, d), axis=0)

            training_data[idx: idx + size], training_label[idx: idx + size] = a, b

            return training_data, training_label, test_data, test_label


        def cross_validate(training_data, training_label, test_):
            print("This is not necessary as we have large dataset and it's expensive to do!")


        train(num_iterations=12000, train_batch_size=200)
        saver.save(session, model_directory)
        test()
       # trainTestSwap(training_data, training_label, test_data, test_label,  1, size=250)
        print('End session')
